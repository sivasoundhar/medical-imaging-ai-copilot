"""LUNA16 CT candidate dataset: manifest building from `candidates.csv`,
series-level split validation, 3D patch extraction, and the torch Dataset
wrapper for nodule-candidate classification training (Day 6).

Dataset-specific notes (LUNA16, confirmed against the real files present
in `Data/` — see PROGRESS_LOG.md Day 6 entry for the exact numbers):
  - `candidates.csv` spans the FULL LUNA16 dataset (10 subsets, ~551k
    rows); only subset0's 89 `.mhd`/`.raw` volumes are available locally.
    `load_candidates()` filters to seriesuids actually present on disk —
    use `available_series_ids()` for that, don't trust the CSV alone.
  - `candidates.csv`'s `class` column (0 = non-nodule, 1 = nodule
    candidate) is the label source for this classification task — NOT
    `annotations.csv`, which lists only confirmed nodules (with
    diameter) and has no negative examples, so it can't train a
    classifier by itself. `annotations.csv` is kept around for Day 7
    localization evaluation, not wired into training here.
  - Severe class imbalance is real, not a hypothetical to guard against:
    within subset0's candidates, confirmed 56,816 non-nodule vs 122
    nodule rows (~466:1). Not addressed inside this module — that's a
    training-time (sampling/loss-weighting) concern, handled in
    `training/train_3d.py`, mirroring how `dataset_2d.py` stays split/
    manifest logic only and leaves DataLoader construction to the
    training script.
"""
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from src.preprocessing.preprocess_3d import (
    DEFAULT_HU_WINDOW,
    DEFAULT_SPACING_MM,
    apply_hu_window,
    hu_convert,
    load_ct_volume,
    resample_isotropic,
)

CROP_SIZE_VOXELS = (32, 48, 48)  # (D, H, W), LUNA16-standard candidate patch size at 1mm spacing


class LeakageError(ValueError):
    """Raised when the same seriesuid appears in more than one split."""


@dataclass
class CandidateEntry:
    seriesuid: str
    coord_xyz: tuple[float, float, float]  # world coordinates (mm), CSV column order
    label: int  # 0 = non-nodule, 1 = nodule candidate (candidates.csv "class")


def available_series_ids(dataset_root: str | Path) -> set[str]:
    """seriesuids for every `.mhd` volume actually present under
    `dataset_root` (e.g. `Data/subset0`)."""
    return {p.stem for p in Path(dataset_root).glob("*.mhd")}


def load_candidates(
    csv_path: str | Path, available_series: set[str] | None = None
) -> list[CandidateEntry]:
    """Parse `candidates.csv`. Pass `available_series` (see
    `available_series_ids`) to filter LUNA16's full-dataset CSV down to
    just the series whose volumes actually exist locally — required
    since we only have subset0, not all 10 subsets."""
    entries = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if available_series is not None and row["seriesuid"] not in available_series:
                continue
            entries.append(
                CandidateEntry(
                    seriesuid=row["seriesuid"],
                    coord_xyz=(
                        float(row["coordX"]),
                        float(row["coordY"]),
                        float(row["coordZ"]),
                    ),
                    label=int(row["class"]),
                )
            )
    return entries


def validate_no_leakage(*splits: list[CandidateEntry]) -> None:
    """Raise LeakageError if any seriesuid appears in more than one of
    the given splits."""
    seen: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        series_ids = {entry.seriesuid for entry in split}
        for sid in series_ids:
            if sid in seen and seen[sid] != split_index:
                raise LeakageError(
                    f"Series '{sid}' appears in more than one split "
                    f"(split {seen[sid]} and split {split_index}) — leakage."
                )
            seen[sid] = split_index


def build_series_safe_splits(
    candidates: list[CandidateEntry],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[CandidateEntry], list[CandidateEntry], list[CandidateEntry]]:
    """Split by seriesuid (one series = one patient scan), not by
    candidate row — otherwise candidates from the same scan could leak
    across splits. Same principle as `dataset_2d.build_patient_safe_splits`,
    applied to the unit that actually identifies a patient here."""
    series_ids = sorted({entry.seriesuid for entry in candidates})
    rng = np.random.default_rng(seed)
    rng.shuffle(series_ids)

    n_val = max(1, round(len(series_ids) * val_fraction))
    n_test = max(1, round(len(series_ids) * test_fraction))

    val_set = set(series_ids[:n_val])
    test_set = set(series_ids[n_val : n_val + n_test])

    train = [e for e in candidates if e.seriesuid not in val_set | test_set]
    val = [e for e in candidates if e.seriesuid in val_set]
    test = [e for e in candidates if e.seriesuid in test_set]

    validate_no_leakage(train, val, test)
    return train, val, test


def class_distribution(entries: list[CandidateEntry]) -> dict[str, int]:
    counts = {"non_nodule": 0, "nodule": 0}
    for entry in entries:
        counts["nodule" if entry.label == 1 else "non_nodule"] += 1
    return counts


def _crop_or_pad(
    array: np.ndarray,
    center: tuple[int, int, int],
    size: tuple[int, int, int],
    pad_value: float,
) -> np.ndarray:
    """Extract a fixed-`size` patch centered on `center` from `array`,
    padding with `pad_value` wherever the patch would run outside the
    volume (candidates near the scan edge are real and must not crash
    or silently shrink the batch's tensor shape)."""
    out = np.full(size, pad_value, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for dim in range(3):
        half = size[dim] // 2
        src_start = center[dim] - half
        src_end = src_start + size[dim]
        clipped_start = max(0, src_start)
        clipped_end = min(array.shape[dim], src_end)
        dst_start = clipped_start - src_start
        dst_end = dst_start + max(0, clipped_end - clipped_start)
        src_slices.append(slice(clipped_start, clipped_end))
        dst_slices.append(slice(dst_start, dst_end))

    if all(s.stop > s.start for s in src_slices):
        out[tuple(dst_slices)] = array[tuple(src_slices)]
    return out


def crop_patch_from_resampled(
    resampled_image: sitk.Image,
    coord_xyz: tuple[float, float, float],
    crop_size: tuple[int, int, int] = CROP_SIZE_VOXELS,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    array: np.ndarray | None = None,
) -> np.ndarray:
    """Crop + HU-window a patch from an *already-resampled* volume.

    Split out from `extract_candidate_patch` so `CTCandidateDataset` can
    cache the expensive resample step across candidates from the same
    series (see its docstring) — this half is cheap (index math + a
    small array copy), the resample half is not.

    `array` lets a caller pass a pre-computed array (e.g. a cached
    `GetArrayViewFromImage(...)`) instead of recomputing it here.
    """
    if array is None:
        array = sitk.GetArrayViewFromImage(resampled_image)  # view, not a copy

    # World (x, y, z) mm -> continuous voxel index (x, y, z) -> integer (z, y, x)
    # for numpy indexing. Using the *resampled* image's own transform
    # (not hand-computed from spacing) is what keeps this correct
    # regardless of direction/origin — resample_isotropic preserves both.
    ix, iy, iz = resampled_image.TransformPhysicalPointToContinuousIndex(coord_xyz)
    center = (int(round(iz)), int(round(iy)), int(round(ix)))

    patch = _crop_or_pad(array, center, crop_size, pad_value=hu_window[0])
    return apply_hu_window(patch, hu_window)


def extract_candidate_patch(
    image: sitk.Image,
    coord_xyz: tuple[float, float, float],
    crop_size: tuple[int, int, int] = CROP_SIZE_VOXELS,
    target_spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> np.ndarray:
    """Resample `image` to isotropic spacing, crop a fixed-size 3D patch
    centered on `coord_xyz` (world mm coordinates), HU-window + normalize
    the crop. Cropping happens on the small patch, not the full
    HU-windowed volume, to avoid the unnecessary full-volume elementwise
    pass (spec's "avoid loading unnecessary [volume] into memory").

    Returns a (D, H, W) float32 array in [0, 1], always exactly
    `crop_size` (air-padded at volume edges, matching
    `resample_isotropic`'s own edge-padding choice).

    One-shot convenience wrapper (resample + crop) — `CTCandidateDataset`
    below does NOT call this directly, since it needs to cache the
    resample step across candidates; it calls `resample_isotropic` and
    `crop_patch_from_resampled` separately instead.
    """
    resampled = resample_isotropic(image, target_spacing_mm)
    return crop_patch_from_resampled(resampled, coord_xyz, crop_size, hu_window)


class CTCandidateDataset(Dataset):
    """Wraps candidate-patch extraction + label lookup for a manifest of
    `CandidateEntry`.

    Caches the resampled volume (image + array) per seriesuid, keyed by
    seriesuid, LRU-bounded by `cache_size` (`OrderedDict`-based).
    `cache_size` defaults to the number of DISTINCT seriesuids in
    `entries` — i.e. every series this dataset will ever touch stays
    cached for its whole lifetime, nothing gets evicted.

    That default matters and is not arbitrary. An earlier version of
    this cache used a small fixed size (4), verified only against
    *sequential* same-series access — which is NOT how training actually
    reads data. `training/train_3d.py` uses a `WeightedRandomSampler`,
    which shuffles across the WHOLE split every draw; consecutive
    candidates are essentially random distinct series. Measured directly
    against real subset0 data under that realistic random-order access
    pattern: `cache_size=4` gave 0.405s/candidate — statistically no
    better than no caching at all (~0.41s/candidate measured earlier),
    because 4 slots against dozens of distinct series in play evicts
    almost everything before it's ever reused. Auto-sizing to cover
    every distinct series in `entries` fixed it: over a full epoch, each
    series' expensive resample is paid at most ONCE total (not once per
    epoch — this dataset object persists across epochs when
    `num_workers=0`/`persistent_workers=True`), and every other access
    is a cheap in-memory crop.

    Memory tradeoff (deliberate, not incidental): a resident cache entry
    is ~88MB (measured on real subset0 volumes — the `sitk.Image`'s own
    buffer; the cached array is a `GetArrayViewFromImage` VIEW into that
    same buffer, not a second copy — an earlier version of this cache
    used `GetArrayFromImage(...).astype(np.float32)`, which silently
    allocated a full second ~175MB float32 copy per entry alongside the
    image, nearly tripling real memory to ~263MB/entry; caught because a
    real Colab run swap-thrashed instead of erroring, not by any test —
    see PROGRESS_LOG.md Day 7 entry). Caching all of subset0's 89 series
    is therefore ~8GB host RAM for a single dataset instance — affordable
    on Colab, since this is a host-RAM cache, not GPU memory (doesn't
    conflict with the spec's "avoid loading unnecessary [volume] into
    GPU memory", Day 6 "Important" section, which is about what reaches
    the GPU). Pass an explicit smaller `cache_size` to trade completeness
    for a memory ceiling — e.g. for a future multi-subset (thousands of
    series) scale-up, where caching everything would no longer be
    affordable and this default would need revisiting.

    Note: each DataLoader worker process (`num_workers > 0`) gets its
    own copy of this dataset and therefore its own independent cache —
    total host RAM scales with `num_workers * cache_size`. For subset0's
    scale, prefer `num_workers=0` (one resident cache, not N of them)
    over parallelizing across workers — see PROGRESS_LOG.md Day 7 entry.
    """

    def __init__(
        self,
        entries: list[CandidateEntry],
        dataset_root: str | Path,
        crop_size: tuple[int, int, int] = CROP_SIZE_VOXELS,
        cache_size: int | None = None,
    ):
        self.entries = entries
        self.dataset_root = Path(dataset_root)
        self.crop_size = crop_size
        self.cache_size = (
            cache_size if cache_size is not None else len({e.seriesuid for e in entries})
        )
        self._cache: OrderedDict[str, tuple[sitk.Image, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def _resampled_volume(self, seriesuid: str) -> tuple[sitk.Image, np.ndarray]:
        if seriesuid in self._cache:
            self._cache.move_to_end(seriesuid)  # mark as most-recently-used
            return self._cache[seriesuid]

        volume_path = self.dataset_root / f"{seriesuid}.mhd"
        image = hu_convert(load_ct_volume(volume_path))
        resampled = resample_isotropic(image)
        # A VIEW into the image's own buffer, not a copy -- GetArrayFromImage
        # (+ the float32 cast) would allocate a second full-volume buffer
        # alongside the sitk.Image already held in the cache, nearly
        # tripling real memory per cached entry (measured: 87.6MB image +
        # 175MB float32 copy = ~263MB, vs ~87.6MB for image+view). The
        # small per-crop assignment in `_crop_or_pad` already upcasts to
        # float32 on the tiny cropped patch, so no full-volume cast is
        # needed here at all.
        array = sitk.GetArrayViewFromImage(resampled)

        self._cache[seriesuid] = (resampled, array)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        return resampled, array

    def __getitem__(self, index: int):
        entry = self.entries[index]
        resampled, array = self._resampled_volume(entry.seriesuid)
        patch = crop_patch_from_resampled(
            resampled, entry.coord_xyz, self.crop_size, array=array
        )
        return patch[np.newaxis, ...], entry.label  # add channel dim -> (1, D, H, W)
