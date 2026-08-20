# Day 11: rebuilt from Day 1's minimal version -- the app has grown
# storage/, knowledge_base/, and model checkpoints since then, none of
# which the original image copied in.
#
# python:3.14-slim, not 3.11 -- reconciles a mismatch flagged since Day
# 1. Local dev has run on 3.14 since Day 2 (confirmed compatible with
# the full stack, including PyTorch 2.13/torchvision 0.28 -- see
# PROGRESS_LOG.md), so this moves Docker to match local, not the other
# way around.
FROM python:3.14-slim

# libgomp1: required by SimpleITK/PyTorch's OpenMP-linked native code --
# missing on the slim base image, absent here causes an import-time
# crash, not a warning. Kept minimal otherwise (no full build-essential
# -- requirements.txt installs from prebuilt wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# No CUDA index configured (intentional, see requirements.txt's own
# comment) -- resolves to CPU wheels here, which is correct for a
# container with no GPU passthrough.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY storage/ storage/
COPY knowledge_base/ knowledge_base/

# Trained model checkpoints (Day 4/7) -- baked into the image
# deliberately, not mounted, so `docker compose up` is self-contained
# for a demo/interview rather than requiring the weights to be sourced
# separately. Gitignored from source control either way (94MB+ each);
# this COPY only works if they're present in the local build context
# (see .dockerignore's comment on why they're NOT excluded there).
COPY training/checkpoints/ training/checkpoints/

# SQLite DB lives here at runtime (storage/database.py's default path)
# -- created on first request if missing; a named volume in
# docker-compose.yml persists it across container restarts.
RUN mkdir -p /app/reports/generated

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
