import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import NewStudy from './pages/NewStudy'
import ReportsHistory from './pages/ReportsHistory'
import ReportView from './pages/ReportView'
import Dashboard from './pages/Dashboard'
import AllStudies from './pages/AllStudies'
import Analytics from './pages/Analytics'
import CopilotChat from './pages/CopilotChat'
import Settings from './pages/Settings'

export default function App() {
  // Real bug reported live: clicking "New Study" in the sidebar while
  // already on /new-study (e.g. mid-analysis, before generating a
  // report) did nothing -- React Router reuses the same NewStudy
  // instance for a same-path navigation, so its step/analysis/etc.
  // state never resets. The only way out was to finish generating a
  // report first. `location.key` is a fresh value on every navigation,
  // including to the same path, so keying the route on it forces a real
  // remount (and therefore a full state reset) every time.
  const location = useLocation()
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/new-study" element={<NewStudy key={location.key} />} />
        <Route path="/all-studies" element={<AllStudies />} />
        <Route path="/copilot" element={<CopilotChat />} />
        <Route path="/reports" element={<ReportsHistory />} />
        <Route path="/reports/:reportId" element={<ReportView />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Layout>
  )
}
