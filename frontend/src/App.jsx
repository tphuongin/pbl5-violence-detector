import { Routes, Route, Navigate, Outlet } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import CameraPage from './pages/CameraPage';
import DetectionHistoryPage from './pages/DetectionHistoryPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';
import LoginPage from './pages/LoginPage';

function ProtectedLayout() {
  const token = localStorage.getItem('token');
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return (
    <>
      <Header />
      <Outlet />
    </>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/detection-history" element={<DetectionHistoryPage />} />
        <Route path="/video-analysis" element={<VideoAnalysisPage />} />
      </Route>
    </Routes>
  );
}

export default App;
