import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import CameraPage from './pages/CameraPage';
import NotificationHistoryPage from './pages/NotificationHistoryPage';
import DetectionHistoryPage from './pages/DetectionHistoryPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';

function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/notification-history" element={<NotificationHistoryPage />} />
        <Route path="/detection-history" element={<DetectionHistoryPage />} />
        <Route path="/video-analysis" element={<VideoAnalysisPage />} />
      </Routes>
    </>
  );
}

export default App;
