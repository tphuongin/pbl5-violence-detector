import { useNavigate, useLocation } from 'react-router-dom';
import { ShieldAlert, LogOut } from 'lucide-react';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleCameraClick = () => {
    navigate('/camera');
  };
  const handleDetectionHistoryClick = () => {
    navigate('/detection-history');
  };
  const handleVideoAnalysisClick = () => {
    navigate('/video-analysis');
  };

  const handleLogoClick = () => {
    navigate('/');
  };

  const isCameraActive = location.pathname === '/camera';
  const isDetectionHistoryActive = location.pathname === '/detection-history';
  const isVideoAnalysisActive = location.pathname === '/video-analysis';

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div 
          className="brand" 
          onClick={handleLogoClick} 
          role="button" 
          tabIndex={0} 
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <ShieldAlert size={22} style={{ color: 'var(--primary)' }} />
          <span>Violence Detector System</span>
        </div>

        <div className="topbar-actions">
          <button
            className={`nav-link ${isCameraActive ? 'active' : ''}`}
            type="button"
            onClick={handleCameraClick}
          >
            Camera
          </button>
          <button
            className={`nav-link ${isVideoAnalysisActive ? 'active' : ''}`}
            type="button"
            onClick={handleVideoAnalysisClick}
          >
            Phân tích video
          </button>
          <button
            className={`nav-link ${isDetectionHistoryActive ? 'active' : ''}`}
            type="button"
            onClick={handleDetectionHistoryClick}
          >
            Lịch sử phát hiện
          </button>
          <button 
            className="logout-btn" 
            type="button"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <LogOut size={16} />
            <span>Đăng xuất</span>
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;
