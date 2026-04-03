import { useNavigate, useLocation } from 'react-router-dom';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleCameraClick = () => {
    navigate('/camera');
  };
  const handleNotificationHistoryClick = () => {
    navigate('/notification-history');
  };
  const handleDetectionHistoryClick = () => {
    navigate('/detection-history');
  };

  const handleLogoClick = () => {
    navigate('/');
  };

  const isCameraActive = location.pathname === '/camera';
  const isNotificationHistoryActive = location.pathname === '/notification-history';
  const isDetectionHistoryActive = location.pathname === '/detection-history';

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand" onClick={handleLogoClick} style={{ cursor: 'pointer' }}>
          ABC
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
            className={`nav-link ${isNotificationHistoryActive ? 'active' : ''}`}   
            type="button"
            onClick={handleNotificationHistoryClick}
          >
            Lịch sử thông báo
          </button>
          <button
            className={`nav-link ${isDetectionHistoryActive ? 'active' : ''}`}
            type="button"
            onClick={handleDetectionHistoryClick}
          >
            Lịch sử phát hiện
          </button>
          <button className="logout-btn" type="button">
            Đăng xuất
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;
