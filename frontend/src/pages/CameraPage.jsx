import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

function CameraPage() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCameras();
  }, []);

  const loadCameras = async () => {
    try {
      setLoading(true);
      const data = await apiService.getCameras();
      setCameras(data.data || []);
      setError(null);
    } catch (err) {
      setError('Không thể tải danh sách camera');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <main className="page-shell">
        <p>Đang tải dữ liệu...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="page-shell">
        <p style={{ color: 'red' }}>{error}</p>
      </main>
    );
  }

  return (
    <main className="page-shell">
      <section className="camera-card" aria-label="Danh sách camera">
        {cameras.length === 0 ? (
          <p>Không có camera nào</p>
        ) : (
          cameras.map((camera) => (
            <div key={camera.CameraID} className="camera-row">
              <div>
                <div className="camera-name">{camera.CameraName}</div>
                <div style={{ fontSize: '0.9em', color: '#666' }}>
                  IP: {camera.CameraIP} | Trạng thái: {camera.CameraStatus ? '✓ Hoạt động' : '✗ Offline'}
                </div>
              </div>
              <button className="view-btn" type="button">
                Xem
              </button>
            </div>
          ))
        )}
      </section>
    </main>
  );
}

export default CameraPage;
