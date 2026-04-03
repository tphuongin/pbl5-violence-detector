import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import VideoPlayer from '../components/VideoPlayer';

function CameraPage() {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
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
      
      // Auto-select first camera
      if (data.data && data.data.length > 0) {
        setSelectedCameraId(data.data[0].CameraID);
      }
      setError(null);
    } catch (err) {
      setError('Không thể tải danh sách camera');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const selectedCamera = cameras.find(c => c.CameraID === selectedCameraId);

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
      <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '20px', height: '100%' }}>
        {/* Camera List */}
        <section className="camera-card" aria-label="Danh sách camera">
          <h3 style={{ marginTop: 0, marginBottom: '15px' }}>Cameras</h3>
          {cameras.length === 0 ? (
            <p>Không có camera nào</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {cameras.map((camera) => (
                <button
                  key={camera.CameraID}
                  onClick={() => setSelectedCameraId(camera.CameraID)}
                  style={{
                    padding: '12px',
                    border: selectedCameraId === camera.CameraID ? '2px solid #2196f3' : '1px solid #ddd',
                    borderRadius: '6px',
                    backgroundColor: selectedCameraId === camera.CameraID ? '#e3f2fd' : '#f9f9f9',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all 0.3s ease',
                  }}
                >
                  <div style={{ fontWeight: 'bold', fontSize: '0.95em' }}>
                    {camera.CameraName}
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#666', marginTop: '4px' }}>
                    {camera.CameraStatus ? '✓ Online' : '✗ Offline'}
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Video Stream */}
        <section>
          {selectedCamera ? (
            <div>
              <h3 style={{ marginTop: 0, marginBottom: '15px' }}>
                {selectedCamera.CameraName}
              </h3>
              <div style={{ marginBottom: '15px' }}>
                <VideoPlayer 
                  cameraId={selectedCamera.CameraID} 
                  cameraName={selectedCamera.CameraName}
                />
              </div>
              
              {/* Camera Details */}
              <div style={{
                backgroundColor: '#f5f5f5',
                padding: '15px',
                borderRadius: '8px',
                fontSize: '0.9em',
              }}>
                <div style={{ marginBottom: '8px' }}>
                  <strong>IP Address:</strong> {selectedCamera.CameraIP}
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <strong>Phone:</strong> {selectedCamera.CameraPhoneNum}
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <strong>Status:</strong> {' '}
                  <span style={{
                    color: selectedCamera.CameraStatus ? '#4caf50' : '#f44336',
                    fontWeight: 'bold'
                  }}>
                    {selectedCamera.CameraStatus ? '🟢 Online' : '🔴 Offline'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: '#999',
            }}>
              <p>Chọn một camera để xem stream</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default CameraPage;
