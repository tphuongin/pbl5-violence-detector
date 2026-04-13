import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import VideoPlayer from '../components/VideoPlayer';

function CameraPage() {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [detectionUpdatedAt, setDetectionUpdatedAt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    if (!selectedCameraId) {
      return undefined;
    }

    let active = true;

    const loadLatestDetection = async () => {
      const data = await apiService.getLatestDetection(selectedCameraId);
      if (!active || !data) {
        return;
      }
      setLatestDetection(data);
      setDetectionUpdatedAt(Date.now());
    };

    setLatestDetection(null);
    setDetectionUpdatedAt(null);
    loadLatestDetection();

    const intervalId = window.setInterval(loadLatestDetection, 1000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [selectedCameraId]);

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
  const isViolence = latestDetection?.label === 'VIOLENCE';
  const hasDetection = Boolean(latestDetection);
  const thumbnailSrc = latestDetection?.thumb_b64
    ? `data:image/jpeg;base64,${latestDetection.thumb_b64}`
    : null;

  const formatProb = (value) => {
    if (typeof value !== 'number') {
      return '--';
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatNumber = (value, suffix = '') => {
    if (typeof value !== 'number') {
      return '--';
    }
    return `${value}${suffix}`;
  };

  const hasValue = (value) => value !== undefined && value !== null && value !== '';

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
              <div className={`live-frame-shell ${isViolence ? 'danger' : 'safe'}`} style={{ marginBottom: '15px' }}>
                {isViolence && (
                  <div className="violence-alert-banner">
                    CANH BAO: PHAT HIEN BAO LUC
                  </div>
                )}

                <div className={`live-status-chip ${isViolence ? 'danger' : 'safe'}`}>
                  {hasDetection ? (isViolence ? 'TRANG THAI: VIOLENCE' : 'TRANG THAI: NORMAL') : 'TRANG THAI: CHUA CO DU LIEU'}
                </div>

                <VideoPlayer
                  cameraId={selectedCamera.CameraID}
                  cameraName={selectedCamera.CameraName}
                />
              </div>

              <div style={{
                backgroundColor: '#fff',
                border: '1px solid #d9e2ec',
                borderRadius: '12px',
                padding: '16px',
                marginBottom: '15px',
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '12px',
                  flexWrap: 'wrap',
                  gap: '10px',
                }}>
                  <h4 style={{ margin: 0 }}>Live Detection</h4>
                  <span style={{
                    backgroundColor: isViolence ? '#ffebee' : '#e8f5e9',
                    color: isViolence ? '#c62828' : '#2e7d32',
                    border: `1px solid ${isViolence ? '#ffcdd2' : '#c8e6c9'}`,
                    borderRadius: '999px',
                    padding: '6px 12px',
                    fontSize: '0.85em',
                    fontWeight: 'bold',
                  }}>
                    {latestDetection?.label || 'No Data'}
                  </span>
                </div>

                {!latestDetection ? (
                  <p style={{ margin: 0, color: '#666' }}>
                    Chưa có dữ liệu detection realtime từ Jetson cho camera này.
                  </p>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 160px', gap: '14px' }}>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                      gap: '10px',
                    }}>
                      {hasValue(latestDetection.timestamp) && (
                        <div><strong>Timestamp:</strong> {latestDetection.timestamp}</div>
                      )}
                      {typeof latestDetection.prob_raw === 'number' && (
                        <div><strong>Prob Raw:</strong> {formatProb(latestDetection.prob_raw)}</div>
                      )}
                      {typeof latestDetection.prob_smooth === 'number' && (
                        <div><strong>Prob Smooth:</strong> {formatProb(latestDetection.prob_smooth)}</div>
                      )}
                      {(hasValue(latestDetection.confirm_count) || hasValue(latestDetection.confirm_needed)) && (
                        <div><strong>Confirm:</strong> {latestDetection.confirm_count ?? '--'} / {latestDetection.confirm_needed ?? '--'}</div>
                      )}
                      {typeof latestDetection.threshold === 'number' && (
                        <div><strong>Threshold:</strong> {formatNumber(latestDetection.threshold)}</div>
                      )}
                      {typeof latestDetection.conf_thresh === 'number' && (
                        <div><strong>Conf Thresh:</strong> {formatNumber(latestDetection.conf_thresh)}</div>
                      )}
                      {typeof latestDetection.infer_fps === 'number' && (
                        <div><strong>Infer FPS:</strong> {formatNumber(latestDetection.infer_fps, ' fps')}</div>
                      )}
                      {typeof latestDetection.cam_fps === 'number' && (
                        <div><strong>Cam FPS:</strong> {formatNumber(latestDetection.cam_fps, ' fps')}</div>
                      )}
                      {typeof latestDetection.infer_ms === 'number' && (
                        <div><strong>Infer Time:</strong> {formatNumber(latestDetection.infer_ms, ' ms')}</div>
                      )}
                      {typeof latestDetection.uptime === 'number' && (
                        <div><strong>Uptime:</strong> {formatNumber(latestDetection.uptime, ' s')}</div>
                      )}
                      {typeof latestDetection.window_countdown === 'number' && (
                        <div><strong>Window:</strong> {formatNumber(latestDetection.window_countdown, ' s')}</div>
                      )}
                      {typeof latestDetection.window_sec === 'number' && (
                        <div><strong>Window Sec:</strong> {formatNumber(latestDetection.window_sec, ' s')}</div>
                      )}
                      {typeof latestDetection.alert_until === 'number' && (
                        <div><strong>Alert Until:</strong> {formatNumber(latestDetection.alert_until, ' s')}</div>
                      )}
                      {typeof latestDetection.alert_sec === 'number' && (
                        <div><strong>Alert Sec:</strong> {formatNumber(latestDetection.alert_sec, ' s')}</div>
                      )}
                      {typeof latestDetection.num_frames === 'number' && (
                        <div><strong>Num Frames:</strong> {formatNumber(latestDetection.num_frames)}</div>
                      )}
                      <div><strong>Cập nhật:</strong> {detectionUpdatedAt ? new Date(detectionUpdatedAt).toLocaleTimeString('vi-VN') : '--'}</div>
                    </div>

                    <div style={{
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      backgroundColor: '#fafafa',
                      minHeight: '90px',
                    }}>
                      {thumbnailSrc ? (
                        <img
                          src={thumbnailSrc}
                          alt="Detection thumbnail"
                          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                        />
                      ) : (
                        <div style={{
                          color: '#777',
                          height: '100%',
                          minHeight: '90px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          textAlign: 'center',
                          padding: '8px',
                          fontSize: '0.85em',
                        }}>
                          No Thumbnail
                        </div>
                      )}
                    </div>
                  </div>
                )}
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
