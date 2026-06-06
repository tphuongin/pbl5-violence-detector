import { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/apiService';
import VideoPlayer from '../components/VideoPlayer';
import { ShieldAlert, AlertTriangle } from 'lucide-react';

function CameraPage() {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [detectionUpdatedAt, setDetectionUpdatedAt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [toastMessage, setToastMessage] = useState('');
  const [showToast, setShowToast] = useState(false);
  const toastTimerRef = useRef(null);
  const latestHistoryIdRef = useRef(null);
  const historyCacheKey = 'violenceHistoryCacheV1';

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    let active = true;
    let intervalId = null;

    const checkHistory = async () => {
      const data = await apiService.getViolenceHistory(1, 1);
      if (!active) {
        return;
      }
      const record = data?.data?.[0];
      if (!record) {
        return;
      }

      if (latestHistoryIdRef.current && record.HistoryID !== latestHistoryIdRef.current) {
        const locationLabel = record.CameraName || record.Location || 'Không rõ vị trí';
        setToastMessage(`Đã lưu vào lịch sử bạo lực: ${locationLabel}`);
        setShowToast(true);
        if (toastTimerRef.current) {
          window.clearTimeout(toastTimerRef.current);
        }
        toastTimerRef.current = window.setTimeout(() => {
          setShowToast(false);
        }, 4500);

        apiService.getViolenceHistory(1, 9).then((pageData) => {
          if (pageData?.data) {
            window.localStorage.setItem(
              historyCacheKey,
              JSON.stringify({
                updatedAt: Date.now(),
                records: pageData.data,
                total: pageData.total || 0,
              })
            );
          }
        });
      }

      latestHistoryIdRef.current = record.HistoryID;
    };

    checkHistory();
    intervalId = window.setInterval(checkHistory, 4000);

    return () => {
      active = false;
      if (intervalId) {
        window.clearInterval(intervalId);
      }
      if (toastTimerRef.current) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
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
        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--muted)', fontWeight: '600' }}>
          Đang tải dữ liệu...
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="page-shell">
        <div style={{
          backgroundColor: 'var(--danger-light)',
          border: '1px solid var(--danger)',
          borderRadius: '12px',
          padding: '20px',
          color: 'var(--danger)',
          fontWeight: '600',
          textAlign: 'center'
        }}>{error}</div>
      </main>
    );
  }

  return (
    <main className="page-shell">
      <div className={`toast-stack ${showToast ? 'show' : ''}`}>
        {toastMessage && (
          <div 
            className={`toast ${showToast ? 'show' : ''}`} 
            role="status"
          >
            <div className="toast-icon-wrapper">
              <AlertTriangle size={20} />
            </div>
            <div className="toast-content">
              <div className="toast-title">Cảnh Báo Hệ Thống</div>
              <div className="toast-desc">{toastMessage}</div>
            </div>
          </div>
        )}
      </div>

      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* Video Stream Main Area */}
        {selectedCamera ? (
          <div>
            <h3 className="stream-title">
              {selectedCamera.CameraName}
            </h3>
            
            <div className={`live-frame-shell ${isViolence ? 'danger' : 'safe'}`} style={{ marginBottom: '20px' }}>
              {isViolence && (
                <div className="violence-alert-banner" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <ShieldAlert size={16} />
                  <span>PHÁT HIỆN BẠO LỰC</span>
                </div>
              )}

              <div className={`live-status-chip ${isViolence ? 'danger' : 'safe'}`}>
                {hasDetection ? (isViolence ? 'VIOLENCE' : 'NORMAL') : 'CHƯA CÓ DỮ LIỆU'}
              </div>

              <VideoPlayer
                cameraId={selectedCamera.CameraID}
                cameraName={selectedCamera.CameraName}
              />
            </div>

            {/* Realtime Detection Info Card */}
            <div className="camera-card" style={{ marginBottom: '20px' }}>
              <div className="live-detection-header">
                <h4>Thông Số Nhận Diện Realtime</h4>
                <span className={`detection-badge ${isViolence ? 'violence' : 'normal'}`}>
                  {latestDetection?.label || 'Chờ Dữ Liệu'}
                </span>
              </div>

              {!latestDetection ? (
                <p className="detection-no-data">
                  Đang đợi dữ liệu phân tích realtime từ thiết bị Jetson cho camera này...
                </p>
              ) : (
                <div className="realtime-grid">
                  {hasValue(latestDetection.timestamp) && (
                    <div className="realtime-item">
                      Thời gian <strong>{latestDetection.timestamp}</strong>
                    </div>
                  )}
                  {typeof latestDetection.prob_raw === 'number' && (
                    <div className="realtime-item">
                      Xác suất Gốc <strong>{formatProb(latestDetection.prob_raw)}</strong>
                    </div>
                  )}
                  {typeof latestDetection.prob_smooth === 'number' && (
                    <div className="realtime-item">
                      Xác suất Lọc <strong>{formatProb(latestDetection.prob_smooth)}</strong>
                    </div>
                  )}
                  {(hasValue(latestDetection.confirm_count) || hasValue(latestDetection.confirm_needed)) && (
                    <div className="realtime-item">
                      Xác nhận <strong>{latestDetection.confirm_count ?? '--'} / {latestDetection.confirm_needed ?? '--'}</strong>
                    </div>
                  )}
                  {typeof latestDetection.threshold === 'number' && (
                    <div className="realtime-item">
                      Ngưỡng Cảnh Báo <strong>{formatNumber(latestDetection.threshold)}</strong>
                    </div>
                  )}
                  {typeof latestDetection.conf_thresh === 'number' && (
                    <div className="realtime-item">
                      Ngưỡng Tin Cậy <strong>{formatNumber(latestDetection.conf_thresh)}</strong>
                    </div>
                  )}
                  {typeof latestDetection.infer_fps === 'number' && (
                    <div className="realtime-item">
                      FPS Phân Tích <strong>{formatNumber(latestDetection.infer_fps, ' fps')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.cam_fps === 'number' && (
                    <div className="realtime-item">
                      FPS Camera <strong>{formatNumber(latestDetection.cam_fps, ' fps')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.infer_ms === 'number' && (
                    <div className="realtime-item">
                      Thời Gian Xử Lý <strong>{formatNumber(latestDetection.infer_ms, ' ms')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.uptime === 'number' && (
                    <div className="realtime-item">
                      Uptime <strong>{formatNumber(latestDetection.uptime, ' s')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.window_countdown === 'number' && (
                    <div className="realtime-item">
                      Đếm Ngược Window <strong>{formatNumber(latestDetection.window_countdown, ' s')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.window_sec === 'number' && (
                    <div className="realtime-item">
                      Thời Gian Window <strong>{formatNumber(latestDetection.window_sec, ' s')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.alert_until === 'number' && (
                    <div className="realtime-item">
                      Giữ Cảnh Báo <strong>{formatNumber(latestDetection.alert_until, ' s')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.alert_sec === 'number' && (
                    <div className="realtime-item">
                      Thời Gian Cảnh Báo <strong>{formatNumber(latestDetection.alert_sec, ' s')}</strong>
                    </div>
                  )}
                  {typeof latestDetection.num_frames === 'number' && (
                    <div className="realtime-item">
                      Số Frame <strong>{formatNumber(latestDetection.num_frames)}</strong>
                    </div>
                  )}
                  <div className="realtime-item">
                    Cập Nhật Lúc <strong>{detectionUpdatedAt ? new Date(detectionUpdatedAt).toLocaleTimeString('vi-VN') : '--'}</strong>
                  </div>
                </div>
              )}
            </div>
            
            {/* Camera Metadata Card */}
            <div className="camera-details-card">
              <div className="camera-details-item">
                <span>Địa chỉ IP</span>
                <strong>{selectedCamera.CameraIP}</strong>
              </div>
              <div className="camera-details-item">
                <span>Số điện thoại nhận cảnh báo</span>
                <strong>{selectedCamera.CameraPhoneNum}</strong>
              </div>
              <div className="camera-details-item">
                <span>Trạng thái kết nối</span>
                <strong style={{ color: selectedCamera.CameraStatus ? 'var(--success)' : 'var(--danger)' }}>
                  {selectedCamera.CameraStatus ? '● Trực tuyến' : '○ Ngoại tuyến'}
                </strong>
              </div>
            </div>
          </div>
        ) : (
          <div className="camera-card" style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '350px',
            color: 'var(--muted)',
          }}>
            <p style={{ margin: 0, fontWeight: '600' }}>Không tìm thấy camera hoạt động nào</p>
          </div>
        )}
      </div>
    </main>
  );
}

export default CameraPage;
