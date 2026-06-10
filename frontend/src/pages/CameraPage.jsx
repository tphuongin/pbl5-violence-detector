import { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/apiService';
import VideoPlayer from '../components/VideoPlayer';
import { ShieldAlert, AlertTriangle, Plus, Edit2, Trash2, X, AlertCircle } from 'lucide-react';
import './CameraPage.css';

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

  const isAdmin = localStorage.getItem('username') === 'admin';

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [modalData, setModalData] = useState({
    CameraName: '',
    CameraIP: '',
    CameraPhoneNum: '',
    CameraStatus: true,
  });
  const [modalError, setModalError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleOpenAddModal = () => {
    setModalData({
      CameraName: '',
      CameraIP: '',
      CameraPhoneNum: '',
      CameraStatus: true,
    });
    setModalError('');
    setIsAddModalOpen(true);
  };

  const handleOpenEditModal = (camera) => {
    setEditingCamera(camera);
    setModalData({
      CameraName: camera.CameraName || '',
      CameraIP: camera.CameraIP || '',
      CameraPhoneNum: camera.CameraPhoneNum || '',
      CameraStatus: camera.CameraStatus !== false,
    });
    setModalError('');
    setIsEditModalOpen(true);
  };

  const handleModalSubmit = async (e) => {
    e.preventDefault();
    if (!modalData.CameraName.trim()) {
      setModalError('Tên camera không được để trống');
      return;
    }

    setIsSubmitting(true);
    setModalError('');

    try {
      if (isAddModalOpen) {
        const newCam = await apiService.createCamera(modalData);
        setCameras((prev) => [...prev, newCam]);
        setSelectedCameraId(newCam.CameraID);
        setIsAddModalOpen(false);
      } else if (isEditModalOpen && editingCamera) {
        const updatedCam = await apiService.updateCamera(editingCamera.CameraID, modalData);
        setCameras((prev) =>
          prev.map((c) => (c.CameraID === updatedCam.CameraID ? updatedCam : c))
        );
        setIsEditModalOpen(false);
      }
    } catch (err) {
      setModalError(err.message || 'Thao tác thất bại');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteCamera = async (cameraId) => {
    const cam = cameras.find((c) => c.CameraID === cameraId);
    const confirmMsg = `Bạn có chắc chắn muốn xóa camera "${cam?.CameraName || 'này'}" không? Việc này sẽ xóa toàn bộ lịch sử bạo lực và dữ liệu cuộc gọi liên quan đến camera này.`;
    if (!window.confirm(confirmMsg)) {
      return;
    }

    try {
      await apiService.deleteCamera(cameraId);
      setCameras((prev) => prev.filter((c) => c.CameraID !== cameraId));
      if (selectedCameraId === cameraId) {
        const remaining = cameras.filter((c) => c.CameraID !== cameraId);
        if (remaining.length > 0) {
          setSelectedCameraId(remaining[0].CameraID);
        } else {
          setSelectedCameraId(null);
        }
      }
    } catch (err) {
      alert(err.message || 'Không thể xóa camera');
    }
  };


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

      <div className="camera-layout">
        {/* Left Column: Sidebar list of cameras */}
        <div>
          <h3 className="camera-list-title">Danh sách Camera</h3>
          {isAdmin && (
            <button className="add-camera-btn" onClick={handleOpenAddModal}>
              <Plus size={16} />
              <span>Thêm Camera</span>
            </button>
          )}
          <div className="camera-list-container">
            {cameras.length === 0 ? (
              <p style={{ margin: 0, padding: '10px', color: 'var(--muted)', fontSize: '0.9rem', fontStyle: 'italic' }}>
                Không có camera
              </p>
            ) : (
              cameras.map((cam) => (
                <div key={cam.CameraID} className="camera-sidebar-item">
                  <button
                    className={`camera-list-btn ${selectedCameraId === cam.CameraID ? 'active' : ''}`}
                    onClick={() => setSelectedCameraId(cam.CameraID)}
                    style={{ flexGrow: 1 }}
                  >
                    <span className="camera-list-btn-name">{cam.CameraName}</span>
                    <span className={`camera-list-btn-status ${cam.CameraStatus ? 'online' : 'offline'}`}>
                      {cam.CameraStatus ? 'Online' : 'Offline'}
                    </span>
                  </button>
                  {isAdmin && (
                    <div className="camera-sidebar-actions">
                      <button
                        className="camera-action-icon edit"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOpenEditModal(cam);
                        }}
                        title="Chỉnh sửa camera"
                      >
                        <Edit2 size={14} />
                      </button>
                      <button
                        className="camera-action-icon delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCamera(cam.CameraID);
                        }}
                        title="Xóa camera"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Column: Active camera detail */}
        <div style={{ minWidth: 0, width: '100%' }}>
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
      </div>

      {/* Admin Camera Add/Edit Modals */}
      {(isAddModalOpen || isEditModalOpen) && (
        <div className="modal-backdrop" onClick={() => { setIsAddModalOpen(false); setIsEditModalOpen(false); }}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{isAddModalOpen ? 'Thêm Camera Mới' : 'Chỉnh Sửa Camera'}</h3>
              <button 
                className="modal-close-btn" 
                onClick={() => { setIsAddModalOpen(false); setIsEditModalOpen(false); }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              {modalError && (
                <div className="modal-error-alert">
                  <AlertCircle size={16} />
                  <span>{modalError}</span>
                </div>
              )}
              <form onSubmit={handleModalSubmit} className="modal-form">
                <div className="form-group">
                  <label htmlFor="modalCamName">Tên Camera *</label>
                  <input
                    id="modalCamName"
                    type="text"
                    placeholder="VD: Camera Sân Sau"
                    value={modalData.CameraName}
                    onChange={(e) => setModalData({ ...modalData, CameraName: e.target.value })}
                    required
                    disabled={isSubmitting}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="modalCamIP">Địa chỉ IP</label>
                  <input
                    id="modalCamIP"
                    type="text"
                    placeholder="VD: 192.168.1.100"
                    value={modalData.CameraIP}
                    onChange={(e) => setModalData({ ...modalData, CameraIP: e.target.value })}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="modalCamPhone">Số điện thoại nhận cảnh báo</label>
                  <input
                    id="modalCamPhone"
                    type="text"
                    placeholder="VD: 0987654321"
                    value={modalData.CameraPhoneNum}
                    onChange={(e) => setModalData({ ...modalData, CameraPhoneNum: e.target.value })}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="form-group-row">
                  <input
                    id="modalCamStatus"
                    type="checkbox"
                    checked={modalData.CameraStatus}
                    onChange={(e) => setModalData({ ...modalData, CameraStatus: e.target.checked })}
                    disabled={isSubmitting}
                  />
                  <label htmlFor="modalCamStatus">Kích hoạt hoạt động (Online)</label>
                </div>

                <div style={{ display: 'none' }}>
                  <button type="submit" id="modalSubmitBtn"></button>
                </div>
              </form>
            </div>
            <div className="modal-footer">
              <button 
                type="button" 
                className="btn-secondary" 
                onClick={() => { setIsAddModalOpen(false); setIsEditModalOpen(false); }}
                disabled={isSubmitting}
              >
                Hủy bỏ
              </button>
              <button 
                type="button" 
                className="btn-primary" 
                onClick={() => document.getElementById('modalSubmitBtn').click()}
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Đang lưu...' : 'Lưu lại'}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default CameraPage;
