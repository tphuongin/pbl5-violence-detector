import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiService, WS_BASE_URL } from '../services/apiService';
import { 
  UploadCloud, 
  FileVideo, 
  Trash2, 
  AlertCircle, 
  CheckCircle2, 
  AlertOctagon, 
  Activity, 
  Film, 
  Clock, 
  Percent, 
  Cpu 
} from 'lucide-react';
import './VideoAnalysisPage.css';

const MAX_FILE_MB = 500;
const MAX_FRAMES = 150;
const MAX_POINTS = 360;
const THRESHOLD = 0.5;

const initialState = {
  jobId: null,
  status: 'idle',
  progress: 0,
  filename: '',
  frames: [],
  summary: null,
  error: null,
  processed: 0,
  totalFrames: 0,
  label: 'NORMAL',
  prob: 0,
  latestThumb: null,
};

function formatVideoTime(seconds) {
  if (Number.isNaN(seconds) || seconds === null || seconds === undefined) {
    return '--';
  }
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${mins}:${secs}`;
}

function thumbToSrc(b64) {
  return b64 ? `data:image/jpeg;base64,${b64}` : '';
}

function getLabelColor(label) {
  return label === 'VIOLENCE' ? '#ef4444' : '#22c55e';
}

function downsamplePoints(points, maxPoints) {
  if (points.length <= maxPoints) {
    return points;
  }
  const stride = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % stride === 0);
}

function useVideoAnalysisSocket(jobId, enabled, onMessage, onStatus) {
  const lastMessageAtRef = useRef(Date.now());

  useEffect(() => {
    if (!jobId || !enabled) {
      return undefined;
    }

    let socket = null;
    let reconnectTimer = null;
    let watchdogTimer = null;
    let closedManually = false;

    const connect = () => {
      const token = localStorage.getItem('token');
      socket = new WebSocket(`${WS_BASE_URL}/ws/video-analysis/${jobId}?token=${token}`);

      socket.onopen = () => {
        lastMessageAtRef.current = Date.now();
        onStatus?.({ state: 'connected' });
      };

      socket.onmessage = (event) => {
        lastMessageAtRef.current = Date.now();
        onStatus?.({ state: 'active' });
        onMessage?.(event.data);
      };

      socket.onclose = () => {
        if (closedManually) {
          return;
        }
        onStatus?.({ state: 'disconnected' });
        reconnectTimer = window.setTimeout(connect, 2000);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    watchdogTimer = window.setInterval(() => {
      const delta = Date.now() - lastMessageAtRef.current;
      if (delta > 60000) {
        onStatus?.({ state: 'stale' });
      }
    }, 10000);

    return () => {
      closedManually = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      if (watchdogTimer) {
        window.clearInterval(watchdogTimer);
      }
      if (socket && socket.readyState < 2) {
        socket.close();
      }
    };
  }, [jobId, enabled, onMessage, onStatus]);
}

function VideoAnalysisPage() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analysis, setAnalysis] = useState(initialState);
  const [chartPoints, setChartPoints] = useState([]);
  const [connectionWarning, setConnectionWarning] = useState(null);
  const [retryAt, setRetryAt] = useState(null);

  const canStart = Boolean(selectedVideo) && analysis.status === 'idle';
  const isWorking = ['uploading', 'queued', 'running'].includes(analysis.status);

  const resetState = (clearFile = false) => {
    setAnalysis(initialState);
    setChartPoints([]);
    setConnectionWarning(null);
    setRetryAt(null);
    if (clearFile) {
      setSelectedVideo(null);
    }
  };

  const updateFromFrame = useCallback((payload) => {
    const nextFrame = {
      frame_idx: payload.frame_idx,
      frame_time_s: payload.frame_time_s,
      prob_smooth: payload.prob_smooth,
      label: payload.label,
      thumb_b64: payload.thumb_b64,
      timestamp: payload.timestamp,
      total_frames: payload.total_frames,
      progress: payload.progress,
    };

    setAnalysis((prev) => {
      const frames = [nextFrame, ...prev.frames].slice(0, MAX_FRAMES);
      return {
        ...prev,
        status: prev.status === 'queued' ? 'running' : prev.status,
        progress: payload.progress ?? prev.progress,
        processed: payload.frame_idx ?? prev.processed,
        totalFrames: payload.total_frames ?? prev.totalFrames,
        frames,
        label: payload.label ?? prev.label,
        prob: payload.prob_smooth ?? prev.prob,
        latestThumb: payload.thumb_b64 ?? prev.latestThumb,
      };
    });

    if (typeof payload.frame_time_s === 'number' && typeof payload.prob_smooth === 'number') {
      setChartPoints((prev) => downsamplePoints([...prev, {
        t: payload.frame_time_s,
        p: payload.prob_smooth,
      }], MAX_POINTS));
    }
  }, []);

  const handleSocketMessage = useCallback((raw) => {
    let payload = null;
    try {
      payload = JSON.parse(raw);
    } catch (error) {
      return;
    }

    if (payload.type === 'frame') {
      updateFromFrame(payload);
    }

    if (payload.type === 'summary') {
      setAnalysis((prev) => ({
        ...prev,
        status: 'done',
        progress: 1,
        summary: payload,
      }));
    }
  }, [updateFromFrame]);

  const handleSocketStatus = useCallback(({ state }) => {
    if (state === 'stale') {
      setConnectionWarning('Mất kết nối, đang thử lại...');
      return;
    }
    if (state === 'disconnected') {
      setConnectionWarning('Mất kết nối, đang thử lại...');
      return;
    }
    if (state === 'connected' || state === 'active') {
      setConnectionWarning(null);
    }
  }, []);

  useVideoAnalysisSocket(analysis.jobId, isWorking, handleSocketMessage, handleSocketStatus);

  const handleVideoChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    if (!file.type.startsWith('video/')) {
      alert('Vui lòng chọn tệp video hợp lệ');
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      alert('Dung lượng video vượt quá 500MB');
      return;
    }
    setSelectedVideo(file);
  };

  const handleRemoveVideo = () => {
    setSelectedVideo(null);
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      alert('Vui lòng chọn một video để phân tích');
      return;
    }

    setAnalysis((prev) => ({
      ...prev,
      status: 'uploading',
      error: null,
      filename: selectedVideo.name,
    }));

    try {
      const { status, payload } = await apiService.analyzeVideo(selectedVideo);
      if (status === 200) {
        setAnalysis((prev) => ({
          ...prev,
          jobId: payload.job_id,
          status: 'queued',
        }));
        return;
      }

      if (status === 409) {
        setAnalysis((prev) => ({
          ...prev,
          status: 'error',
          error: 'Jetson đang xử lý video khác, vui lòng chờ',
        }));
        setRetryAt(Date.now() + 5000);
        return;
      }

      const detail = payload?.detail || 'Không thể gửi video lên Jetson';
      setAnalysis((prev) => ({
        ...prev,
        status: 'error',
        error: detail,
      }));
    } catch (error) {
      setAnalysis((prev) => ({
        ...prev,
        status: 'error',
        error: 'Không thể kết nối đến backend',
      }));
    }
  };

  const handleCancel = async () => {
    if (!analysis.jobId) {
      resetState();
      return;
    }

    try {
      await apiService.cancelVideoAnalysis(analysis.jobId);
    } catch (error) {
      // Ignore cancel errors and reset UI state anyway.
    }
    resetState();
  };

  const handleRetry = () => {
    resetState(true);
  };

  useEffect(() => {
    if (!retryAt) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      if (Date.now() >= retryAt) {
        setRetryAt(null);
      }
    }, 500);
    return () => window.clearInterval(timer);
  }, [retryAt]);

  const chartMeta = useMemo(() => {
    const maxTime = analysis.summary?.video_duration_s
      || chartPoints[chartPoints.length - 1]?.t
      || 1;
    return { maxTime };
  }, [analysis.summary, chartPoints]);

  const latestFrame = analysis.frames[0];
  const latestLabel = analysis.label || latestFrame?.label || 'NORMAL';
  const latestProb = typeof analysis.prob === 'number' ? analysis.prob : latestFrame?.prob_smooth;

  const formatProb = (value) => {
    if (typeof value !== 'number') {
      return '--';
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const renderChart = () => {
    const width = 720;
    const height = 180;
    const maxTime = chartMeta.maxTime || 1;
    const thresholdY = height - THRESHOLD * height;

    if (chartPoints.length === 0) {
      return (
        <div className="chart-placeholder">Chưa có dữ liệu biểu đồ</div>
      );
    }

    const toX = (t) => (t / maxTime) * width;
    const toY = (p) => height - p * height;

    const linePath = chartPoints
      .map((point, index) => `${index === 0 ? 'M' : 'L'} ${toX(point.t)} ${toY(point.p)}`)
      .join(' ');

    const areas = [];
    for (let i = 0; i < chartPoints.length - 1; i += 1) {
      const p1 = chartPoints[i];
      const p2 = chartPoints[i + 1];
      const above1 = p1.p >= THRESHOLD;
      const above2 = p2.p >= THRESHOLD;

      if (!above1 && !above2) {
        continue;
      }

      let startT = p1.t;
      let startP = p1.p;
      let endT = p2.t;
      let endP = p2.p;

      if (above1 && !above2) {
        const ratio = (THRESHOLD - p1.p) / (p2.p - p1.p);
        endT = p1.t + ratio * (p2.t - p1.t);
        endP = THRESHOLD;
      }

      if (!above1 && above2) {
        const ratio = (THRESHOLD - p1.p) / (p2.p - p1.p);
        startT = p1.t + ratio * (p2.t - p1.t);
        startP = THRESHOLD;
      }

      const x1 = toX(startT);
      const x2 = toX(endT);
      const y1 = toY(startP);
      const y2 = toY(endP);
      const path = `M ${x1} ${y1} L ${x2} ${y2} L ${x2} ${thresholdY} L ${x1} ${thresholdY} Z`;
      areas.push(path);
    }

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg">
        <line x1="0" y1={thresholdY} x2={width} y2={thresholdY} className="chart-threshold" />
        {areas.map((area, index) => (
          <path key={`${area}-${index}`} d={area} className="chart-area" />
        ))}
        <path d={linePath} className="chart-line" />
      </svg>
    );
  };

  return (
    <main className="page-shell">
      <div className="page-banner">
        <h1>Phân tích Video Offline</h1>
        <p>Tải lên video để AI phân tích toàn bộ khung hình, vẽ biểu đồ timeline xác suất bạo lực và bóc tách các phân đoạn sự cố.</p>
      </div>

      <section className="video-analysis-section">
        <div className="video-analysis-container">
          {analysis.status === 'error' && (
            <div className="error-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
              <AlertCircle size={32} style={{ color: 'var(--danger)', marginBottom: '10px' }} />
              <div className="error-title">Có lỗi xảy ra</div>
              <p>{analysis.error || 'Không thể phân tích video'}</p>
              <div className="error-actions">
                <button
                  onClick={handleRetry}
                  className="retry-btn"
                  disabled={Boolean(retryAt)}
                >
                  {retryAt ? 'Vui lòng chờ...' : 'Thử lại'}
                </button>
              </div>
            </div>
          )}

          <div className="upload-area-wrapper">
            {!selectedVideo ? (
              <label
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file) {
                    handleVideoChange({ target: { files: [file] } });
                  }
                }}
              >
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoChange}
                  style={{ display: 'none' }}
                />
                <UploadCloud size={48} className="upload-icon" style={{ color: 'var(--primary)', marginBottom: '12px' }} />
                <div className="upload-text">
                  <p className="upload-title">Chọn hoặc kéo thả video</p>
                  <p className="upload-hint">MP4, AVI, MOV, MKV - tối đa 500MB</p>
                </div>
              </label>
            ) : (
              <div className="video-selected">
                <div className="video-info">
                  <div className="video-icon-wrapper">
                    <FileVideo size={20} style={{ color: 'var(--primary)' }} />
                  </div>
                  <div className="video-details">
                    <p className="video-name">{selectedVideo.name}</p>
                    <p className="video-size">
                      {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleRemoveVideo}
                  className="remove-btn"
                  title="Xóa video"
                  disabled={isWorking}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!canStart}
            className={`analyze-btn ${canStart ? 'enabled' : 'disabled'}`}
          >
            {analysis.status === 'uploading' ? 'Đang tải lên...' : 'Bắt đầu phân tích'}
          </button>

          {isWorking && (
            <div className="progress-card">
              <div className="progress-header">
                <div>
                  <h3>Tiến trình phân tích</h3>
                  <p className="progress-subtitle">{analysis.filename || selectedVideo?.name}</p>
                </div>
                <span className={`status-pill ${analysis.status}`}>
                  {analysis.status === 'queued' ? 'Đang chờ' : 'Đang chạy'}
                </span>
              </div>

              {connectionWarning && (
                <div className="connection-warning">{connectionWarning}</div>
              )}

              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.round((analysis.progress || 0) * 100)}%` }}
                />
              </div>
              <div className="progress-meta">
                <span>{Math.round((analysis.progress || 0) * 100)}%</span>
                <span>{analysis.processed || 0}/{analysis.totalFrames || '--'} frames</span>
              </div>

              <div className="frame-panel">
                <div className="thumb-box">
                  {analysis.latestThumb ? (
                    <img
                      src={thumbToSrc(analysis.latestThumb)}
                      alt="Latest frame"
                    />
                  ) : (
                    <div className="thumb-placeholder">Chưa có ảnh</div>
                  )}
                </div>
                <div className="frame-meta">
                  <span
                    className="label-pill"
                    style={{ background: getLabelColor(latestLabel) }}
                  >
                    {latestLabel}
                  </span>
                  <div className="frame-detail">
                    <span>Confidence:</span>
                    <strong>{formatProb(latestProb)}</strong>
                  </div>
                  {latestFrame?.timestamp && (
                    <div className="frame-detail">
                      <span>Time:</span>
                      <strong>{latestFrame.timestamp}</strong>
                    </div>
                  )}
                </div>
                <button className="cancel-btn" onClick={handleCancel}>Huỷ phân tích</button>
              </div>

              <div className="timeline-card">
                <div className="timeline-header">
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                    <Activity size={16} style={{ color: 'var(--primary)' }} />
                    <span>Timeline xác suất</span>
                  </h4>
                  <span>{formatVideoTime(chartMeta.maxTime)}s</span>
                </div>
                {renderChart()}
              </div>
            </div>
          )}

          {analysis.status === 'done' && analysis.summary && (
            <div className="summary-card">
              <div 
                className={`verdict ${analysis.summary.verdict === 'VIOLENCE' ? 'danger' : 'safe'}`}
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
              >
                {analysis.summary.verdict === 'VIOLENCE' ? (
                  <>
                    <AlertOctagon size={20} />
                    <span>PHÁT HIỆN BẠO LỰC</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={20} />
                    <span>BÌNH THƯỜNG</span>
                  </>
                )}
              </div>

              <div className="summary-grid">
                <div>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Film size={14} />
                    Tổng frames
                  </span>
                  <strong>{analysis.summary.total_frames}</strong>
                </div>
                <div>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} />
                    Thời lượng
                  </span>
                  <strong>{analysis.summary.video_duration_s.toFixed(1)}s</strong>
                </div>
                <div>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Percent size={14} />
                    Tỉ lệ bạo lực
                  </span>
                  <strong>{(analysis.summary.violence_ratio * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={14} />
                    Xác suất cao nhất
                  </span>
                  <strong>{(analysis.summary.max_prob * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Cpu size={14} />
                    Thời gian xử lý
                  </span>
                  <strong>{analysis.summary.processing_time_s.toFixed(1)}s</strong>
                </div>
              </div>

              <div className="timeline-card">
                <div className="timeline-header">
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                    <Activity size={16} style={{ color: 'var(--primary)' }} />
                    <span>Timeline xác suất</span>
                  </h4>
                  <span>{formatVideoTime(chartMeta.maxTime)}s</span>
                </div>
                {renderChart()}
              </div>

              <div className="segment-table">
                <div className="segment-row header">
                  <span>Bắt đầu</span>
                  <span>Kết thúc</span>
                  <span>Độ dài</span>
                  <span>Xác suất cao nhất</span>
                </div>
                {analysis.summary.violence_segments?.length ? (
                  analysis.summary.violence_segments.map((segment, index) => (
                    <div className="segment-row" key={`${segment.start_s}-${index}`}>
                      <span>{formatVideoTime(segment.start_s)}</span>
                      <span>{formatVideoTime(segment.end_s)}</span>
                      <span>{(segment.end_s - segment.start_s).toFixed(1)}s</span>
                      <span>{(segment.max_prob * 100).toFixed(0)}%</span>
                    </div>
                  ))
                ) : (
                  <div className="segment-row empty">Không có đoạn bạo lực</div>
                )}
              </div>

              <button className="reset-btn" onClick={() => resetState(true)}>Phân tích video khác</button>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

export default VideoAnalysisPage;
