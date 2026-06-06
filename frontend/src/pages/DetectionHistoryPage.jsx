import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { Calendar, MapPin, ExternalLink, Database } from 'lucide-react';

function DetectionHistoryPage() {
  const [violenceRecords, setViolenceRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 9;
  const [hasCache, setHasCache] = useState(false);
  const cacheKey = 'violenceHistoryCacheV1';

  useEffect(() => {
    const raw = window.localStorage.getItem(cacheKey);
    if (raw) {
      try {
        const cached = JSON.parse(raw);
        if (Array.isArray(cached?.records)) {
          setViolenceRecords(cached.records);
          setTotal(cached.total || 0);
          setHasCache(true);
          setLoading(false);
        }
      } catch (err) {
        // Ignore cache parsing errors.
      }
    }

    loadViolenceHistory(page, { silent: hasCache });
  }, [page]);

  const loadViolenceHistory = async (nextPage, options = {}) => {
    try {
      if (!options.silent) {
        setLoading(true);
      }
      const data = await apiService.getViolenceHistory(nextPage, pageSize);
      setViolenceRecords(data.data || []);
      setTotal(data.total || 0);
      setError(null);

      if (nextPage === 1) {
        window.localStorage.setItem(
          cacheKey,
          JSON.stringify({
            updatedAt: Date.now(),
            records: data.data || [],
            total: data.total || 0,
          })
        );
        setHasCache(true);
      }
    } catch (err) {
      setError('Không thể tải lịch sử phát hiện');
      console.error(err);
    } finally {
      if (!options.silent) {
        setLoading(false);
      }
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

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
      {/* Unified Banner */}
      <div className="page-banner danger-theme">
        <h1>Lịch Sử Phát Hiện Bạo Lực</h1>
        <p>
          {total > 0 
            ? `Tổng cộng đã phát hiện ${total} sự cố bạo lực trên hệ thống` 
            : 'Chưa có sự cố bạo lực nào được ghi nhận'}
        </p>
      </div>

      {/* Content */}
      <section style={{ padding: 0 }}>
        {violenceRecords.length === 0 ? (
          <div className="camera-card" style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: 'var(--muted)',
            fontWeight: '600',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '12px'
          }}>
            <Database size={40} style={{ color: 'var(--muted)' }} />
            <p style={{ margin: 0 }}>Không có dữ liệu sự cố bạo lực nào được ghi nhận</p>
          </div>
        ) : (
          <div className="history-grid">
            {violenceRecords.map((record) => (
              <div key={record.HistoryID} className="history-card">
                {record.ClipURL && (
                  <video
                    src={record.ClipURL}
                    controls
                    preload="metadata"
                    className="history-video"
                  />
                )}
                
                {/* Location Badge */}
                <div className="history-location-badge" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <MapPin size={12} />
                  <span>{record.CameraName || record.Location || 'Không rõ vị trí'}</span>
                </div>

                {/* Timestamp */}
                <div className="history-time" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Calendar size={14} style={{ color: 'var(--muted)' }} />
                  <span>
                    {record.Timestamp
                      ? new Date(record.Timestamp).toLocaleString('vi-VN')
                      : 'Không rõ thời gian'}
                  </span>
                </div>

                {/* Confidence bar */}
                <div className="history-confidence-label">
                  <span>Độ tin cậy</span>
                  <span style={{
                    color: record.Confidence > 0.9 ? 'var(--danger)' : 'var(--warning)',
                    fontWeight: '700'
                  }}>
                    {(record.Confidence * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="history-confidence-bar-bg">
                  <div
                    className="history-confidence-bar-fill"
                    style={{
                      width: `${record.Confidence * 100}%`,
                      backgroundColor: record.Confidence > 0.9 ? 'var(--danger)' : 'var(--warning)'
                    }}
                  />
                </div>

                {/* Action Button */}
                {record.ClipURL && (
                  <a
                    href={record.ClipURL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="history-action-btn"
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                  >
                    <ExternalLink size={16} />
                    <span>Mở Clip chi tiết</span>
                  </a>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Unified Pagination */}
        {totalPages > 1 && (
          <div className="pagination-container">
            <button
              type="button"
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              disabled={page === 1}
              className="pagination-btn"
            >
              Trang trước
            </button>
            <span className="pagination-text">
              Trang {page} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page === totalPages}
              className="pagination-btn"
            >
              Trang sau
            </button>
          </div>
        )}
      </section>
    </main>
  );
}

export default DetectionHistoryPage;
