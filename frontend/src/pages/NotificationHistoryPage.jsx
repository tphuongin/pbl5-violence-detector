import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

function NotificationHistoryPage() {
  const [calls, setCalls] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCalls();
  }, []);

  const loadCalls = async () => {
    try {
      setLoading(true);
      const data = await apiService.getCalls();
      setCalls(data.data || []);
      setError(null);
    } catch (err) {
      setError('Không thể tải lịch sử cuộc gọi');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

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
      <div className="page-banner">
        <h1>Lịch Sử Cuộc Gọi Cảnh Báo</h1>
        <p>
          {calls.length > 0 
            ? `Tổng cộng đã gửi ${calls.length} cuộc gọi khẩn cấp cho cơ quan chức năng hoặc người giám sát` 
            : 'Chưa có cuộc gọi cảnh báo nào được thực hiện'}
        </p>
      </div>

      {/* Content */}
      <section style={{ padding: '0' }}>
        {calls.length === 0 ? (
          <div className="camera-card" style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: 'var(--muted)',
            fontWeight: '600'
          }}>
            <p style={{ margin: 0 }}>Không có cuộc gọi cảnh báo nào</p>
          </div>
        ) : (
          <div className="history-grid">
            {calls.map((call, index) => (
              <div key={call.CallID} className="history-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '160px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <span style={{
                      backgroundColor: 'var(--danger-light)',
                      color: 'var(--danger)',
                      padding: '4px 12px',
                      borderRadius: '99px',
                      fontSize: '0.75rem',
                      fontWeight: '800',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase'
                    }}>
                      📞 Cuộc gọi khẩn
                    </span>
                    <span style={{
                      fontSize: '0.85rem',
                      fontWeight: '800',
                      color: 'var(--muted)',
                      backgroundColor: 'var(--surface-strong)',
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1px solid var(--line)'
                    }}>
                      #{index + 1}
                    </span>
                  </div>

                  {/* Call Date/Time */}
                  <div style={{
                    color: 'var(--text)',
                    fontSize: '0.95rem',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    margin: '12px 0 6px 0'
                  }}>
                    <span style={{ color: 'var(--danger)' }}>⏱</span>
                    {new Date(call.CallDate).toLocaleString('vi-VN')}
                  </div>
                </div>

                {/* Call Status Indicator */}
                <div style={{
                  paddingTop: '12px',
                  borderTop: '1px solid var(--line)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginTop: '12px'
                }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    backgroundColor: 'var(--success)',
                    borderRadius: '50%',
                    boxShadow: '0 0 0 2px rgba(16, 185, 129, 0.2)'
                  }}></div>
                  <span style={{
                    fontSize: '0.8rem',
                    color: 'var(--muted)',
                    fontWeight: '700'
                  }}>Đã hoàn thành</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

export default NotificationHistoryPage;
