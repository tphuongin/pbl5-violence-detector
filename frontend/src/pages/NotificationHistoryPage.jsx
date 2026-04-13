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
        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div style={{
            fontSize: '1.1em',
            color: '#666',
            fontWeight: '500'
          }}>Đang tải dữ liệu...</div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="page-shell">
        <div style={{
          backgroundColor: '#ffebee',
          border: '1px solid #ef5350',
          borderRadius: '8px',
          padding: '20px',
          color: '#c62828',
          margin: '20px'
        }}>{error}</div>
      </main>
    );
  }

  return (
    <main className="page-shell">
      <section style={{ padding: '0' }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
          color: 'white',
          padding: '30px 20px',
          borderRadius: '8px 8px 0 0',
          marginBottom: '0'
        }}>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '1.8em', fontWeight: '600' }}>Lịch sử cuộc gọi cảnh báo</h1>
          <p style={{ margin: '0', fontSize: '0.95em', opacity: '0.9' }}>
            {calls.length > 0 
              ? `Tổng cộng ${calls.length} cuộc gọi cảnh báo` 
              : 'Chưa có cuộc gọi cảnh báo nào'}
          </p>
        </div>

        {/* Content */}
        <div style={{ padding: '20px' }}>
          {calls.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '60px 20px',
              color: '#999'
            }}>
              <div style={{ fontSize: '3em', marginBottom: '15px' }}>-</div>
              <p style={{ fontSize: '1em', margin: '0' }}>Không có cuộc gọi cảnh báo nào</p>
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '20px'
            }}>
              {calls.map((call, index) => (
                <div
                  key={call.CallID}
                  style={{
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    padding: '20px',
                    backgroundColor: '#fff',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
                    transition: 'all 0.3s ease',
                    cursor: 'default',
                    overflow: 'hidden',
                    position: 'relative'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.08)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  {/* Order Number */}
                  <div style={{
                    position: 'absolute',
                    top: '12px',
                    right: '12px',
                    backgroundColor: '#f5576c',
                    color: 'white',
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.85em',
                    fontWeight: '600'
                  }}>
                    {index + 1}
                  </div>

                  {/* Call Type */}
                  <div style={{
                    marginBottom: '16px',
                    paddingRight: '45px'
                  }}>
                    <div style={{
                      display: 'inline-block',
                      backgroundColor: '#f5576c',
                      color: 'white',
                      padding: '8px 16px',
                      borderRadius: '20px',
                      fontSize: '0.85em',
                      fontWeight: '600'
                    }}>
                      Cuộc gọi cảnh báo
                    </div>
                  </div>

                  {/* Call Time */}
                  <div style={{
                    color: '#555',
                    fontSize: '0.9em',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <span style={{
                      color: '#f5576c',
                      fontWeight: '600',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '20px',
                      height: '20px',
                      fontSize: '0.85em'
                    }}>→</span>
                    {new Date(call.CallDate).toLocaleString('vi-VN')}
                  </div>

                  {/* Call Status Indicator */}
                  <div style={{
                    marginTop: '16px',
                    paddingTop: '12px',
                    borderTop: '1px solid #f0f0f0',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      backgroundColor: '#4caf50',
                      borderRadius: '50%',
                      animation: 'pulse 2s infinite'
                    }}></div>
                    <span style={{
                      fontSize: '0.8em',
                      color: '#666',
                      fontWeight: '500'
                    }}>Đã gửi</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </main>
  );
}

export default NotificationHistoryPage;
