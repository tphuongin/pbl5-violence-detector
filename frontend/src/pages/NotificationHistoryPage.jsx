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
      <section className="info-page-card">
        {calls.length === 0 ? (
          <p className="info-page-text">Các cuộc gọi cảnh báo, nếu có, sẽ hiển thị tại đây</p>
        ) : (
          <div>
            <p className="info-page-text">Tổng cộng: {calls.length} cuộc gọi</p>
            <div style={{ marginTop: '20px' }}>
              {calls.map((call) => (
                <div
                  key={call.CallID}
                  style={{
                    border: '1px solid #ddd',
                    borderRadius: '8px',
                    padding: '15px',
                    marginBottom: '15px',
                    backgroundColor: '#f9f9f9',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: '1.1em' }}>
                        📞 Cuộc gọi cảnh báo
                      </div>
                      <div style={{ color: '#666', fontSize: '0.9em', marginTop: '5px' }}>
                        ⏰ {new Date(call.CallDate).toLocaleString('vi-VN')}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default NotificationHistoryPage;
