import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

function DetectionHistoryPage() {
  const [violenceRecords, setViolenceRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadViolenceHistory();
  }, []);

  const loadViolenceHistory = async () => {
    try {
      setLoading(true);
      const data = await apiService.getViolenceHistory();
      setViolenceRecords(data.data || []);
      setError(null);
    } catch (err) {
      setError('Không thể tải lịch sử phát hiện');
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
        {violenceRecords.length === 0 ? (
          <p className="info-page-text">Các cảnh bạo lực, nếu có, sẽ hiển thị tại đây</p>
        ) : (
          <div>
            <p className="info-page-text">Tổng cộng: {violenceRecords.length} cảnh bạo lực được phát hiện</p>
            <div style={{ marginTop: '20px' }}>
              {violenceRecords.map((record) => (
                <div
                  key={record.HistoryID}
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
                        📍 {record.Location}
                      </div>
                      <div style={{ color: '#666', fontSize: '0.9em', marginTop: '5px' }}>
                        ⏰ {new Date(record.Timestamp).toLocaleString('vi-VN')}
                      </div>
                      <div style={{ marginTop: '10px' }}>
                        <span
                          style={{
                            display: 'inline-block',
                            backgroundColor: record.Confidence > 0.9 ? '#ff6b6b' : '#ffa500',
                            color: 'white',
                            padding: '5px 10px',
                            borderRadius: '4px',
                            fontSize: '0.85em',
                          }}
                        >
                          Độ tin cậy: {(record.Confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {record.ClipURL && (
                      <a
                        href={record.ClipURL}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          display: 'inline-block',
                          padding: '8px 16px',
                          backgroundColor: '#007bff',
                          color: 'white',
                          textDecoration: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                        }}
                      >
                        Xem clip
                      </a>
                    )}
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

export default DetectionHistoryPage;
