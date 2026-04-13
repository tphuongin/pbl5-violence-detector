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
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '30px 20px',
          borderRadius: '8px 8px 0 0',
          marginBottom: '0'
        }}>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '1.8em', fontWeight: '600' }}>Lịch sử phát hiện bạo lực</h1>
          <p style={{ margin: '0', fontSize: '0.95em', opacity: '0.9' }}>
            {violenceRecords.length > 0 
              ? `Tổng cộng ${violenceRecords.length} cảnh bạo lực được phát hiện` 
              : 'Chưa có cảnh bạo lực nào được phát hiện'}
          </p>
        </div>

        {/* Content */}
        <div style={{ padding: '20px' }}>
          {violenceRecords.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '60px 20px',
              color: '#999'
            }}>
              <div style={{ fontSize: '3em', marginBottom: '15px' }}>-</div>
              <p style={{ fontSize: '1em', margin: '0' }}>Không có dữ liệu phát hiện bạo lực</p>
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '20px'
            }}>
              {violenceRecords.map((record) => (
                <div
                  key={record.HistoryID}
                  style={{
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    padding: '20px',
                    backgroundColor: '#fff',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
                    transition: 'all 0.3s ease',
                    cursor: 'default',
                    overflow: 'hidden'
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
                  {/* Location Badge */}
                  <div style={{
                    display: 'inline-block',
                    backgroundColor: '#667eea',
                    color: 'white',
                    padding: '6px 12px',
                    borderRadius: '20px',
                    fontSize: '0.8em',
                    fontWeight: '600',
                    marginBottom: '12px'
                  }}>
                    {record.Location}
                  </div>

                  {/* Timestamp */}
                  <div style={{
                    color: '#666',
                    fontSize: '0.85em',
                    marginBottom: '12px',
                    paddingBottom: '12px',
                    borderBottom: '1px solid #f0f0f0'
                  }}>
                    {new Date(record.Timestamp).toLocaleString('vi-VN')}
                  </div>

                  {/* Confidence */}
                  <div style={{ marginBottom: '16px' }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '6px'
                    }}>
                      <span style={{ fontSize: '0.85em', color: '#666' }}>Độ tin cậy</span>
                      <span style={{
                        fontWeight: '600',
                        color: record.Confidence > 0.9 ? '#d32f2f' : '#f57c00',
                        fontSize: '0.9em'
                      }}>{(record.Confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{
                      width: '100%',
                      height: '6px',
                      backgroundColor: '#e0e0e0',
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${record.Confidence * 100}%`,
                        height: '100%',
                        backgroundColor: record.Confidence > 0.9 ? '#d32f2f' : '#f57c00',
                        borderRadius: '3px',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>

                  {/* Action Button */}
                  {record.ClipURL && (
                    <a
                      href={record.ClipURL}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        display: 'block',
                        width: '100%',
                        padding: '10px',
                        backgroundColor: '#667eea',
                        color: 'white',
                        textDecoration: 'none',
                        borderRadius: '6px',
                        textAlign: 'center',
                        fontSize: '0.9em',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s ease'
                      }}
                      onMouseEnter={(e) => e.target.style.backgroundColor = '#5568d3'}
                      onMouseLeave={(e) => e.target.style.backgroundColor = '#667eea'}
                    >
                      Xem clip
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

export default DetectionHistoryPage;
