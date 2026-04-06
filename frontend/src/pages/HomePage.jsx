import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

function HomePage() {
  const [stats, setStats] = useState({
    cameras: 0,
    violenceRecords: 0,
    calls: 0,
    users: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const [camerasData, violenceData, callsData, usersData] = await Promise.all([
        apiService.getCameras(),
        apiService.getViolenceHistory(),
        apiService.getCalls(),
        apiService.getUsers(),
      ]);

      setStats({
        cameras: camerasData.count || 0,
        violenceRecords: violenceData.count || 0,
        calls: callsData.count || 0,
        users: usersData.count || 0,
      });
    } catch (err) {
      console.error('Error loading stats:', err);
    } finally {
      setLoading(false);
      console.log("data", camerasData)
    }
  };

  return (
    <main className="page-shell">
      <section className="intro-section">
        {!loading && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '15px',
              marginTop: '30px',
            }}
          >
            <div
              style={{
                backgroundColor: '#e3f2fd',
                padding: '20px',
                borderRadius: '8px',
                textAlign: 'center',
                border: '2px solid #2196f3',
              }}
            >
              <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#2196f3' }}>
                {stats.cameras}
              </div>
              <div style={{ color: '#666', marginTop: '8px' }}>Camera</div>
            </div>

            <div
              style={{
                backgroundColor: '#fff3e0',
                padding: '20px',
                borderRadius: '8px',
                textAlign: 'center',
                border: '2px solid #ff9800',
              }}
            >
              <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#ff9800' }}>
                {stats.violenceRecords}
              </div>
              <div style={{ color: '#666', marginTop: '8px' }}>Phát hiện bạo lực</div>
            </div>

            <div
              style={{
                backgroundColor: '#f3e5f5',
                padding: '20px',
                borderRadius: '8px',
                textAlign: 'center',
                border: '2px solid #9c27b0',
              }}
            >
              <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#9c27b0' }}>
                {stats.calls}
              </div>
              <div style={{ color: '#666', marginTop: '8px' }}>Cuộc gọi</div>
            </div>

            <div
              style={{
                backgroundColor: '#e8f5e9',
                padding: '20px',
                borderRadius: '8px',
                textAlign: 'center',
                border: '2px solid #4caf50',
              }}
            >
              <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#4caf50' }}>
                {stats.users}
              </div>
              <div style={{ color: '#666', marginTop: '8px' }}>Người dùng</div>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default HomePage;
