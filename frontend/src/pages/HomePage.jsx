import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { Video, AlertTriangle, PhoneCall, Users } from 'lucide-react';

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

      console.log('Fetched data:', { camerasData, violenceData, callsData, usersData });

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
    }
  };

  return (
    <main className="page-shell">
      <div className="page-banner">
        <h1>Hệ thống Giám sát & Phát hiện Bạo lực Realtime</h1>
        <p>Giải pháp AI tiên tiến giúp phát hiện hành vi bạo lực qua camera giám sát và tự động phát cảnh báo cuộc gọi tức thời.</p>
      </div>

      <section className="intro-section">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--muted)', fontWeight: '600' }}>
            Đang tải dữ liệu thống kê...
          </div>
        ) : (
          <div className="stats-grid">
            <div className="stat-card cameras">
              <Video size={36} style={{ color: '#2563eb', marginBottom: '12px' }} />
              <div className="stat-number">{stats.cameras}</div>
              <div className="stat-label">Camera Kết Nối</div>
            </div>

            <div className="stat-card violence">
              <AlertTriangle size={36} style={{ color: '#d97706', marginBottom: '12px' }} />
              <div className="stat-number">{stats.violenceRecords}</div>
              <div className="stat-label">Cảnh Báo Bạo Lực</div>
            </div>

            <div className="stat-card calls">
              <PhoneCall size={36} style={{ color: '#7c3aed', marginBottom: '12px' }} />
              <div className="stat-number">{stats.calls}</div>
              <div className="stat-label">Cuộc Gọi Đã Gửi</div>
            </div>

            <div className="stat-card users">
              <Users size={36} style={{ color: '#059669', marginBottom: '12px' }} />
              <div className="stat-number">{stats.users}</div>
              <div className="stat-label">Người Dùng</div>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default HomePage;
