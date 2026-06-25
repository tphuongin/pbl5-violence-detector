import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/apiService';
import { ShieldAlert, Lock, User, AlertCircle, Loader2 } from 'lucide-react';
import './LoginPage.css';

function LoginPage() {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError('Vui lòng nhập đầy đủ thông tin');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const data = await apiService.login(username, password);
      localStorage.setItem('token', data.token);
      localStorage.setItem('username', data.user.Username);
      localStorage.setItem('userId', data.user.UserID);
      
      // Redirect to home
      navigate('/');
    } catch (err) {
      setError(err.message || 'Tên đăng nhập hoặc mật khẩu không chính xác');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-overlay"></div>
      <div className="login-card">
        <div className="login-brand">
          <div className="login-logo-wrapper">
            <ShieldAlert size={36} />
          </div>
          <h2>Violence Detector</h2>
          <p>Hệ thống giám sát và phát hiện bạo lực</p>
        </div>

        {error && (
          <div className="login-error-alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleLogin} className="login-form">
          <div className="input-group">
            <label htmlFor="username">Tên đăng nhập</label>
            <div className="input-wrapper">
              <User size={18} className="input-icon" />
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Nhập tài khoản"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="password">Mật khẩu</label>
            <div className="input-wrapper">
              <Lock size={18} className="input-icon" />
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Nhập mật khẩu"
                disabled={isLoading}
              />
            </div>
          </div>

          <button type="submit" className="login-submit-btn" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 size={18} className="spin-icon" />
                <span>Đang xác thực...</span>
              </>
            ) : (
              <span>Đăng nhập</span>
            )}
          </button>
        </form>

        <div className="login-hint">
          <p>Tài khoản dùng thử mặc định:</p>
          <code>Tên đăng nhập: <strong>admin</strong> | Mật khẩu: <strong>admin123</strong></code>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
