export const API_BASE_URL = 'http://localhost:8000/api';
// const API_BASE_URL = 'http://192.168.137.1:8000/api'; 

export const WS_BASE_URL = API_BASE_URL
  .replace(/\/api\/?$/, '')
  .replace(/^http/, 'ws');

console.log('API Base URL:', API_BASE_URL);

// Helper for authenticated requests
export const authFetch = async (url, options = {}) => {
  const token = localStorage.getItem('token');
  const headers = { ...options.headers };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return fetch(url, { ...options, headers });
};

export const apiService = {
  // Auth
  login: async (username, password) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Đăng nhập thất bại');
      }
      return data;
    } catch (error) {
      console.error('Error during login:', error);
      throw error;
    }
  },

  logout: async () => {
    try {
      await authFetch(`${API_BASE_URL}/auth/logout`, {
        method: 'POST',
      });
    } catch (error) {
      console.error('Error during logout:', error);
    } finally {
      localStorage.removeItem('token');
      localStorage.removeItem('username');
      localStorage.removeItem('userId');
    }
  },

  // Cameras
  getCameras: async () => {
    try {
      console.log('Fetching cameras from:', `${API_BASE_URL}/cameras`);
      const response = await authFetch(`${API_BASE_URL}/cameras`);
      console.log('Cameras response status:', response.status);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      console.log('Cameras data:', data);
      return data;
    } catch (error) {
      console.error('Error fetching cameras:', error);
      return { count: 0, data: [] };
    }
  },

  getCameraById: async (cameraId) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/cameras/${cameraId}`);
      if (!response.ok) throw new Error('Failed to fetch camera');
      return await response.json();
    } catch (error) {
      console.error('Error fetching camera:', error);
      return null;
    }
  },

  createCamera: async (cameraData) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cameraData),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Không thể tạo camera mới');
      }
      return await response.json();
    } catch (error) {
      console.error('Error creating camera:', error);
      throw error;
    }
  },

  updateCamera: async (cameraId, cameraData) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/cameras/${cameraId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cameraData),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Không thể cập nhật camera');
      }
      return await response.json();
    } catch (error) {
      console.error('Error updating camera:', error);
      throw error;
    }
  },

  deleteCamera: async (cameraId) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/cameras/${cameraId}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Không thể xóa camera');
      }
      return await response.json();
    } catch (error) {
      console.error('Error deleting camera:', error);
      throw error;
    }
  },

  muteCameraBuzzer: async (cameraId) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/cameras/${cameraId}/mute`, {
        method: 'POST',
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Không thể tắt còi');
      }
      return await response.json();
    } catch (error) {
      console.error('Error muting camera buzzer:', error);
      throw error;
    }
  },

  getLatestDetection: async (cameraId) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/detection/${cameraId}/latest`);
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) throw new Error(`Failed to fetch detection: HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching latest detection:', error);
      return null;
    }
  },

  // Violence History
  getViolenceHistory: async (page = 1, pageSize = 12) => {
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
      });
      const response = await authFetch(`${API_BASE_URL}/violence-history?${params.toString()}`);
      if (!response.ok) throw new Error('Failed to fetch violence history');
      return await response.json();
    } catch (error) {
      console.error('Error fetching violence history:', error);
      return { total: 0, data: [] };
    }
  },

  getViolenceById: async (historyId) => {
    try {
      const response = await authFetch(`${API_BASE_URL}/violence-history/${historyId}`);
      if (!response.ok) throw new Error('Failed to fetch violence record');
      return await response.json();
    } catch (error) {
      console.error('Error fetching violence record:', error);
      return null;
    }
  },

  // Users
  getUsers: async () => {
    try {
      const response = await authFetch(`${API_BASE_URL}/users`);
      if (!response.ok) throw new Error('Failed to fetch users');
      return await response.json();
    } catch (error) {
      console.error('Error fetching users:', error);
      return { count: 0, data: [] };
    }
  },

  // Calls
  getCalls: async () => {
    try {
      const response = await authFetch(`${API_BASE_URL}/calls`);
      if (!response.ok) throw new Error('Failed to fetch calls');
      return await response.json();
    } catch (error) {
      console.error('Error fetching calls:', error);
      return { count: 0, data: [] };
    }
  },

  analyzeVideo: async (file) => {
    const formData = new FormData();
    formData.append('video', file);

    const response = await authFetch(`${API_BASE_URL}/analyze-video`, {
      method: 'POST',
      body: formData,
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = { detail: 'Invalid response from server' };
    }

    return { status: response.status, payload };
  },

  getVideoAnalysisStatus: async (jobId) => {
    const response = await authFetch(`${API_BASE_URL}/video-analysis/${jobId}/status`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  },

  cancelVideoAnalysis: async (jobId) => {
    const response = await authFetch(`${API_BASE_URL}/video-analysis/${jobId}/cancel`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  },
};
