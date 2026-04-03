const API_BASE_URL = 'http://localhost:8000/api';

console.log('API Base URL:', API_BASE_URL);

export const apiService = {
  // Cameras
  getCameras: async () => {
    try {
      console.log('Fetching cameras from:', `${API_BASE_URL}/cameras`);
      const response = await fetch(`${API_BASE_URL}/cameras`);
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
      const response = await fetch(`${API_BASE_URL}/cameras/${cameraId}`);
      if (!response.ok) throw new Error('Failed to fetch camera');
      return await response.json();
    } catch (error) {
      console.error('Error fetching camera:', error);
      return null;
    }
  },

  // Violence History
  getViolenceHistory: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/violence-history`);
      if (!response.ok) throw new Error('Failed to fetch violence history');
      return await response.json();
    } catch (error) {
      console.error('Error fetching violence history:', error);
      return { count: 0, data: [] };
    }
  },

  getViolenceById: async (historyId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/violence-history/${historyId}`);
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
      const response = await fetch(`${API_BASE_URL}/users`);
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
      const response = await fetch(`${API_BASE_URL}/calls`);
      if (!response.ok) throw new Error('Failed to fetch calls');
      return await response.json();
    } catch (error) {
      console.error('Error fetching calls:', error);
      return { count: 0, data: [] };
    }
  },
};
