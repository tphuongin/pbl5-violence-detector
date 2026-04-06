import { useState, useEffect, useRef } from 'react';

function VideoPlayer({ cameraId, cameraName }) {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const imgRef = useRef(null);
  const wsRef = useRef(null);
  const retryTimeoutRef = useRef(null);

  useEffect(() => {
    setIsLoading(true);
    setIsConnected(false);

    const connectWebSocket = () => {
      const wsUrl = `ws://192.168.137.1:8000/ws/view/${cameraId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setIsLoading(false);
        if (retryTimeoutRef.current) {
          clearTimeout(retryTimeoutRef.current);
        }
      };

      wsRef.current.onmessage = (event) => {
        const arrayBuffer = event.data;
        const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        if (imgRef.current) {
          imgRef.current.src = url;
        }
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        setIsLoading(false);
        if (retryTimeoutRef.current) {
          clearTimeout(retryTimeoutRef.current);
        }
        retryTimeoutRef.current = window.setTimeout(() => {
          connectWebSocket();
        }, 2000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        setIsLoading(false);
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [cameraId]);

  return (
    <div
      style={{
        width: '100%',
        backgroundColor: '#000',
        borderRadius: '8px',
        overflow: 'hidden',
        position: 'relative',
        aspectRatio: '16 / 9',
      }}
    >
      {isLoading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#000',
            zIndex: 10,
          }}
        >
          <div style={{ color: '#fff', textAlign: 'center' }}>
            <p style={{ fontSize: '1.2em', marginBottom: '10px' }}>⏳ Đang kết nối...</p>
            <p style={{ fontSize: '0.9em', color: '#aaa' }}>{cameraName}</p>
          </div>
        </div>
      )}

      {!isLoading && !isConnected && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#000',
            zIndex: 10,
          }}
        >
          <div style={{ color: '#f44336', textAlign: 'center' }}>
            <p style={{ fontSize: '1.5em', marginBottom: '10px' }}>No Signal</p>
            <p style={{ fontSize: '0.9em', color: '#aaa' }}>{cameraName}</p>
            <p style={{ fontSize: '0.8em', color: '#888', marginTop: '10px' }}>
              Đang cố gắng kết nối...
            </p>
          </div>
        </div>
      )}

      <img
        ref={imgRef}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: isLoading || !isConnected ? 'none' : 'block',
        }}
        alt={cameraName}
      />
    </div>
  );
}

export default VideoPlayer;
