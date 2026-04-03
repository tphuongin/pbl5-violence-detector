import { useState, useEffect, useRef } from 'react';

function VideoPlayer({ cameraId, cameraName }) {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const imgRef = useRef(null);
  const connectionTimeoutRef = useRef(null);

  useEffect(() => {
    setIsLoading(true);
    setIsConnected(false);

    // Start refreshing frames from MJPEG stream
    const refreshFrame = () => {
      if (imgRef.current) {
        // Add timestamp to bypass cache
        const timestamp = new Date().getTime();
        imgRef.current.src = `http://localhost:8000/api/stream/${cameraId}/live?t=${timestamp}`;
      }
    };

    // Initial frame load
    refreshFrame();
    setIsLoading(false);

    // Set up connection timeout
    connectionTimeoutRef.current = setTimeout(() => {
      setIsConnected(false);
    }, 5000);

    const handleImageLoad = () => {
      setIsConnected(true);
      clearTimeout(connectionTimeoutRef.current);
      // Refresh every 100ms for smooth playback
      setTimeout(refreshFrame, 100);
    };

    const handleImageError = () => {
      setIsConnected(false);
      // Retry after 2 seconds
      setTimeout(refreshFrame, 2000);
    };

    if (imgRef.current) {
      imgRef.current.addEventListener('load', handleImageLoad);
      imgRef.current.addEventListener('error', handleImageError);
    }

    return () => {
      if (imgRef.current) {
        imgRef.current.removeEventListener('load', handleImageLoad);
        imgRef.current.removeEventListener('error', handleImageError);
      }
      clearTimeout(connectionTimeoutRef.current);
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
