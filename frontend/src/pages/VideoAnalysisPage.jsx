import { useState } from 'react';
import './VideoAnalysisPage.css';

function VideoAnalysisPage() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleVideoChange = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedVideo(file);
    } else {
      alert('Vui lòng chọn tệp video hợp lệ');
    }
  };

  const handleRemoveVideo = () => {
    setSelectedVideo(null);
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      alert('Vui lòng chọn một video để phân tích');
      return;
    }

    // TODO: Implement video analysis API call
    console.log('Analyzing video:', selectedVideo.name);
    alert('Bắt đầu phân tích video: ' + selectedVideo.name);
  };

  return (
    <main className="page-shell">
      <section className="video-analysis-section">
        <div className="video-analysis-header">
          <h1>Phân tích video</h1>
          <p className="subtitle">Chọn video bạn muốn phân tích</p>
        </div>

        <div className="video-analysis-container">
          <div className="upload-area-wrapper">
            {!selectedVideo ? (
              <label
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file && file.type.startsWith('video/')) {
                    setSelectedVideo(file);
                  } else {
                    alert('Vui lòng chọn tệp video hợp lệ');
                  }
                }}
              >
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoChange}
                  style={{ display: 'none' }}
                />
                <div className="upload-icon">📹</div>
                <div className="upload-text">
                  <p className="upload-title">Chọn hoặc kéo thả video</p>
                  <p className="upload-hint">Nhấp vào đây để chọn tệp video từ máy tính</p>
                </div>
              </label>
            ) : (
              <div className="video-selected">
                <div className="video-info">
                  <div className="video-icon-wrapper">
                    <div className="video-icon">📹</div>
                  </div>
                  <div className="video-details">
                    <p className="video-name">{selectedVideo.name}</p>
                    <p className="video-size">
                      {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleRemoveVideo}
                  className="remove-btn"
                  title="Xóa video"
                >
                  ✕
                </button>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!selectedVideo}
            className={`analyze-btn ${selectedVideo ? 'enabled' : 'disabled'}`}
          >
            Phân tích
          </button>
        </div>
      </section>
    </main>
  );
}

export default VideoAnalysisPage;
