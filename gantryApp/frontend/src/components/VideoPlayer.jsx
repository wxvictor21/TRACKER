import React from 'react';

const VideoPlayer = () => {
  return (
    <div>
      <h2>Live Video</h2>
      <img src="http://localhost:5000/video_feed" alt="Video Stream" width="640" height="480" />
    </div>
  );
};

export default VideoPlayer;
