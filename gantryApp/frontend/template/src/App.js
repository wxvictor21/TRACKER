import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [images, setImages] = useState([
    '/provisional_images/1_20250915_121411.jpg',
    '/provisional_images/2_20250915_121411.jpg',
    '/provisional_images/3_20250915_121411.jpg',
    '/provisional_images/4_20250915_121411.jpg',
  ]);
  const [photosTakenTime, setPhotosTakenTime] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [zoomedImage, setZoomedImage] = useState(null);

  // Data for Overall Performance section
  const goodPieces = 85;
  const badPieces = 15;
  const totalPieces = goodPieces + badPieces;
  const goodPercentage = (goodPieces / totalPieces) * 100;
  const badPercentage = (badPieces / totalPieces) * 100;
  const objectivePercentage = 90; // Target: 90% good pieces

  useEffect(() => {
    if (images.length > 0) {
      const firstImageFilename = images[0].split('/').pop();
      const timestampMatch = firstImageFilename.match(/\d{8}_\d{6}/);
      if (timestampMatch) {
        const timestamp = timestampMatch[0];
        const year = timestamp.substring(0, 4);
        const month = timestamp.substring(4, 6);
        const day = timestamp.substring(6, 8);
        const hour = timestamp.substring(9, 11);
        const minute = timestamp.substring(11, 13);
        const second = timestamp.substring(13, 15);
        const formattedTime = `${day}/${month}/${year.substring(2)} ${hour}:${minute}:${second}`;
        setPhotosTakenTime(formattedTime);
      }
    }
  }, [images]);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleImageClick = (image) => {
    setZoomedImage(image);
  };

  const handleCloseZoom = () => {
    setZoomedImage(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src="/logoATRIA.png" className="logo" alt="ATRIA Logo" />
        <h1>Quality Control</h1>
        <img src="/logoCIE.png" className="logo" alt="CIE Logo" />
      </header>
      <div className="clock-container">
        <div className="datetime-wrapper">
          <p className="current-date">{currentTime.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: '2-digit' })}</p>
          <p className="current-time">{currentTime.toLocaleTimeString()}</p>
        </div>
      </div>

      <main>
        <h2>Last Images</h2>
        {photosTakenTime && <p className="photos-taken-time">Photos taken on: {photosTakenTime}</p>}
        <div className="image-gallery">
          {images.length > 0 ? (
            images.map((image, index) => (
              <div 
                key={index} 
                className={`image-item ${index === 0 ? 'image-item-red' : 'image-item-green'}`}
                onClick={() => handleImageClick(image)}
              >
                <img 
                  src={image} 
                  alt={`Inspection ${index + 1}`} 
                  className={'image-green-border'} 
                />
                <p className="image-filename">id: {image.split('/').pop().split('_')[0]}</p>
                <p className="image-full-filename">{image.split('/').pop()}</p>
              </div>
            ))
          ) : (
            <p>No images found.</p>
          )}
        </div>

        {/* Overall Performance Section */}
        <div className="graph-section">
          <h2>Overall Performance</h2>
          <p className="objective-text">Objective: {objectivePercentage}%</p>
          <div className="graph-data">
            <p>Total Pieces Inspected: {totalPieces}</p>
            <p>Good Pieces: {goodPieces} ({goodPercentage.toFixed(1)}%)</p>
            <p>Bad Pieces: {badPieces} ({badPercentage.toFixed(1)}%)</p>
          </div>
          <div className="bar-chart">
            <div className="bar good-bar" style={{ width: `${goodPercentage}%` }}>
              Good
            </div>
            <div className="bar bad-bar" style={{ width: `${badPercentage}%` }}>
              Bad
            </div>
            <div className="objective-line" style={{ left: `${objectivePercentage}%` }}>
            </div>
            <div className="objective-pointer-triangle" style={{ left: `${objectivePercentage}%` }}></div>
          </div>
        </div>
      </main>

      {zoomedImage && (
        <div className="zoomed-overlay" onClick={handleCloseZoom}>
          <img src={zoomedImage} alt="Zoomed Image" className="zoomed-image" />
        </div>
      )}
    </div>
  );
}

export default App;