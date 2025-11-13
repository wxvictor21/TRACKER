import React, { useState, useEffect } from 'react';
import { gallery, API_BASE } from '../services/api';

export default function Gallery() {
  const [photos, setPhotos] = useState([]);

  useEffect(() => {
    async function loadGallery() {
      const res = await gallery();
      if (res.photos) {
        setPhotos(res.photos);
      }
    }
    loadGallery();
  }, []);

  return (
    <div>
      <h1>Gallery</h1>
      {photos.length > 0 ? (
        <div>
          {photos.map(photo => (
            <div key={photo} style={{display: 'inline-block', textAlign: 'center', margin: '1rem'}}>
              <img src={API_BASE + photo} style={{maxWidth: 400, border: '1px solid #ccc', padding: '0.5rem', borderRadius: '15px'}} /><br />
              <span>{photo}</span>
            </div>
          ))}
        </div>
      ) : (
        <p>No photos captured yet.</p>
      )}
    </div>
  );
}
