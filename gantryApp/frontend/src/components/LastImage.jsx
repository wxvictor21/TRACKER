import React, { useState, useEffect } from 'react';
import { gallery, API_BASE } from '../services/api';

export default function LastImage() {
  const [lastImage, setLastImage] = useState(null);

  useEffect(() => {
    const fetchLastImage = async () => {
      try {
        const response = await gallery();
        if (response.photos && response.photos.length > 0) {
          const last = response.photos[response.photos.length - 1];
          setLastImage(last);
        }
      } catch (error) {
        console.error('Error fetching last image:', error);
      }
    };

    fetchLastImage();
    const interval = setInterval(fetchLastImage, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ marginLeft: '2rem' }}>
      <h2>Última Imagen Guardada</h2>
      {lastImage ? (
        <img src={API_BASE + lastImage} alt="Last saved" style={{ width: '100%', border: '1px solid #ccc', padding: '0.5rem', borderRadius: '15px' }} />
      ) : (
        <p>No hay imágenes guardadas.</p>
      )}
    </div>
  );
}
