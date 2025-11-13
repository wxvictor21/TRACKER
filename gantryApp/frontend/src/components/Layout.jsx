import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Layout.css';

export default function Layout({ children }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div>
      <head>
        <title>ABENICS CHAMBER</title>
      </head>
      <nav style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--header-bg-color)', color: 'var(--header-text-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src="/logoATRIA.png" alt="Atria logo" style={{ height: '50px', marginRight: '1rem' }} />
        </div>
        <h1 style={{ margin: '0', flexGrow: 1, textAlign: 'center' }}>ABENICS CHAMBER</h1>
        
        <div 
          style={{ position: 'relative', display: 'inline-block' }}
          onMouseEnter={() => setIsMenuOpen(true)}
          onMouseLeave={() => setIsMenuOpen(false)}
        >
          <img src="/menu_icon.jpg" alt="Menu" style={{ height: '40px', cursor: 'pointer', borderRadius: '5px' }} />
          
          {isMenuOpen && (
            <div style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              backgroundColor: 'white',
              border: '1px solid #ccc',
              borderRadius: '5px',
              boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
              zIndex: 1000,
              padding: '0.5rem 0',
              minWidth: '120px'
            }}>
              <Link to="/" style={{ display: 'block', padding: '0.5rem 1rem', color: 'black', textDecoration: 'none' }}>Control</Link>
              <Link to="/gallery" style={{ display: 'block', padding: '0.5rem 1rem', color: 'black', textDecoration: 'none' }}>Gallery</Link>
            </div>
          )}
        </div>

      </nav>
      <div style={{padding:20}}>
        {children}
      </div>
    </div>
  );
}
