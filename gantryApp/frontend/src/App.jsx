import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import MotorControl from './components/MotorControl';
import Gallery from './components/Gallery';
import VideoPlayer from './components/VideoPlayer';

export default function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={
            <div>
              <h1>Control</h1>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '2rem' }}>
                <div style={{ flex: 1 }}>
                  <MotorControl />
                </div>
                <div style={{ flex: 1 }}>
                  <VideoPlayer />
                </div>
              </div>
            </div>
          } />
          <Route path="/gallery" element={<Gallery />} />
        </Routes>
      </Layout>
    </Router>
  );
}
