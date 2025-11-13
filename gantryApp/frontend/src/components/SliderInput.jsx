import React, { useState, useEffect } from 'react';

export default function SliderInput({ label, value, onChange, min = -100, max = 100 }) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(value);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleValueClick = () => {
    setIsEditing(true);
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    setIsEditing(false);
    const numValue = Number(inputValue);
    if (!isNaN(numValue)) {
      onChange(Math.max(min, Math.min(max, numValue)));
    } else {
      setInputValue(value); // Revert to original if invalid
    }
  };

  const handleSliderChange = (e) => {
    onChange(Number(e.target.value));
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
      <label style={{ minWidth: '30px' }}>{label}:</label>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={handleSliderChange}
        style={{ flex: 1, cursor: 'pointer' }}
      />
      {isEditing ? (
        <input
          type="number"
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyPress={(e) => { if (e.key === 'Enter') handleInputBlur(); }}
          autoFocus
          style={{ width: '60px', padding: '0.3rem', textAlign: 'center' }}
        />
      ) : (
        <span
          onClick={handleValueClick}
          style={{ cursor: 'pointer', minWidth: '60px', textAlign: 'center', padding: '0.3rem', border: '1px solid var(--border-color)', borderRadius: '4px' }}
        >
          {value}
        </span>
      )}
    </div>
  );
}
