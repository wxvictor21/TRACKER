
import React, {useState} from 'react'
import { moveCamera, captureImage } from '../services/api'
import SliderInput from './SliderInput'

export default function MotorControl(){
  const [x,setX]=useState(-150)
  const [y,setY]=useState(-180)
  const [status,setStatus]=useState('')

  async function onCapture(){
    setStatus('Moving...')
    const moveRes = await moveCamera(Number(x), Number(y), 1500)
    setStatus(JSON.stringify(moveRes))
    if (!moveRes.error) {
      setStatus('Capturing...')
      const captureRes = await captureImage()
      setStatus(JSON.stringify(captureRes))
    }
  }

  return (
    <div style={{border: '1px solid var(--border-color)', padding: '1rem', borderRadius: '5px', marginBottom: '1rem'}}>
      <h2>Manual Control</h2>
      <SliderInput label="X" value={x} onChange={setX} min={-150} max={150} />
      <SliderInput label="Y" value={y} onChange={setY} min={-180} max={180} />
      <div style={{display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1rem'}}>
        <button style={{padding: '0.5rem 1rem', backgroundColor: 'var(--button-bg-color)', color: 'white', border: 'none', borderRadius: '5px'}} onClick={onCapture}>Take</button>
      </div>
      <pre>{status}</pre>
    </div>
  )
}
