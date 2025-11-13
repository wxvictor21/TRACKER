
export const API_BASE = "http://192.168.2.67:5000"; // Adjust to your Raspberry Pi IP and port

export async function moveCamera(x,y,f=1500){
  const res = await fetch(`${API_BASE}/api/move`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({x,y,f})
  });
  return res.json();
}

export async function captureImage(){
  const res = await fetch(`${API_BASE}/api/capture`, {method:'POST'}).catch(e=>null);
  // Fallback to /api/gallery capture behaviour - many camera modules save image on capture and return file
  const j = res ? await res.json() : {};
  return j;
}

export async function gallery(){
  const res = await fetch(`${API_BASE}/api/gallery`);
  return res.json();
}
