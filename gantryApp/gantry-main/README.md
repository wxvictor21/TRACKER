# Gantry Project - Flask API + React frontend (starter)

This repository contains your original Flask-based gantry control code (in `main.py`, `grbl_module.py`, `camera_module.py`) and a new API wrapper (`api.py`) plus a starter React frontend skeleton in `frontend/`.

## What I added
- `api.py`: a Flask JSON API that exposes endpoints:
  - `GET /api/status` -> returns GRBL status
  - `POST /api/move` -> JSON body `{"x":..,"y":..,"f":..}` to move motors
  - `GET /api/gallery` -> lists saved photos
  - `GET /api/photos/<filename>` -> serves saved images
  - `POST /api/sequence` -> run a simple capture sequence (body params: num_shots, step_x, step_y, start_x, start_y)

- `requirements.txt`: dependencies for the server side.

- `frontend/`: starter React + Vite project skeleton (files created, run `npm install` to install dependencies).

## How to run the backend (development)
1. Create a Python virtualenv (recommended) and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   (On Raspberry Pi you may need system packages for Aravis and OpenCV.)
3. Run the API:
   ```bash
   python api.py
   ```
   The API will listen on port 5000.

## How to run the frontend (development)
1. Go to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies and run dev server:
   ```bash
   npm install
   npm run dev
   ```
   Vite will start and serve the React app (usually on port 5173). Configure `src/services/api.js` to point to your Raspberry Pi IP.

## Production notes
- Do not use `CORS(app)` with `allow_origins="*"` in production. Restrict origins.
- Use a proper WSGI server (gunicorn) and a reverse proxy (nginx) for serving static files and handling SSL.
- Build the React app (`npm run build`) and serve `frontend/dist` with nginx or copy the files into the Flask `static/` directory and serve from there.

## Next steps I can do for you (pick any)
- Implement a full React UI that replaces the Flask templates and uses the API.
- Add WebSocket (Socket.IO) support to stream status/position updates in real time.
- Create a production-ready deployment script (systemd + gunicorn + nginx + SSL).
- Adapt the API to FastAPI if you prefer async support.