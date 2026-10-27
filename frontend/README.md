# Road hazard frontend

Independent React/Vite application. Run it on a computer or web host **separate
from the Jetson Nano**. It needs no model files, Python environment, source files,
or shared filesystem from the detection API project.

## Development

Use Node.js 20.19+:

```sh
npm ci
cp .env.example .env
# Set VITE_API_BASE_URL to the Nano's address, e.g. http://192.168.1.50:8000
npm run dev
```

Open `http://localhost:5173`. You can also edit the API address in the UI without
rebuilding. On the detection server, allow the exact origin in the browser:

```sh
python3 -m road_hazard serve --config configs/jetson.json --host 0.0.0.0 --cors-origin http://localhost:5173
```

If you open the frontend using `http://192.168.1.20:5173`, allow that origin instead.
The API address must be reachable **from the browser**. `localhost` refers to
the browser's computer, not the Nano. Ports, scheme, and hostname matter for CORS.

Select an image or video, set confidence and frame stride, and run detection.
The UI uploads the raw file, polls the asynchronous job, shows the annotated
image or first video frame, and links to all output files. Keep the tab open
while processing; job IDs are not persisted across reloads. A connection failure
can be retried with **Refresh job** without uploading again. Finished jobs can
be deleted from the server using the result panel.

Video outputs are MJPEG/AVI without audio, provided as downloads rather than
requiring browser codec support. Higher strides leave skipped frames unmarked.
Video counts are observations across analyzed frames, not unique hazards.

## Build and host

```sh
npm test
npm run build
```

Deploy only `dist/` to any static web host. Vite substitutes `VITE_API_BASE_URL`
at **build time**; change it before building, or edit the address in the UI.
Do not deploy `vite preview` as a production server.

For an HTTPS frontend, the browser also needs an HTTPS API. Use a reverse proxy
on the frontend host to forward `/api/` to the Nano over the private network.
Customize `deploy/nginx.conf`, set `VITE_API_BASE_URL` to the full public
`https://your-host/api` address, configure TLS, and allow `https://your-host` in
the API CORS origins. This avoids mixed-content failures and keeps Node/Vite off
the Nano. The example's unauthenticated HTTP listener is for a trusted LAN;
add authentication before exposing detection remotely.

## API contract

This client consumes version 1 of the detection API:

- `GET /health`
- `POST /v1/detect/image?confidence=…` or `/v1/detect/video?confidence=…&stride=…`
  with a File/Blob as the raw body. The response is HTTP 202 with `id` and `status`.
- `GET /v1/jobs/{id}` returns progress and `result` when completed, or `error`
  when failed.
- `GET /v1/jobs/{id}/files/{filename}` downloads an advertised result artifact.
- `DELETE /v1/jobs/{id}` removes completed/failed jobs.

The backend runs independently and does not serve this application. There is no
Vite filesystem plugin, localhost-only proxy, mapping dataset, or cloud service.
