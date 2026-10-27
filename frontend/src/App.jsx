import { useEffect, useRef, useState } from 'react';
import { artifactUrl, normalizeBase, request } from './client.js';

const defaultAddress = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const fileAccept = 'image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska,.mkv,.avi';

export default function App() {
  const [address, setAddress] = useState(defaultAddress);
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState('image');
  const [confidence, setConfidence] = useState(0.25);
  const [stride, setStride] = useState(1);
  const [job, setJob] = useState(null);
  const [jobBase, setJobBase] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [pollVersion, setPollVersion] = useState(0);
  const uploadController = useRef(null);
  const jobPending = job && ['queued', 'running'].includes(job.status);
  const locked = busy || jobPending;

  useEffect(() => () => uploadController.current?.abort(), []);

  useEffect(() => {
    if (!jobPending) return;
    let stopped = false;
    let timer;
    const controller = new AbortController();
    async function poll() {
      try {
        const latest = await request(jobBase, `/v1/jobs/${job.id}`, { signal: controller.signal });
        if (stopped) return;
        setJob(latest);
        if (latest.status === 'failed') setError(latest.error || 'Detection failed.');
        if (['queued', 'running'].includes(latest.status)) timer = setTimeout(poll, 1500);
      } catch (err) {
        if (!stopped) setError(`Could not read job status: ${err.message}. Use “Refresh job” to reconnect.`);
      }
    }
    poll();
    return () => { stopped = true; clearTimeout(timer); controller.abort(); };
  }, [job?.id, Boolean(jobPending), jobBase, pollVersion]);

  async function checkConnection() {
    setError('');
    setHealth(null);
    try {
      setHealth(await request(normalizeBase(address), '/health', { signal: AbortSignal.timeout(10000) }));
    } catch (err) { setError(err.message); }
  }

  async function run(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError('');
    setJob(null);
    const controller = new AbortController();
    uploadController.current = controller;
    try {
      const base = normalizeBase(address);
      const created = await request(base, `/v1/detect/${kind}?confidence=${confidence}&stride=${stride}`, {
        method: 'POST', headers: { 'Content-Type': file.type || 'application/octet-stream' },
        body: file, signal: controller.signal,
      });
      setJobBase(base);
      setJob(created);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); uploadController.current = null; }
  }

  async function deleteResult() {
    setBusy(true);
    setError('');
    try {
      await request(jobBase, `/v1/jobs/${job.id}`, { method: 'DELETE' });
      setJob(null);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  const result = job?.result;
  const preview = result?.artifacts.image || result?.artifacts.preview;
  const counts = result?.counts || result?.detection_observations || {};

  return <main>
    <header>
      <div className="brand-mark" aria-hidden="true">↗</div>
      <div><p className="eyebrow">ROAD INTELLIGENCE</p><h1>Hazard detection</h1></div>
      <span className="header-tag">Local inference</span>
    </header>
    <p className="intro">Analyze road images and videos. Your detection service processes each file and saves the annotated results.</p>

    <section className="connection" aria-label="API connection">
      <label htmlFor="address">Detection API</label>
      <div className="input-row"><input id="address" type="url" value={address} disabled={locked}
        onChange={event => { setAddress(event.target.value); setHealth(null); }} />
        <button type="button" className="secondary" onClick={checkConnection} disabled={busy}>Check connection</button></div>
      <p className={health ? 'connected' : 'muted'}>{health
        ? `Connected · ${health.backend} · ${health.models.join(', ')} · upload limit ${health.max_upload_mb} MB`
        : 'Enter the network address of your detection service.'}</p>
    </section>

    <div className="workspace">
      <form onSubmit={run} className="panel controls">
        <p className="eyebrow">01 / INPUT</p><h2>Choose your footage</h2>
        <label className="upload" htmlFor="file"><span className="upload-icon" aria-hidden="true">↑</span>
          <strong>{file ? file.name : 'Select an image or video'}</strong>
          <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'JPG, PNG, WebP, MP4, MOV, AVI or WebM'}</span>
          <input id="file" type="file" accept={fileAccept} required disabled={locked} onChange={event => {
            const chosen = event.target.files?.[0];
            setJob(null);
            setError('');
            setFile(chosen || null);
            setKind(chosen && (chosen.type.startsWith('video/') || /\.(mp4|mov|avi|webm|mkv)$/i.test(chosen.name)) ? 'video' : 'image');
          }} />
        </label>
        <label htmlFor="kind">Media type</label><select id="kind" value={kind} disabled={locked} onChange={event => setKind(event.target.value)}>
          <option value="image">Image</option><option value="video">Video</option></select>
        <label htmlFor="confidence">Minimum confidence <span>{Math.round(confidence * 100)}%</span></label>
        <input id="confidence" type="range" min="0.05" max="0.95" step="0.05" value={confidence} disabled={locked} onChange={event => setConfidence(Number(event.target.value))} />
        {kind === 'video' && <><label htmlFor="stride">Analyze every Nth frame</label>
          <input id="stride" type="number" min="1" max="10000" value={stride} disabled={locked} onChange={event => setStride(event.target.value)} required />
          <p className="muted">Use 1 to analyze every frame. Higher values reduce processing time; skipped frames remain unmarked. Output has no audio.</p></>}
        <button className="primary" type="submit" disabled={!file || locked}>{busy ? 'Uploading…' : jobPending ? 'Processing…' : 'Run detection →'}</button>
      </form>

      <section className="panel results" aria-label="Detection results" aria-live="polite">
        <div className="result-heading"><div><p className="eyebrow">02 / RESULTS</p><h2>{result ? 'Analysis complete' : 'Detection output'}</h2></div>
          {job && <span className={`status ${job.status}`}>{job.status}</span>}</div>
        {!job && <div className="empty"><span aria-hidden="true">⌖</span><h3>Ready when you are</h3><p>Annotated footage, detections, and downloadable files will appear here.</p></div>}
        {jobPending && <div className="empty"><div className="spinner" /><h3>{job.status === 'queued' ? 'Waiting for the detector' : 'Analyzing your footage'}</h3>
          <p>{job.progress ? `${job.progress.frames_processed} ${job.progress.total_frames ? `of ${job.progress.total_frames}` : ''} frames processed` : 'You can leave this running while the service processes the file.'}</p>
          <button type="button" className="secondary" onClick={() => { setError(''); setPollVersion(value => value + 1); }}>Refresh job</button></div>}
        {result && <>
          {preview && <img className="preview" src={artifactUrl(jobBase, job.id, preview)} alt="Road footage with detected hazards marked" />}
          {result.kind === 'video' && <p className="muted">Preview of the first frame. Download the annotated AVI to view the complete video.</p>}
          <div className="counts">{Object.keys(counts).length ? Object.entries(counts).map(([name, count]) => <div key={name}><strong>{count}</strong><span>{name}</span></div>) : <p>No hazards detected at this confidence.</p>}</div>
          <p className="muted">{result.elapsed_seconds.toFixed(1)} seconds · {result.width} × {result.height}
            {result.kind === 'video' && ` · ${result.inferred_frames} of ${result.frames} frames analyzed. Counts are observations across frames, not unique hazards.`}</p>
          <nav className="downloads" aria-label="Download results">{Object.entries(result.artifacts).map(([label, filename]) => <a key={label} href={artifactUrl(jobBase, job.id, filename)} target="_blank" rel="noreferrer">↓ {filename}</a>)}</nav>
        </>}
        {job && !jobPending && <button className="text-button" type="button" onClick={deleteResult} disabled={busy}>Delete this job and its saved files</button>}
      </section>
    </div>
    {error && <div className="error" role="alert">{error}</div>}
    <footer>Processed on your detection server · No cloud inference</footer>
  </main>;
}
