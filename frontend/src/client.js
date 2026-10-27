export function normalizeBase(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Enter an HTTP or HTTPS API address without credentials, a query, or a fragment.');
  }
  return url.href.replace(/\/$/, '');
}

export function artifactUrl(base, id, filename) {
  return `${base}/v1/jobs/${encodeURIComponent(id)}/files/${encodeURIComponent(filename)}`;
}

export async function request(base, path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error || `API request failed (${response.status})`);
  if (!body) throw new Error('The server returned an invalid response. Check the API address.');
  return body;
}
