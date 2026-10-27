import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBase, artifactUrl, request } from '../src/client.js';

test('API URLs support a remote host and reverse proxy prefix', () => {
  assert.equal(normalizeBase('http://192.168.1.50:8000/'), 'http://192.168.1.50:8000');
  assert.equal(normalizeBase('https://example.com/api/'), 'https://example.com/api');
  for (const url of ['file:///tmp/file', 'ftp://host', 'http://user:pass@host', 'http://host?query=1']) {
    assert.throws(() => normalizeBase(url));
  }
  assert.equal(artifactUrl('https://example.com/api', 'abc', 'result.json'), 'https://example.com/api/v1/jobs/abc/files/result.json');
});

test('HTTP errors are surfaced and 204 deletion needs no JSON', async () => {
  const original = global.fetch;
  try {
    global.fetch = async () => new Response(JSON.stringify({ error: 'Detection queue is full' }), { status: 429 });
    await assert.rejects(request('http://jetson', '/v1/detect/image'), /queue is full/);
    global.fetch = async () => new Response(null, { status: 204 });
    assert.equal(await request('http://jetson', '/v1/jobs/abc', { method: 'DELETE' }), null);
  } finally { global.fetch = original; }
});
