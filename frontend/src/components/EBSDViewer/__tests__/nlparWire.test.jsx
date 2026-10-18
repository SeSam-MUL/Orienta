// @vitest-environment node
/**
 * What actually goes on the wire for NLPAR, and that it matches the backend.
 *
 * The component tests mock `services/api` wholesale, so they can only prove what
 * the panel ASKED the client for - never what the client sent. These use the
 * real axios-facing client with axios replaced, and then hold the request
 * field names and the paths against the backend route module itself.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { ebsdApi } from '../../../services/api';

const here = dirname(fileURLToPath(import.meta.url));
const routeFile = resolve(here, '../../../../../backend/api/routes/nlpar.py');
const apiFile = resolve(here, '../../../services/api.js');

const lastPost = (url) => {
  const call = [...axios.__inst.post.mock.calls].reverse().find((c) => c[0] === url);
  return call ? call[1] : null;
};

beforeEach(() => { vi.clearAllMocks(); });

describe('POST /api/ebsd/nlpar', () => {
  it('sends the radius and the request id the progress poll and Cancel need', async () => {
    await ebsdApi.nlpar(2, 'rid-7');
    expect(lastPost('/api/ebsd/nlpar')).toEqual({ radius: 2, request_id: 'rid-7' });
  });

  it('omits request_id when there is none, and defaults to radius 3', async () => {
    await ebsdApi.nlpar();
    const body = lastPost('/api/ebsd/nlpar');
    expect(body).toEqual({ radius: 3 });
    expect('request_id' in body).toBe(false);
  });

  it('never sends a lambda: the strength is always estimated by the backend', async () => {
    await ebsdApi.nlpar(1, 'x');
    const body = lastPost('/api/ebsd/nlpar');
    expect(Object.keys(body).sort()).toEqual(['radius', 'request_id']);
  });
});

describe('GET /api/ebsd/nlpar/advice', () => {
  it('asks the advice endpoint, with no parameters', async () => {
    await ebsdApi.nlparAdvice();
    expect(axios.__inst.get).toHaveBeenCalledWith('/api/ebsd/nlpar/advice');
  });
});

describe('POST /api/ebsd/nlpar/cancel/{id}', () => {
  it('puts the id in the path, encoded', async () => {
    await ebsdApi.nlparCancel('a b/c');
    const urls = axios.__inst.post.mock.calls.map((c) => c[0]);
    expect(urls).toContain('/api/ebsd/nlpar/cancel/a%20b%2Fc');
  });
});

describe.skipIf(!existsSync(routeFile))('the client against the backend route module', () => {
  const py = existsSync(routeFile) ? readFileSync(routeFile, 'utf8') : '';
  const js = readFileSync(apiFile, 'utf8');

  it('every field the client sends is a field of NlparRequest', () => {
    const model = py.match(/class NlparRequest\(BaseModel\):([\s\S]*?)\n\n/)[1];
    const fields = [...model.matchAll(/^\s{4}(\w+)\s*:/gm)].map((m) => m[1]);
    expect(fields).toContain('radius');
    const sent = js.match(/nlpar: \(radius = 3, requestId = null\) =>\s*api\.post\('\/api\/ebsd\/nlpar', \{([^}]*)\}/);
    expect(sent, 'ebsdApi.nlpar not found in api.js').toBeTruthy();
    const keys = [...sent[1].matchAll(/(?:^|[\s,.])(radius|request_id)\b/g)].map((m) => m[1]);
    expect(keys.length).toBeGreaterThanOrEqual(2);
    for (const k of new Set(keys)) expect(fields, `client sends ${k}`).toContain(k);
  });

  it('the paths are the ones the router serves', () => {
    expect(py).toContain('@router.post("/nlpar")');
    expect(py).toContain('@router.post("/nlpar/cancel/{request_id}")');
    expect(js).toContain("'/api/ebsd/nlpar'");
    expect(js).toContain('/api/ebsd/nlpar/cancel/${encodeURIComponent(requestId)}');
  });

  it('the advice route exists and answers with every field and value the panel reads', () => {
    const service = readFileSync(resolve(here, '../../../../../backend/api/services/nlpar.py'), 'utf8');
    expect(py).toContain('@router.get("/nlpar/advice")');
    expect(js).toContain("'/api/ebsd/nlpar/advice'");
    const advice = py.slice(py.indexOf('async def nlpar_advice'));
    for (const field of ['recommendation', 'applied', 'applied_radius', 'lam', 'sigma', 'threshold', 'reason']) {
      expect(advice, `advice answer has no "${field}"`).toContain(`"${field}"`);
    }
    for (const value of ['applied', 'unavailable', 'too_small', 'hex', 'uniform']) {
      expect(advice, `advice never answers "${value}"`).toContain(`"${value}"`);
    }
    for (const value of ['recommended', 'not_recommended']) {
      expect(service, `advice_for never answers "${value}"`).toContain(`"${value}"`);
    }
    const view = readFileSync(resolve(here, '../NlparControls.jsx'), 'utf8');
    for (const value of ['recommended', 'not_recommended', 'applied', 'too_small', 'hex', 'uniform']) {
      expect(view, `adviceView does not handle "${value}"`).toContain(value);
    }
  });

  it('the radius range the input allows is the one the backend accepts', async () => {
    const { MIN_RADIUS, MAX_RADIUS, DEFAULT_RADIUS } = await import('../useNlpar');
    const service = readFileSync(resolve(here, '../../../../../backend/api/services/nlpar.py'), 'utf8');
    const sLo = Number(service.match(/^MIN_RADIUS\s*=\s*(\d+)/m)[1]);
    const sHi = Number(service.match(/^MAX_RADIUS\s*=\s*(\d+)/m)[1]);
    const sDef = Number(service.match(/^DEFAULT_RADIUS\s*=\s*(\d+)/m)[1]);
    expect([MIN_RADIUS, MAX_RADIUS, DEFAULT_RADIUS]).toEqual([sLo, sHi, sDef]);
  });
});
