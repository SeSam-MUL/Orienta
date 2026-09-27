/**
 * The add-on endpoints.
 *
 * Its own file rather than another block in `api.js`, which is already 1600
 * lines — but on the SAME axios instance, so every failure here reaches the
 * interceptor, the breadcrumb trail and the diagnostics zip. An add-on failing
 * is exactly the kind of thing a user reports with a screenshot, and a call
 * that bypassed the instance would leave nothing in the log to read.
 *
 * All six return the axios promise. `mapImage` in particular is NOT a URL
 * builder: an `<img src>` gives the caller neither the `{image, scale}` it
 * needs to draw a legend nor a catchable 404 when a map has been evicted, and
 * it leaves no breadcrumb.
 */
import api from './api';

const BASE = '/api/addons';

export const addonsApi = {
  list: () => api.get(BASE),

  setEnabled: (name, enabled) =>
    api.post(`${BASE}/${encodeURIComponent(name)}/enabled`, { enabled }),

  runJob: (name, body) =>
    api.post(`${BASE}/${encodeURIComponent(name)}/run-job`, body),

  job: (jobId) => api.get(`${BASE}/jobs/${encodeURIComponent(jobId)}`),

  jobResult: (jobId) =>
    api.get(`${BASE}/jobs/${encodeURIComponent(jobId)}/result`),

  // Every segment is encoded: a name and an analysis key are an add-on
  // author's strings. The manifest rules keep them URL-safe, but this file
  // must not be the place that assumes a rule stays true.
  mapImage: (name, resultId, analysisKey, key) =>
    api.get(`${BASE}/${encodeURIComponent(name)}/outputs/`
      + `${encodeURIComponent(resultId)}/${encodeURIComponent(analysisKey)}/`
      + `${encodeURIComponent(key)}/image`),
};

export default addonsApi;
