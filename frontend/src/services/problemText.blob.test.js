// @vitest-environment jsdom
/**
 * `blobProblemText`: the error text of a request whose response type is `blob`.
 *
 * axios hands the body of a failed blob request over as a Blob, so `response.data.detail` is
 * undefined and the old readers showed "Request failed with status code 400". The code header is
 * readable all the same, and the JSON in the blob holds the prose.
 */
import { describe, it, expect } from 'vitest';
import { blobProblemText, codedText, codedProblemText } from './problemText';

const MISS_FREE = (dict) => (key, opts) => (key in dict ? dict[key] : opts?.defaultValue);

const blobError = (body, code, message = 'Request failed with status code 400') => ({
  message,
  response: {
    status: 400,
    headers: code ? { 'x-orienta-code': code } : {},
    data: new Blob([body], { type: 'application/json' }),
  },
});

describe('blobProblemText', () => {
  it('words a coded refusal from the header, in the language of t, not from the blob', async () => {
    const t = MISS_FREE({ 'common:errors.variantNotSupported': 'Diese Quelle kann hier nicht gelesen werden.' });
    const err = blobError(JSON.stringify({ detail: 'English prose from the server' }), 'variantNotSupported');
    expect(await blobProblemText(err, t, 'indexing'))
      .toBe('Diese Quelle kann hier nicht gelesen werden.');
  });

  it('prefers the page namespace over common, as problemText does', async () => {
    const t = MISS_FREE({
      'common:errors.x': 'shared', 'indexing:errors.x': 'own',
    });
    expect(await blobProblemText(blobError('{}', 'x'), t, 'indexing')).toBe('own');
  });

  it('reads detail from the JSON in the blob when there is no code', async () => {
    const t = MISS_FREE({});
    const err = blobError(JSON.stringify({ detail: 'No indexing result available.' }));
    expect(await blobProblemText(err, t, 'indexing')).toBe('No indexing result available.');
  });

  it('reads detail.message when a route answers with a dict', async () => {
    const t = MISS_FREE({});
    const err = blobError(JSON.stringify({ detail: { code: 'z', message: 'a sentence' } }));
    expect(await blobProblemText(err, t, 'indexing')).toBe('a sentence');
  });

  it('falls back to the blob JSON for a code this build has no text for', async () => {
    const t = MISS_FREE({});
    const err = blobError(JSON.stringify({ detail: 'the prose' }), 'aFutureCode');
    expect(await blobProblemText(err, t, 'indexing')).toBe('the prose');
  });

  it('ends in the error message when the blob is not JSON', async () => {
    const err = blobError('<html>Bad gateway</html>');
    expect(await blobProblemText(err, MISS_FREE({}), 'indexing'))
      .toBe('Request failed with status code 400');
  });

  it('ends in the error message when the JSON has no usable detail', async () => {
    const err = blobError(JSON.stringify({ detail: [{ loc: ['body'], msg: 'bad' }] }));
    expect(await blobProblemText(err, MISS_FREE({}), 'indexing'))
      .toBe('Request failed with status code 400');
  });

  it('reads a plain (non-blob) body like problemText does', async () => {
    const err = { message: 'm', response: { headers: {}, data: { detail: 'plain prose' } } };
    expect(await blobProblemText(err, MISS_FREE({}), 'indexing')).toBe('plain prose');
  });

  it('never throws: a network error, null, or a t that throws', async () => {
    expect(await blobProblemText({ message: 'Network Error' }, MISS_FREE({}), 'indexing')).toBe('Network Error');
    expect(await blobProblemText(null, MISS_FREE({}), 'indexing')).toBe('');
    const boom = () => { throw new Error('t broke'); };
    expect(await blobProblemText(blobError('{}', 'x'), boom, 'indexing'))
      .toBe('Request failed with status code 400');
  });
});

describe('codedText / codedProblemText', () => {
  it('is null without a code, without t, and for a code nobody has text for', () => {
    const t = MISS_FREE({ 'common:errors.k': 'text' });
    expect(codedText(null, t, 'indexing')).toBeNull();
    expect(codedText('k', null, 'indexing')).toBeNull();
    expect(codedText('unknown', t, 'indexing')).toBeNull();
    expect(codedText('k', t, 'indexing')).toBe('text');
    expect(codedProblemText({ response: { headers: { 'x-orienta-code': 'k' } } }, t, 'analysis')).toBe('text');
    expect(codedProblemText({ response: { headers: {} } }, t, 'analysis')).toBeNull();
  });
});
