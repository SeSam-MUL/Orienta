// @vitest-environment node
/**
 * What actually goes on the wire when a group changes.
 *
 * Asked for by b9, and the reason is a message that turned out to be half
 * right. "Send ids from now on" held for `parent`, `active`, `hidden` and
 * `?name=` on /resolve, and NOT for the five writing routes, which looked
 * their target up by NAME -- so an id sent there was a 404, and `DELETE /`
 * answered "not found" without anything looking broken. Measured against a
 * real collection "Al systems", whose id is `Al_systems`: the two differ in
 * the normal case, and getting it wrong would have made every group
 * unmanageable.
 *
 * FOUR of those five now take EITHER (c1, 21f58a73) -- `delete`, `assign`,
 * `unassign_from` and `rename`, all through `resolve_ref`. `PUT /update`,
 * the NESTING route, does NOT: `routes/phase_collections.py` looks it up in
 * a name-keyed dict and 404s on an id, not even casefolded. An earlier
 * version of this comment said all five, which is the kind of sentence
 * somebody migrates a codebase on; nesting would have broken and the
 * breakage blamed on the migration rather than the claim. Reported to c1.
 *
 * So what is pinned below is no longer the only thing that works for four
 * of the routes -- it is what this screen actually sends, uniformly by
 * name, rather than ids to four routes and a name to the fifth. That is
 * still worth a test: a silent switch to ids would be invisible in every
 * other test in the feature, and the day the two spellings diverge again
 * this file is where it shows.
 *
 * A contract in a test rather than in a message.
 *
 * KEPT APART FROM THE OTHER TESTS ON PURPOSE. `groupPersistence.test.js`
 * mocks `services/api` wholesale, so it can only prove what the store asked
 * the client for -- never what the client sent. That is the exact shape of
 * the defect this file exists to stop.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: { collections: [] } })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    patch: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { backendStore } from './groupPersistence';

const inst = () => axios.__inst;
const lastCall = (verb, url) => {
  const call = [...inst()[verb].mock.calls].reverse().find((c) => c[0] === url);
  return call || null;
};

beforeEach(() => { vi.clearAllMocks(); });

// The two spellings a group answers to. Every assertion below says which one
// went on the wire, so a silent switch to the other one fails here.
const NAME = 'Al systems';
const ID = 'Al_systems';

describe('adding phases to a group', () => {
  it('names the group by NAME, and lists the keys', async () => {
    await backendStore.apply({ verb: 'added', groupId: ID, groupName: NAME,
      keys: ['Al', 'Al13Fe4'] });
    const [, body] = lastCall('post', '/api/phase-collections/members');
    expect(body).toEqual({ name: NAME, keys: ['Al', 'Al13Fe4'], position: null });
  });

  it('and never the id', async () => {
    await backendStore.apply({ verb: 'added', groupId: ID, groupName: NAME,
      keys: ['Al'] });
    const [, body] = lastCall('post', '/api/phase-collections/members');
    expect(body.name).not.toBe(ID);
    expect(JSON.stringify(body)).not.toContain(ID);
  });
});

describe('taking phases out of a group', () => {
  it('goes to DELETE /members with the name and the keys in the body', async () => {
    await backendStore.apply({ verb: 'removed', groupId: ID, groupName: NAME,
      keys: ['Al'] });
    const [, config] = lastCall('delete', '/api/phase-collections/members');
    expect(config.data).toEqual({ name: NAME, keys: ['Al'] });
  });

  it('and NOT to the un-file-everywhere helper', async () => {
    // `assign(keys, None)` takes a phase out of every group. Removing it
    // from A must not lose it from B.
    await backendStore.apply({ verb: 'removed', groupId: ID, groupName: NAME,
      keys: ['Al'] });
    expect(lastCall('post', '/api/phase-collections/members')).toBe(null);
  });
});

describe('making, renaming and deleting', () => {
  it('create sends a name and nothing else', async () => {
    await backendStore.apply({ verb: 'created', groupId: ID, groupName: NAME });
    const [, body] = lastCall('post', '/api/phase-collections/');
    expect(body).toEqual({ name: NAME });
  });

  it('rename sends the OLD name and the new one', async () => {
    // The old name, because that is what the folder knows it by -- there is
    // no id in this request and there cannot be one yet.
    await backendStore.apply({ verb: 'renamed', groupId: ID,
      fromName: 'Al systems', groupName: 'Al-Fe systems' });
    const [, body] = lastCall('patch', '/api/phase-collections/rename');
    expect(body).toEqual({ name: 'Al systems', new_name: 'Al-Fe systems' });
  });

  it('delete sends the name in the body, not in the path', async () => {
    // Group names carry spaces and Greek; this repo already has a route
    // that broke on a `{name}` path segment.
    await backendStore.apply({ verb: 'deleted', groupId: ID, groupName: NAME });
    const [url, config] = lastCall('delete', '/api/phase-collections/');
    expect(url).toBe('/api/phase-collections/');
    expect(config.data).toEqual({ name: NAME });
  });
});

describe('a move, now that the route has the flag', () => {
  it('is ONE request, carrying move: true', async () => {
    await backendStore.apply({ verb: 'moved', groupId: ID, groupName: 'Cubic',
      keys: ['Al'], fromNames: ['Al systems', 'Working set'] });
    const [, added] = lastCall('post', '/api/phase-collections/members');
    expect(added).toEqual(
      { name: 'Cubic', keys: ['Al'], position: null, move: true });
    // No removes at all: the server takes it out of the others.
    expect(inst().delete.mock.calls
      .filter((c) => c[0] === '/api/phase-collections/members'))
      .toEqual([]);
  });

  it('and a plain ADD still does not carry it', async () => {
    // The flag on an add would silently turn "file this here too" into
    // "file this here only", which is the tag model's whole distinction.
    await backendStore.apply({ verb: 'added', groupId: ID, groupName: 'Cubic',
      keys: ['Al'] });
    const [, body] = lastCall('post', '/api/phase-collections/members');
    expect('move' in body).toBe(false);
  });
});

describe('the two inverses, which nothing on the wire had covered', () => {
  // `unmoved` and `recreated` are where findings 1, 3 and 4 of the final
  // review lived, and this file -- the one that pins what actually goes
  // over the wire -- did not exercise either of them.

  it('undoing a move adds per source group and never uses the flag', async () => {
    await backendStore.apply({ verb: 'unmoved', groupId: ID, groupName: 'G',
      keys: ['Al', 'Si'],
      fromMap: { X: ['Si'] }, removeHereKeys: ['Si'] });
    const posts = inst().post.mock.calls
      .filter((c) => c[0] === '/api/phase-collections/members')
      .map((c) => c[1]);
    expect(posts).toEqual([{ name: 'X', keys: ['Si'], position: null }]);
    expect(inst().delete.mock.calls
      .filter((c) => c[0] === '/api/phase-collections/members')
      .map((c) => c[1].data))
      .toEqual([{ name: 'G', keys: ['Si'] }]);
  });

  it('undoing a delete restores the tree, by name', async () => {
    await backendStore.apply({ verb: 'recreated', groupId: ID,
      groupName: 'Al systems', keys: ['Al'],
      parentName: null, childNames: ['Al-Fe'] });
    expect(inst().put.mock.calls
      .filter((c) => c[0] === '/api/phase-collections/update')
      .map((c) => c[1]))
      .toEqual([{ name: 'Al-Fe', parent: 'Al systems' }]);
  });
});

describe('reading the groups', () => {
  it('is one GET, and the page does not ask per group', async () => {
    await backendStore.load();
    expect(inst().get.mock.calls.map((c) => c[0]))
      .toEqual(['/api/phase-collections/']);
  });
});

describe('nesting a group under another', () => {
  it('sends the PARENT NAME, because that is what the folder stores', async () => {
    // `parent` is a reference the folder keeps as a name; the id lives only
    // on the screen. Sending `Al_systems` here would set a parent that
    // resolves to nothing.
    await backendStore.apply({ verb: 'nested', groupId: 'Mg_systems',
      groupName: 'Mg systems', parentName: NAME });
    const [, body] = lastCall('put', '/api/phase-collections/update');
    expect(body).toEqual({ name: 'Mg systems', parent: NAME });
  });

  it('promoting to the top level needs its own flag, not a null parent', async () => {
    // The route's own comment: `parent: null` means "leave it alone", so
    // null cannot also mean "remove it". Sending null would silently do
    // nothing and the group would stay a child.
    await backendStore.apply({ verb: 'nested', groupId: 'x',
      groupName: 'Al-Fe phases', parentName: null });
    const [, body] = lastCall('put', '/api/phase-collections/update');
    expect(body).toEqual({ name: 'Al-Fe phases', clear_parent: true });
    expect('parent' in body).toBe(false);
  });
});
