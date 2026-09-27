// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import i18n from '../../i18n';

// Plain keys, and the plural ones with the count they are always called with.
// i18next returns the BARE key on a miss, not "namespace:key", so asserting
// against the prefixed form passes for a key that does not exist — the first
// draft of this guard did exactly that and an invented key sailed through.
const KEYS = [
  ['picker.allPhases', { count: 3 }],
  ['picker.label', {}],
  ['picker.manage', {}],
  ['picker.workingSet', {}],
  ['picker.hiddenNotice', { count: 2 }],
  ['counts.usableHere', { usable: 1, total: 2, collection: 'X' }],
  ['counts.notUsableHere', { count: 1 }],
  ['counts.showAll', {}],
  ['counts.missingFromLibrary', { count: 2 }],
  // CollectionManager.jsx (Task 10) — create/rename/delete, the working set,
  // the suggestion run, and the two verbs a collection member offers.
  ['manager.title', {}],
  ['manager.close', {}],
  ['manager.namePlaceholder', {}],
  ['manager.parentLabel', {}],
  ['manager.parentNone', {}],
  ['manager.create', {}],
  ['manager.createWorkingSet', {}],
  ['manager.createWorkingSetTooltip', {}],
  ['manager.suggest', {}],
  ['manager.suggesting', {}],
  ['manager.suggestTooltip', {}],
  ['manager.suggestCreated', { count: 2 }],
  ['manager.suggestSkipped', { count: 1, names: 'Matrix' }],
  ['manager.suggestEmpty', {}],
  ['manager.suggestCount', { count: 3 }],
  ['manager.suggestDiscard', {}],
  ['manager.suggestApply', {}],
  ['manager.noCollectionsYet', {}],
  ['manager.unassignedTitle', { count: 4 }],
  ['manager.addTo', {}],
  ['manager.add', {}],
  ['manager.moveTo', {}],
  ['manager.moveToTooltip', {}],
  ['manager.removeMemberTooltip', {}],
  ['manager.moveUp', {}],
  ['manager.moveDown', {}],
  ['manager.notInLibrary', {}],
  ['manager.emptyCollection', {}],
  ['manager.collapse', {}],
  ['manager.expand', {}],
  ['manager.renameTooltip', {}],
  ['manager.rename', {}],
  ['manager.renameTitle', { name: 'Matrix' }],
  ['manager.deleteTooltip', {}],
  ['manager.delete', {}],
  ['manager.deleteTitle', {}],
  ['manager.deleteMessage', { name: 'Matrix' }],
];

// CAVEAT, measured: the app sets `fallbackLng: 'en'`, so this loop passes even
// if ja/collections.json and zh/collections.json do not exist — every lookup
// falls back to the English string. What actually enforces four files is
// `src/locales/localeParity.test.js`, which readFileSync's each one and throws
// ENOENT. Run both; neither alone is the guard.
describe('collections namespace', () => {
  it('has every key this feature renders, in all four languages', async () => {
    for (const lng of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lng);
      for (const [k, opts] of KEYS) {
        const v = i18n.t(`collections:${k}`, opts);
        expect(v, `${lng} ${k}`).not.toBe(k);          // bare key = miss
        expect(v, `${lng} ${k}`).not.toBe(`collections:${k}`);
        expect(v, `${lng} ${k}`).not.toBe('');
      }
    }
    await i18n.changeLanguage('en');
  });

  it('the guard can actually fail', async () => {
    await i18n.changeLanguage('en');
    const v = i18n.t('collections:picker.thisKeyDoesNotExist');
    expect(v).toBe('picker.thisKeyDoesNotExist');
  });
});
