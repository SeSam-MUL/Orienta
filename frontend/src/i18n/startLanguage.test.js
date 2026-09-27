import { describe, it, expect } from 'vitest';
import { resolveStartLanguage } from './startLanguage';

const supported = ['en', 'de', 'ja', 'zh'];
const decide = (stored, search) => resolveStartLanguage({ stored, search, supported });

describe('the language the app starts in', () => {
  it('is English when nothing is stored and nothing was handed over — unchanged', () => {
    expect(decide(null, '')).toMatchObject({ lang: 'en', persist: false });
  });

  it('takes the setup wizard\'s language on a first start', () => {
    // Before this, a German user went from a German setup into English.
    expect(decide(null, '?lang=de')).toMatchObject({ lang: 'de', persist: true });
  });

  it('keeps a language chosen inside the app against an automatic hand-over', () => {
    // A repair after a reinstall shows the wizard in the Windows language;
    // that must not quietly undo a choice the user made in the app.
    expect(decide('ja', '?lang=de')).toMatchObject({ lang: 'ja', persist: false });
  });

  it('lets an explicit choice in the wizard win — it is the newest choice', () => {
    expect(decide('ja', '?lang=de&langExplicit=1')).toMatchObject({ lang: 'de', persist: true });
  });

  it('ignores a language it does not have', () => {
    expect(decide(null, '?lang=fr')).toMatchObject({ lang: 'en', persist: false });
    expect(decide('de', '?lang=xx&langExplicit=1')).toMatchObject({ lang: 'de' });
  });

  it('ignores a stored value it does not have', () => {
    expect(decide('fr', '')).toMatchObject({ lang: 'en' });
  });

  it('removes the hand-over from the address and keeps everything else', () => {
    // So a reload or a bookmark does not apply it a second time.
    expect(decide(null, '?lang=de&langExplicit=1').cleanSearch).toBe('');
    expect(decide(null, '?view=polefigure&lang=de').cleanSearch).toBe('?view=polefigure');
    expect(decide(null, '?view=polefigure').hadHandOver).toBe(false);
  });
});
