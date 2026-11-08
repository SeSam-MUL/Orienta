// @vitest-environment jsdom
//
// The note under a finished indexing run (Hough on a trigonal space group PyEBSDIndex
// gets wrong). Real i18n instance and the real locale files, not a mocked `t`; the
// backend code is read from the backend source so a rename on either side turns red.
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import i18n from '../../i18n';
import ResultCaveats, { caveatsFrom, caveatText, SOFT_CODES } from './ResultCaveats';

const ROOT = path.resolve(__dirname, '../../../..');
const read = (...p) => fs.readFileSync(path.join(ROOT, ...p), 'utf8');
const CODE = 'houghTrigonalSetting';
const HEX = 'houghHexagonalVariants';
const LAUE = 'houghHexagonalLaue';

const caveat = (phases, extra = {}) => ({ code: CODE, phases, text: 'English text of the backend', ...extra });

beforeEach(async () => { await i18n.changeLanguage('en'); });
afterEach(async () => { cleanup(); await i18n.changeLanguage('en'); });

describe('the contract with the backend', () => {
  it('uses the code the backend writes', () => {
    const src = read('hough_caveats.py');
    expect(src).toContain(`HOUGH_TRIGONAL_SETTING = "${CODE}"`);
    expect(src).toContain(`HOUGH_HEXAGONAL_VARIANTS = "${HEX}"`);
    expect(src).toContain(`HOUGH_HEXAGONAL_LAUE = "${LAUE}"`);
  });

  it('treats as soft the codes the backend calls soft', () => {
    const src = read('hough_caveats.py');
    expect(src).toContain('SOFT_CODES = frozenset({HOUGH_HEXAGONAL_VARIANTS})');
    expect([...SOFT_CODES]).toEqual([HEX]);
  });

  it('the page reads the key the two answers of the backend carry', () => {
    const routes = read('backend', 'api', 'routes', 'indexing.py');
    expect(routes).toContain('task_result["caveats"] = _caveats');
    expect(routes).toContain('resp["caveats"] = _caveats');
  });

  it('the Indexing page wires the banner to the hook (behaviour: useActiveResultCaveats.test.jsx)', () => {
    const page = read('frontend', 'src', 'components', 'Indexing', 'IndexingPage.jsx');
    expect(page).toContain('useActiveResultCaveats({ enabled: !!isActive && !running })');
    expect(page).toContain('applyResultCaveats(d)');
    expect(page).toContain('clearResultCaveats()');
    expect(page).toContain('<ResultCaveats caveats={resultCaveats} />');
  });
});

describe('what is shown', () => {
  it('shows nothing without caveats', () => {
    for (const none of [undefined, null, [], [{ code: 7 }], 'text']) {
      const { container, unmount } = render(<ResultCaveats caveats={none} />);
      expect(container.firstChild).toBeNull();
      unmount();
    }
  });

  it('says it in English, naming the phase, for one phase', () => {
    const { getByRole } = render(<ResultCaveats caveats={[caveat(['Cr2N'])]} />);
    const text = getByRole('status').textContent;
    expect(text).toContain('Phase Cr2N: For this space group PyEBSDIndex uses the two-fold axes of the other trigonal setting');
    expect(text).toContain('Hough often cannot tell apart orientations 60° apart about c.');
    expect(text).toContain('Spherical or Dictionary indexing may help.');
    expect(text).not.toContain('therefore');
    expect(text).not.toContain('Pattern Match');
    expect(text).not.toContain('English text of the backend');
    expect(text).not.toContain('caveats.');
    expect(text).not.toContain('%');
  });

  it('uses the plural sentence for several phases and lists them', () => {
    const { getByRole } = render(<ResultCaveats caveats={[caveat(['Cr2N', 'Mg2Si'])]} />);
    const text = getByRole('status').textContent;
    expect(text).toContain('Phases Cr2N, Mg2Si: ');
    expect(text).toContain('For these space groups PyEBSDIndex uses');
    expect(text).not.toContain('this space group');
  });

  it.each([
    ['de', 'Phase Cr2N: Für diese Raumgruppe verwendet PyEBSDIndex die Zweizähler der anderen trigonalen Aufstellung', 'um 60° um c gegeneinander gedreht sind, oft nicht unterscheiden'],
    ['ja', '相 Cr2N：該当する空間群では、PyEBSDIndex はもう一方の三方晶設定の 2 回軸を使用しており', 'c 軸まわりに 60° 異なる方位を区別できないことがよくあります'],
    ['zh', '相 Cr2N：对上述相的空间群，PyEBSDIndex 使用的是另一种三方晶设定的二重轴', '常常无法区分绕 c 轴相差 60° 的取向'],
  ])('is in the language of the user (%s)', async (lang, start, mid) => {
    await i18n.changeLanguage(lang);
    const { getByRole } = render(<ResultCaveats caveats={[caveat(['Cr2N'])]} />);
    const text = getByRole('status').textContent;
    expect(text).toContain(start);
    expect(text).toContain(mid);
    expect(text).not.toContain('Hough often cannot');
  });

  it('has its own sentence for one and for several phases in a language that has a singular', async () => {
    const NAME = 'Zq9'; // not a substring of any sentence
    let checked = 0;
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      // only a language whose plural rules separate 1 from the rest has two forms
      if (new Intl.PluralRules(lang).select(1) !== 'one') continue;
      await i18n.changeLanguage(lang);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(caveat([NAME]), t, exists);
      const many = caveatText(caveat([NAME, NAME]), t, exists);
      // same phase name in both: the sentences themselves must differ
      expect(one.replace(NAME, '')).not.toBe(many.replace(`${NAME}, ${NAME}`, ''));
      checked += 1;
    }
    expect(checked).toBeGreaterThanOrEqual(2); // en and de
  });

  it('says nothing about a number in a language without a singular (ja, zh)', async () => {
    const plural = { ja: ['これらの', 'この相'], zh: ['这些', '该相'] };
    for (const lang of ['ja', 'zh']) {
      expect(new Intl.PluralRules(lang).select(1)).toBe('other');
      await i18n.changeLanguage(lang);
      const res = i18n.getResourceBundle(lang, 'indexing').caveats;
      // the parity test wants the `_one` key too; it is never read in these languages
      // (plural rule: always `other`), so it must say exactly what `_other` says
      expect(res.houghTrigonalSetting_one).toBe(res.houghTrigonalSetting_other);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(caveat(['Zq9']), t, exists);
      const many = caveatText(caveat(['Zq9', 'Mg2Si']), t, exists);
      for (const word of plural[lang]) {
        expect(one).not.toContain(word);
        expect(many).not.toContain(word);
      }
      expect(many).toContain('Zq9, Mg2Si');
    }
  });

  it('falls back to the English text of the backend for a code it has no sentence for', async () => {
    for (const lang of ['en', 'de']) {
      await i18n.changeLanguage(lang);
      const { getByRole, unmount } = render(
        <ResultCaveats caveats={[{ code: 'aCodeNoBuildHas', phases: ['A'], text: 'the backend said why' }]} />);
      expect(getByRole('status').textContent).toContain('the backend said why');
      expect(getByRole('status').textContent).not.toContain('caveats.aCodeNoBuildHas');
      unmount();
    }
  });

  it('shows one row per caveat', () => {
    const { getByRole } = render(<ResultCaveats caveats={[
      caveat(['A']), { code: 'other', phases: [], text: 'another note' }]} />);
    expect(getByRole('status').children).toHaveLength(2);
  });

  it('does not take a code that is not a plain word as a key', () => {
    expect(caveatsFrom({ caveats: [{ code: 'a.b', text: 'x' }, { code: '__proto__' }, caveat(['A'])] }))
      .toHaveLength(1);
  });
});

// ---- the softer note on the hexagonal variants ------------------------------------------

const hexCaveat = (phases) => ({ code: HEX, phases, text: 'English text of the backend' });

describe('the soft note (houghHexagonalVariants)', () => {
  it('says it in English for one and for several phases, without a number', () => {
    const one = render(<ResultCaveats caveats={[hexCaveat(['Brucite'])]} />).getByRole('status').textContent;
    expect(one).toContain('Phase Brucite: Hough cannot always tell apart orientations related by a rotation');
    expect(one).toContain('a symmetry of the hexagonal cell but not of this crystal');
    expect(one).toContain('(60° about c)');
    expect(one).not.toContain('basal plane');
    expect(one).toContain('Spherical or Dictionary indexing may help.');
    expect(one).not.toContain('English text of the backend');
    expect(one).not.toContain('%');
    cleanup();
    const many = render(<ResultCaveats caveats={[hexCaveat(['Brucite', 'Quartz'])]} />).getByRole('status').textContent;
    expect(many).toContain('Phases Brucite, Quartz: Hough cannot always tell apart');
    expect(many).toContain('but not of these crystals');
    expect(many).not.toContain('this crystal');
  });

  it.each([
    ['de', 'Phase Brucite: Hough kann Orientierungen nicht immer sicher voneinander unterscheiden', 'nicht dieses Kristalls ist (60° um c)'],
    ['ja', '相 Brucite：Hough では、六方晶の単位格子の対称操作ではあっても結晶そのものの対称操作ではない回転', '（c 軸まわりの 60°）'],
    ['zh', '相 Brucite：Hough 标定不一定能区分彼此相差一个旋转的取向', '（绕 c 轴 60°）'],
  ])('is in the language of the user (%s)', async (lang, start, mid) => {
    await i18n.changeLanguage(lang);
    const text = render(<ResultCaveats caveats={[hexCaveat(['Brucite'])]} />).getByRole('status').textContent;
    expect(text).toContain(start);
    expect(text).toContain(mid);
    expect(text).not.toContain('Hough cannot always');
    expect(text).not.toContain('English text of the backend');
    expect(text).not.toContain('caveats.');
  });

  it('has a singular and a plural sentence in en and de', async () => {
    const plural = { en: 'these crystals', de: 'dieser Kristalle' };
    const singular = { en: 'this crystal', de: 'dieses Kristalls' };
    for (const lang of ['en', 'de']) {
      await i18n.changeLanguage(lang);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(hexCaveat(['A']), t, exists);
      const many = caveatText(hexCaveat(['A', 'B']), t, exists);
      expect(one).toContain(singular[lang]);
      expect(one).not.toContain(plural[lang]);
      expect(many).toContain(plural[lang]);
      expect(many).not.toContain(singular[lang]);
    }
  });

  it('names no number of phases in ja and zh, and _one equals _other there', async () => {
    const plural = { ja: ['これらの', 'この相', 'この結晶'], zh: ['这些', '该相', '此晶体'] };
    for (const lang of ['ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const res = i18n.getResourceBundle(lang, 'indexing').caveats;
      expect(res.houghHexagonalVariants_one).toBe(res.houghHexagonalVariants_other);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(hexCaveat(['Zq9']), t, exists);
      const many = caveatText(hexCaveat(['Zq9', 'Mg2Si']), t, exists);
      for (const word of plural[lang]) {
        expect(one).not.toContain(word);
        expect(many).not.toContain(word);
      }
      expect(many).toContain('Zq9, Mg2Si');
    }
  });

  it('is shown weaker than the warning: info sign and accent colour, not the warning sign', () => {
    const { getByRole } = render(<ResultCaveats caveats={[caveat(['Cr2N']), hexCaveat(['Brucite'])]} />);
    const [strong, soft] = getByRole('status').children;
    expect(strong.getAttribute('data-severity')).toBe('warning');
    expect(soft.getAttribute('data-severity')).toBe('info');
    expect(strong.textContent.startsWith('⚠')).toBe(true);
    expect(soft.textContent.startsWith('⚠')).toBe(false);
    expect(soft.textContent.startsWith('ⓘ')).toBe(true);
    expect(strong.style.borderLeft).toContain('accent-yellow');
    expect(soft.style.borderLeft).toContain('accent-cyan');
  });

  it('shows two rows when both codes come, each with its own phases', () => {
    const { getByRole } = render(<ResultCaveats caveats={[caveat(['Cr2N']), hexCaveat(['Brucite', 'Quartz'])]} />);
    const rows = getByRole('status').children;
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain('Phase Cr2N: For this space group PyEBSDIndex');
    expect(rows[1].textContent).toContain('Phases Brucite, Quartz: Hough cannot');
  });
});

// ---- the stronger note on the Laue classes -3 and 6/m ------------------------------------

const laueCaveat = (phases) => ({ code: LAUE, phases, text: 'English text of the backend' });

describe('the strong hexagonal note (houghHexagonalLaue)', () => {
  it('says it in English for one and for several phases, without a number or a cause', () => {
    const one = render(<ResultCaveats caveats={[laueCaveat(['Berlinite'])]} />).getByRole('status').textContent;
    expect(one).toContain('Phase Berlinite: Hough often returns an orientation related to the true one by a rotation');
    expect(one).toContain('a symmetry of the hexagonal cell but not of this crystal');
    expect(one).toContain('(depending on the crystal class, 180° about an axis in the basal plane or 60° about c)');
    expect(one).toContain('Spherical or Dictionary indexing may help.');
    expect(one).not.toContain('English text of the backend');
    expect(one).not.toContain('%');
    expect(one).not.toContain('therefore');
    cleanup();
    const many = render(<ResultCaveats caveats={[laueCaveat(['Berlinite', 'Apatite'])]} />).getByRole('status').textContent;
    expect(many).toContain('Phases Berlinite, Apatite: Hough often returns');
    expect(many).toContain('but not of these crystals');
    expect(many).not.toContain('this crystal');
  });

  it.each([
    ['de', 'Phase Berlinite: Hough liefert oft eine Orientierung, die sich von der wahren durch eine Drehung unterscheidet', 'je nach Kristallklasse 180° um eine Achse in der Basisebene oder 60° um c'],
    ['ja', '相 Berlinite：Hough は、真の方位に対して', '結晶類によって、底面内の軸まわりの 180°、または c 軸まわりの 60°'],
    ['zh', '相 Berlinite：Hough 标定常常给出与真实取向相差一个旋转的取向', '取决于晶类，为绕基面内某一轴 180°，或绕 c 轴 60°'],
  ])('is in the language of the user (%s)', async (lang, start, mid) => {
    await i18n.changeLanguage(lang);
    const text = render(<ResultCaveats caveats={[laueCaveat(['Berlinite'])]} />).getByRole('status').textContent;
    expect(text).toContain(start);
    expect(text).toContain(mid);
    expect(text).not.toContain('Hough often returns');
    expect(text).not.toContain('English text of the backend');
    expect(text).not.toContain('caveats.');
  });

  it('has a singular and a plural sentence in en and de', async () => {
    const plural = { en: 'these crystals', de: 'dieser Kristalle' };
    const singular = { en: 'this crystal', de: 'dieses Kristalls' };
    for (const lang of ['en', 'de']) {
      await i18n.changeLanguage(lang);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(laueCaveat(['A']), t, exists);
      const many = caveatText(laueCaveat(['A', 'B']), t, exists);
      expect(one).toContain(singular[lang]);
      expect(one).not.toContain(plural[lang]);
      expect(many).toContain(plural[lang]);
      expect(many).not.toContain(singular[lang]);
    }
  });

  it('names no number of phases in ja and zh, and _one equals _other there', async () => {
    const plural = { ja: ['これらの', 'この相', 'この結晶'], zh: ['这些', '该相', '此晶体'] };
    for (const lang of ['ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const res = i18n.getResourceBundle(lang, 'indexing').caveats;
      expect(res.houghHexagonalLaue_one).toBe(res.houghHexagonalLaue_other);
      const t = i18n.t.bind(i18n);
      const exists = i18n.exists.bind(i18n);
      const one = caveatText(laueCaveat(['Zq9']), t, exists);
      const many = caveatText(laueCaveat(['Zq9', 'Mg2Si']), t, exists);
      for (const word of plural[lang]) {
        expect(one).not.toContain(word);
        expect(many).not.toContain(word);
      }
      expect(many).toContain('Zq9, Mg2Si');
    }
  });

  it('is a warning like the first code: only the variants hint is soft', () => {
    expect([...SOFT_CODES]).toEqual([HEX]);
    const { getByRole } = render(<ResultCaveats caveats={[
      caveat(['Cr2N']), laueCaveat(['Berlinite']), hexCaveat(['Brucite'])]} />);
    const rows = [...getByRole('status').children];
    expect(rows.map((r) => r.getAttribute('data-severity'))).toEqual(['warning', 'warning', 'info']);
    expect(rows[1].textContent.startsWith('⚠')).toBe(true);
    expect(rows[1].style.borderLeft).toContain('accent-yellow');
    expect(rows[2].textContent.startsWith('ⓘ')).toBe(true);
  });

  it('has a sentence in every language for each of the three codes, none left to the fallback', async () => {
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const res = i18n.getResourceBundle(lang, 'indexing').caveats;
      for (const code of [CODE, LAUE, HEX]) {
        for (const form of ['one', 'other']) {
          expect(typeof res[`${code}_${form}`]).toBe('string');
          expect(res[`${code}_${form}`]).toContain('{{phases}}');
        }
      }
    }
  });
});
