/**
 * The German locale must be written with real umlauts.
 *
 * ASCII transliteration ("fuer", "waehlen", "Groesse") has shipped by mistake
 * more than once: it reads as broken to a German user, and it is invisible to
 * anyone reviewing in English. A guard that covered a single key of one
 * namespace let 153 words through in three others.
 *
 * The check is an ALLOW-list, not a deny-list, and that is the whole point.
 * A deny-list can only confirm the fix it was written from: the first version
 * of this file listed the words that had just been corrected, so it passed
 * while `Chemie-Uebereinstimmung` and `Die Ueberspannung` sat two lines away
 * from words it did catch. Here, every word containing ae / oe / ue must be
 * one of the words below, which are the ones where those pairs are genuinely
 * correct German or English: Quelle, neue, manuell, dauern, value, Issue.
 * A new offender is reported because it is unknown, not because someone
 * thought of it.
 *
 * When a real new word trips this, add it to ALLOWED with the umlaut written
 * out if it needs one. The list is short on purpose.
 */
import { describe, it, expect } from 'vitest';

const de = import.meta.glob('./de/*.json', { eager: true });

/** Words where ae / oe / ue is correct as written. Lower-case. */
const ALLOWED = new Set([
  'aktuell', 'aktuelle', 'aktuellem', 'aktuellen', 'aktueller', 'aktuelles',
  'aufbauen', 'bauen', 'datenquelle', 'dauerhaft', 'dauern', 'dauert',
  'genaue', 'genauer', 'genauere', 'graue', 'issue', 'kartensteuerung',
  // From the Latin, not a flattened umlaut: Koeffizient has no ö.
  'koeffizient', 'koeffizienten', 'laue',
  // Greek rhombos + hedron: no umlaut was flattened here either.
  'rhomboedrisch', 'rhomboeder',
  'literaturquelle', 'manuell', 'manuelle', 'manuellen', 'manuelles',
  'musterquelle', 'neue', 'neuem', 'neuen', 'neuer', 'neuere', 'neues',
  'neueste', 'neuesten', 'neuexport', 'niederfrequenten', 'quell', 'quellcode',
  'quelldatei', 'quelle', 'quellen', 'quellgröße', 'quellkopie', 'quellordner',
  'querrichtung', 'rescued', 'sauerstoff', 'sequenz', 'ungenauer', 'value',
  'traue', 'trauen', 'values', 'vertraue', 'vertrauen',
  'vertrauensbereich', 'vertrauenswürdig',
  'virtuell', 'virtuelle', 'virtuellen', 'virtuelles',
  'vertrauenswürdige', 'vorschauen', 'zuerst',
]);

/**
 * Transliterated sharp s. These cannot be found by the ae/oe/ue rule above,
 * and there are few enough of them to name.
 */
const SHARP_S = new Set([
  'ausschliesslich', 'schliesst', 'heisst', 'weiss', 'gemaess', 'groesse',
  'strasse', 'fliesst', 'geniesst', 'massgeblich', 'schliesslich',
]);

const VOWEL_PAIR = /ae|oe|ue/i;
const WORDS = /[A-Za-zÄÖÜäöüß]+/g;

/**
 * i18next interpolation placeholders are not German text.
 *
 * `{{gpuEstimate}}` is a variable NAME chosen in JavaScript; the "ue" in it is
 * not a missing umlaut, and nothing a translator does can change it. Reported
 * as two offenders in de/indexing.json while the German around them was
 * correct, which is a guard crying wolf at the code rather than the prose.
 */
const PLACEHOLDERS = /\{\{[^}]*\}\}/g;

function offenders(value, path, found) {
  if (typeof value === 'string') {
    (value.replace(PLACEHOLDERS, ' ').match(WORDS) || []).forEach((raw) => {
      const word = raw.toLowerCase();
      if (SHARP_S.has(word)) {
        found.push(`${path}: ${raw}`);
      } else if (VOWEL_PAIR.test(word) && !ALLOWED.has(word)) {
        found.push(`${path}: ${raw}`);
      }
    });
    return found;
  }
  if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, child]) =>
      offenders(child, path ? `${path}.${key}` : key, found),
    );
  }
  return found;
}

describe('German locale', () => {
  const files = Object.entries(de);

  it('covers every German namespace', () => {
    expect(files.length).toBeGreaterThan(15);
  });

  it('reports a transliteration nobody listed in advance', () => {
    // the two words the previous deny-list version of this guard missed
    expect(offenders({ a: 'Chemie-Uebereinstimmung' }, '', [])).toEqual(['a: Uebereinstimmung']);
    expect(offenders({ a: 'Die Ueberspannung' }, '', [])).toEqual(['a: Ueberspannung']);
    // and one that exists nowhere in this repo
    expect(offenders({ a: 'unverzueglich' }, '', [])).toEqual(['a: unverzueglich']);
  });

  it('accepts the umlauts and leaves correct German and English alone', () => {
    expect(offenders({ a: 'Chemie-Übereinstimmung, Überspannung' }, '', [])).toEqual([]);
    expect(offenders(
      { a: 'Quelle, neue Messung, manuell, dauern, Passwort, value, Issue, Laue' }, '', [],
    )).toEqual([]);
  });

  it('catches a transliterated sharp s, which the vowel rule cannot see', () => {
    expect(offenders({ a: 'ausschliesslich' }, '', [])).toEqual(['a: ausschliesslich']);
    expect(offenders({ a: 'ausschließlich' }, '', [])).toEqual([]);
    expect(offenders({ a: 'dass das Ergebnis passt' }, '', [])).toEqual([]);
  });

  files.forEach(([file, mod]) => {
    it(`${file} is written with real umlauts`, () => {
      expect(offenders(mod.default ?? mod, '', [])).toEqual([]);
    });
  });
});

/**
 * One form of address, throughout.
 *
 * The M5 tester noticed that "Problem melden" used "du" while the rest of the
 * window used "Sie" (report, section 5, point 4). It was not three strings:
 * 32 of them addressed the reader informally, across ten namespaces, against
 * 43 that were formal. Mixed address reads as two people wrote the interface.
 */
describe('German address', () => {
  // Two things carry the informal address in German, and the first version of
  // this guard checked only one. Pronouns are the easy half; the address lives
  // just as much in the VERB, and a sweep that fixes "du" but leaves "Wähle
  // die Ebene" has done half the job — which is what a review found here.
  const PRONOUN = /\b(du|dich|dir|dein|deine|deinen|deinem|deiner|deines)\b/i;

  // A bare imperative stem at the start of a sentence. Listed rather than
  // derived: German imperatives are not regular enough to match by shape, and
  // a list also says which verbs this interface actually uses.
  const IMPERATIVE = new RegExp(
    '(?:^|[.!?:;—–]\\s|\\n)('
    + 'Fang|Erzeuge|Wähle|Nimm|Klicke|Markiere|Vergleiche|Stelle|Öffne|Suche'
    + '|Nutze|Teste|Trenne|Prüfe|Setze|Drücke|Gib|Schreibe|Ziehe|Schau|Achte'
    + '|Beachte|Halte|Lass|Mach|Zeige|Starte|Speichere|Füge|Entferne'
    + ')\\s+\\S',
  );

  function informal(value, path, found) {
    if (typeof value === 'string') {
      const p = value.match(PRONOUN);
      if (p) found.push(`${path}: ${p[0]}`);
      const v = value.match(IMPERATIVE);
      if (v) found.push(`${path}: ${v[1]}`);
      return found;
    }
    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([k, v]) =>
        informal(v, path ? `${path}.${k}` : k, found));
    }
    return found;
  }

  it('catches the informal pronoun', () => {
    expect(informal({ a: 'Was hast du gemacht?' }, '', [])).toEqual(['a: du']);
    expect(informal({ a: 'an deine Fehlermeldung' }, '', [])).toEqual(['a: deine']);
    expect(informal({ a: 'Was haben Sie gemacht?' }, '', [])).toEqual([]);
    // and does not trip on ordinary words that contain the letters
    expect(informal({ a: 'Dudenstraße, Dublette, dieser Duktus' }, '', [])).toEqual([]);
  });

  it('catches the informal verb, which carries the address just as much', () => {
    // A review of the first sweep found exactly these: pronoun fixed, verb left.
    expect(informal({ a: 'Fang hier an: den Knopf drücken.' }, '', [])).toEqual(['a: Fang']);
    expect(informal({ a: 'Ihre Muster sind 60x60. Erzeuge eines dafür.' }, '', []))
      .toEqual(['a: Erzeuge']);
    expect(informal({ a: 'Wähle die zu rendernde Ebene.' }, '', [])).toEqual(['a: Wähle']);
    expect(informal({ a: 'Wählen Sie die zu rendernde Ebene.' }, '', [])).toEqual([]);
  });

  it('does not mistake a noun or a mid-sentence word for an imperative', () => {
    expect(informal({ a: 'Test 3/10' }, '', [])).toEqual([]);
    expect(informal({ a: 'Die Suche läuft, das Laden dauert.' }, '', [])).toEqual([]);
    expect(informal({ a: 'Wird zurückgesetzt …' }, '', [])).toEqual([]);
  });

  Object.entries(de).forEach(([file, mod]) => {
    it(`${file} addresses the reader formally`, () => {
      expect(informal(mod.default ?? mod, '', [])).toEqual([]);
    });
  });
});
