import { describe, it, expect } from 'vitest';
import {
  ORIENTATION_SOURCE_BADGES,
  orientationSourceBadge,
  orientationSourceIsSuspicious,
} from './orientationSource';

import en from '../../locales/en/indexing.json';
import de from '../../locales/de/indexing.json';
import ja from '../../locales/ja/indexing.json';
import zh from '../../locales/zh/indexing.json';

describe('orientation provenance badge', () => {
  it('shows nothing for a map whose orientations are the spherical ones', () => {
    expect(orientationSourceBadge('spherical')).toBeNull();
    expect(orientationSourceBadge(undefined)).toBeNull();
    expect(orientationSourceIsSuspicious('spherical')).toBe(false);
  });

  it('shows the Hough badge when the whole map came from Hough', () => {
    expect(orientationSourceBadge('hough')).toEqual({
      labelKey: 'matchesDialog.orientationFromHough',
      tipKey: 'matchesDialog.orientationHoughTip',
    });
    expect(orientationSourceIsSuspicious('hough')).toBe(true);
  });

  it('shows a DIFFERENT badge for a mixed map', () => {
    // The backend reports "mixed" as soon as the render arbitration hands some
    // pixels back to the sphere. Before this existed the badge fell through to
    // "nothing", i.e. a map with Hough orientations in it looked untouched.
    const mixed = orientationSourceBadge('mixed');
    expect(mixed).not.toBeNull();
    expect(mixed.labelKey).not.toBe(ORIENTATION_SOURCE_BADGES.hough.labelKey);
    expect(orientationSourceIsSuspicious('mixed')).toBe(true);
  });

  it('an unknown source is not badged rather than badged wrongly', () => {
    expect(orientationSourceBadge('seed')).toBeNull();
    expect(orientationSourceBadge('')).toBeNull();
  });
});

describe('the keys the badge asks for exist in every language', () => {
  const bundles = { en, de, ja, zh };
  const keys = Object.values(ORIENTATION_SOURCE_BADGES)
    .flatMap(b => [b.labelKey, b.tipKey]);

  it.each(Object.keys(bundles))('%s', (lang) => {
    for (const key of keys) {
      const [ns, k] = key.split('.');
      const value = bundles[lang][ns]?.[k];
      expect(value, `${lang}: missing ${key}`).toBeTruthy();
      // A copy-paste of the English string in ja/zh is a missing translation,
      // not a translation.
      if (lang !== 'en') {
        expect(value, `${lang}: ${key} is still the English text`)
          .not.toBe(bundles.en[ns][k]);
      }
    }
  });
});
