import { describe, it, expect } from 'vitest';
import { pickHandoffFiles, handoffFileKindKey } from './phaseMapHandoff';

// Records as the discovery returns them: `linked_cif` is the basename of the CIF
// the file belongs to (a CIF is its own).
const cif = (name) => ({
  path: `/db/CIF_Library/${name}`, filename: name, linked_cif: name, file_type: 'phase',
});
const sht = (name, linked) => ({
  path: `/db/EBSD_SHT_Database/${name}`, filename: name, linked_cif: linked,
});
const master = (name, linked) => ({
  path: `/db/EBSD_H5_Cache/${name}`, filename: name, linked_cif: linked, file_type: 'master',
});
const dictionary = (name, linked, extra = {}) => ({
  path: `/db/Dictionary_Library/${name}`, filename: name, linked_cif: linked,
  file_type: 'dictionary', detector_shape: [118, 118], resolution_deg: 2, ...extra,
});

describe('pickHandoffFiles', () => {
  it('Hough: the CIFs of the phase map, and the phases that have no CIF', () => {
    const files = [cif('Al.cif'), cif('Si.cif'), cif('Fe.cif')];
    const r = pickHandoffFiles(files, 'hough', ['Al.cif', 'Si.cif', 'Ni.cif']);
    expect(r.matched.map((f) => f.filename)).toEqual(['Al.cif', 'Si.cif']);
    expect(r.missing).toEqual(['Ni.cif']);
    expect(r.needsDictionary).toEqual([]);
    expect(r.several).toEqual([]);
  });

  it('Hough: ignores SHT files even when they are linked to a phase-map CIF', () => {
    const files = [sht('Al.sht', 'Al.cif')];
    const { matched, missing } = pickHandoffFiles(files, 'hough', ['Al.cif']);
    expect(matched).toEqual([]);
    expect(missing).toEqual(['Al.cif']);
  });

  it('Spherical: the SHT linked to each phase-map CIF, in the phase map order', () => {
    const files = [sht('Si.sht', 'Si.cif'), sht('Al.sht', 'Al.cif'), sht('Fe.sht', 'Fe.cif')];
    const { matched, missing } = pickHandoffFiles(files, 'spherical', ['Al.cif', 'Si.cif']);
    expect(matched.map((f) => f.filename)).toEqual(['Al.sht', 'Si.sht']);
    expect(missing).toEqual([]);
  });

  it('Spherical: joins by the linked CIF, never by how the filenames look', () => {
    // "Al7FeCu2 ..." starts with "Al" and belongs to another phase
    const files = [
      sht('Al7FeCu2 (Al7FeCu2) [tI40] {20kV}.sht', 'Al7FeCu2.cif'),
      sht('Aluminium (Al) [cF4] {20kV}.sht', 'Al.cif'),
    ];
    const { matched, missing } = pickHandoffFiles(files, 'spherical', ['Al.cif']);
    expect(matched.map((f) => f.filename)).toEqual(['Aluminium (Al) [cF4] {20kV}.sht']);
    expect(missing).toEqual([]);
  });

  it('Spherical: a phase whose SHT is not in the library is missing, not guessed', () => {
    const files = [sht('Al.sht', 'Al.cif'), sht('Other.sht', null)];
    const { matched, missing } = pickHandoffFiles(files, 'spherical', ['Al.cif', 'Si.cif']);
    expect(matched.map((f) => f.filename)).toEqual(['Al.sht']);
    expect(missing).toEqual(['Si.cif']);
  });

  it('Spherical: ONE file per phase, and which one was taken is reported', () => {
    const files = [sht('Al 20kV.sht', 'Al.cif'), sht('Al 30kV.sht', 'Al.cif')];
    const { matched, several } = pickHandoffFiles(files, 'spherical', ['Al.cif']);
    expect(matched.map((f) => f.filename)).toEqual(['Al 20kV.sht']);
    expect(several).toEqual([{ name: 'Al.cif', taken: 'Al 20kV.sht', others: ['Al 30kV.sht'] }]);
  });

  describe('Dictionary', () => {
    const ctx = { currentPc: [0.5, 0.5, 0.5], detectorShape: [118, 118], geom: null };

    it('a phase with only a master pattern is NOT matched: a dictionary must be generated first', () => {
      const files = [master('Al_master.h5', 'Al.cif')];
      const r = pickHandoffFiles(files, 'dictionary', ['Al.cif'], ctx);
      expect(r.matched).toEqual([]);
      expect(r.needsDictionary).toEqual(['Al.cif']);
      expect(r.missing).toEqual([]);
    });

    it('a phase with neither master nor dictionary is missing', () => {
      const r = pickHandoffFiles([master('Si_master.h5', 'Si.cif')], 'dictionary', ['Al.cif'], ctx);
      expect(r.missing).toEqual(['Al.cif']);
      expect(r.needsDictionary).toEqual([]);
    });

    it('selects the dictionary, and ships exactly that one (path -> path)', () => {
      const d = dictionary('Al_master_dict_2deg.h5', 'Al.cif');
      const files = [master('Al_master.h5', 'Al.cif'), d];
      const r = pickHandoffFiles(files, 'dictionary', ['Al.cif'], ctx);
      expect(r.matched).toEqual([d]);
      expect(r.dictPaths).toEqual({ [d.path]: d.path });
    });

    it('takes the dictionary that fits the detector, not the first one', () => {
      const wrong = dictionary('Ni_128x156.h5', 'Ni.cif', { detector_shape: [128, 156] });
      const right = dictionary('Ni_118x118.h5', 'Ni.cif');
      const r = pickHandoffFiles([wrong, right], 'dictionary', ['Ni.cif'], ctx);
      expect(r.matched.map((f) => f.filename)).toEqual(['Ni_118x118.h5']);
      expect(r.several).toEqual([]);                  // the other one does not fit
    });

    it('dictionaries that exist but fit another detector count as needing a dictionary', () => {
      const wrong = dictionary('Ni_128x156.h5', 'Ni.cif', { detector_shape: [128, 156] });
      const r = pickHandoffFiles([wrong], 'dictionary', ['Ni.cif'], ctx);
      expect(r.matched).toEqual([]);
      expect(r.needsDictionary).toEqual(['Ni.cif']);
    });

    it('Ni 2.0deg and 5.0deg both fit: one is taken (the closer PC), the other reported', () => {
      const a = dictionary('Ni_2.0deg.h5', 'Ni.cif', { pc: [0.9, 0.9, 0.9], resolution_deg: 2 });
      const b = dictionary('Ni_5.0deg.h5', 'Ni.cif', { pc: [0.5, 0.5, 0.5], resolution_deg: 5 });
      const r = pickHandoffFiles([a, b], 'dictionary', ['Ni.cif'], ctx);
      expect(r.matched).toEqual([b]);
      expect(r.several).toEqual([{ name: 'Ni.cif', taken: 'Ni_5.0deg.h5', others: ['Ni_2.0deg.h5'] }]);
    });

    it('twin phases get their OWN dictionary even when their formulas are equal', () => {
      const d1 = dictionary('MgAl_a.h5', 'Mg17Al12_a.cif', { formula: 'Mg17Al12' });
      const d2 = dictionary('MgAl_b.h5', 'Mg17Al12_b.cif', { formula: 'Mg17Al12' });
      const r = pickHandoffFiles([d1, d2], 'dictionary', ['Mg17Al12_a.cif', 'Mg17Al12_b.cif'], ctx);
      expect(r.matched).toEqual([d1, d2]);
      expect(r.dictPaths).toEqual({ [d1.path]: d1.path, [d2.path]: d2.path });
    });
  });

  it('never matches across file kinds: a master list has no SHT in it', () => {
    const files = [sht('Al.sht', 'Al.cif')];
    const { matched, missing } = pickHandoffFiles(files, 'dictionary', ['Al.cif']);
    expect(matched).toEqual([]);
    expect(missing).toEqual(['Al.cif']);
  });

  it('copes with an empty list: every phase is missing', () => {
    const r = pickHandoffFiles([], 'spherical', ['Al.cif']);
    expect(r.matched).toEqual([]);
    expect(r.missing).toEqual(['Al.cif']);
    const none = pickHandoffFiles(undefined, 'hough', undefined);
    expect(none.matched).toEqual([]);
    expect(none.missing).toEqual([]);
  });
});

describe('handoffFileKindKey', () => {
  it('names a locale key per method (the text is in the user\'s language)', () => {
    expect(handoffFileKindKey('hough')).toBe('messages.handoffKindHough');
    expect(handoffFileKindKey('spherical')).toBe('messages.handoffKindSpherical');
    expect(handoffFileKindKey('dictionary')).toBe('messages.handoffKindDictionary');
  });
});
