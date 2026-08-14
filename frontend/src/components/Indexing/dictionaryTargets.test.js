import { describe, it, expect } from 'vitest';
import {
  mastersForPhase,
  dictsForPhase,
  masterPathForPhase,
  dictionaryTargets,
  detectorShapeFromSignalShape,
  resolveDictPathForPhase,
  resolvePhasePaths,
  unresolvedDictPhases,
  selectBestDict,
  dictShapeMatches,
  dictGeometryMatches,
  dictGeometryUnknown,
} from './dictionaryTargets';

// Shape copied from a real GET /api/indexing/files/dictionary response.
const FILES = [
  {
    path: 'C:/db/EBSD_H5_Cache/Al/Al_master_E20kV_npx500.h5',
    filename: 'Al_master_E20kV_npx500.h5',
    file_type: 'master', formula: 'Al',
  },
  {
    path: 'C:/db/EBSD_H5_Cache/Si/Si_master_E20kV_npx500.h5',
    filename: 'Si_master_E20kV_npx500.h5',
    file_type: 'master', formula: 'Si',
  },
  {
    path: 'C:/db/Dictionary_Library/Al/Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    filename: 'Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    file_type: 'dictionary', formula: 'Al',
  },
];

// Masters only — no dictionaries. Lets the shape rules be tested in isolation.
const MASTERS = FILES.slice(0, 2);

const AL = { path: FILES[0].path, formula: 'Al', display_label: 'Al', file_type: 'master' };
const SI = { path: FILES[1].path, formula: 'Si', display_label: 'Si', file_type: 'master' };

describe('masterPathForPhase', () => {
  it('resolves each phase to ITS OWN master, not the first one in the list', () => {
    // The bug: the page seeded the dialog with `phaseFiles.find(...h5...)`,
    // which is always Al when Al is listed first — Si could never be generated.
    expect(masterPathForPhase(SI, FILES)).toBe(FILES[1].path);
    expect(masterPathForPhase(AL, FILES)).toBe(FILES[0].path);
  });

  it('falls back to the phase entry when it is itself a master', () => {
    const lone = { path: 'C:/x/Ni_master_E20kV_npx500.h5', formula: 'Ni', file_type: 'master' };
    expect(masterPathForPhase(lone, [])).toBe(lone.path);
  });

  it('returns nothing for a phase that only has a dictionary', () => {
    const dictOnly = {
      path: 'C:/db/Dictionary_Library/Fe/Fe_master_dict_20kV_60x60_5.0deg.h5',
      formula: 'Fe', file_type: 'dictionary',
    };
    expect(masterPathForPhase(dictOnly, [])).toBe('');
  });

  it('does not mistake a _dict_ file for a master', () => {
    const asPhase = { path: FILES[2].path, formula: 'Zz' };  // formula matches nothing
    expect(masterPathForPhase(asPhase, FILES)).toBe('');
  });
});

describe('mastersForPhase / dictsForPhase', () => {
  it('matches on formula, case-insensitively', () => {
    expect(mastersForPhase({ formula: 'al' }, FILES).map(f => f.filename))
      .toEqual(['Al_master_E20kV_npx500.h5']);
    expect(dictsForPhase({ formula: 'AL' }, FILES)).toHaveLength(1);
  });

  it('returns nothing for a phase without a formula', () => {
    expect(mastersForPhase({}, FILES)).toEqual([]);
    expect(dictsForPhase({ formula: '' }, FILES)).toEqual([]);
  });

  it('keeps Si dictionaries out of the Al card', () => {
    expect(dictsForPhase(SI, FILES)).toEqual([]);
  });
});

describe('dictionaryTargets', () => {
  it('yields one entry per selected phase with its own master', () => {
    const t = dictionaryTargets([AL, SI], FILES);
    expect(t.map(x => x.label)).toEqual(['Al', 'Si']);
    expect(t.map(x => x.masterPath)).toEqual([FILES[0].path, FILES[1].path]);
    expect(t.every(x => x.hasMaster)).toBe(true);
  });

  it('keeps a master-less phase but flags it, instead of hiding it', () => {
    const noMaster = { path: 'C:/x/Fe.cif', formula: 'Fe' };
    const t = dictionaryTargets([AL, noMaster], FILES);
    expect(t).toHaveLength(2);
    expect(t[1].hasMaster).toBe(false);
    expect(t[1].masterPath).toBe('');
  });

  it('carries the master filename for display', () => {
    expect(dictionaryTargets([SI], FILES)[0].masterFilename)
      .toBe('Si_master_E20kV_npx500.h5');
  });
});

describe('detectorShapeFromSignalShape', () => {
  it('reverses hyperspy (width, height) into pipeline (rows, cols)', () => {
    // A transposed detector shape silently produces a useless dictionary.
    expect(detectorShapeFromSignalShape([156, 128])).toEqual([128, 156]);
  });

  it('handles the square EDAX case', () => {
    expect(detectorShapeFromSignalShape([118, 118])).toEqual([118, 118]);
  });

  it('returns null rather than a bogus default when unknown', () => {
    expect(detectorShapeFromSignalShape(undefined)).toBeNull();
    expect(detectorShapeFromSignalShape([60])).toBeNull();
    expect(detectorShapeFromSignalShape([0, 60])).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Which dictionary actually gets indexed
// ---------------------------------------------------------------------------

const DICT_118 = {
  path: 'C:/lib/Si/Si_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
  file_type: 'dictionary', formula: 'Si',
  detector_shape: [118, 118], pc: [0.547, 0.465, 0.609], resolution_deg: 5.0,
};
const DICT_128 = {
  path: 'C:/lib/Al/Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
  file_type: 'dictionary', formula: 'Al',
  detector_shape: [128, 156], pc: [0.5, 0.5, 0.5], resolution_deg: 5.0,
};
const DICT_118_AL = {
  path: 'C:/lib/Al/Al_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
  file_type: 'dictionary', formula: 'Al',
  detector_shape: [118, 118], pc: [0.547, 0.465, 0.609], resolution_deg: 5.0,
};

describe('resolveDictPathForPhase', () => {
  it('falls back to the best dictionary when the user never clicked one', () => {
    // THE BUG: the card showed "chosen" from this same fallback, but the
    // request read only the explicit map -> it shipped the MASTER path and the
    // backend died with "phase_list required for Dictionary indexing".
    const files = [...FILES, DICT_118];
    expect(resolveDictPathForPhase(SI, files, { currentPc: [0.547, 0.465, 0.609] }))
      .toBe(DICT_118.path);
  });

  it("respects the user's explicit click over the fallback", () => {
    const files = [...FILES, DICT_118_AL, DICT_128];
    const explicit = { [AL.path]: DICT_128.path };
    expect(resolveDictPathForPhase(AL, files, { explicit })).toBe(DICT_128.path);
  });

  it('returns nothing when the phase has no dictionary at all', () => {
    expect(resolveDictPathForPhase(SI, FILES)).toBe('');
  });

  it('never picks a dictionary built for a different detector', () => {
    // 128x156 dict, 118x118 data -> the indexer rejects it outright.
    expect(resolveDictPathForPhase(AL, [...MASTERS, DICT_128], { detectorShape: [118, 118] }))
      .toBe('');
  });

  it('still offers a dictionary whose detector shape is unknown', () => {
    // The user's 2.0deg Al dictionary lost its JSON sidecar, so discovery
    // reports no detector_shape. We cannot judge it — withholding it would
    // hide a possibly fine dictionary. It fails loud at index time if wrong.
    const noSidecar = { ...FILES[2] };
    expect(noSidecar.detector_shape).toBeUndefined();
    expect(resolveDictPathForPhase(AL, [...MASTERS, noSidecar], { detectorShape: [118, 118] }))
      .toBe(noSidecar.path);
  });

  it('prefers the fitting detector over the better PC', () => {
    const wrongShapePerfectPc = { ...DICT_128, pc: [0.547, 0.465, 0.609] };
    const rightShapeWorsePc = { ...DICT_118_AL, pc: [0.4, 0.4, 0.4] };
    const got = resolveDictPathForPhase(AL, [...FILES, wrongShapePerfectPc, rightShapeWorsePc], {
      currentPc: [0.547, 0.465, 0.609], detectorShape: [118, 118],
    });
    expect(got).toBe(rightShapeWorsePc.path);
  });
});

describe('resolvePhasePaths', () => {
  it('substitutes each phase master with ITS dictionary', () => {
    const files = [...FILES, DICT_118, DICT_118_AL];
    const got = resolvePhasePaths([AL.path, SI.path], [AL, SI], files, {
      currentPc: [0.547, 0.465, 0.609], detectorShape: [118, 118],
    });
    expect(got).toEqual([DICT_118_AL.path, DICT_118.path]);
  });

  it('keeps the phase path when nothing resolves, so it can be reported', () => {
    const got = resolvePhasePaths([AL.path, SI.path], [AL, SI], [...FILES, DICT_118_AL], {});
    expect(got).toEqual([DICT_118_AL.path, SI.path]);
  });
});

describe('unresolvedDictPhases', () => {
  it('names the phase that would be sent without a dictionary', () => {
    const bad = unresolvedDictPhases([AL, SI], [...FILES, DICT_118_AL], {});
    expect(bad.map(b => b.label)).toEqual(['Si']);
    expect(bad[0].reason).toBe('none');
  });

  it('distinguishes "wrong detector" from "none at all"', () => {
    const bad = unresolvedDictPhases([AL], [...MASTERS, DICT_128], { detectorShape: [118, 118] });
    expect(bad[0].reason).toBe('shape');
    expect(bad[0].shapes).toEqual(['128x156']);
    expect(bad[0].needed).toBe('118x118');
  });

  it('is empty when every phase resolves', () => {
    const files = [...FILES, DICT_118, DICT_118_AL];
    expect(unresolvedDictPhases([AL, SI], files, { detectorShape: [118, 118] })).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Real payload, copied verbatim from discover_files_for_method(DICTIONARY,
// current_pc=[0.547, 0.465, 0.609]) on the user's library, 2026-08-14.
// This is the exact situation that produced "phase_list required" for Si.
// ---------------------------------------------------------------------------

const REAL_DISCOVERY = [
  { path: '/db/Al_master_E20kV_npx500.h5', file_type: 'master', formula: 'Al' },
  { path: '/db/Si_master_E20kV_npx500.h5', file_type: 'master', formula: 'Si' },
  { path: '/lib/Al/Al_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
    filename: 'Al_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
    file_type: 'dictionary', formula: 'Al',
    detector_shape: [118, 118], resolution_deg: 5.0, pc: [0.547, 0.465, 0.609] },
  { path: '/lib/Al/Al_master_E20kV_npx500_dict_20kV_128x156_2.0deg.h5',
    filename: 'Al_master_E20kV_npx500_dict_20kV_128x156_2.0deg.h5',
    file_type: 'dictionary', formula: 'Al',
    detector_shape: null, resolution_deg: null, pc: null },   // sidecar lost
  { path: '/lib/Al/Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    filename: 'Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    file_type: 'dictionary', formula: 'Al',
    detector_shape: [128, 156], resolution_deg: 5.0, pc: [0.5, 0.5, 0.5] },
  { path: '/lib/Si/Si_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
    filename: 'Si_master_E20kV_npx500_dict_20kV_118x118_pc547_465_609_5.0deg.h5',
    file_type: 'dictionary', formula: 'Si',
    detector_shape: [118, 118], resolution_deg: 5.0, pc: [0.547, 0.465, 0.609] },
];

const REAL_OPTS = { currentPc: [0.547, 0.465, 0.609], detectorShape: [118, 118] };

describe('the run that failed: Al + Si, 118x118 EDAX data', () => {
  const AL_P = REAL_DISCOVERY[0], SI_P = REAL_DISCOVERY[1];

  it('sends a DICTIONARY for Si, not its master (the actual crash)', () => {
    const paths = resolvePhasePaths(
      [AL_P.path, SI_P.path],
      [{ ...AL_P, display_label: 'Al' }, { ...SI_P, display_label: 'Si' }],
      REAL_DISCOVERY, REAL_OPTS,
    );
    expect(paths[1]).toBe(REAL_DISCOVERY[5].path);
    expect(paths[1]).not.toBe(SI_P.path);
  });

  it('picks the 118x118 dictionary for Al, never the 128x156 one', () => {
    const got = resolveDictPathForPhase({ ...AL_P, display_label: 'Al' }, REAL_DISCOVERY, REAL_OPTS);
    expect(got).toBe(REAL_DISCOVERY[2].path);
  });

  it('marks the 128x156 dictionaries as not fitting the detector', () => {
    expect(dictShapeMatches(REAL_DISCOVERY[4], [118, 118])).toBe(false);
    expect(dictShapeMatches(REAL_DISCOVERY[2], [118, 118])).toBe(true);
  });

  it('nothing is left unresolved, so the run is not blocked', () => {
    const bad = unresolvedDictPhases(
      [{ ...AL_P, display_label: 'Al' }, { ...SI_P, display_label: 'Si' }],
      REAL_DISCOVERY, REAL_OPTS,
    );
    expect(bad).toEqual([]);
  });

  it('reads the resolution from resolution_deg — `resolution` is always null', () => {
    expect(REAL_DISCOVERY[2].resolution).toBeUndefined();
    expect(selectBestDict([REAL_DISCOVERY[2], REAL_DISCOVERY[4]], REAL_OPTS.currentPc, [118, 118]))
      .toBe(REAL_DISCOVERY[2]);
  });
});

describe('camera-tilt matching', () => {
  const GEOM = { sampleTilt: 70, detectorTilt: 3.44, azimuthal: 0 };
  const D_FLAT   = { ...DICT_118, path: '/lib/Si/flat.h5',   detector_tilt: 0 };
  const D_TILTED = { ...DICT_118, path: '/lib/Si/tilted.h5', detector_tilt: 3.44 };
  const D_OLD    = { ...DICT_118, path: '/lib/Si/old.h5',    detector_tilt: null };

  it('rejects a dictionary simulated for the wrong camera tilt', () => {
    // Measured: 3.44 deg off -> NCC 0.017 for the SAME orientation.
    expect(dictGeometryMatches(D_FLAT, GEOM)).toBe(false);
    expect(dictGeometryMatches(D_TILTED, GEOM)).toBe(true);
  });

  it('treats a missing detector_tilt as unknown, not as flat', () => {
    // Old sidecars predate the field; hiding them would be a false negative.
    expect(dictGeometryMatches(D_OLD, GEOM)).toBe(true);
  });

  it('tolerates sub-half-degree differences', () => {
    expect(dictGeometryMatches({ ...DICT_118, detector_tilt: 3.2 }, GEOM)).toBe(true);
    expect(dictGeometryMatches({ ...DICT_118, detector_tilt: 2.0 }, GEOM)).toBe(false);
  });

  it('never picks the flat dictionary when a matching one exists', () => {
    const files = [...MASTERS, D_FLAT, D_TILTED];
    expect(resolveDictPathForPhase(SI, files, { detectorShape: [118, 118], geom: GEOM }))
      .toBe(D_TILTED.path);
  });

  it('blocks the run with reason "tilt" when only flat ones exist', () => {
    const bad = unresolvedDictPhases([SI], [...MASTERS, D_FLAT],
      { detectorShape: [118, 118], geom: GEOM });
    expect(bad[0].reason).toBe('tilt');
    expect(bad[0].tilts).toEqual(['0.00°']);
    expect(bad[0].neededTilt).toBe('3.44°');
  });
});

describe('dictionaries written before the tilt fix', () => {
  const GEOM = { sampleTilt: 70, detectorTilt: 3.44, azimuthal: 0 };
  const NO_TILT_FIELD = { ...DICT_118, path: '/lib/Si/pre_fix.h5' };   // sidecar has none
  const VERIFIED      = { ...DICT_118, path: '/lib/Si/new.h5', detector_tilt: 3.44 };

  it('flags an unrecorded camera tilt on a tilted detector', () => {
    // Every dictionary this app wrote before the fix used 0 deg.
    expect(dictGeometryUnknown(NO_TILT_FIELD, GEOM)).toBe(true);
    expect(dictGeometryUnknown(VERIFIED, GEOM)).toBe(false);
  });

  it('does not flag anything on a flat detector', () => {
    const flat = { sampleTilt: 70, detectorTilt: 0, azimuthal: 0 };
    expect(dictGeometryUnknown(NO_TILT_FIELD, flat)).toBe(false);
  });

  it('prefers a verified tilt over an unverifiable one, even at worse PC', () => {
    const unverifiedPerfectPc = { ...NO_TILT_FIELD, pc: [0.547, 0.465, 0.609] };
    const verifiedWorsePc = { ...VERIFIED, pc: [0.4, 0.4, 0.4] };
    const got = selectBestDict(
      [unverifiedPerfectPc, verifiedWorsePc], [0.547, 0.465, 0.609], [118, 118], GEOM,
    );
    expect(got.path).toBe(VERIFIED.path);
  });

  it('still uses an unverifiable dictionary when it is the only one', () => {
    // Not proof it is wrong — the legacy CPU generator got a real detector.
    const got = selectBestDict([NO_TILT_FIELD], null, [118, 118], GEOM);
    expect(got.path).toBe(NO_TILT_FIELD.path);
  });
});

describe('the Ni library dictionaries that collapsed the map', () => {
  // Verbatim from Database/Dictionary_Library/Ni/*.json and HiGainNi.h5.
  const HIGAIN_NI = { sampleTilt: 75.7, detectorTilt: 10.0, azimuthal: 0 };
  const BROKEN = {          // camera tilt written into the sample-tilt slot
    path: '/lib/Ni/Ni_master_E20kV_npx500_dict_20kV_60x60_2.0deg.h5',
    file_type: 'dictionary', formula: 'Ni',
    detector_shape: [60, 60], pc: [0.507, 0.262, 0.558],
    sample_tilt: 10.0, detector_tilt: null, resolution_deg: 2.0,
  };
  const GOOD = {            // written by the fixed generator
    path: '/lib/Ni/Ni_master_E20kV_npx500_dict_20kV_60x60_pc513_267_555_2.0deg.h5',
    file_type: 'dictionary', formula: 'Ni',
    detector_shape: [60, 60], pc: [0.513, 0.267, 0.555],
    sample_tilt: 75.7, detector_tilt: 10.0, resolution_deg: 2.0,
  };
  const NI = { path: '/db/Ni_master_E20kV_npx500.h5', formula: 'Ni',
               display_label: 'Ni', file_type: 'master' };

  it('rejects a 65.7 deg sample-tilt error', () => {
    // Indexing against this produced a single-orientation map while Hough and
    // Spherical resolved the grains on the same file.
    expect(dictGeometryMatches(BROKEN, HIGAIN_NI)).toBe(false);
    expect(dictGeometryMatches(GOOD, HIGAIN_NI)).toBe(true);
  });

  it('picks the correct dictionary over the two broken ones', () => {
    const files = [NI, BROKEN, { ...BROKEN, path: '/lib/Ni/5deg.h5', resolution_deg: 5.0 }, GOOD];
    expect(resolveDictPathForPhase(NI, files, {
      detectorShape: [60, 60], geom: HIGAIN_NI, currentPc: [0.507, 0.262, 0.558],
    })).toBe(GOOD.path);
  });

  it('blocks the run when only the broken ones exist', () => {
    const bad = unresolvedDictPhases([NI], [NI, BROKEN],
      { detectorShape: [60, 60], geom: HIGAIN_NI });
    expect(bad).toHaveLength(1);
    expect(bad[0].reason).toBe('tilt');
  });

  it('still accepts a dictionary that records no sample tilt at all', () => {
    const { sample_tilt, ...noTilt } = BROKEN;
    expect(dictGeometryMatches({ ...noTilt, detector_tilt: 10.0 }, HIGAIN_NI)).toBe(true);
  });
});
