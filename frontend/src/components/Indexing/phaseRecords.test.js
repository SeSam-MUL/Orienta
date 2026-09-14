/**
 * The record form the indexing request carries, and why the lists alone lie.
 *
 * Held against the SAME flat path list the page builds, so the test states
 * the defect rather than describing it: three independent `.filter()` passes
 * over [Al.cif, alpha.sht, Cu.h5] give three lists of length 1 whose first
 * entries are three different phases, and the backend's whole-list precedence
 * (`cif_paths or master_h5_paths or sht_paths`) then names the spherical run's
 * only phase after the CIF of a phase that is not in it.
 */
import { describe, it, expect } from 'vitest';
import { buildPhaseRecords, phaseFileField } from './phaseRecords';

const RAGGED = ['D:/db/Al.cif', 'D:/db/alpha.sht', 'D:/db/Cu_master.h5'];

describe('phaseFileField', () => {
  it('routes each extension to the field its method reads', () => {
    expect(phaseFileField('x/Al.cif')).toBe('cif');
    expect(phaseFileField('x/Al (Al) [cF4] {20kV}.SHT')).toBe('sht');
    expect(phaseFileField('x/Al_master.h5')).toBe('master');
    expect(phaseFileField('x/Al_master.hdf5')).toBe('master');
  });

  it('claims nothing for a file no method runs on', () => {
    expect(phaseFileField('x/notes.txt')).toBeNull();
    expect(phaseFileField('')).toBeNull();
    expect(phaseFileField(undefined)).toBeNull();
  });
});

describe('buildPhaseRecords', () => {
  it('keeps one record per phase, in the order the page shows them', () => {
    expect(buildPhaseRecords(RAGGED)).toEqual([
      { cif: 'D:/db/Al.cif' },
      { sht: 'D:/db/alpha.sht' },
      { master: 'D:/db/Cu_master.h5' },
    ]);
  });

  it('carries the page label when there is one, for the log only', () => {
    const records = buildPhaseRecords(['D:/db/alpha.sht'],
      { 'D:/db/alpha.sht': 'α-Al(FeMn)Si' });
    expect(records[0]).toEqual({ sht: 'D:/db/alpha.sht', name: 'α-Al(FeMn)Si' });
  });

  it('omits a file no method runs on rather than emitting an empty record', () => {
    // Nothing indexes `phases` positionally — the backend derives both the
    // per-method list and the per-phase name source from the records — so an
    // omission shifts nothing. An EMPTY record, on the other hand, is a phase
    // nothing can run: it would quietly reduce the phase count of the run, so
    // the backend refuses it (PhaseFiles._must_bring_a_file) and the page must
    // not produce one. The picker accepts any file for method "embedding".
    const records = buildPhaseRecords(['D:/db/Al.cif', 'D:/db/notes.txt']);
    expect(records).toEqual([{ cif: 'D:/db/Al.cif' }]);
  });

  it('says which phase each file belongs to, where the three lists cannot', () => {
    const records = buildPhaseRecords(RAGGED);
    const cifs = RAGGED.filter((f) => f.toLowerCase().endsWith('.cif'));
    const shts = RAGGED.filter((f) => f.toLowerCase().endsWith('.sht'));

    // The lists as the page used to send them: both length 1, and their
    // first entries are DIFFERENT phases.
    expect(cifs).toHaveLength(1);
    expect(shts).toHaveLength(1);
    expect(records.findIndex((r) => r.cif === cifs[0])).toBe(0);
    expect(records.findIndex((r) => r.sht === shts[0])).toBe(1);
  });

  it('survives an empty or missing list', () => {
    expect(buildPhaseRecords([])).toEqual([]);
    expect(buildPhaseRecords(undefined)).toEqual([]);
  });
});
