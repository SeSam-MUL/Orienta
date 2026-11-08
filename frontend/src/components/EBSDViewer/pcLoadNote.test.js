import { describe, it, expect, vi } from 'vitest';
import { pcLoadNote, announceImportedPc } from './pcLoadNote';

describe('pcLoadNote', () => {
  it('warns only when the store itself has no PC', () => {
    expect(pcLoadNote({ pc_defaulted: true, pc_source: 'default', pc: null }))
      .toEqual({ level: 'warning', key: 'pcDefaultWarning', params: {} });
  });
  it('does not warn about the placeholder when a restored typed PC is in use', () => {
    const n = pcLoadNote({ pc_defaulted: false, pc_source: 'manual', pc: [0.48, 0.52, 0.6], pc_notice: 'user_pc_kept' });
    expect(n).toEqual({ level: 'info', key: 'pcUserKept', params: { pc: '0.4800, 0.5200, 0.6000' } });
  });
  it('says that the file now carries a PC which replaced the earlier one', () => {
    const n = pcLoadNote({ pc_defaulted: false, pc_source: 'osc', pc: [0.55, 0.5, 0.7], pc_notice: 'file_pc_wins',
      pc_discarded: [0.48, 0.52, 0.6] });
    expect(n.key).toBe('pcFileWins');
    expect(n.params.pc).toBe('0.5500, 0.5000, 0.7000');
    expect(n.params.discarded).toBe('0.4800, 0.5200, 0.6000');   // the PC the user typed, to type again
  });
  it('says that the /Detector/pc of an export was not applied because the header wins', () => {
    const n = pcLoadNote({ pc_defaulted: false, pc_source: null, pc: [0.5241, 0.4133, 0.6317],
      pc_notice: 'export_pc_ignored', pc_export_ignored: [0.51, 0.33, 0.84] });
    expect(n).toEqual({ level: 'info', key: 'pcExportIgnored',
      params: { pc: '0.5241, 0.4133, 0.6317', exported: '0.5100, 0.3300, 0.8400' } });
  });
  it('keeps the plain .osc line and says nothing for an ordinary header PC', () => {
    expect(pcLoadNote({ pc_defaulted: false, pc_source: 'osc', pc: [0.5, 0.5, 0.7] }).key).toBe('pcFromOsc');
    expect(pcLoadNote({ pc_defaulted: false, pc_source: null, pc: [0.5, 0.5, 0.7] })).toBeNull();
    expect(pcLoadNote(null)).toBeNull();
  });
});

describe('announceImportedPc', () => {
  const t = (key, params) => `${key}|${JSON.stringify(params)}`;
  const mk = () => ({ warning: vi.fn(), info: vi.fn() });

  it('toasts the notice of the import answer at its level, with the viewer texts', () => {
    const toast = mk();
    const msg = announceImportedPc({ pc_defaulted: false, pc_source: 'osc', pc: [0.55, 0.5, 0.7],
      pc_notice: 'file_pc_wins', pc_discarded: [0.48, 0.52, 0.6] }, t, toast);
    expect(toast.info).toHaveBeenCalledTimes(1);
    expect(toast.info.mock.calls[0][0]).toMatch(/^ebsdviewer:logMessages\.pcFileWins\|/);
    expect(toast.info.mock.calls[0][0]).toContain('0.4800, 0.5200, 0.6000');
    expect(msg).toBe(toast.info.mock.calls[0][0]);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('warns for a file without a PC and says nothing for an ordinary one', () => {
    const toast = mk();
    announceImportedPc({ pc_defaulted: true, pc_source: 'default' }, t, toast);
    expect(toast.warning).toHaveBeenCalledTimes(1);
    const quiet = mk();
    expect(announceImportedPc({ pc_defaulted: false, pc_source: null, pc: [0.5, 0.5, 0.7] }, t, quiet)).toBeNull();
    expect(announceImportedPc({}, t, quiet)).toBeNull();      // an import that loaded no signal has no PC fields
    expect(quiet.info).not.toHaveBeenCalled();
    expect(quiet.warning).not.toHaveBeenCalled();
  });
});
