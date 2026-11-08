// "Add file..." on the Phase Map loads a rich/light .h5 through /api/indexing/import-h5. The
// answer carries the PC state of the loaded file; the page has to pass it to the same notice
// the EBSD viewer shows (pcLoadNote). The page is too large to mount here, so, as the repo does
// for such wires, the source is read: the call must stand in the import branch, after the answer.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const src = readFileSync(fileURLToPath(new URL('./PhaseMapPage.jsx', import.meta.url)), 'utf8');

describe('Phase Map import-h5 and the PC notice', () => {
  it('announces the PC state of the import answer in the import branch', () => {
    const imp = src.indexOf("api.post('/api/indexing/import-h5'");
    const call = src.indexOf('announceImportedPc(d, t, toast)');
    const elseBranch = src.indexOf('} else {', call);
    expect(imp).toBeGreaterThan(-1);
    expect(call).toBeGreaterThan(imp);
    expect(elseBranch).toBeGreaterThan(call);          // still inside the .h5 branch
    expect(src).toContain("import { announceImportedPc } from '../EBSDViewer/pcLoadNote'");
  });
});
