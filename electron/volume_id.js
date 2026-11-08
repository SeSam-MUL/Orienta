/**
 * Which VOLUME a data folder was on when it was recorded (Windows).
 *
 * A drive letter is not an identity. A folder recorded as `E:\Orienta` on one
 * external disk is "gone" once E: is another disk, although nobody deleted it. The
 * type of the drive cannot tell: .NET's DriveInfo reports an external USB hard
 * disk as Fixed, exactly like the system disk. The volume serial number can --
 * it belongs to the file system, and differs between two disks.
 *
 * So the pointer's folder gets a small record beside it (`home.volume`, JSON),
 * written when the folder is chosen: `{ root, serial, folder }`. At startup a
 * missing folder is called "deleted" only if the SAME volume answers for that
 * letter now. A pointer written before this record existed has none, and for it
 * the drive type is all there is (a documented limit, see home_choice.js).
 *
 * The serial comes from `vol E:` -- the same text in every language apart from
 * the label, and a pattern (XXXX-XXXX) that is language-free. No native module,
 * no PowerShell for this one.
 */
const fs = require('node:fs');
const path = require('node:path');

const RECORD = 'home.volume';

/** The file beside the pointer file. */
function volumeRecordFile(pointerFile) {
  return path.join(path.dirname(pointerFile), RECORD);
}

/** The drive root ("E:\\") of an absolute Windows path, or null (UNC, relative). */
function driveRootOf(dir) {
  const root = path.parse(String(dir)).root;
  return /^[A-Za-z]:[\\/]$/.test(root) ? root : null;
}

/** `XXXX-XXXX` of the volume at `root`, or null if it cannot be told. */
function windowsVolumeSerial(root) {
  const letter = String(root).slice(0, 2);
  if (!/^[A-Za-z]:$/.test(letter)) return null;
  try {
    const cmd = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'cmd.exe');
    const out = require('node:child_process').execFileSync(
      cmd, ['/d', '/c', 'vol', letter],
      { encoding: 'utf8', timeout: 5000, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
    const match = /\b([0-9A-F]{4}-[0-9A-F]{4})\b/i.exec(String(out));
    return match ? match[1].toUpperCase() : null;
  } catch {
    return null;
  }
}

/** The record, or null when there is none or it is unusable. Never throws. */
function readVolumeRecord(pointerFile, io = fs) {
  try {
    const rec = JSON.parse(io.readFileSync(volumeRecordFile(pointerFile), 'utf8'));
    if (rec && typeof rec.root === 'string' && typeof rec.serial === 'string' && rec.serial) return rec;
  } catch { /* none, or unreadable: as if there were none */ }
  return null;
}

/**
 * Record the volume `dir` is on (best effort: a record that cannot be written
 * must never be the reason a folder cannot be chosen), or forget it.
 */
function writeVolumeRecord(pointerFile, dir, { volumeSerial = windowsVolumeSerial, io = fs } = {}) {
  const file = volumeRecordFile(pointerFile);
  try {
    const root = dir ? driveRootOf(dir) : null;
    const serial = root ? volumeSerial(root) : null;
    if (!root || !serial) {
      io.rmSync(file, { force: true });
      return null;
    }
    io.writeFileSync(file, `${JSON.stringify({ root, serial, folder: dir })}\n`, 'utf8');
    return { root, serial, folder: dir };
  } catch {
    try { io.rmSync(file, { force: true }); } catch { /* ignore */ }
    return null;
  }
}

module.exports = {
  RECORD,
  volumeRecordFile,
  driveRootOf,
  windowsVolumeSerial,
  readVolumeRecord,
  writeVolumeRecord,
};
