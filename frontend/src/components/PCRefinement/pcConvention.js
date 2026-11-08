// The small rules behind the PC fields in three conventions. Numbers are never converted here: that is
// pcApi.convert (kikuchipy's own conversion); this file only reads what was typed and names the fields.

/** The three kinds of numbers a PC is written in. The page keeps and applies Bruker's. */
export const CONVENTIONS = ['bruker', 'tsl', 'oxford'];

/**
 * What a person typed as a number, or NaN. A dot or a comma may be the decimal mark; anything else
 * (letters, two dots, an empty field, "1e-3") is not a number here, so a typo is never half-read.
 */
export function parseNum(text) {
  const s = String(text ?? '').trim().replace(',', '.');
  return /^[-+]?(\d+\.?\d*|\.\d+)$/.test(s) ? parseFloat(s) : NaN;
}

/** Four decimals, the way every PC on the page is written. */
export const fmt4 = (v) => Number(v).toFixed(4);

/** `[rows, columns]` when both are whole numbers above zero, else null (no file, or no size known). */
export function patternSizeOf(rows, cols) {
  const r = Number(rows);
  const c = Number(cols);
  return Number.isInteger(r) && Number.isInteger(c) && r > 0 && c > 0 ? [r, c] : null;
}

/** The convention to preselect for an open file: its own when the backend knows it, else Bruker's. */
export function preselectedConvention(pcInUse) {
  const c = pcInUse?.convention;
  return pcInUse?.convention_known && CONVENTIONS.includes(c) ? c : null;
}
