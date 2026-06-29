// Disambiguate a set of file display names by stripping the longest
// whole-token prefix and suffix common to ALL of them, leaving only the part
// that actually differs.
//
// Oxford H5OINA exports all share a long, identical tail (e.g. every file ends
// in "... Arbeitsbereich N Elementverteilungsdaten M") and several share the
// same head ("HIgh_MG_EBSD ..."). Front-truncation made the rows all read
// "…Elementverteilungsdaten 1", indistinguishable. This returns the middle
// discriminating segment instead. Falls back to the full name when only one
// name is given or nothing distinguishes them.
export function disambiguateNames(names) {
  if (!Array.isArray(names) || names.length <= 1) return names ? [...names] : [];
  const split = names.map((n) => String(n).split(/\s+/));
  // longest common prefix (whole tokens)
  let pre = 0;
  while (split.every((t) => t[pre] !== undefined && t[pre] === split[0][pre])) pre++;
  // longest common suffix (whole tokens), not overlapping the prefix
  let suf = 0;
  while (split.every((t) => {
    const i = t.length - 1 - suf;
    return i >= pre && t[i] === split[0][split[0].length - 1 - suf];
  })) suf++;
  return split.map((toks, idx) => {
    const mid = toks.slice(pre, toks.length - suf).join(' ').trim();
    return mid || names[idx]; // never return empty
  });
}

export default disambiguateNames;
