/**
 * Source-link resolution for PhaseMap layered viewer.
 *
 * Determines whether the active indexing result has a usable H5OINA
 * source file backing it (for SE/EDS/VBSE layers) and whether the
 * pixel geometries align.
 */

// What the backend actually calls these things.
//
//   /api/ebsd/info      → { file_path, data_shape: [rows, cols, detRows, detCols] }
//   /api/indexing/results → { source_file, original_shape }
//
// This module used to read `path`/`shape`/`source_path`, none of which exist,
// so every result resolved to "no source file available" and the whole H5OINA
// group — every EDS element map and electron image — silently never appeared
// in the layer list. The aliases are read in order; the old names stay first so
// anything already passing them keeps working.
function pickPath(obj) {
  return obj?.source_path ?? obj?.path ?? obj?.file_path ?? obj?.source_file ?? null;
}

// A pattern array is [rows, cols, detectorRows, detectorCols]; the map is the
// first two. A map shape is already [rows, cols] and passes through.
function pickShape(obj) {
  const raw = obj?.shape ?? obj?.data_shape ?? obj?.original_shape ?? null;
  if (!Array.isArray(raw) || raw.length < 2) return null;
  return [raw[0], raw[1]];
}

function shapesEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b)
    && a.length === b.length
    && a.every((v, i) => v === b[i]);
}

/**
 * @param {object} args
 * @param {object|null} args.resultEntry      Gallery entry for current result
 * @param {object|null} args.ebsdInfo         /api/ebsd/info response
 * @param {object|null} args.analysisStatus   /api/analysis/status response
 * @param {object|null} args.manualOverride   User-picked file (overrides auto)
 * @param {number[]|null} args.resultShape    [nRows, nCols] of the result
 * @returns {{linked: boolean, sourcePath: string|null, reason: string|null,
 *           sourceShape: number[]|null}}
 */
export function resolveSource({
  resultEntry,
  ebsdInfo,
  analysisStatus,         // reserved for future use (KAM/GOS layers)
  manualOverride,
  resultShape = null,
} = {}) {
  void analysisStatus;

  // 1. Manual override always wins
  if (manualOverride?.path) {
    const sourceShape = manualOverride.shape ?? null;
    if (resultShape && sourceShape && !shapesEqual(resultShape, sourceShape)) {
      return {
        linked: true,
        sourcePath: manualOverride.path,
        reason: 'Shape mismatch — some layers may auto-crop or be disabled.',
        sourceShape,
      };
    }
    return { linked: true, sourcePath: manualOverride.path, reason: null, sourceShape };
  }

  // 2. resultEntry.source_path + ebsdInfo agree → auto-link
  const sourcePath = pickPath(resultEntry?.data) ?? pickPath(ebsdInfo);
  const sourceShape = pickShape(ebsdInfo);

  if (!sourcePath) {
    return {
      linked: false,
      sourcePath: null,
      reason: 'No source file available. Load the source H5OINA in the EBSD page or link manually.',
      sourceShape: null,
    };
  }

  // 3. Shape check
  if (resultShape && sourceShape && !shapesEqual(resultShape, sourceShape)) {
    return {
      linked: false,
      sourcePath,
      reason: `Shape mismatch (result ${resultShape.join('×')} vs source ${sourceShape.join('×')}). Re-run indexing on full map, or link a different source.`,
      sourceShape,
    };
  }

  return { linked: true, sourcePath, reason: null, sourceShape };
}

/**
 * Compute the alignment between result and source shapes.
 * @returns {{aligned: boolean, mode: 'exact'|'crop'|null, crop?: object, reason?: string}}
 */
export function alignShape({ resultShape, sourceShape, cropOffset = null } = {}) {
  if (!resultShape || !sourceShape) {
    return { aligned: false, mode: null, reason: 'Missing shape information.' };
  }
  if (shapesEqual(resultShape, sourceShape)) {
    return { aligned: true, mode: 'exact' };
  }
  if (cropOffset && Array.isArray(cropOffset) && cropOffset.length === 2) {
    const [r0, c0] = cropOffset;
    const [rH, cW] = resultShape;
    const [sH, sW] = sourceShape;
    if (r0 >= 0 && c0 >= 0 && r0 + rH <= sH && c0 + cW <= sW) {
      return {
        aligned: true,
        mode: 'crop',
        crop: { row0: r0, row1: r0 + rH, col0: c0, col1: c0 + cW },
      };
    }
    return {
      aligned: false,
      mode: null,
      reason: 'Crop offset exceeds source bounds.',
    };
  }
  return {
    aligned: false,
    mode: null,
    reason: `Shape mismatch (${resultShape.join('×')} vs ${sourceShape.join('×')}, no crop offset).`,
  };
}
