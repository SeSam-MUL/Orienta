/**
 * Source-link resolution for PhaseMap layered viewer.
 *
 * Determines whether the active indexing result has a usable H5OINA
 * source file backing it (for SE/EDS/VBSE layers) and whether the
 * pixel geometries align.
 */

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
  const sourcePath = resultEntry?.data?.source_path ?? ebsdInfo?.path ?? null;
  const sourceShape = ebsdInfo?.shape ?? null;

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
