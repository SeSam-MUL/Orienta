// Maps the backend `source` field to an i18n key for the provenance label.
export function qualityProvenanceKey(source) {
  return source === 'native' ? 'quality.nativeBandContrast' : 'quality.computedPatternQuality'
}
