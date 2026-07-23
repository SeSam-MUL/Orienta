import { describe, it, expect } from 'vitest'
import { qualityProvenanceKey } from './qualityProvenance'

describe('qualityProvenanceKey', () => {
  it('maps native source to the native band-contrast key', () => {
    expect(qualityProvenanceKey('native')).toBe('quality.nativeBandContrast')
  })

  it('maps computed source to the computed pattern-quality key', () => {
    expect(qualityProvenanceKey('computed')).toBe('quality.computedPatternQuality')
  })

  it('falls back to computed for anything else', () => {
    expect(qualityProvenanceKey(undefined)).toBe('quality.computedPatternQuality')
    expect(qualityProvenanceKey(null)).toBe('quality.computedPatternQuality')
    expect(qualityProvenanceKey('anything')).toBe('quality.computedPatternQuality')
  })
})
