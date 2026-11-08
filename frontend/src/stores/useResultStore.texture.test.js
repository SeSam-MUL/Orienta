// The Texture tab's phase choice in the store.
import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';

const state = () => useResultStore.getState();
const PHASES = [{ id: 0, name: 'Si' }, { id: 1, name: 'Al' }];
const STATUS = { loaded: true, shape: [4, 5], has_grains: false, phase_options: PHASES, dataset_token: 't1' };

beforeEach(() => { state().clearResults(); });

describe('texturePhase', () => {
  it('starts at the phase with the most indexed pixels', () => {
    expect(state().texturePhase).toBe('most_measured');
  });

  it('a status answer of the same map takes the backend\'s choice', () => {
    state().applyBackendStatus(STATUS);
    state().applyBackendStatus({ ...STATUS, texture_phase: 1 });
    expect(state().texturePhase).toBe(1);
    state().applyBackendStatus({ ...STATUS, texture_phase: 'most_measured' });
    expect(state().texturePhase).toBe('most_measured');
  });

  it("a pick that no run has applied yet survives status answers that carry the backend's choice", () => {
    state().applyBackendStatus({ ...STATUS, texture_phase: 'most_measured' });
    state().setTexturePhase(0);                                       // picked on the tab, not run yet
    expect(state().textureEdited).toBe(true);
    state().applyBackendStatus({ ...STATUS, texture_phase: 'most_measured' });    // what the real backend answers
    expect(state().texturePhase).toBe(0);
    state().applyBackendStatus({ ...STATUS, texture_phase: 1 });
    expect(state().texturePhase).toBe(0);
  });

  it('a run that applied the pick hands the choice back to the backend', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(0);
    state().markTextureApplied({ phase: 0, preset: 'FCC_Rolling', tolerance: '10.0' });
    expect(state().textureEdited).toBe(false);
    state().applyBackendStatus({ ...STATUS, texture_phase: 1 });
    expect(state().texturePhase).toBe(1);
  });

  it('a run for an earlier pick does not apply the one picked meanwhile', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(0);
    state().setTexturePhase(1);
    state().markTextureApplied({ phase: 0, preset: 'FCC_Rolling', tolerance: '10.0' });
    expect(state().textureEdited).toBe(true);
    state().applyBackendStatus({ ...STATUS, texture_phase: 'most_measured' });
    expect(state().texturePhase).toBe(1);
  });

  it('another map drops a pending pick', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(1);
    state().applyBackendStatus({ ...STATUS, dataset_token: 't2', texture_phase: 'most_measured' });
    expect(state().texturePhase).toBe('most_measured');
    expect(state().textureEdited).toBe(false);
  });

  it('takes the backend\'s preset and tolerance, keeps the typed text of the same number, and the default for nonsense', () => {
    state().applyBackendStatus({ ...STATUS, texture_preset: 'BCC_Rolling', texture_tolerance_deg: 4 });
    expect([state().texturePreset, state().textureTolerance]).toEqual(['BCC_Rolling', '4']);
    state().setTextureTolerance('4.0');
    state().markTextureApplied({ phase: 'most_measured', preset: 'BCC_Rolling', tolerance: '4.0' });
    state().applyBackendStatus({ ...STATUS, texture_preset: 'BCC_Rolling', texture_tolerance_deg: 4 });
    expect(state().textureTolerance).toBe('4.0');
    state().applyBackendStatus({ ...STATUS, texture_preset: 'HCP_Rolling', texture_tolerance_deg: 99 });
    expect([state().texturePreset, state().textureTolerance]).toEqual(['FCC_Rolling', '10.0']);
  });

  it('a preset or tolerance typed on the tab and not run yet survives status answers that carry the backend\'s', () => {
    state().applyBackendStatus({ ...STATUS, texture_preset: 'FCC_Rolling', texture_tolerance_deg: 10 });
    state().setTexturePreset('BCC_Rolling');
    state().setTextureTolerance('3');
    expect(state().textureEdited).toBe(true);
    state().applyBackendStatus({ ...STATUS, texture_preset: 'FCC_Rolling', texture_tolerance_deg: 10 });
    expect([state().texturePreset, state().textureTolerance]).toEqual(['BCC_Rolling', '3']);
    state().markTextureApplied({ phase: 'most_measured', preset: 'BCC_Rolling', tolerance: '3' });
    state().applyBackendStatus({ ...STATUS, texture_preset: 'FCC_Rolling', texture_tolerance_deg: 10 });
    expect([state().texturePreset, state().textureTolerance]).toEqual(['FCC_Rolling', '10']);     // the backend's number
  });

  it('another map puts preset and tolerance back at the defaults', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePreset('BCC_Rolling');
    state().setTextureTolerance('3');
    state().applyBackendStatus({ ...STATUS, dataset_token: 't2', texture_preset: 'FCC_Rolling', texture_tolerance_deg: 10 });
    expect([state().texturePreset, state().textureTolerance, state().textureEdited]).toEqual(['FCC_Rolling', '10', false]);
    state().setTextureTolerance('3');
    state().resetGrainStatsSettings();
    expect(state().textureTolerance).toBe('10.0');
  });

  it('an answer without the field keeps what is chosen', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(0);
    state().applyBackendStatus(STATUS);
    expect(state().texturePhase).toBe(0);
  });

  it('a phase id the map does not have becomes the default', () => {
    state().applyBackendStatus({ ...STATUS, texture_phase: 7 });
    expect(state().texturePhase).toBe('most_measured');
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(1);
    state().applyBackendStatus({ ...STATUS, phase_options: [{ id: 5, name: 'Cu' }], dataset_token: 't2' });
    expect(state().texturePhase).toBe('most_measured');
  });

  it('another map, a load and a lost map put it back', () => {
    state().applyBackendStatus(STATUS);
    state().setTexturePhase(1);
    state().applyBackendStatus({ ...STATUS, dataset_token: 't2' });
    expect(state().texturePhase).toBe('most_measured');
    state().setTexturePhase(1);
    state().resetGrainStatsSettings();
    expect(state().texturePhase).toBe('most_measured');
    state().setTexturePhase(1);
    state().applyBackendStatus({ loaded: false });
    expect(state().texturePhase).toBe('most_measured');
  });
});
