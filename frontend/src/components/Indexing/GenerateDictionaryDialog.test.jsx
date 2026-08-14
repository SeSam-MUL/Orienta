// @vitest-environment jsdom
/**
 * The user could not generate a usable dictionary because the dialog
 *  (a) offered no way to say WHICH phase, and
 *  (b) defaulted to a 60x60 detector at PC (0.5, 0.5, 0.5) regardless of the
 *      loaded dataset, so the result matched nothing, and
 *  (c) sent no save_to_library, so the file landed in tasks/ where the
 *      Indexing page's discovery never looks.
 * These tests pin all three.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const generate = vi.fn();
vi.mock('../../services/api', () => ({
  dictionaryGpuApi: { generate: (...a) => generate(...a) },
}));

import GenerateDictionaryDialog from './GenerateDictionaryDialog';

const TARGETS = [
  {
    index: 0, phasePath: 'C:/db/Al_master_E20kV_npx500.h5', label: 'Al', formula: 'Al',
    masterPath: 'C:/db/Al_master_E20kV_npx500.h5',
    masterFilename: 'Al_master_E20kV_npx500.h5', hasMaster: true,
  },
  {
    index: 1, phasePath: 'C:/db/Si_master_E20kV_npx500.h5', label: 'Si', formula: 'Si',
    masterPath: 'C:/db/Si_master_E20kV_npx500.h5',
    masterFilename: 'Si_master_E20kV_npx500.h5', hasMaster: true,
  },
];

function renderDialog(props = {}) {
  return render(
    <GenerateDictionaryDialog
      isOpen
      onClose={() => {}}
      taskId={null}
      setTaskId={() => {}}
      progress={null}
      setProgress={() => {}}
      phaseTargets={TARGETS}
      detectorShape={[118, 118]}
      pc={[0.547, 0.465, 0.609]}
      energyKv={20}
      resolutionDeg={2.0}
      datasetName="AL_SI_x3000"
      initialMasterPath={TARGETS[0].masterPath}
      {...props}
    />
  );
}

/** The master-path text input (MasterSelector renders the only text input). */
const masterInput = () =>
  Array.from(document.querySelectorAll('input')).find(i => i.type === 'text');

beforeEach(() => {
  generate.mockReset();
  generate.mockResolvedValue({ data: { task_id: 't-1', output_path: 'C:/lib/Si/x.h5' } });
});
afterEach(() => cleanup());

describe('phase selection', () => {
  it('lists every selected phase with its own master', () => {
    renderDialog();
    const select = screen.getByLabelText(/Phase:/);
    const labels = Array.from(select.options).map(o => o.textContent);
    expect(labels).toContain('Al — Al_master_E20kV_npx500.h5');
    expect(labels).toContain('Si — Si_master_E20kV_npx500.h5');
  });

  it('switching the phase switches the master that will be simulated', () => {
    renderDialog();
    expect(masterInput().value).toBe(TARGETS[0].masterPath);
    fireEvent.change(screen.getByLabelText(/Phase:/), { target: { value: '1' } });
    expect(masterInput().value).toBe(TARGETS[1].masterPath);
  });

  it('preselects the phase whose master the page passed in', () => {
    renderDialog({ initialMasterPath: TARGETS[1].masterPath });
    // Per-phase button on the Si card -> dialog opens on Si, not on Al.
    expect(screen.getByLabelText(/Phase:/).value).toBe('1');
    expect(masterInput().value).toBe(TARGETS[1].masterPath);
  });

  it('re-seeds when reopened for a different phase', () => {
    const { rerender } = renderDialog({ initialMasterPath: TARGETS[0].masterPath });
    expect(masterInput().value).toBe(TARGETS[0].masterPath);

    const props = {
      isOpen: false, onClose: () => {}, taskId: null, setTaskId: () => {},
      progress: null, setProgress: () => {}, phaseTargets: TARGETS,
      detectorShape: [118, 118], pc: [0.547, 0.465, 0.609],
      energyKv: 20, resolutionDeg: 2.0, datasetName: 'AL_SI_x3000',
    };
    rerender(<GenerateDictionaryDialog {...props} initialMasterPath={TARGETS[0].masterPath} />);
    rerender(<GenerateDictionaryDialog {...props} isOpen initialMasterPath={TARGETS[1].masterPath} />);
    expect(masterInput().value).toBe(TARGETS[1].masterPath);
  });

  it('marks a phase without a master as unselectable instead of hiding it', () => {
    const noMaster = { index: 2, label: 'Fe', masterPath: '', masterFilename: '', hasMaster: false };
    renderDialog({ phaseTargets: [...TARGETS, noMaster] });
    const opt = Array.from(screen.getByLabelText(/Phase:/).options).find(o => o.value === '2');
    expect(opt).toBeTruthy();
    expect(opt.disabled).toBe(true);
  });
});

describe('geometry seeding', () => {
  it('uses the loaded dataset detector size, not the old hardcoded 60x60', () => {
    renderDialog();
    const nums = Array.from(document.querySelectorAll('input[type="number"]')).map(i => i.value);
    expect(nums.slice(0, 2)).toEqual(['118', '118']);
  });

  it("uses the page's current PC, not (0.5, 0.5, 0.5)", () => {
    renderDialog();
    const nums = Array.from(document.querySelectorAll('input[type="number"]')).map(i => i.value);
    expect(nums.slice(2, 5)).toEqual(['0.547', '0.465', '0.609']);
  });

  it('says where the geometry came from', () => {
    renderDialog();
    expect(screen.getByText(/AL_SI_x3000/)).toBeTruthy();
    expect(screen.getByText(/118x118/)).toBeTruthy();
  });

  it('falls back to 60x60 and warns when no dataset is loaded', () => {
    renderDialog({ detectorShape: null, pc: null, datasetName: '' });
    const nums = Array.from(document.querySelectorAll('input[type="number"]')).map(i => i.value);
    expect(nums.slice(0, 2)).toEqual(['60', '60']);
    expect(screen.getByText(/No dataset loaded/i)).toBeTruthy();
  });
});

describe('submitted payload', () => {
  it('carries phase master, real geometry, energy and save_to_library', async () => {
    renderDialog();
    fireEvent.change(screen.getByLabelText(/Phase:/), { target: { value: '1' } });
    fireEvent.click(screen.getByText('Generate'));

    await waitFor(() => expect(generate).toHaveBeenCalled());
    expect(generate.mock.calls[0][0]).toEqual({
      master_path: TARGETS[1].masterPath,
      detector_shape: [118, 118],
      pc: [0.547, 0.465, 0.609],
      sample_tilt: 70,
      energy_kv: 20,
      resolution_deg: 2,
      normalize: false,
      save_to_library: true,
      backend: 'gpu',
      detector_tilt_deg: 0,
      azimuthal_deg: 0,
    });
  });

  it('sends the REAL camera tilt, not a hardcoded 0', async () => {
    // The whole reason the first generated dictionary was useless: the
    // detector sits at 3.44 deg and generation simulated a flat one, which
    // measured NCC 0.017 against the correct pattern for the SAME orientation.
    renderDialog({ detectorTilt: 3.44 });
    fireEvent.click(screen.getByText('Generate'));
    await waitFor(() => expect(generate).toHaveBeenCalled());
    expect(generate.mock.calls[0][0].detector_tilt_deg).toBe(3.44);
  });

  it('seeds the camera-tilt field from the loaded dataset', () => {
    renderDialog({ detectorTilt: 3.44 });
    expect(screen.getByLabelText(/Camera tilt:/).value).toBe('3.44');
  });

  it('re-seeds the camera tilt when reopened', () => {
    const base = {
      onClose: () => {}, taskId: null, setTaskId: () => {}, progress: null,
      setProgress: () => {}, phaseTargets: TARGETS, detectorShape: [118, 118],
      pc: [0.547, 0.465, 0.609], energyKv: 20, resolutionDeg: 2.0,
      initialMasterPath: TARGETS[0].masterPath,
    };
    const { rerender } = render(<GenerateDictionaryDialog {...base} isOpen detectorTilt={0} />);
    expect(screen.getByLabelText(/Camera tilt:/).value).toBe('0');
    rerender(<GenerateDictionaryDialog {...base} isOpen={false} detectorTilt={3.44} />);
    rerender(<GenerateDictionaryDialog {...base} isOpen detectorTilt={3.44} />);
    expect(screen.getByLabelText(/Camera tilt:/).value).toBe('3.44');
  });

  it('routes to the CPU backend when CPU is picked', async () => {
    renderDialog();
    fireEvent.click(screen.getByLabelText(/CPU/));
    fireEvent.click(screen.getByText('Generate'));

    await waitFor(() => expect(generate).toHaveBeenCalled());
    expect(generate.mock.calls[0][0].backend).toBe('cpu');
  });

  it('lets CPU actually be selected — it used to be a disabled radio', () => {
    renderDialog();
    const cpu = screen.getByLabelText(/CPU/);
    expect(cpu.disabled).toBe(false);
    fireEvent.click(cpu);
    expect(cpu.checked).toBe(true);
    // …and Generate must not stay greyed out on CPU.
    expect(screen.getByText('Generate').disabled).toBe(false);
  });

  it('refuses a non-positive energy instead of writing a mislabelled file', async () => {
    renderDialog();
    fireEvent.change(screen.getByLabelText(/Energy:/), { target: { value: '0' } });
    fireEvent.click(screen.getByText('Generate'));

    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy());
    expect(generate).not.toHaveBeenCalled();
  });
});
