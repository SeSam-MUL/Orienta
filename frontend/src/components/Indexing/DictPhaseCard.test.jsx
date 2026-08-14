// @vitest-environment jsdom
/**
 * "ich kann auch nicht auswählen für welche phase ich eines erstelle" —
 * each phase card now carries its own Generate button, and it hands back
 * THAT phase's master.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react';
import DictPhaseCard from './DictPhaseCard';
import SelectedPhasesList from './SelectedPhasesList';

afterEach(() => cleanup());

const AL_MASTER = 'C:/db/EBSD_H5_Cache/Al/Al_master_E20kV_npx500.h5';
const SI_MASTER = 'C:/db/EBSD_H5_Cache/Si/Si_master_E20kV_npx500.h5';

const FILES = [
  { path: AL_MASTER, filename: 'Al_master_E20kV_npx500.h5', file_type: 'master', formula: 'Al' },
  { path: SI_MASTER, filename: 'Si_master_E20kV_npx500.h5', file_type: 'master', formula: 'Si' },
  {
    path: 'C:/db/Dictionary_Library/Al/Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    filename: 'Al_master_E20kV_npx500_dict_20kV_128x156_5.0deg.h5',
    file_type: 'dictionary', formula: 'Al', pc: [0.5, 0.5, 0.5], resolution: 5.0,
  },
];

const AL = { path: AL_MASTER, formula: 'Al', display_label: 'Al', file_type: 'master', space_group: 'Fm-3m' };
const SI = { path: SI_MASTER, formula: 'Si', display_label: 'Si', file_type: 'master', space_group: 'Fd-3m' };

describe('DictPhaseCard generate button', () => {
  it('is offered even when the phase already has dictionaries', () => {
    // The user's Al has two dictionaries — both for the wrong detector. The
    // card must still let them build a matching one.
    render(<DictPhaseCard phase={AL} allFiles={FILES} onGenerateDict={() => {}} />);
    expect(screen.getByText('+ Generate dictionary')).toBeTruthy();
  });

  it('hands back the phase and ITS OWN master', () => {
    const onGenerateDict = vi.fn();
    render(<DictPhaseCard phase={SI} allFiles={FILES} onGenerateDict={onGenerateDict} />);
    fireEvent.click(screen.getByText('+ Generate dictionary'));
    expect(onGenerateDict).toHaveBeenCalledWith(SI, SI_MASTER);
  });

  it('disables and explains itself when the phase has no master', () => {
    const onGenerateDict = vi.fn();
    const noMaster = { path: 'C:/db/CIF_Library/Fe.cif', formula: 'Fe', display_label: 'Fe' };
    render(<DictPhaseCard phase={noMaster} allFiles={FILES} onGenerateDict={onGenerateDict} />);

    const btn = screen.getByText('+ Generate dictionary');
    expect(btn.disabled).toBe(true);
    fireEvent.click(btn);
    expect(onGenerateDict).not.toHaveBeenCalled();
    expect(screen.getByText(/no master pattern for this phase/i)).toBeTruthy();
  });

  it('stays out of the way when the page passes no handler', () => {
    render(<DictPhaseCard phase={AL} allFiles={FILES} />);
    expect(screen.queryByText('+ Generate dictionary')).toBeNull();
  });

  it('still keeps each phase to its own dictionaries', () => {
    // Guard for the refactor onto the shared dictionaryTargets helpers.
    const { container } = render(<DictPhaseCard phase={SI} allFiles={FILES} />);
    expect(within(container).getByText(/No dictionary generated/i)).toBeTruthy();
  });
});

describe('SelectedPhasesList wiring', () => {
  it('gives every dictionary phase its own button', () => {
    const onGenerateDict = vi.fn();
    render(
      <SelectedPhasesList
        phases={[AL, SI]}
        method="dictionary"
        allDiscoveredFiles={FILES}
        onGenerateDict={onGenerateDict}
      />
    );
    const buttons = screen.getAllByText('+ Generate dictionary');
    expect(buttons).toHaveLength(2);

    fireEvent.click(buttons[1]);
    expect(onGenerateDict).toHaveBeenCalledWith(SI, SI_MASTER);
  });
});
