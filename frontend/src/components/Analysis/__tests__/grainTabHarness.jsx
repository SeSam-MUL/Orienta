// GrainTab inside the page-level state and the page's watch of a running reconstruction it gets in AnalysisPage (the result store): grains, version,
// reconstruction parameters and statistics settings. Not a test file; the tests import it.
import useResultStore from '../../../stores/useResultStore';
import { GrainTab } from '../AnalysisPage';
import { useReconstructionWatch } from '../useReconstructionWatch';

export function GrainTabInPage() {
  const store = useResultStore();
  useReconstructionWatch();      // the page polls a running reconstruction; the tab only shows it
  return (
    <GrainTab
      analysisLoaded
      hasGrains={store.hasGrains}
      grainsVersion={store.grainsVersion}
      onGrainsReconstructed={store.markGrainsReconstructed}
      onComplete={() => {}}
      statsSettings={store.grainStatsSettings}
      onStatsSettingsChange={store.setGrainStatsSettings}
      onGrainMapsStale={store.bumpGrainMaps}
    />
  );
}
