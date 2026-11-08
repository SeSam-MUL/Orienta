// Whether a phase is loaded on the PC page, for the buttons far below the page that need one ("Show the bands over a
// pattern" draws the bands of the loaded phase). The page provides it; a part used alone has a phase.
import { createContext, useContext } from 'react';

export const PhaseLoadedContext = createContext(true);

export const usePhaseLoaded = () => useContext(PhaseLoadedContext);
