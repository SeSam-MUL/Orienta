/**
 * EDS element color store — single source of truth for element colors.
 * Persists to localStorage so user customizations survive page reloads.
 * Replaces hardcoded color maps in App.jsx, EDSPage.jsx, EBSDViewer.jsx,
 * and EdsOverlayPanel.jsx.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const DEFAULT_COLORS = {
  H: '#636e72',
  Li: '#fab005', Be: '#e8590c',
  B: '#a9e34b', C: '#44475a', N: '#bd93f9', O: '#f8f8f2', F: '#20c997', Ne: '#495057',
  Na: '#74c0fc', Mg: '#e599f7',
  Al: '#8be9fd', Si: '#50fa7b', P: '#69db7c', S: '#ffd43b', Cl: '#51cf66', Ar: '#495057',
  K: '#d0bfff', Ca: '#ffd43b',
  Sc: '#3bc9db', Ti: '#6272a4', V: '#3bc9db', Cr: '#bd93f9', Mn: '#ffb86c',
  Fe: '#ff5555', Co: '#ff6b6b', Ni: '#ff79c6', Cu: '#f1fa8c', Zn: '#20c997',
  Ga: '#a9e34b', Ge: '#868e96', As: '#868e96', Se: '#fab005', Br: '#e64980', Kr: '#495057',
  Rb: '#d0bfff', Sr: '#ffd43b',
  Y: '#3bc9db', Zr: '#74c0fc', Nb: '#e599f7', Mo: '#fab005', Tc: '#868e96',
  Ru: '#868e96', Rh: '#868e96', Pd: '#ced4da', Ag: '#ced4da', Cd: '#adb5bd',
  In: '#a9e34b', Sn: '#adb5bd', Sb: '#868e96', Te: '#fab005', I: '#e64980', Xe: '#495057',
  Cs: '#d0bfff', Ba: '#ffd43b',
  Hf: '#74c0fc', Ta: '#74c0fc', W: '#fd7e14', Re: '#868e96',
  Os: '#868e96', Ir: '#868e96', Pt: '#ced4da', Au: '#fcc419', Hg: '#adb5bd',
  Tl: '#a9e34b', Pb: '#868e96', Bi: '#868e96', Po: '#868e96', At: '#868e96', Rn: '#495057',
  Fr: '#d0bfff', Ra: '#ffd43b',
  La: '#66d9e8', Ce: '#66d9e8', Pr: '#66d9e8', Nd: '#66d9e8', Pm: '#66d9e8',
  Sm: '#66d9e8', Eu: '#66d9e8', Gd: '#66d9e8', Tb: '#66d9e8', Dy: '#66d9e8',
  Ho: '#66d9e8', Er: '#66d9e8', Tm: '#66d9e8', Yb: '#66d9e8', Lu: '#66d9e8',
  Ac: '#ffa8a8', Th: '#ffa8a8', Pa: '#ffa8a8', U: '#ffa8a8', Np: '#ffa8a8',
  Pu: '#ffa8a8', Am: '#ffa8a8', Cm: '#ffa8a8', Bk: '#ffa8a8', Cf: '#ffa8a8',
  Es: '#ffa8a8', Fm: '#ffa8a8', Md: '#ffa8a8', No: '#ffa8a8', Lr: '#ffa8a8',
  Rf: '#868e96', Db: '#868e96', Sg: '#868e96', Bh: '#868e96',
  Hs: '#868e96', Mt: '#868e96', Ds: '#868e96', Rg: '#868e96',
  Cn: '#868e96', Nh: '#868e96', Fl: '#868e96', Mc: '#868e96',
  Lv: '#868e96', Ts: '#868e96', Og: '#495057',
  He: '#495057',
};

const useEdsColorStore = create(
  persist(
    (set, get) => ({
      colors: { ...DEFAULT_COLORS },
      setColor: (symbol, color) =>
        set((s) => ({ colors: { ...s.colors, [symbol]: color } })),
      resetColors: () => set({ colors: { ...DEFAULT_COLORS } }),
      getColor: (symbol) => get().colors[symbol] || '#bd93f9',
    }),
    { name: 'eds-element-colors' }
  )
);

export { DEFAULT_COLORS };
export default useEdsColorStore;
