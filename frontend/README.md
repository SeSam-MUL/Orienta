# Orienta — Frontend (React + Vite)

This directory holds the **React + Vite** single-page application that is the user
interface for Orienta, an EBSD pattern-analysis desktop app. The UI is a thin
presentation layer: it renders the workspace (sidebar navigation plus one page per
analysis module), manages local UI state with [zustand](https://github.com/pmndrs/zustand)
stores, and talks to the FastAPI backend exclusively through `src/services/api.js`
over HTTP and a WebSocket. The same bundle runs in a normal browser (Vite dev
server) and inside the Electron desktop shell.

## Directory layout

| Path | Description |
| --- | --- |
| [`index.html`](index.html) | Vite HTML entry point; mounts the app into `#root`. |
| [`vite.config.js`](vite.config.js) | Vite + React config. Sets the dev server to port `5173` and proxies `/api` → `http://127.0.0.1:8000` and `/ws` → the backend WebSocket. |
| [`package.json`](package.json) | Scripts and dependencies (React 19, axios, i18next, zustand, plotly, three, react-router, react-window). |
| [`eslint.config.js`](eslint.config.js) | Flat ESLint config (`npm run lint`). |
| [`src/main.jsx`](src/main.jsx) | App bootstrap: imports `./i18n`, wraps the tree in `ThemeProvider`, and renders `App` (or the detached pole-figure window when `?view=polefigure`). |
| [`src/App.jsx`](src/App.jsx) | Main shell: sidebar + lazy-loaded module pages, health check, WebSocket wiring, language switcher, error boundary. |
| [`src/index.css`](src/index.css), [`src/App.css`](src/App.css) | Global styles. |
| `src/components/` | One subfolder per UI area (Dashboard, EBSDViewer, Indexing, PhaseMap, EDS, Analysis, Simulation, PatternMatch, PoleFigure, DatabaseBrowser, CrystalDatabase, CrystalHint, MLHub, Settings, Batch, Refinement, DictionaryGPU, PCRefinement, HDF5Viewer) plus `shared/` (Sidebar, StatusBar, toasts) and `common/`. |
| `src/stores/` | zustand stores — the app's client-side state (see below). |
| `src/services/` | Backend client. `api.js` is the single place every endpoint is defined; `__tests__/` and `api.frame.test.js` cover it. |
| `src/i18n/` | i18next setup (`index.js`) — auto-discovers locale JSON and exposes `LANGUAGES` / `setLanguage()`. |
| `src/locales/` | Translation resources, one folder per language (`en`, `de`, `ja`, `zh`); see [`src/locales/README.md`](src/locales/README.md). |
| `src/theme/` | Theming: `ThemeProvider.jsx`, `tokens.js` (design tokens / colors), `themes.js`, `components.jsx`. |
| `src/hooks/` | Reusable hooks (e.g. `useDevLogs.js`). Most page-specific hooks live inside their component folders. |
| `src/test/` | Vitest setup (`i18n-setup.js`, loaded via `vite.config.js`). |
| `src/assets/` | Static images (hero, logos). |

## Build & dev commands

Run all commands from this `frontend/` directory.

| Command | What it does |
| --- | --- |
| `npm install` | Install dependencies (first-time setup). |
| `npm run dev` | Start the Vite dev server on `http://localhost:5173` with HMR. Requires the backend running on port `8000` (the dev server proxies `/api` and `/ws` to it). |
| `npm run build` | Production build to `dist/` (`vite build`). |
| `npm run preview` | Serve the built `dist/` locally to sanity-check the production bundle. |
| `npm run lint` | Run ESLint over the source. |
| `npm run test` | Run the Vitest suite once (`npm run test:watch` for watch mode). |
| `npm run electron:dev` | Start Vite and launch the Electron shell against it once the dev server is up. |
| `npm run electron:build` | Build the bundle and package the desktop app with electron-builder. |

The pages in `App.jsx` are code-split with `React.lazy`, so only the Dashboard is in
the initial bundle and each module loads its own chunk on first navigation.

## How the UI talks to the backend — `src/services/api.js`

`api.js` is the **only** module that knows how to reach the backend. It creates a
single axios instance whose base URL is `import.meta.env.VITE_API_URL || ''` —
empty in dev so requests go through the Vite proxy, and a full URL when bundled in
Electron. Endpoints are grouped into namespaced objects, e.g. `ebsdApi`, `h5Api`,
`indexApi`, `phaseMapApi`, `analysisApi`, `edsApi`, `simApi`, `dbApi`,
`pcApi`, `refinementApi`, `settingsApi`, `frameApi`, `poleFigureApi`,
`dictionaryGpuApi`, plus helpers like `healthCheck()` and `getGpuStatus()`.

Real-time backend pushes (progress, status) arrive over a WebSocket created by
`createWebSocket(onMessage)`, which derives the `/ws` URL from the API base and
defensively ignores non-JSON frames. Components and stores should call these
functions rather than using `fetch`/`axios` directly, so URL conventions and the
dev/Electron base-URL switch stay in one place.

## State — `src/stores/` (zustand)

State that crosses module boundaries lives in small zustand stores (replacing the
PyQt signal/slot wiring of the original desktop app). Key stores:

- **`useDataStore`** — the loaded EBSD file: path, format, grid/pattern shape,
  available features (patterns, EDS, electron images), current pixel position.
- **`useResultStore`** — indexing/analysis results shared across pages.
- **`useProgressStore`** — long-running job progress (fed by the WebSocket).
- **`useLoadedFilesStore`** — registry of loaded files for the file switcher.
- **`useFrameStore`** — reference-frame / coordinate-system selection.
- **`useResultStore`**, **`useCockpitStore`**, **`useBatchStore`**,
  **`useBatchConfigStore`**, **`useRefinementStore`**, **`usePhaseColorStore`**,
  **`useEdsColorStore`**, **`useToastStore`** — per-feature state (batch jobs,
  refinement settings, phase/EDS color maps, toasts, HDF5 cockpit view).

## Internationalization — `src/i18n/` + `src/locales/`

`src/i18n/index.js` initializes react-i18next and **auto-discovers** every locale
file via `import.meta.glob('../locales/*/*.json')`, so adding a translation only
means dropping a JSON file into the right folder — no edit to the setup is needed.
Resources are organized as `src/locales/<lng>/<namespace>.json`, with one namespace
per UI area (`nav`, `shell`, `eds`, `indexing`, …) plus a shared `common`.

Languages: **en** (source of truth), **de**, **ja**, **zh**. The active language is
persisted to `localStorage`; switch it via `setLanguage(code)` (exported from
`src/i18n/index.js`, available as `LANGUAGES`). Translation conventions (which
acronyms stay untranslated, interpolation style) are documented in
[`src/locales/README.md`](src/locales/README.md).

## Where this fits in Orienta

```
Electron shell (../electron) ─┐
Browser ──────────────────────┴─► React frontend (this dir)
                                     │  HTTP /api  +  WebSocket /ws
                                     ▼
                                  FastAPI backend (../backend)
                                     ▼
                                  Scientific layer (kikuchipy / orix / diffsims)
```

This directory is purely the presentation layer. All EBSD computation, indexing,
simulation, and file I/O happen in the Python backend; see the sibling
[`../backend`](../backend) directory and the project root
[`../README.md`](../README.md) for the full architecture and how to start the
backend the dev server proxies to.

## Run notes specific to this directory

- The dev server proxy expects the backend at `127.0.0.1:8000`; start the backend
  first or API/WebSocket calls will fail.
- `?view=polefigure` in the URL renders the standalone pole-figure window instead
  of the full app (used by the detached Electron window).
- Tests run under Vitest with jsdom; `src/test/i18n-setup.js` initializes i18n so
  `useTranslation()` returns real (English) strings in component tests.
