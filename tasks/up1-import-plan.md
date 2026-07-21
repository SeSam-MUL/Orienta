# Task: EDAX UP1 Pattern-Import verdrahten

## Context
Neue Testdaten: EDAX `.up1` (8-bit rohe Patterns) + `.osc`-Beileger in `Test_data/new test/`.
User will diese Dateien laden **und damit arbeiten** (viewen, prozessieren, indexieren).
kikuchipy 0.11.3 liest `.up1` bereits nativ + lazy — verifiziert an den echten Dateien.

## Befund aus der Untersuchung (echte Dateien)
- **Scan5.up1**: UP1 v1, 79×79 Patterns, 22801 Punkte. v1-Header trägt KEINE Map-Form
  → kikuchipy liefert flach `(22801|79,79)`. Reshape via `nav_shape=(151,151)` funktioniert.
- **map…cropped.up1**: UP1 v3, 114×114 Patterns → kikuchipy liefert korrekt `(123,91|114,114)`
  (v3-Header trägt die Form selbst — kein Reshape nötig).
- **.osc-Beileger**: Offsets 16/20/24 = `ncols-1`, `nrows-1`, `npoints`. Auf beiden Dateien
  bestätigt: (150+1)²=22801, (122+1)·(90+1)=11193. Robuste Validierung: `(o16+1)*(o20+1)==o24`.
- Geladenes Signal bringt **Default-Detektor** (pc=0.5/0.5/0.5, sample_tilt=70°, shape aus Patterns),
  `xmap=None`, `static_background=None`, keine EDS. → Indexing startet mit Default-PC (wie üblich
  per PC-Refinement zu kalibrieren); EDS/Static-BG-Features degradieren wie bei EDS-losen h5.
- Random-Access-Lesen: 4,6 ms/Pattern (rohes contiguous Layout).

## Approach
Einziger Choke-Point ist `safe_loader.load_ebsd_safe` (nutzen sowohl die async-Route
`/api/ebsd/load` als auch der sync-Batch-Pfad `load_ebsd_file`). Dort UP1 erkennen,
für v1 die `nav_shape` aus dem `.osc` (Fallback: perfektes Quadrat) auflösen und an
`kp.load` durchreichen. v3 unangetastet lassen. Frontend-Datei-Filter um `up1`/`up2`
erweitern. Keine Änderung an Indexing/Detector nötig (Defaults greifen).

## Steps
- [ ] **1. Neues Modul `edax_up1.py`** (klein, testbar):
      - `read_up1_header(path) -> (version, pw, ph, npat, data_offset)`
      - `read_osc_grid(osc_path) -> (ncols, nrows) | None` (validiert `(o16+1)*(o20+1)==o24`)
      - `resolve_up1_nav_shape(up1_path) -> tuple | None`
        (v1: .osc → sonst perfektes Quadrat → sonst None+Warnung; v3: None)
- [ ] **2. `safe_loader.load_ebsd_safe`**: UP1-Zweig VOR dem generischen kp.load —
      bei `.up1/.up2` `nav_shape` auflösen und `kp.load(path, lazy=use_lazy, nav_shape=…)`.
      Fail-loud, wenn v1 ohne .osc und keine Quadratzahl (klare Fehlermeldung an den User).
- [ ] **3. Frontend-Filter**: `frontend/src/components/EBSDViewer/EBSDViewer.jsx`
      - Zeile ~1011 Datei-Dialog-Filter: `up1`, `up2` ergänzen (+ eigener Filtereintrag "EDAX Patterns")
      - Zeile ~2319 Drag&Drop-Regex `/\.(h5oina|h5|hdf5)$/i` → `up1|up2` ergänzen
- [ ] **4. Graceful-Check**: sicherstellen, dass EDS-Auto-Probe / `useDefaultLayers` / h5-Session
      bei up1 nicht crasht (fail-soft) — sonst dünn abfangen.
- [ ] **5. Tests** `tests/test_up1_import.py`:
      - `resolve_up1_nav_shape` auf echten Dateien (Scan5 → (151,151); map → None)
      - `load_ebsd_safe` auf beiden up1 → 2D nav, dtype uint8, Detektor vorhanden
      - .osc-Grid-Parsing + Validierungs-Guard (falsche/fehlende .osc → Fallback)
- [ ] **6. Verifikation**: Backend-Restart, echte up1 über die UI laden, Overview + Pattern
      anzeigen, eine kleine ROI mit Hough/Dictionary indexieren (Default-PC) → plausibel.

## Risks
- v1 ohne .osc und keine Quadratzahl → nicht auto-reshapebar. Mitigation: fail-loud mit
  Hinweis „Grid unbekannt, .osc fehlt" statt stiller Flach-Ladung.
- `.osc`-Format ist versionsabhängig; ich nutze NUR die 3 robust verifizierten Felder
  (ncols-1/nrows-1/npoints) mit Konsistenz-Check — kein Voll-Reverse-Engineering (PC/Orientierungen
  aus .osc bewusst NICHT, zu riskant).
- up1 = nur Patterns: kein EDS, keine Aztec-Vorindizierung, kein static-BG. Bewusste
  Einschränkung, kein Bug — muss dem User klar sein (Info-Hinweis beim Laden optional).

## Done When
- Beide echten up1 laden über die normale „Load"-UI mit korrekter 2D-Map.
- Overview + Einzelpattern-Anzeige funktionieren.
- Indexing einer ROI läuft mit Default-PC durch (Ergebnis positionsrichtig zurückgeschrieben).
- `tests/test_up1_import.py` grün; bestehende Loader-Tests unverändert grün.
