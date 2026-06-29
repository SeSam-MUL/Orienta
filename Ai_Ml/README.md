> **⚠️ Status: work in progress — to be finished.** This module ships with the
> Orienta source release but is not yet complete; its APIs and results may change.

# EBSD-AI: Phasenvorhersage fuer EBSD-Diffraktionsmuster

Ein Python-Programm, das kuenstliche Intelligenz nutzt, um Kristallphasen in Materialien automatisch zu erkennen.

---

## Was macht dieses Programm?

Dieses Programm analysiert spezielle Bilder von Materialien, die mit einem Elektronenmikroskop aufgenommen wurden (sogenannte "EBSD-Muster"), und sagt vorher, welche Kristallstruktur (Phase) das Material an dieser Stelle hat.

### Einfach erklaert

Wenn Sie ein Stueck Metall oder ein anderes kristallines Material unter einem speziellen Mikroskop untersuchen, entstehen charakteristische Muster (wie Fingerabdruecke des Materials). Dieses Programm "lernt", diese Muster zu erkennen und kann dann automatisch sagen: "Das ist wahrscheinlich Eisen in dieser Form" oder "Das ist Stahl mit dieser Struktur".

### Was ist EBSD?

EBSD steht fuer "Electron Backscatter Diffraction" (Elektronenrueckstreubeugung). Das ist eine Technik, bei der:
- Ein Elektronenmikroskop Elektronen auf eine Materialprobe schiesst
- Diese Elektronen werden vom Kristallgitter zurueckgestreut
- Dabei entstehen charakteristische Streifenmuster (wie Barcodes)
- Jede Kristallphase hat ihr eigenes typisches Muster

### Was sind Kristallphasen?

Viele Materialien (besonders Metalle) koennen in verschiedenen Kristallformen existieren, zum Beispiel:
- Eisen kann als "Ferrit" (wuerfelfoermige Atomanordnung) vorliegen
- Oder als "Austenit" (anders gepackte Wuerfel)
- Stahl kann verschiedene Mischungen und Phasen haben

Diese unterschiedlichen Anordnungen nennt man "Phasen", und sie haben verschiedene Eigenschaften (Haerte, Magnetismus, usw.).

### Wie funktioniert die KI?

Das Programm nutzt ein "neuronales Netz" - eine Art kuenstliches Gehirn, das aus Beispielen lernt:
1. Sie zeigen dem Programm viele EBSD-Muster mit Beschriftungen ("das ist Phase A", "das ist Phase B")
2. Das Programm lernt die typischen Merkmale jeder Phase
3. Bei neuen, unbeschrifteten Mustern kann es dann vorhersagen, welche Phase es ist

Zusaetzlich kann das Programm auch chemische Informationen (EDS-Daten) nutzen - also welche chemischen Elemente im Material vorhanden sind.

---

## Voraussetzungen

Bevor Sie beginnen, brauchen Sie:

1. **Einen Computer** mit Windows, Linux oder macOS
2. **Python Version 3.10 oder neuer** ([Download hier](https://www.python.org/downloads/))
   - Pruefen Sie Ihre Python-Version in der Kommandozeile: `python --version`
3. **Grundlegende Kenntnisse der Kommandozeile** (Terminal/CMD)
   - Wie man einen Ordner oeffnet
   - Wie man Befehle eintippt und mit Enter bestaetigt

### Was ist die Kommandozeile?

- **Windows**: Programm "CMD" oder "PowerShell" (Suche im Startmenue)
- **Mac**: Programm "Terminal" (in Programme > Dienstprogramme)
- **Linux**: Terminal-Anwendung (meist Strg+Alt+T)

---

## Installation

### Schritt 1: Projekt herunterladen

Laden Sie dieses Projekt auf Ihren Computer herunter:

```bash
# Wenn Sie Git installiert haben:
git clone https://github.com/IHR-BENUTZERNAME/Ai_Ml.git
cd Ai_Ml

# ODER: Laden Sie die ZIP-Datei herunter und entpacken Sie sie,
# dann oeffnen Sie die Kommandozeile in diesem Ordner
```

### Schritt 2: Virtuelle Umgebung erstellen

Eine "virtuelle Umgebung" ist wie ein separater Raum fuer dieses Projekt, damit es nicht mit anderen Python-Programmen auf Ihrem Computer kollidiert.

```bash
# Virtuelle Umgebung erstellen (nur einmal noetig):
python -m venv .venv
```

**Was passiert hier?**
- Python erstellt einen Ordner namens `.venv`
- Darin werden alle Programmbibliotheken fuer dieses Projekt gespeichert

### Schritt 3: Virtuelle Umgebung aktivieren

Jedes Mal wenn Sie mit dem Projekt arbeiten, muessen Sie die Umgebung aktivieren:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Erfolgreich?**
- Sie sehen jetzt `(.venv)` am Anfang Ihrer Kommandozeile
- Das bedeutet: Die virtuelle Umgebung ist aktiv

### Schritt 4: Programm installieren

Jetzt installieren wir das Programm und alle benoetigten Bibliotheken:

```bash
pip install -e .
```

**Was passiert hier?**
- `pip` ist der Python-Paketmanager (laedt Bibliotheken herunter)
- `-e` bedeutet "editable" - Sie koennen den Code bearbeiten und die Aenderungen gelten sofort
- `.` bedeutet "installiere das Projekt im aktuellen Ordner"

**Installierte Bibliotheken:**
- `torch` - Die KI-Bibliothek (PyTorch)
- `numpy` - Mathematik und Arrays
- `h5py` - Speichert grosse Datenmengen effizient
- `scikit-image` - Bildverarbeitung
- `scipy` - Wissenschaftliche Berechnungen
- `matplotlib` - Erstellt Grafiken

**Hinweis:** Die Installation kann 5-10 Minuten dauern, besonders PyTorch ist gross (mehrere GB).

### Schritt 5: Installation ueberpruefen

Testen Sie, ob alles funktioniert:

```bash
python -c "import ebsd_ai; print('Installation erfolgreich!')"
```

**Erwartete Ausgabe:**
```
Installation erfolgreich!
```

**Wenn Sie eine Fehlermeldung sehen:**
- Siehe Abschnitt "Fehlerbehebung" unten

---

## Erste Schritte: Einfache Beispiele

### Beispiel 1: Trainingsdaten speichern

Dieses Programm braucht Trainingsdaten, um zu lernen. So speichern Sie EBSD-Muster:

```python
# Datei: mein_erstes_beispiel.py
import numpy as np
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.config import DetectorInfo, DataSource

# 1. Erstellen Sie einen Datenspeicher
store = TrainingStore(local_path="./meine_daten", samples_per_shard=1000)

# 2. Erstellen Sie ein Beispiel-EBSD-Muster (normalerweise laden Sie echte Daten)
beispiel_muster = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# 3. Beispiel-EDS-Daten (chemische Zusammensetzung)
eds_daten = {
    "Fe": 70.0,  # 70% Eisen
    "C": 2.0,    # 2% Kohlenstoff
    "Cr": 18.0,  # 18% Chrom
    "Ni": 10.0,  # 10% Nickel
}

# 4. Informationen ueber den Detektor
detektor = DetectorInfo(
    manufacturer="Oxford",
    convention="TSL",
    pixel_count=307200,
)

# 5. Speichern Sie die Probe
store.add_sample(
    pattern=beispiel_muster,
    eds_dict=eds_daten,
    detector=detektor,
    confirmed_phase="Austenit",  # Die bekannte Phase
    orientation=(1.0, 0.0, 0.0, 0.0),  # Quaternion
    confidence=0.85,  # Wie sicher ist die Beschriftung? (0-1)
    source=DataSource.MANUAL_INDEXING,
    source_file="beispiel_scan_01.h5",
)

print(f"Gespeichert! Statistiken: {store.get_dataset_stats()}")
```

**Was passiert hier?**
1. Wir erstellen einen Ordner `./meine_daten` fuer die Trainingsdaten
2. Ein EBSD-Muster wird gespeichert (hier zufaellig generiert - normalerweise laden Sie echte Daten)
3. EDS-Daten geben die chemische Zusammensetzung an (welche Elemente, wie viel Prozent)
4. Wir sagen dem Programm, dass dies "Austenit" ist (zum Trainieren)
5. Alles wird komprimiert in einer HDF5-Datei gespeichert

**Fuehren Sie es aus:**
```bash
python mein_erstes_beispiel.py
```

**Erwartete Ausgabe:**
```
Gespeichert! Statistiken: {'total_samples': 1, 'phases': ['Austenit'], 'samples_with_eds': 1}
```

### Beispiel 2: Ein KI-Modell erstellen

So erstellen Sie ein Modell, das Phasen vorhersagen kann:

```python
# Datei: modell_erstellen.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig

# 1. Konfiguration: Welche Phasen soll das Modell unterscheiden koennen?
phasen = ["Ferrit", "Austenit", "Martensit", "Perlit"]

config = ModelConfig(
    n_phases=len(phasen),        # Anzahl der Phasen
    pattern_size=128,             # Bildgroesse (128x128 Pixel)
    pattern_feature_dim=256,      # Wie viele Merkmale aus dem Bild extrahieren
    eds_feature_dim=64,           # Wie viele Merkmale aus EDS-Daten
    fused_feature_dim=128,        # Kombinierte Merkmale
    dropout=0.2,                  # Verhindert Ueberanpassung (20%)
)

# 2. Modell erstellen
modell = PhaseClassifier(config=config, phase_names=phasen)

# 3. Zeige Modellstruktur
print(f"Modell erstellt mit {sum(p.numel() for p in modell.parameters())} Parametern")
print(f"Phasen: {modell.phase_names}")

# 4. Modell speichern
torch.save(modell.get_save_dict(), "mein_modell.pt")
print("Modell gespeichert in: mein_modell.pt")
```

**Was passiert hier?**
1. Wir definieren 4 Phasen, die das Modell unterscheiden soll
2. `ModelConfig` legt die Architektur fest (wie gross das neuronale Netz ist)
3. `PhaseClassifier` erstellt das KI-Modell
4. Das Modell wird als Datei gespeichert (zum spaeteren Laden)

**Fuehren Sie es aus:**
```bash
python modell_erstellen.py
```

**Erwartete Ausgabe:**
```
Modell erstellt mit 2847940 Parametern
Phasen: ['Ferrit', 'Austenit', 'Martensit', 'Perlit']
Modell gespeichert in: mein_modell.pt
```

**Was sind Parameter?**
- Das sind die "Gewichte" im neuronalen Netz, die beim Training angepasst werden
- Etwa 2.8 Millionen Parameter - das Modell lernt aus Daten, welche Werte diese haben sollen

### Beispiel 3: Modell trainieren

Die Trainings-Pipeline ist jetzt vollstaendig implementiert. So trainieren Sie ein Modell auf Ihren Daten:

```python
# Datei: modell_trainieren.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

# 1. Daten laden
#    (Stellen Sie sicher, dass Sie vorher Daten gespeichert haben - siehe Beispiel 1)
store = TrainingStore(local_path="./meine_daten")
dataset = EBSDPhaseDataset(store)

# 2. Modell erstellen
phasen = dataset.phase_names  # Phasen werden automatisch aus den Daten erkannt
config = ModelConfig(n_phases=len(phasen))
modell = PhaseClassifier(config=config, phase_names=phasen)

# 3. Training konfigurieren
training_config = TrainingConfig(
    learning_rate=0.001,         # Lernrate (wie schnell das Modell lernt)
    batch_size=64,               # Wie viele Muster gleichzeitig verarbeiten
    epochs=50,                   # Wie oft alle Daten durchlaufen werden
    early_stopping_patience=10,  # Stopp wenn 10 Epochen keine Verbesserung
    mixed_precision=True,        # Schnelleres Training auf GPU (automatisch aus auf CPU)
    val_fraction=0.2,            # 20% der Daten fuer Validierung verwenden
)

# 4. Trainer erstellen und Training starten
trainer = Trainer(
    model=modell,
    config=training_config,
    output_dir="checkpoints",    # Hier werden Modell-Zwischenstaende gespeichert
    device="auto",               # Verwendet GPU wenn vorhanden, sonst CPU
)

# 5. Training starten
ergebnis = trainer.train(dataset)

# 6. Ergebnisse anzeigen
print(f"Training abgeschlossen nach {ergebnis.epochs_completed} Epochen")
print(f"Beste Epoche: {ergebnis.best_epoch + 1}")
print(f"Bester Validierungs-Verlust: {ergebnis.best_val_loss:.4f}")
print(f"Fruehzeitig gestoppt: {'Ja' if ergebnis.early_stopped else 'Nein'}")
print(f"Dauer: {ergebnis.elapsed_seconds:.1f} Sekunden")
```

**Was passiert hier?**
1. Wir laden die gespeicherten Trainingsdaten aus dem TrainingStore
2. Ein neues KI-Modell wird erstellt - die Phasen werden automatisch erkannt
3. `TrainingConfig` legt fest, wie das Training ablaufen soll:
   - **Lernrate**: Wie stark das Modell bei jedem Schritt angepasst wird
   - **Batch-Groesse**: Wie viele Muster gleichzeitig verarbeitet werden
   - **Epochen**: Wie oft der gesamte Datensatz durchlaufen wird
   - **Early Stopping**: Stoppt automatisch, wenn das Modell aufhoert, besser zu werden
4. Der `Trainer` uebernimmt das Training vollautomatisch:
   - Er teilt die Daten in Training (80%) und Validierung (20%)
   - Er speichert das beste Modell automatisch als `checkpoints/best.pt`
   - Er speichert den letzten Stand als `checkpoints/last.pt`
   - Bei GPU wird automatisch "Mixed Precision" verwendet (schneller)
5. Am Ende erhalten Sie ein `TrainingResult` mit allen Statistiken

**Fuehren Sie es aus:**
```bash
python modell_trainieren.py
```

**Erwartete Ausgabe (Beispiel):**
```
Training abgeschlossen nach 35 Epochen
Beste Epoche: 25
Bester Validierungs-Verlust: 0.3421
Fruehzeitig gestoppt: Ja
Dauer: 142.3 Sekunden
```

**Wichtiger Hinweis:** Sie brauchen genuegend Trainingsdaten (mindestens einige hundert Muster pro Phase), damit das Modell sinnvoll lernen kann.

### Beispiel 4: Training fortsetzen

Wenn das Training unterbrochen wurde oder Sie laenger trainieren moechten:

```python
# Datei: training_fortsetzen.py
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

# 1. Daten laden
store = TrainingStore(local_path="./meine_daten")
dataset = EBSDPhaseDataset(store)

# 2. Modell aus letztem Checkpoint laden
modell = PhaseClassifier.from_save_dict(
    __import__("torch").load("checkpoints/last.pt", weights_only=False)["model"]
)

# 3. Trainer erstellen
training_config = TrainingConfig(epochs=100)  # Jetzt 100 Epochen insgesamt
trainer = Trainer(model=modell, config=training_config, output_dir="checkpoints")

# 4. Vom letzten Checkpoint fortsetzen
trainer.resume("checkpoints/last.pt")

# 5. Weiter trainieren
ergebnis = trainer.train(dataset)
print(f"Fortgesetzt und {ergebnis.epochs_completed} weitere Epochen trainiert")
```

**Was passiert hier?**
1. Wir laden das Modell und den Trainingsstand aus dem letzten Checkpoint
2. `resume()` stellt den Zustand des Optimierers wieder her (Lernrate, Momentum, etc.)
3. Das Training geht dort weiter, wo es aufgehoert hat

### Beispiel 5: Fortschritt beobachten

Sie koennen eine Callback-Funktion uebergeben, um den Trainingsfortschritt live zu sehen:

```python
# Datei: training_mit_fortschritt.py
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

def fortschritt_anzeigen(epoche, gesamt, train_loss, val_loss):
    """Wird nach jeder Epoche aufgerufen."""
    balken = "#" * int(20 * (epoche + 1) / gesamt)
    leer = "-" * (20 - len(balken))
    print(
        f"[{balken}{leer}] Epoche {epoche+1}/{gesamt} | "
        f"Training: {train_loss:.4f} | Validierung: {val_loss:.4f}"
    )

# Daten und Modell vorbereiten (wie in Beispiel 3)
store = TrainingStore(local_path="./meine_daten")
dataset = EBSDPhaseDataset(store)
phasen = dataset.phase_names
modell = PhaseClassifier(
    config=ModelConfig(n_phases=len(phasen)), phase_names=phasen
)

# Trainer mit Callback erstellen
trainer = Trainer(
    model=modell,
    config=TrainingConfig(epochs=20),
    output_dir="checkpoints",
    progress_callback=fortschritt_anzeigen,  # <-- Hier uebergeben
)

ergebnis = trainer.train(dataset)
```

**Erwartete Ausgabe (Beispiel):**
```
[#-------------------] Epoche 1/20  | Training: 1.3862 | Validierung: 1.3845
[##------------------] Epoche 2/20  | Training: 1.2501 | Validierung: 1.2130
[###-----------------] Epoche 3/20  | Training: 1.0843 | Validierung: 1.0612
...
[####################] Epoche 20/20 | Training: 0.2145 | Validierung: 0.3102
```

### Beispiel 6: Modell evaluieren (Qualitaet pruefen)

Nach dem Training sollten Sie pruefen, wie gut Ihr Modell wirklich ist. Der Evaluator berechnet verschiedene Qualitaetsmetriken:

```python
# Datei: modell_evaluieren.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_test_split
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.evaluator import Evaluator, format_classification_report

# 1. Daten laden und in Train/Val/Test aufteilen
store = TrainingStore(local_path="./meine_daten")
dataset = EBSDPhaseDataset(store, eds_dropout_rate=0.0)

# Aufteilen: 70% Training, 15% Validierung, 15% Test
train_ds, val_ds, test_ds = train_val_test_split(
    dataset, val_fraction=0.15, test_fraction=0.15
)

# 2. Trainiertes Modell laden
save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
modell = PhaseClassifier.from_save_dict(save_dict)

# 3. Evaluator erstellen
evaluator = Evaluator(
    model=modell,
    device="auto",       # GPU wenn vorhanden, sonst CPU
    k=3,                 # Top-3 Genauigkeit berechnen
    n_calibration_bins=10,  # Kalibrierungsanalyse mit 10 Intervallen
    batch_size=64,
)

# 4. Auf Testdaten evaluieren
ergebnis = evaluator.evaluate(test_ds)

# 5. Bericht anzeigen
print(format_classification_report(ergebnis))

# 6. Einzelne Metriken abfragen
print(f"\nTop-1 Genauigkeit: {ergebnis.top1_accuracy * 100:.1f}%")
print(f"Top-3 Genauigkeit: {ergebnis.topk_accuracy * 100:.1f}%")
print(f"Mittlere Konfidenz: {ergebnis.mean_confidence * 100:.1f}%")
print(f"Attention Alpha:    {ergebnis.mean_attention_alpha:.2f}")
print(f"Anzahl Testproben:  {ergebnis.n_samples}")

# 7. Confusion Matrix anzeigen (welche Phasen werden verwechselt?)
print("\nConfusion Matrix (Zeilen = wahre Phase, Spalten = vorhergesagte Phase):")
print(f"{'':>15}", end="")
for name in ergebnis.phase_names:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(ergebnis.phase_names):
    print(f"{name:>15}", end="")
    for j in range(len(ergebnis.phase_names)):
        print(f"{ergebnis.confusion_matrix[i, j]:>12d}", end="")
    print()
```

**Was passiert hier?**
1. Die Daten werden in drei Teile aufgeteilt: Training, Validierung und Test
2. Das trainierte Modell wird aus dem besten Checkpoint geladen
3. Der `Evaluator` berechnet automatisch alle wichtigen Qualitaetsmetriken
4. `format_classification_report()` erstellt einen uebersichtlichen Textbericht

**Fuehren Sie es aus:**
```bash
python modell_evaluieren.py
```

**Erwartete Ausgabe (Beispiel):**
```
Phase                Precision     Recall         F1    Support
--------------------------------------------------------------
Ferrit                  0.8500     0.8947     0.8718         19
Austenit                0.9091     0.8333     0.8696         12
Martensit               0.7500     0.8571     0.8000         14
--------------------------------------------------------------
Top-1 Accuracy:  0.8444
Top-3 Accuracy:  0.9778
Mean Confidence: 0.7234
Mean Alpha:      0.6812
Samples:         45

Top-1 Genauigkeit: 84.4%
Top-3 Genauigkeit: 97.8%
Mittlere Konfidenz: 72.3%
Attention Alpha:    0.68
Anzahl Testproben:  45

Confusion Matrix (Zeilen = wahre Phase, Spalten = vorhergesagte Phase):
                      Ferrit     Austenit    Martensit
        Ferrit            17            1            1
       Austenit            1           10            1
      Martensit            2            0           12
```

**Die Metriken erklaert:**
- **Top-1 Genauigkeit**: Wie oft die beste Vorhersage korrekt ist (z.B. 84.4% = bei 84 von 100 Proben richtig)
- **Top-3 Genauigkeit**: Wie oft die richtige Phase unter den 3 wahrscheinlichsten ist (wichtig fuer den Vorfilter-Einsatz)
- **Precision** (Praezision): Wenn das Modell "Ferrit" sagt, wie oft stimmt das?
- **Recall** (Trefferquote): Von allen echten Ferrit-Proben, wie viele hat das Modell gefunden?
- **F1-Score**: Kombiniert Precision und Recall zu einer Zahl (1.0 = perfekt, 0.0 = schlecht)
- **Confusion Matrix**: Zeigt genau, welche Phasen miteinander verwechselt werden
- **Kalibrierung**: Prueft, ob ein Modell, das "80% sicher" sagt, auch in 80% der Faelle richtig liegt
- **Attention Alpha**: Wie stark das Modell dem EBSD-Muster vertraut (1.0 = nur Muster, 0.0 = nur EDS)

### Beispiel 7: Eine Vorhersage machen (PhasePredictor)

Der `PhasePredictor` ist die empfohlene Schnittstelle fuer Vorhersagen. Er uebernimmt automatisch das Laden des Modells, die Vorverarbeitung der Daten und die Umwandlung in das richtige Format.

```python
# Datei: vorhersage.py
import numpy as np
from ebsd_ai.inference.predictor import PhasePredictor

# 1. Predictor aus einem Checkpoint laden (bestes Modell aus dem Training)
predictor = PhasePredictor("checkpoints/best.pt", device="auto")

# 2. Pruefen, ob das Modell geladen wurde
print(f"Modell geladen: {predictor.has_model}")
print(f"Phasen: {predictor.phase_names}")

# 3. Beispiel-EBSD-Muster (beliebige Groesse - wird automatisch skaliert)
muster = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# 4. EDS-Daten (chemische Zusammensetzung) - ganz einfach als Dictionary
eds_daten = {
    "Fe": 70.0,  # 70% Eisen
    "C": 2.0,    # 2% Kohlenstoff
    "Cr": 18.0,  # 18% Chrom
    "Ni": 10.0,  # 10% Nickel
}

# 5. Vorhersage durchfuehren - eine einzige Zeile!
ergebnis = predictor.predict_phase(
    pattern=muster,
    eds_data=eds_daten,
    k=3,  # Top-3 Vorhersagen
)

# 6. Ergebnisse anzeigen
print("\nTop-3 Vorhersagen:")
for phase, wahrscheinlichkeit in ergebnis.top_k:
    print(f"  {phase}: {wahrscheinlichkeit*100:.1f}%")

print(f"\nKonfidenz: {ergebnis.confidence*100:.1f}%")
print(f"EDS-Beitrag: {ergebnis.eds_contribution*100:.1f}%")
```

**Was passiert hier?**
1. `PhasePredictor` laedt das Modell aus dem Checkpoint -- Sie muessen sich nicht um `torch.load`, `eval()` oder Tensoren kuemmern
2. Ihr EBSD-Muster kann jede Groesse haben -- es wird automatisch auf 128x128 skaliert und normalisiert
3. EDS-Daten geben Sie einfach als Dictionary an (Element-Name und Prozent)
4. `predict_phase()` erledigt alles: Vorverarbeitung, Tensorumwandlung, Vorhersage
5. Das Ergebnis ist ein `PhasePrediction`-Objekt mit den Top-k Phasen, Konfidenz und EDS-Beitrag

**Fuehren Sie es aus:**
```bash
python vorhersage.py
```

**Erwartete Ausgabe (Beispiel):**
```
Modell geladen: True
Phasen: ['Ferrit', 'Austenit', 'Martensit', 'Perlit']

Top-3 Vorhersagen:
  Ferrit: 72.3%
  Austenit: 18.5%
  Martensit: 6.1%

Konfidenz: 72.3%
EDS-Beitrag: 27.0%
```

**Interpretation:**
- Das Modell denkt zu 72.3%, dass es Ferrit ist
- 27% des Ergebnisses basieren auf den EDS-Daten, 73% auf dem EBSD-Muster

**Gut zu wissen:** Der PhasePredictor funktioniert auch, wenn nur ein Teil der Daten vorhanden ist:
```python
# Nur EBSD-Muster (ohne EDS-Daten)
ergebnis = predictor.predict_phase(pattern=muster)

# Nur EDS-Daten (ohne EBSD-Muster)
ergebnis = predictor.predict_phase(eds_data=eds_daten)
```

### Beispiel 8: Einen ganzen EBSD-Scan vorhersagen (Batch)

Wenn Sie einen kompletten EBSD-Scan haben (viele Messpunkte auf einer Flaeche), koennen Sie alle Punkte auf einmal vorhersagen:

```python
# Datei: scan_vorhersage.py
import numpy as np
from ebsd_ai.inference.predictor import PhasePredictor

# 1. Predictor laden
predictor = PhasePredictor("checkpoints/best.pt", device="auto")

# 2. EBSD-Scan laden (4D-Array: Zeilen x Spalten x Hoehe x Breite)
#    Beispiel: 100x200 Messpunkte, jedes Muster 480x640 Pixel
n_zeilen, n_spalten = 100, 200
scan = np.random.randint(0, 255, (n_zeilen, n_spalten, 480, 640), dtype=np.uint8)

# 3. EDS-Maps (optional): Element -> flaches Array mit einem Wert pro Messpunkt
eds_maps = {
    "Fe": np.random.uniform(60, 80, n_zeilen * n_spalten),
    "Cr": np.random.uniform(10, 20, n_zeilen * n_spalten),
    "Ni": np.random.uniform(5, 15, n_zeilen * n_spalten),
}

# 4. Optional: Nur bestimmte Pixel vorhersagen (z.B. mit genuegend Konfidenz)
maske = np.ones((n_zeilen, n_spalten), dtype=bool)  # Alle Pixel
# maske[0:10, :] = False  # Die ersten 10 Zeilen ueberspringen

# 5. Batch-Vorhersage starten
scan_ergebnis = predictor.predict_scan(
    patterns_4d=scan,
    eds_maps=eds_maps,
    selection_mask=maske,
    k=3,
)

# 6. Ergebnisse anzeigen
print(f"Scan-Groesse: {scan_ergebnis.scan_shape}")
print(f"Vorhergesagte Pixel: {scan_ergebnis.n_pixels}")
print(f"Phasen: {scan_ergebnis.phase_names}")
print(f"Wahrscheinlichkeiten-Shape: {scan_ergebnis.probabilities.shape}")
print(f"Konfidenz-Shape: {scan_ergebnis.confidence.shape}")
print(f"EDS-Beitrag-Shape: {scan_ergebnis.eds_contribution.shape}")
```

**Was passiert hier?**
1. Der Scan ist ein 4D-Array: (Zeilen, Spalten, Muster-Hoehe, Muster-Breite)
2. EDS-Maps geben die chemische Zusammensetzung an jedem Messpunkt an
3. Die `selection_mask` bestimmt, welche Pixel vorhergesagt werden sollen (spart Rechenzeit)
4. `predict_scan()` verarbeitet alle Messpunkte effizient in Batches
5. Das Ergebnis enthaelt Wahrscheinlichkeiten, Konfidenz und EDS-Beitrag fuer jeden Messpunkt

**Fuehren Sie es aus:**
```bash
python scan_vorhersage.py
```

**Erwartete Ausgabe (Beispiel):**
```
Scan-Groesse: (100, 200)
Vorhergesagte Pixel: 20000
Phasen: ['Ferrit', 'Austenit', 'Martensit', 'Perlit']
Wahrscheinlichkeiten-Shape: (20000, 4)
Konfidenz-Shape: (20000,)
EDS-Beitrag-Shape: (20000,)
```

### Beispiel 9: Modell kontinuierlich verbessern (Online Learning)

Wenn Sie regelmaessig neue Indexierungsergebnisse erhalten, kann das Modell sich automatisch verbessern, ohne komplett neu trainiert zu werden:

```python
# Datei: online_lernen.py
import torch
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.online_learner import OnlineLearner, OnlineLearnerConfig

# 1. Bestehendes Modell und Trainingsdaten laden
store = TrainingStore(local_path="./meine_daten")
save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
modell = PhaseClassifier.from_save_dict(save_dict)

# 2. Online Learner konfigurieren
config = OnlineLearnerConfig(
    learning_rate=0.0001,     # Kleine Lernrate (vorsichtiges Update)
    steps_per_update=50,      # 50 Gradient-Schritte pro Update
    batch_size=32,            # Mini-Batch-Groesse
    replay_fraction=0.5,      # 50% alte Daten mischen (gegen Vergessen)
    ewc_lambda=0.5,           # EWC-Staerke (schuetzt wichtige Gewichte)
)

# 3. Learner erstellen
learner = OnlineLearner(
    model=modell,
    store=store,
    config=config,
    device="auto",
    checkpoint_dir="checkpoints",  # Speichert nach jedem Update
)

# 4. Update ausfuehren (z.B. nach neuen Indexierungsergebnissen)
ergebnis = learner.update()

print(f"Schritte: {ergebnis.steps_completed}")
print(f"Mittlerer Verlust: {ergebnis.mean_loss:.4f}")
print(f"Neue Phasen entdeckt: {ergebnis.new_phases_added}")
print(f"Verarbeitete Proben: {ergebnis.samples_used}")
```

**Was passiert hier?**
1. Das bestehende trainierte Modell wird geladen
2. Der `OnlineLearner` fuehrt ein vorsichtiges Update durch:
   - **Experience Replay**: Mischt neue und alte Daten, damit das Modell alte Phasen nicht vergisst
   - **EWC (Elastic Weight Consolidation)**: Schuetzt wichtige Modell-Gewichte vor zu grossen Aenderungen
   - **Phasenerweiterung**: Wenn neue Phasen in den Daten auftauchen, wird das Modell automatisch erweitert
3. Ein Checkpoint wird gespeichert (`checkpoints/online_latest.pt`)

**Typischer Einsatz:** Nach jedem EBSD-Scan mit hoher Konfidenz (CI > 0.3) speichern Sie die Ergebnisse im TrainingStore und rufen dann `learner.update()` auf. So wird das Modell mit jeder Messung besser.

### Beispiel 10: EBSD-Muster verbessern (Pattern Enhancement)

Verrauschte EBSD-Muster koennen mit dem PatternEnhancer verbessert werden, bevor sie klassifiziert oder indexiert werden:

```python
# Datei: muster_verbessern.py
import numpy as np
import torch
from ebsd_ai.models.pattern_enhancer import PatternEnhancer, EnhancerConfig
from ebsd_ai.config import DetectorInfo, DetectorConvention, DetectorManufacturer

# 1. Enhancer erstellen (oder aus Checkpoint laden)
config = EnhancerConfig(
    base_channels=32,   # Kanalanzahl im ersten Block (32 -> 64 -> 128 -> 256)
    depth=4,             # 4 Encoder-/Decoder-Stufen
    use_film=True,       # Detektor-Metadaten fuer Konditionierung nutzen
)
enhancer = PatternEnhancer(config=config)

# 2. Ein einzelnes Muster verbessern (beliebige Groesse)
verrauschtes_muster = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# Ohne Detektor-Informationen:
verbessertes = enhancer.enhance(verrauschtes_muster)
print(f"Ausgabe-Shape: {verbessertes.shape}")  # (128, 128)

# Mit Detektor-Informationen (fuer bessere Ergebnisse):
detektor = DetectorInfo(
    manufacturer=DetectorManufacturer.OXFORD,
    pc=(0.5, 0.2, 0.6),
    pc_convention=DetectorConvention.OXFORD,
    kv=20.0,
    working_distance=15.0,
    sample_tilt=70.0,
)
verbessertes = enhancer.enhance(verrauschtes_muster, detector_info=detektor)

# 3. Einen ganzen Scan verbessern
n_zeilen, n_spalten = 10, 20
scan = np.random.randint(0, 255, (n_zeilen, n_spalten, 120, 160), dtype=np.uint8)

verbesserter_scan = enhancer.enhance_scan(
    patterns_4d=scan,
    detector_info=detektor,
    batch_size=32,  # Wie viele Muster gleichzeitig verarbeiten
)
print(f"Scan-Shape: {verbesserter_scan.shape}")  # (10, 20, 128, 128)

# 4. Optional: Nur bestimmte Bereiche verbessern
maske = np.zeros((n_zeilen, n_spalten), dtype=bool)
maske[2:8, 5:15] = True  # Nur einen Ausschnitt verbessern
teil_scan = enhancer.enhance_scan(scan, detector_info=detektor, selection_mask=maske)

# 5. Modell speichern/laden
torch.save(enhancer.get_save_dict(), "enhancer.pt")

# Spaeter laden:
save_dict = torch.load("enhancer.pt", weights_only=False)
geladener_enhancer = PatternEnhancer.from_save_dict(save_dict)
```

**Was passiert hier?**
1. Der `PatternEnhancer` ist ein U-Net (Encoder-Decoder Netzwerk mit Skip-Verbindungen)
2. Er nimmt ein verrauschtes EBSD-Muster und gibt ein verbessertes 128x128 Muster zurueck
3. Optional kann er mit FiLM-Schichten auf Detektor-Metadaten konditioniert werden (kV, PC, Detektortyp)
4. `enhance_scan()` verarbeitet ganze Scans effizient in Batches
5. Die verbesserten Muster koennen dann fuer bessere Dictionary/Spherical Indexierung verwendet werden

**Wichtig:** Der Enhancer muss zuerst auf gepaarten Daten (verrauscht → sauber) trainiert werden, bevor er sinnvolle Ergebnisse liefert.

### Beispiel 11: Trainingsdaten synchronisieren (Server Sync)

Wenn Sie im Team arbeiten, koennen Sie Trainingsdaten ueber ein gemeinsames Netzlaufwerk teilen:

```python
# Datei: daten_synchronisieren.py
from ebsd_ai.sync.server_sync import ServerSync

# 1. ServerSync erstellen (lokaler Ordner + gemeinsamer Server-Pfad)
sync = ServerSync(
    local_dir="./meine_daten",          # Ihr lokaler TrainingStore-Ordner
    server_dir="//server/share/ebsd_ml/", # Gemeinsamer Netzlaufwerk-Pfad
)

# 2. Eigene Daten zum Server hochladen (Push)
push_ergebnis = sync.push()
print(f"Hochgeladen: {push_ergebnis.n_copied} Dateien")
print(f"Uebersprungen: {push_ergebnis.n_skipped} (bereits vorhanden)")

# 3. Daten anderer Nutzer herunterladen (Pull)
pull_ergebnis = sync.pull()
print(f"Heruntergeladen: {pull_ergebnis.n_copied} Dateien")

# 4. Oder beides auf einmal (bidirektional)
push_res, pull_res = sync.full_sync()

# 5. Statistiken anzeigen
print(f"Lokale Shards: {sync.local_stats()['n_shards']}")
print(f"Server Shards: {sync.server_stats()['n_shards']}")

# 6. Mit Fortschrittsanzeige
def fortschritt(aktuell, gesamt, dateiname):
    print(f"  [{aktuell}/{gesamt}] {dateiname}")

sync.push(progress_callback=fortschritt)
```

**Was passiert hier?**
1. `ServerSync` verbindet einen lokalen TrainingStore-Ordner mit einem Netzlaufwerk
2. `push()` kopiert neue lokale Shard-Dateien zum Server (bereits vorhandene werden uebersprungen)
3. `pull()` kopiert neue Server-Dateien in den lokalen Ordner
4. Die Kopien sind **atomar**: Eine Datei wird erst unter einem temporaeren Namen geschrieben und dann umbenannt, sodass ein Abbruch nie zu einer beschaedigten Datei fuehrt
5. Jede Datei wird unabhaengig kopiert — wenn eine fehlschlaegt, werden die anderen trotzdem uebertragen

**Fuehren Sie es aus:**
```bash
python daten_synchronisieren.py
```

**Erwartete Ausgabe (Beispiel):**
```
Hochgeladen: 2 Dateien
Uebersprungen: 1 (bereits vorhanden)
Heruntergeladen: 3 Dateien
Lokale Shards: 6
Server Shards: 6
```

### Beispiel 12: Training ueber die Kommandozeile (CLI)

Statt Python-Code zu schreiben, koennen Sie das Training direkt ueber die Kommandozeile starten:

```bash
# Einfaches Training (Standardeinstellungen)
.venv/bin/python scripts/train.py --data ./meine_daten

# Mit angepassten Einstellungen
.venv/bin/python scripts/train.py \
    --data ./meine_daten \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0005 \
    --patience 15 \
    --output ./mein_modell

# Training fortsetzen (von letztem Checkpoint)
.venv/bin/python scripts/train.py \
    --data ./meine_daten \
    --resume ./mein_modell/last.pt
```

**Verfuegbare Optionen:**

| Option | Standard | Beschreibung |
|--------|----------|-------------|
| `--data` | (erforderlich) | Pfad zum TrainingStore-Ordner |
| `--epochs` | 50 | Maximale Anzahl Trainings-Epochen |
| `--batch-size` | 64 | Mini-Batch-Groesse |
| `--lr` | 0.001 | Lernrate |
| `--patience` | 10 | Early-Stopping-Geduld (Epochen) |
| `--val-fraction` | 0.15 | Anteil Validierungsdaten |
| `--eds-dropout` | 0.3 | EDS-Dropout-Rate |
| `--output` | checkpoints | Ausgabe-Ordner fuer Checkpoints |
| `--device` | auto | Geraet: auto, cpu oder cuda |
| `--resume` | — | Checkpoint zum Fortsetzen |
| `--top-k` | 3 | Top-k fuer Evaluierung |

**Was passiert?**
1. Das Skript laedt die Trainingsdaten und zeigt Statistiken an
2. Ein Modell wird erstellt (oder aus Checkpoint geladen)
3. Das Training laeuft mit Fortschrittsanzeige
4. Am Ende wird das Modell auf den Validierungsdaten evaluiert
5. Die Ergebnisse (Precision, Recall, F1, Confusion Matrix) werden angezeigt
6. Checkpoints werden als `best.pt` und `last.pt` gespeichert

**Erwartete Ausgabe (Beispiel):**
```
Loaded TrainingStore: 500 samples from ./meine_daten
  Phases: {'Ferrit': 200, 'Austenit': 180, 'Martensit': 120}
  Samples with EDS: 300
  Samples without EDS: 200

Created model: 3 phases, 3,010,340 parameters

Training (max 50 epochs, patience 10)...
  Device: cuda
  Output: checkpoints

  [##------------------] Epoch   5/50 | train_loss=0.82 | val_loss=0.91
  ...
  [####################] Epoch  35/50 | train_loss=0.21 | val_loss=0.34

Training complete:
  Epochs completed: 35
  Best epoch:       25
  Best val loss:    0.3102
  Early stopped:    Yes
  Duration:         142.3s

Phase                 Precision     Recall         F1    Support
--------------------------------------------------------------
Ferrit                   0.8500     0.8947     0.8718         19
Austenit                 0.9091     0.8333     0.8696         12
Martensit                0.7500     0.8571     0.8000         14
--------------------------------------------------------------
Top-1 Accuracy:  0.8444
Top-3 Accuracy:  0.9778
```

### Beispiel 13: Modell evaluieren ueber die Kommandozeile

So bewerten Sie ein trainiertes Modell direkt ueber die Kommandozeile:

```bash
# Modell auf allen Daten evaluieren
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./meine_daten

# Nur auf 20% der Daten evaluieren (Train/Test-Split)
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./meine_daten \
    --split 0.2

# Mit Top-5 Genauigkeit und 20 Kalibrierungs-Intervallen
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./meine_daten \
    --top-k 5 \
    --calibration-bins 20
```

**Verfuegbare Optionen:**

| Option | Standard | Beschreibung |
|--------|----------|-------------|
| `--model` | (erforderlich) | Pfad zum Modell-Checkpoint |
| `--data` | (erforderlich) | Pfad zum TrainingStore-Ordner |
| `--top-k` | 3 | Top-k fuer Genauigkeitsberechnung |
| `--batch-size` | 64 | Batch-Groesse |
| `--device` | auto | Geraet: auto, cpu oder cuda |
| `--split` | 0 | Anteil fuer Evaluation (0 = alle Daten) |
| `--calibration-bins` | 10 | Anzahl Kalibrierungs-Intervalle |

**Was wird angezeigt?**
- Klassifikationsbericht mit Precision, Recall, F1 pro Phase
- Absolute und normalisierte Confusion Matrix
- Kalibrierungsanalyse (Konfidenz vs. tatsaechliche Genauigkeit)

### Beispiel 14: EBSD-Scandateien importieren (CLI)

Statt Trainingsdaten manuell im Python-Code zu speichern, koennen Sie EBSD-Scandateien direkt ueber die Kommandozeile in den TrainingStore importieren:

```bash
# Eine einzelne Datei importieren
.venv/bin/python scripts/import_data.py scan.h5oina --store ./meine_daten

# Mehrere Dateien auf einmal
.venv/bin/python scripts/import_data.py *.ctf --store ./meine_daten

# Mit angepasstem CI-Schwellenwert (Standard: 0.3)
.venv/bin/python scripts/import_data.py scan.ang scan.ctf \
    --store ./meine_daten --ci 0.4

# Vorab pruefen, was importiert wuerde (ohne Daten zu schreiben)
.venv/bin/python scripts/import_data.py scan.h5oina \
    --store ./meine_daten --dry-run

# Ausfuehrliche Ausgabe mit Phasen-Details
.venv/bin/python scripts/import_data.py scan.h5oina \
    --store ./meine_daten --ci 0.0 --verbose
```

**Verfuegbare Optionen:**

| Option | Standard | Beschreibung |
|--------|----------|-------------|
| `files` | (erforderlich) | Eine oder mehrere EBSD-Scandateien (.ang, .ctf, .h5, .h5oina) |
| `--store` | (erforderlich) | Pfad zum TrainingStore-Ordner |
| `--ci` | 0.3 | Minimaler Confidence Index fuer Import |
| `--target-size` | 128 | Normalisierte Mustergroesse in Pixeln |
| `--dry-run` | — | Nur Dateien analysieren, nichts speichern |
| `--verbose` / `-v` | — | Detaillierte Ausgabe mit Warnungen und Phasen |

**Unterstuetzte Dateiformate:**
- **.ang** — EDAX/OIM (nur Indexierungsergebnisse, keine Muster → werden uebersprungen)
- **.ctf** — Oxford/HKL Channel Text File (nur Indexierungsergebnisse, keine Muster → werden uebersprungen)
- **.h5oina** — Oxford Instruments HDF5 (mit Mustern, wenn vorhanden)
- **.h5** — kikuchipy oder EMsoft HDF5 (automatische Erkennung)

**Erwartete Ausgabe (Beispiel):**
```
Import target: ./meine_daten
CI threshold:  0.3
Pattern size:  128x128
Files:         2

  scan1.h5oina ... 1850/2000 points (H5OINA)
  scan2.h5oina ... 920/1000 points (H5OINA)

--- Summary ---
Files processed: 2
Files with errors: 0
Total scan points: 3000
Imported: 2770

Phase breakdown:
  Ferrit: 1500
  Austenit: 870
  Martensit: 400
```

**Was passiert hier?**
1. Das Skript erkennt automatisch das Dateiformat jeder Datei
2. Jede Datei wird geparst und in ein einheitliches `ScanData`-Format umgewandelt
3. Nur Messpunkte mit Confidence Index >= Schwellenwert werden importiert
4. Detektor-Metadaten (kV, Hersteller) werden automatisch aus den Dateien gelesen
5. Die Ergebnisse werden komprimiert im TrainingStore gespeichert (HDF5)
6. `--dry-run` ist nuetzlich, um vorher zu pruefen, wie viele Daten importiert wuerden

**Hinweis:** Nur Dateien, die EBSD-Muster enthalten (typischerweise HDF5-Formate), koennen tatsaechlich importiert werden. Reine Textformate (.ang, .ctf) enthalten nur Indexierungsergebnisse ohne Muster und werden mit einer Warnung uebersprungen.

### Beispiel 15: Scandateien im Python-Code importieren

Sie koennen den Import auch direkt aus Python-Code steuern:

```python
# Datei: scan_importieren.py
from ebsd_ai.data.scan_import import parse_scan, import_to_store
from ebsd_ai.data.training_store import TrainingStore

# 1. Scandatei parsen (Format wird automatisch erkannt)
scan = parse_scan("mein_scan.h5oina")

print(f"Format: {scan.source_format.value}")
print(f"Messpunkte: {scan.n_points}")
print(f"Phasen: {scan.phase_names}")
print(f"Hat Muster: {scan.has_patterns}")
print(f"Hat EDS: {scan.has_eds}")

# 2. In TrainingStore importieren (mit CI-Filterung)
store = TrainingStore(local_path="./meine_daten")
ergebnis = import_to_store("mein_scan.h5oina", store, ci_threshold=0.3)

print(f"\nImportiert: {ergebnis.n_points_imported}/{ergebnis.n_points_total}")
print(f"Phasen: {ergebnis.phase_counts}")
if ergebnis.warnings:
    print(f"Warnungen: {ergebnis.warnings}")
```

---

## Projektstruktur

Wenn Sie den Code verstehen oder anpassen moechten:

```
Ai_Ml/
├── ebsd_ai/                      # Hauptpaket (Quellcode)
│   ├── config.py                 # Zentrale Konfiguration
│   ├── data/                     # Datenverarbeitung
│   │   ├── pattern_io.py         # Laden und Normalisieren von EBSD-Mustern
│   │   ├── training_store.py     # Speichern von Trainingsdaten (HDF5)
│   │   ├── augmentation.py       # Datenaugmentierung (mehr Varianz)
│   │   ├── dataset.py            # PyTorch Dataset mit Sampling und Splits
│   │   ├── scan_import.py        # Import-Infrastruktur (ScanData, Euler-Konvertierung, Formaterkennung)
│   │   ├── ang_parser.py         # Parser fuer EDAX/OIM .ang-Dateien
│   │   ├── ctf_parser.py         # Parser fuer Oxford/HKL .ctf-Dateien
│   │   └── hdf5_parser.py        # Parser fuer HDF5-basierte EBSD-Formate (H5OINA, kikuchipy, EMsoft)
│   ├── features/                 # Merkmalsextraktion
│   │   ├── pattern_encoder.py    # CNN fuer EBSD-Muster
│   │   ├── eds_encoder.py        # MLP fuer EDS-Daten
│   │   └── fusion.py             # Kombiniert beide Datenquellen
│   ├── models/                   # KI-Modelle
│   │   ├── phase_classifier.py   # Hauptmodell: Phasenklassifikation
│   │   ├── pattern_enhancer.py   # U-Net fuer Muster-Entrauschung/Verbesserung
│   │   └── losses.py             # Verlustfunktionen fuer das Training
│   ├── training/                 # Trainings-Pipeline
│   │   ├── trainer.py            # Trainingsschleife mit Checkpointing
│   │   ├── evaluator.py          # Evaluierung (Metriken, Confusion Matrix)
│   │   └── online_learner.py     # Kontinuierliches Lernen aus neuen Daten
│   ├── inference/                # Vorhersage-Pipeline
│   │   └── predictor.py          # PhasePredictor fuer Einzel- und Scan-Vorhersagen
│   └── sync/                     # Netzwerk-Synchronisation
│       └── server_sync.py        # Bidirektionaler Datenaustausch ueber Netzlaufwerke
├── scripts/                      # Kommandozeilen-Skripte
│   ├── train.py                 # Training ueber die Kommandozeile
│   ├── evaluate.py              # Evaluierung ueber die Kommandozeile
│   └── import_data.py           # EBSD-Scandateien importieren
├── tests/                        # Automatische Tests
├── pyproject.toml                # Projekt-Konfiguration
└── README.md                     # Diese Datei
```

**Was ist was?**

- **config.py**: Zentrale Einstellungen (welche Elemente gibt es, Modellgroessen, etc.)
- **pattern_io.py**: Laedt EBSD-Bilder, skaliert sie auf 128x128, normalisiert Helligkeit
- **training_store.py**: Speichert Tausende von EBSD-Mustern effizient in HDF5-Dateien
- **augmentation.py**: Erstellt Variationen (Rotation, Rauschen) fuer besseres Training
- **dataset.py**: PyTorch Dataset fuer effizientes Laden von Daten, mit phasenausgewogenem Sampling und Train/Val/Test-Splits
- **pattern_encoder.py**: CNN (Convolutional Neural Network) extrahiert Merkmale aus EBSD-Mustern
- **eds_encoder.py**: MLP (Multi-Layer Perceptron) verarbeitet chemische Daten
- **fusion.py**: Kombiniert EBSD + EDS intelligent mit Attention-Mechanismus
- **phase_classifier.py**: Das Hauptmodell, das alles zusammenfuegt
- **pattern_enhancer.py**: U-Net fuer Muster-Entrauschung -- verbessert verrauschte EBSD-Muster mit optionaler Detektor-Konditionierung via FiLM
- **losses.py**: Verlustfunktionen (messen, wie falsch die Vorhersagen sind, damit das Modell lernt)
- **trainer.py**: Die vollstaendige Trainingsschleife mit automatischem Speichern und Fruehstopp
- **evaluator.py**: Bewertet ein trainiertes Modell mit Genauigkeit, Precision, Recall, F1, Confusion Matrix und Kalibrierungsanalyse
- **online_learner.py**: Kontinuierliches Lernen -- aktualisiert ein bestehendes Modell inkrementell mit neuen Daten, ohne alte Phasen zu vergessen
- **predictor.py**: Die oeffentliche Vorhersage-Schnittstelle -- laedt ein Modell und macht Vorhersagen fuer einzelne Muster oder ganze Scans
- **server_sync.py**: Synchronisiert Trainingsdaten zwischen lokalem Rechner und Netzlaufwerk -- atomare Dateioperationen, Fortschritts-Callback, bidirektional
- **scan_import.py**: Import-Infrastruktur fuer EBSD-Scandaten -- erkennt Dateiformate (ANG, CTF, HDF5), konvertiert Euler-Winkel zu Quaternionen, bietet einen einheitlichen `ScanData`-Container
- **ang_parser.py**: Parser fuer EDAX/OIM .ang-Dateien -- liest Header (Phasen, Gittergroesse, Detektor-Metadaten) und Datenspalten (Euler-Winkel, CI, Phasen-ID)
- **ctf_parser.py**: Parser fuer Oxford/HKL .ctf-Dateien -- liest Channel Text File Header (Phasen mit Gitterparametern, Schrittweiten) und Datenspalten (Euler-Winkel in Grad, MAD, Band Contrast)
- **hdf5_parser.py**: Parser fuer HDF5-basierte EBSD-Formate -- unterstuetzt H5OINA (Oxford Instruments), kikuchipy (Open-Source) und EMsoft (Simulationssoftware) mit automatischer Formaterkennung und Quaternion-zu-Euler-Konvertierung
- **import_data.py** (scripts): CLI-Skript zum Importieren von EBSD-Scandateien -- erkennt Formate automatisch, filtert nach Confidence Index und speichert Daten im TrainingStore

---

## Wichtige Konzepte

### 1. EBSD-Muster

- **Eingabeformat**: Graustufenbilder, typisch 480x640 oder 1024x1024 Pixel
- **Verarbeitung**: Wird auf 128x128 skaliert und normalisiert (Mittelwert 0, Standardabweichung 1)
- **Warum?** Das neuronale Netz braucht einheitliche Eingaben

### 2. EDS-Daten (Chemie)

- **Format**: Ein Vektor mit 92 Zahlen (ein Wert pro chemisches Element)
- **Bedeutung**: Atomprozent jedes Elements (z.B. Fe=70.0 heisst 70% Eisen)
- **Optional**: Wenn keine EDS-Daten vorhanden, werden Nullen verwendet

### 3. Phasen

- **Was sind Phasen?** Verschiedene Kristallstrukturen desselben Materials
- **Beispiele**: Ferrit (kubisch-raumzentriert), Austenit (kubisch-flaechenzentriert)
- **Klassifikation**: Das Modell sagt vorher: "Dieses Muster gehoert zu Phase X mit Y% Wahrscheinlichkeit"

### 4. Training vs. Vorhersage

- **Training**: Das Modell lernt aus beschrifteten Daten (kann Stunden dauern)
- **Vorhersage**: Das trainierte Modell analysiert neue, unbeschriftete Daten (Sekunden)

### 5. Multimodale Fusion

- **Problem**: Wir haben zwei Datenquellen (EBSD-Muster + EDS-Chemie)
- **Loesung**: Attention-Mechanismus lernt, wie stark jede Quelle gewichtet wird
- **Vorteil**: Funktioniert auch, wenn nur EBSD oder nur EDS verfuegbar ist

### 6. Trainings-Pipeline

Die Trainings-Pipeline automatisiert das gesamte Training:

- **Optimierer (Adam)**: Passt die Modell-Parameter an, um Fehler zu reduzieren
- **Lernraten-Planer (OneCycleLR)**: Aendert die Lernrate waehrend des Trainings - startet klein, wird groesser, dann wieder kleiner. Das fuehrt oft zu besseren Ergebnissen.
- **Early Stopping**: Stoppt das Training automatisch, wenn das Modell aufhoert, besser zu werden. Das verhindert "Ueberanpassung" (das Modell lernt die Trainingsdaten auswendig statt allgemeine Muster).
- **Checkpointing**: Speichert den Modellzustand regelmaessig:
  - `best.pt` - Das bisher beste Modell (niedrigster Validierungsverlust)
  - `last.pt` - Der neueste Stand (zum Fortsetzen)
- **Mixed Precision**: Auf Grafikkarten mit NVIDIA GPU wird automatisch mit halber Genauigkeit gerechnet. Das ist ca. doppelt so schnell bei gleichem Ergebnis.
- **Klassen-Balancierung**: Wenn Sie mehr Daten von einer Phase haben als von einer anderen, werden die selteneren Phasen automatisch staerker gewichtet.

### 7. Evaluierung (Qualitaetsbewertung)

Nach dem Training ist es wichtig zu pruefen, wie gut das Modell wirklich funktioniert. Der Evaluator bietet:

- **Top-1 Genauigkeit**: Wie oft die beste Vorhersage korrekt ist
- **Top-k Genauigkeit**: Wie oft die richtige Antwort unter den k wahrscheinlichsten ist (Standard: Top-3). Besonders wichtig, wenn das Modell als Vorfilter eingesetzt wird.
- **Precision, Recall, F1**: Detaillierte Metriken pro Phase - zeigen, wo das Modell stark oder schwach ist
- **Confusion Matrix**: Eine Tabelle, die zeigt, welche Phasen miteinander verwechselt werden. Zeilen sind die wahren Phasen, Spalten die vorhergesagten.
- **Kalibrierung**: Prueft, ob die Konfidenzwerte des Modells vertrauenswuerdig sind. Wenn das Modell "80% sicher" sagt, sollte es auch in ungefaehr 80% der Faelle richtig liegen.

---

## Aktuelle Implementierung

**Stand: Februar 2026**

### Fertig implementiert (Trainings-, Evaluierungs- und Vorhersage-Pipeline komplett)

- Zentrale Konfiguration (Detektoren, Elemente, Modellparameter)
- EBSD-Muster Ein-/Ausgabe (Laden, Skalieren, Normalisieren)
- Trainingsdaten-Speicher (HDF5 mit Kompression und Sharding)
- Datenaugmentierung (Rotation, Rauschen, Helligkeitsvariation)
- Pattern Encoder (ResNet-CNN fuer Bildmerkmale)
- EDS Encoder (MLP fuer chemische Daten)
- Multimodale Fusion (Attention-gewichtete Kombination)
- Phase Classifier (End-to-End-Modell mit dynamischer Phasenerweiterung)
- Modell-Serialisierung (Speichern/Laden von Checkpoints)
- PyTorch Dataset (EBSDPhaseDataset mit EDS-Dropout und phasenausgewogenem Sampling)
- Loss-Funktionen (gewichtete Cross-Entropy mit Klassen-Balancierung und Confidence-Penalty)
- Trainingsschleife (Trainer mit Checkpointing, Early Stopping und Mixed Precision)
- Evaluierung (Evaluator mit Top-1/Top-k Genauigkeit, Precision, Recall, F1, Confusion Matrix und Kalibrierungsanalyse)
- **Predictor API (PhasePredictor fuer Einzel- und Batch-Vorhersagen mit automatischer Vorverarbeitung und Graceful Degradation)**
- **Online Learning (OnlineLearner fuer kontinuierliches Lernen aus neuen Indexierungsergebnissen mit Experience Replay und EWC)**
- **Pattern Enhancement (PatternEnhancer U-Net fuer Muster-Entrauschung mit FiLM-Konditionierung auf Detektor-Metadaten)**
- **Server-Synchronisation (ServerSync fuer bidirektionalen Datenaustausch ueber Netzlaufwerke mit atomaren Dateioperationen)**
- **Training-CLI (scripts/train.py) fuer Training ueber die Kommandozeile mit allen Optionen**
- **Evaluierungs-CLI (scripts/evaluate.py) fuer Modellbewertung mit Confusion Matrix und Kalibrierung**
- **Import-Infrastruktur (data/scan_import.py) mit ScanData-Container, Euler-zu-Quaternion-Konvertierung und automatischer Formaterkennung fuer ANG, CTF und HDF5**
- **ANG-Parser (data/ang_parser.py) fuer EDAX/OIM .ang-Dateien mit Header-Parsing, Phasenerkennung und robuster Fehlerbehandlung**
- **CTF-Parser (data/ctf_parser.py) fuer Oxford/HKL .ctf-Dateien mit Euler-Grad-zu-Radiant-Konvertierung und MAD-zu-Konfidenz-Mapping**
- **HDF5-Parser (data/hdf5_parser.py) fuer H5OINA, kikuchipy und EMsoft HDF5-Formate mit automatischer Formaterkennung**
- **Import-Pipeline (parse_scan + import_to_store) fuer automatischen Import von EBSD-Scandateien in den TrainingStore mit CI-Filterung**
- **Import-CLI (scripts/import_data.py) fuer Dateiimport ueber die Kommandozeile mit Dry-Run-Modus und Multi-Datei-Unterstuetzung**
- **NaN/Inf-Sanitisierung: Automatische Erkennung und Bereinigung von nicht-endlichen Werten (NaN, Inf) an allen Dateneingabepunkten, in Parsern, im Dataset und im Trainer. Verhindert stille numerische Korruption. Inklusive Gradient-Clipping im Training.**
- **Speichereffiziente Verarbeitung grosser Scans: Thread-Locks (RLock) fuer sichere parallele HDF5-Zugriffe, leichtgewichtige Metadaten-Abfrage (get_metadata/iter_metadata), HDF5-Chunking, chunked Import (konfigurierbarer chunk_size), lazy Pattern-Indexierung bei Scan-Vorhersagen, num_workers=0 in DataLoadern fuer h5py-Kompatibilitaet**
- **Saubere oeffentliche API: Alle wichtigen Klassen direkt ueber `from ebsd_ai import ...` importierbar. Einheitliche Fehlermeldungen (RuntimeError mit hilfreichen Hinweisen wenn kein Modell geladen). Konfigurierbare MAD-zu-Konfidenz-Abbildung (`mad_to_confidence(mad, decay=1.0)`) fuer flexible Anpassung an verschiedene Materialien und Detektoren.**

**Das bedeutet:** Alle Module der geplanten Architektur sind jetzt implementiert. Die gesamte Pipeline von Training bis Vorhersage ist funktionsfaehig. Sie koennen Daten vorbereiten, ein Modell erstellen, es automatisch trainieren, die Qualitaet mit detaillierten Metriken bewerten und anschliessend mit dem `PhasePredictor` bequem Vorhersagen machen -- sowohl fuer einzelne Muster als auch fuer ganze EBSD-Scans. Mit dem `OnlineLearner` kann das Modell kontinuierlich aus neuen Daten lernen, ohne zuvor gelernte Phasen zu vergessen. Der `PatternEnhancer` kann verrauschte EBSD-Muster verbessern, bevor sie zur Klassifikation oder Indexierung verwendet werden. Mit `ServerSync` koennen Trainingsdaten ueber ein Netzlaufwerk mit anderen Nutzern geteilt werden.

---

## Tests ausfuehren

Dieses Projekt hat automatische Tests, um sicherzustellen, dass alles funktioniert.

### Alle Tests ausfuehren

```bash
pytest
```

**Erwartete Ausgabe:**
```
======================= test session starts ========================
collected 743 items

tests/test_augmentation.py .........................         [  5%]
tests/test_config.py ......................                  [ 10%]
tests/test_coverage_gaps.py ..........                      [ 12%]
tests/test_dataset.py ..................................................  [ 23%]
tests/test_eds_encoder.py .......                            [ 24%]
tests/test_evaluator.py ........................................  [ 33%]
tests/test_fusion.py ..............                          [ 36%]
tests/test_integration.py ...............                    [ 39%]
tests/test_losses.py .................................       [ 46%]
tests/test_online_learner.py ..........................      [ 52%]
tests/test_pattern_encoder.py ........                       [ 53%]
tests/test_pattern_enhancer.py .......................................  [ 62%]
tests/test_pattern_io.py .............                       [ 65%]
tests/test_phase_classifier.py ..........................    [ 70%]
tests/test_predictor.py ...........................................  [ 79%]
tests/test_server_sync.py .........................          [ 85%]
tests/test_training_store.py ..............                  [ 88%]
tests/test_ang_parser.py ..............................          [ 94%]
tests/test_ctf_parser.py ..............................          [ 89%]
tests/test_hdf5_parser.py ......................................................  [ 97%]
tests/test_import_pipeline.py ...................................  [ 98%]
tests/test_memory_efficiency.py ....................              [ 98%]
tests/test_nan_sanitization.py ............................      [ 98%]
tests/test_public_api.py ...........................             [ 99%]
tests/test_scan_import.py ....................................................  [ 99%]
tests/test_trainer.py ..................................     [100%]

======================= 743 passed in 75.18s =======================
```

**Was bedeutet das?**
- Jeder Punkt (`.`) ist ein erfolgreicher Test
- `743 passed` heisst: Alle 743 Tests haben funktioniert
- Wenn ein Test fehlschlaegt, sehen Sie ein `F` statt `.`

### Nur bestimmte Tests ausfuehren

```bash
# Nur Konfigurations-Tests
pytest tests/test_config.py

# Nur Modell-Tests
pytest tests/test_phase_classifier.py

# Nur Loss-Funktions-Tests
pytest tests/test_losses.py

# Nur Evaluierungs-Tests
pytest tests/test_evaluator.py

# Nur Predictor-Tests (Vorhersage-Schnittstelle)
pytest tests/test_predictor.py

# Nur Online-Learning-Tests
pytest tests/test_online_learner.py

# Nur Pattern-Enhancement-Tests
pytest tests/test_pattern_enhancer.py

# Nur Server-Sync-Tests
pytest tests/test_server_sync.py

# Import-Infrastruktur-Tests (Euler-Konvertierung, Formaterkennung, ScanData)
pytest tests/test_scan_import.py

# ANG-Parser-Tests (EDAX/OIM .ang-Dateien)
pytest tests/test_ang_parser.py

# CTF-Parser-Tests (Oxford/HKL .ctf-Dateien)
pytest tests/test_ctf_parser.py

# HDF5-Parser-Tests (H5OINA, kikuchipy, EMsoft)
pytest tests/test_hdf5_parser.py

# Import-Pipeline-Tests (parse_scan, import_to_store, CLI)
pytest tests/test_import_pipeline.py

# NaN/Inf-Sanitisierung-Tests
pytest tests/test_nan_sanitization.py

# Speichereffizienz-Tests (Thread-Locks, Metadaten, Chunking)
pytest tests/test_memory_efficiency.py

# Oeffentliche API und Fehlermeldungen
pytest tests/test_public_api.py

# End-to-End-Integrationstest (prueft die gesamte Pipeline)
pytest tests/test_integration.py

# Verbose-Modus (mehr Details)
pytest -v
```

---

## Fehlerbehebung

### Problem: "ModuleNotFoundError: No module named 'ebsd_ai'"

**Ursache:** Das Paket wurde nicht installiert oder die virtuelle Umgebung ist nicht aktiviert.

**Loesung:**
```bash
# 1. Aktivieren Sie die virtuelle Umgebung
source .venv/bin/activate  # Mac/Linux
# ODER
.venv\Scripts\activate     # Windows

# 2. Installieren Sie das Paket
pip install -e .
```

### Problem: "torch not found" oder PyTorch-Fehler

**Ursache:** PyTorch wurde nicht korrekt installiert.

**Loesung:**
```bash
# Installieren Sie PyTorch manuell
pip install torch torchvision

# Testen Sie die Installation
python -c "import torch; print(torch.__version__)"
```

**Fuer GPU-Unterstuetzung (optional, fuer schnelleres Training):**
Besuchen Sie https://pytorch.org/ und waehlen Sie Ihre Systemkonfiguration (CUDA-Version).

### Problem: "RuntimeError: CUDA out of memory"

**Ursache:** Die GPU hat nicht genug Speicher (bei GPU-Training).

**Loesung:**
```python
# Reduzieren Sie die Batch-Groesse in Ihrer Konfiguration
training_config = TrainingConfig(
    batch_size=8,  # Statt 64 (Standard)
)

# ODER: Nutzen Sie CPU statt GPU
trainer = Trainer(model=modell, config=training_config, device="cpu")
```

### Problem: Training stoppt sofort (nach 1-2 Epochen)

**Ursache:** Early Stopping greift, weil zu wenige Daten vorhanden sind oder `early_stopping_patience` zu niedrig ist.

**Loesung:**
```python
# Erhoehen Sie die Geduld (Patience)
training_config = TrainingConfig(
    early_stopping_patience=20,  # 20 Epochen ohne Verbesserung abwarten
    epochs=100,                  # Mehr Epochen erlauben
)
```

### Problem: "RuntimeError: expected scalar type Float but found Half"

**Ursache:** Mixed Precision ist aktiviert, aber Ihr Code erwartet volle Genauigkeit.

**Loesung:**
```python
# Mixed Precision ausschalten
training_config = TrainingConfig(
    mixed_precision=False,
)
```

### Problem: Tests schlagen fehl

**Ursache:** Moeglicherweise inkompatible Versionen oder fehlende Abhaengigkeiten.

**Loesung:**
```bash
# Aktualisieren Sie alle Pakete
pip install --upgrade -e .[dev]

# Fuehren Sie Tests einzeln aus, um das Problem zu finden
pytest tests/test_config.py -v
```

### Problem: HDF5-Fehler beim Speichern von Daten

**Ursache:** Beschaedigte HDF5-Datei oder Berechtigungsprobleme.

**Loesung:**
```bash
# Loeschen Sie beschaedigte Dateien
rm -rf meine_daten/*.h5

# Stellen Sie sicher, dass der Ordner schreibbar ist
chmod -R u+w meine_daten  # Linux/Mac
```

### Problem: "ImportError: cannot import name 'X'"

**Ursache:** Veraltete Installation oder Aenderungen am Code.

**Loesung:**
```bash
# Deinstallieren und neu installieren
pip uninstall ebsd-ai
pip install -e .
```

### Problem: Sehr langsames Training

**Ursache:** CPU wird verwendet statt GPU, oder die Batch-Groesse ist zu klein.

**Loesung:**
```python
# Pruefen Sie, ob GPU verfuegbar ist
import torch
print(f"CUDA verfuegbar: {torch.cuda.is_available()}")
print(f"Geraet: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Vergroessern Sie die Batch-Groesse (wenn genug GPU-Speicher vorhanden)
training_config = TrainingConfig(
    batch_size=128,  # Mehr Daten pro Schritt = schneller
)
```

**Hinweis:** Ohne NVIDIA-GPU ist Training langsam. Fuer grosse Datensaetze empfehlen wir einen Computer mit GPU (NVIDIA mit CUDA-Unterstuetzung).

### Problem: Evaluierungsergebnisse zeigen 0% Genauigkeit

**Ursache:** Das Modell wurde nicht richtig trainiert, oder Sie evaluieren auf komplett anderen Phasen als beim Training.

**Loesung:**
```python
# Stellen Sie sicher, dass die Phasen uebereinstimmen
print(f"Modell-Phasen: {modell.phase_names}")
print(f"Daten-Phasen:  {dataset.phase_names}")

# Pruefen Sie, ob genug Trainingsdaten vorhanden waren
# Mindestens einige hundert Muster pro Phase werden empfohlen
```

### Problem: Kalibrierung zeigt schlechte Werte

**Ursache:** Das Modell ist "ueberselbstbewusst" (gibt hohe Konfidenz auch bei falschen Vorhersagen).

**Loesung:**
- Dies ist normal bei kleinen Datensaetzen oder kurzer Trainingszeit
- Laengeres Training mit mehr Daten verbessert die Kalibrierung
- Die Confidence-Penalty-Loss hilft bereits automatisch dagegen

---

## Erweiterte Nutzung

### Neue Phase hinzufuegen

Das Modell kann dynamisch erweitert werden, wenn Sie eine neue Phase entdecken:

```python
# Modell mit 3 Phasen laden
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier

save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
modell = PhaseClassifier.from_save_dict(save_dict)
print(f"Vorher: {modell.phase_names}")

# Neue Phase hinzufuegen
modell.add_phase("Bainit")
print(f"Nachher: {modell.phase_names}")

# Speichern und dann mit neuen Daten weiter trainieren
torch.save(modell.get_save_dict(), "modell_erweitert.pt")
```

### Datenaugmentierung nutzen

Mehr Trainingsdaten durch Variationen erstellen:

```python
import numpy as np
from ebsd_ai.data.augmentation import augment_pattern, augment_eds, AugmentationConfig

# Konfiguration
config = AugmentationConfig(
    rotation_angle=2.0,       # +/-2 Grad Rotation
    brightness_range=0.2,     # +/-20% Helligkeit
    contrast_range=0.2,       # +/-20% Kontrast
    noise_std=0.02,           # 2% Rauschen
    horizontal_flip_prob=0.5, # 50% Chance auf Spiegelung
)

# Original-Daten
muster = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
eds = np.array([70.0, 2.0, 18.0, 10.0] + [0.0]*88)  # Fe, C, Cr, Ni

# Augmentieren (5 Variationen)
for i in range(5):
    aug_muster = augment_pattern(muster, config, rng=np.random.default_rng(i))
    aug_eds = augment_eds(eds, config, rng=np.random.default_rng(i))
    print(f"Variation {i+1}: Muster-Shape {aug_muster.shape}, EDS-Summe {aug_eds.sum():.1f}")
```

### Typischer Arbeitsablauf

Hier ist der empfohlene Ablauf von Anfang bis Ende:

```
1. Daten sammeln     -->  EBSD-Muster mit bekannten Phasen
2. Daten speichern   -->  TrainingStore.add_sample() (Beispiel 1)
3. Modell erstellen  -->  PhaseClassifier (Beispiel 2)
4. Modell trainieren -->  Trainer.train() (Beispiel 3)
5. Modell bewerten   -->  Evaluator.evaluate() (Beispiel 6)
6. Vorhersagen       -->  PhasePredictor.predict_phase() (Beispiel 7)
7. Scan vorhersagen  -->  PhasePredictor.predict_scan() (Beispiel 8)
8. Weiter lernen     -->  OnlineLearner.update() (Beispiel 9)
9. Muster verbessern -->  PatternEnhancer.enhance() (Beispiel 10)
10. Daten teilen     -->  ServerSync.push/pull() (Beispiel 11)
11. Neue Phase?      -->  Wird automatisch erkannt und erweitert
```

---

## Technische Details

### Modellarchitektur

```
Input: EBSD-Muster (128x128) + EDS (92) + Metadaten (15+1)
    |
PatternEncoder (ResNet-CNN)      EDSEncoder (MLP)
    | (256-dim)                      | (64-dim)
    +----------> MultimodalFusion <--+
              (Attention-gated)
                    | (128-dim)
              Linear Classifier
                    |
    Output: Phasen-Logits + Attention-Gewichte
```

### Trainings-, Evaluierungs- und Vorhersage-Pipeline

```
EBSDPhaseDataset
    |
    +-- Automatische Train/Val/Test-Aufteilung (stratifiziert nach Phase)
    |
    +-- Phasen-ausgewogenes Sampling (seltene Phasen werden oefter gezeigt)
    |
    +-- Datenaugmentierung (Rotation, Rauschen, EDS-Dropout)
    |
    v
Trainer
    |
    +-- Adam-Optimierer + OneCycleLR Scheduler
    |
    +-- CombinedLoss (gewichtete Cross-Entropy + Confidence-Penalty)
    |
    +-- Mixed Precision (automatisch auf GPU)
    |
    +-- Early Stopping (stoppt bei keiner Verbesserung)
    |
    +-- Checkpointing (best.pt + last.pt)
    |
    v
TrainingResult (Verlust-Kurven, beste Epoche, Dauer)
    |
    v
Evaluator
    |
    +-- Top-1 und Top-k Genauigkeit
    |
    +-- Precision, Recall, F1 pro Phase
    |
    +-- Confusion Matrix (absolut und normalisiert)
    |
    +-- Kalibrierungsanalyse (Konfidenz vs. tatsaechliche Genauigkeit)
    |
    +-- Textbericht via format_classification_report()
    |
    v
EvaluationResult (alle Metriken in einem Objekt)
    |
    v
PhasePredictor (Vorhersage-Schnittstelle)
    |
    +-- predict_phase(): Einzelne Muster vorhersagen
    |
    +-- predict_scan(): Ganze EBSD-Scans vorhersagen (Batch)
    |
    +-- Automatische Vorverarbeitung (Skalierung, Normalisierung)
    |
    +-- Graceful Degradation (funktioniert mit/ohne EDS)
    |
    v
PhasePrediction / ScanPredictionResult
    |
    v (Ergebnisse zurueck in TrainingStore)
OnlineLearner (Kontinuierliches Lernen)
    |
    +-- Experience Replay (alte + neue Daten mischen)
    |
    +-- EWC (Elastic Weight Consolidation gegen Vergessen)
    |
    +-- Automatische Phasenerweiterung
    |
    v
Verbessertes Modell (zurueck zum PhasePredictor)
```

### Datenspeicherung

- **Format**: HDF5 mit gzip-Kompression (Level 4)
- **Sharding**: Max. 5000 Proben pro Datei (konfigurierbar)
- **Struktur**:
  ```
  shard_00000.h5
  ├── sample_00000000/
  │   ├── pattern_original (uint8/uint16)
  │   ├── pattern_normalized (float32, 128x128)
  │   ├── eds_atomic_pct (float32, 92)
  │   ├── has_eds (bool)
  │   ├── confirmed_phase (string)
  │   └── ... (Metadaten)
  ├── sample_00000001/
  ...
  ```

### Systemanforderungen

- **RAM**: Min. 8 GB (16 GB empfohlen)
- **Speicher**: 10 GB frei (fuer Daten und Modelle)
- **GPU** (optional): NVIDIA mit CUDA fuer schnelles Training
- **CPU**: Funktioniert, aber langsamer

---

## Lizenz und Mitwirkung

Dieses Projekt ist Open Source. Beitraege sind willkommen!

**Entwicklerhinweise:**
- Code-Stil: Folgen Sie PEP 8 (automatisch mit `ruff` geprueft)
- Type Hints: Alle Funktionen sollten Typen haben
- Tests: Neue Features brauchen Tests
- Dokumentation: NumPy-Stil Docstrings

**Code pruefen:**
```bash
# Lint (Stilpruefung)
ruff check .

# Type-Checking
mypy ebsd_ai

# Tests mit Coverage
pytest --cov=ebsd_ai
```

---

## Changelog

### Version 0.1.0 (Februar 2026)

**Implementiert (komplette ML-Pipeline):**
- Zentrale Konfiguration und Datenstrukturen
- EBSD-Muster I/O und Normalisierung
- HDF5-basierter Trainingsdaten-Speicher
- Datenaugmentierung (Rotation, Rauschen, Photometrie)
- Pattern Encoder (ResNet-Architektur)
- EDS Encoder (MLP mit Batch Normalization)
- Multimodale Fusion (Attention-gated)
- Phase Classifier (End-to-End-Modell)
- PyTorch Dataset mit EDS-Dropout, phasenausgewogenem Sampling und stratifizierten Splits
- Loss-Funktionen (gewichtete Cross-Entropy, Confidence-Penalty, kombinierte Loss)
- Trainingsschleife (Adam + OneCycleLR, Checkpointing, Early Stopping, Mixed Precision)
- Evaluierung (Top-1/Top-k Genauigkeit, Precision, Recall, F1, Confusion Matrix, Kalibrierung)
- Predictor-API (PhasePredictor fuer Einzel- und Batch-Vorhersagen)
- Online Learning (OnlineLearner mit Experience Replay, EWC und automatischer Phasenerweiterung)
- Pattern Enhancement (PatternEnhancer U-Net mit FiLM-Konditionierung fuer Muster-Entrauschung)
- Server-Synchronisation (ServerSync fuer bidirektionalen Datenaustausch mit atomaren Dateioperationen)
- Kommandozeilen-Trainingsskript (scripts/train.py) mit argparse, Resume, Evaluierung
- Kommandozeilen-Evaluierungsskript (scripts/evaluate.py) mit Confusion Matrix und Kalibrierung
- End-to-End-Integrationstest (prueft die gesamte Pipeline von Datenspeicherung bis Vorhersage)
- Vollstaendige Type-Checking-Abdeckung (mypy: 0 Fehler in 23 Dateien)
- Vollstaendige Lint-Abdeckung (ruff: 0 Warnungen)
- Test-Coverage-Optimierung (99% Gesamtabdeckung, jedes Modul ueber 96%)
- Import-Infrastruktur (ScanData-Container, Euler-zu-Quaternion-Konvertierung, automatische Formaterkennung)
- ANG-Parser fuer EDAX/OIM .ang-Dateien (Header-Parsing, Phasenerkennung, robuste Fehlerbehandlung)
- CTF-Parser fuer Oxford/HKL .ctf-Dateien (Euler-Grad-Konvertierung, MAD-zu-Konfidenz-Mapping)
- HDF5-Parser fuer H5OINA, kikuchipy und EMsoft (automatische Formaterkennung, Quaternion-Euler-Konvertierung)
- Import-Pipeline (parse_scan + import_to_store) mit CI-Filterung und automatischer Detektor-Erkennung
- Import-CLI (scripts/import_data.py) fuer Dateiimport ueber die Kommandozeile
- NaN/Inf-Sanitisierung an allen Dateneingabepunkten, Parsern, Dataset und Trainer
- Gradient-Clipping und NaN-Loss-Erkennung im Trainer
- Speichereffiziente Verarbeitung grosser Scans (Thread-Locks, leichtgewichtige Metadaten-API, HDF5-Chunking, chunked Import, lazy Pattern-Indexierung)
- Saubere oeffentliche API mit __all__-Exporten und einheitlichen Fehlermeldungen
- Konfigurierbare MAD-zu-Konfidenz-Abbildung (mad_to_confidence mit einstellbarem Decay)
- Vollstaendige Test-Suite (743 Tests)

**Alle Module der geplanten Architektur sind implementiert.**
