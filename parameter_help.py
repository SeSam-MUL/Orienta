# parameter_help.py
"""
Module providing help text and recommendations for indexing and PC parameters.
"""

class ParameterHelp:
    """
    Holds detailed descriptions, impact of high vs low values, and phase-specific recommendations
    for each indexing and detector-parameter.
    """
    _data = {
        "min_d": {
            "label": "min_d (Å)",
            "description": (
                "Minimum lattice plane d-spacing to include for reflector generation."
                "High values (e.g. >1.5 Å): keep only coarse planes, reducing noise and speeding up indexing,"
                "but may ignore useful finer band information."
                "Low values (e.g. <1.0 Å): include more fine-grained planes, improving precision on clean patterns,"
                "but at the cost of increased processing time and potential spurious bands."
            ),
            "recommendation": {
                "Al": "1.0–1.5 Å",
                "Ni": "1.5–2.0 Å",
                "Fe": "1.0–1.5 Å",
                "iron": "1.0–1.5 Å",
                "Ferrite": "1.0–1.5 Å",
                "Austenite": "1.0–1.5 Å",
                "Martensite": "1.0–1.5 Å",
            }
        },
        "f_threshold": {
            "label": "f_threshold",
            "description": (
                "Relative structure-factor threshold for selecting reflectors."
                "Lower thresholds (e.g. 0.05–0.10): include weaker diffraction signals, which can boost"
                "index reliability on noisy data but increase computational effort and noise sensitivity."
                "Higher thresholds (e.g. 0.15–0.20): focus on the strongest reflections only, speeding up"
                "indexing with cleaner inputs but risking too few reference bands on weaker patterns."
            ),
            "recommendation": {
                "Al": "0.10–0.15",
                "Ni": "0.05–0.10",
                "Fe": "0.05–0.10",
                "iron": "0.05–0.10",
                "Ferrite": "0.05–0.10",
                "Austenite": "0.08–0.12",
                "Martensite": "0.05–0.10",
            }
        },
        "max_reflectors": {
            "label": "max_reflectors",
            "description": (
                "Maximum number of lattice plane reflectors to consider after filtering."
                "Lower limits (e.g. 30–50): very fast indexing but may lack sufficient band references,"
                "leading to lower confidence index (CI)."
                "Higher limits (e.g. 80–120): more comprehensive plane set, improving accuracy on"
                "complex patterns, but at the cost of longer compute times."
            ),
            "recommendation": {
                "Al": "50–80",
                "Ni": "60–100",
                "Fe": "60–100",
                "iron": "60–100",
                "Ferrite": "50–80",
                "Austenite": "60–100",
                "Martensite": "50–80",
            }
        },
        "nBands": {
            "label": "nBands",
            "description": (
                "Number of Hough-transform bands (lines) to detect per pattern."
                "A lower count (e.g. 8–10) speeds up detection but can miss weaker or partial bands,"
                "reducing CI accuracy."
                "A higher count (e.g. 15–20) captures more features for better orientation fit,"
                "but increases processing time and potential for false positives."
            ),
            "recommendation": {
                "Al": "10–15",
                "Ni": "12–20",
                "Fe": "10–15",
                "iron": "10–15",
                "Ferrite": "10–15",
                "Austenite": "12–18",
                "Martensite": "10–15",
            }
        },
        "method": {
            "label": "method",
            "description": (
                "Optimization algorithm for refining the pattern center (PC)."
                "PSO (Particle Swarm Optimization): robust global search, handles complex CI landscapes"
                "but is computationally expensive."
                "Nelder-Mead: faster local simplex search, suitable when starting PC is close to optimum,"
                "but may get trapped in local minima if PC error is large."
            ),
            "recommendation": {
                "default": "PSO"
            }
        },
        "search_limit": {
            "label": "search_limit",
            "description": (
                "Maximum relative step size for PC search during optimization."
                "Small values (e.g. 0.01–0.05): fine-tune the center around initial guess quickly."
                "Larger values (e.g. 0.10–0.20): allow broader search of PC space, useful"
                "when initial guess is far off, at the expense of more iterations."
            ),
            "recommendation": {
                "Al": "0.05–0.10",
                "Ni": "0.05–0.15",
                "Fe": "0.05–0.10",
                "iron": "0.05–0.10",
                "Ferrite": "0.05–0.10",
                "Austenite": "0.05–0.10",
                "Martensite": "0.05–0.10",
            }
        }
    }

    @classmethod
    def get_tooltip(cls, key: str, phase: str = None) -> str:
        """Return multiline tooltip text for the given parameter key.

        Includes phase-specific recommendation if a phase name is provided.
        """
        entry = cls._data.get(key)
        if not entry:
            return ""
        # Build tooltip text
        text = entry['label'] + ":\n" + entry['description']
        if phase and phase in entry.get('recommendation', {}):
            text += "\n\nRecommended for {}: {}".format(
            phase,
            entry['recommendation'][phase]
        )
        return text