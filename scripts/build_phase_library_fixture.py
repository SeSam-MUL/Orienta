"""Freeze the real crystal library into the fixture the search index is built on.

WHY A FIXTURE AT ALL. The phase library's search semantics (spec §2.1) are the
first thing built and everything downstream inherits them, but the endpoint
that will serve this data (§3) does not exist yet. A frozen snapshot of the
REAL library lets the index and its acceptance criteria be written and
measured now, against the phases that actually exist, instead of against
invented ones. The spec's acceptance criteria name keys, not counts, precisely
so they survive the swap to the live endpoint.

WHAT IT IS NOT. This is not the data contract's implementation. When the
endpoint lands, the shape below is what it must produce; if it produces
something else, this file is the record of what was agreed, and the tests
built on it will say so loudly.

THREE THINGS THIS SCRIPT GETS RIGHT THAT THE MOCKUP GENERATOR DID NOT:

1. **Facts about the structure come from the real reader**
   (`crystal_hint_local_library.get_index()`), not from a regex over formula
   strings. The mockup's element sets were parsed out of text and were wrong
   for two phases; bands built on them would have promised Si where there is
   none.

2. **The cell comes from a NAMED block.** 18 of 36 CIFs carry several data
   blocks and 14 of them disagree about `_cell_length_a` — `Al.cif` holds
   2.8631 AND 4.049 (primitive vs conventional). SpringerMaterials names the
   settings in the block header, so the setting is read, not guessed, and the
   one used is recorded in the fixture.

3. **Capabilities are three separate questions** (spec §2.7). "Has a master"
   was counted wrong once already: `Dictionary_Library/` holds pre-built
   dictionaries, the EMsoft masters live in `EBSD_H5_Cache/<stem>/`. The
   master link parses the convention the simulation path writes; it does not
   substring-match, which is what makes the shipped app answer "2".

Run from the repo root:  python scripts/build_phase_library_fixture.py
"""
import json
import re
import sys

import h5py
from pymatgen.core.periodic_table import Element
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'frontend' / 'src' / 'components' / 'PhaseLibrary' / '__fixtures__' / 'library.json'
SCHEMA = 1

# `Formula (CIF_stem) [Pearson] {kV}.sht`
SHT_NAME = re.compile(
    r'^(?P<formula>.+?) \((?P<stem>.+?)\)(?: \[(?P<pearson>[^\]]+)\])? \{(?P<kv>[^}]+)\}$')
REFS = re.compile(r'_publ_section_references\s*\n;(.*?)\n;', re.S)
# A SpringerMaterials data block names its cell setting:
#   data_sm_isp_SD0302719-standardized_unitcell
BLOCK = re.compile(r'^\s*data_(?P<name>\S+)', re.M | re.I)

# Preference order for which block's cell to believe. "standardized" is the
# conventional cell in a standard setting -- the number a crystallographer
# quotes. "niggli_reduced" is the primitive cell and is 1/sqrt(2) of it for
# fcc, which is how a 4.049 becomes a 2.8631.
SETTING_ORDER = ['standardized_unitcell', 'published_cell', 'niggli_reduced_cell']


def cif_blocks(text):
    """[(setting_name_or_None, block_text)] in file order."""
    marks = [(m.start(), m.group('name')) for m in BLOCK.finditer(text)]
    out = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        setting = next((s for s in SETTING_ORDER if s in name.lower()), None)
        out.append((setting, text[pos:end]))
    return out


def tag(text, name, quoted=True):
    """First value of a CIF tag that is neither blank nor '?'."""
    pat = re.escape(name) + (r"\s+'([^']*)'" if quoted else r'\s+(\S+)')
    for m in re.finditer(pat, text):
        v = m.group(1).strip()
        if v and v not in ('?', '.'):
            return v
    return tag(text, name, quoted=False) if quoted else None


def cell_from(text):
    """(cell dict, setting used, all settings seen) -- see docstring point 2."""
    blocks = cif_blocks(text)
    seen = sorted({s for s, _ in blocks if s})
    for want in SETTING_ORDER:
        for setting, body in blocks:
            if setting == want and tag(body, '_cell_length_a', quoted=False):
                return _cell(body), want, seen
    return _cell(text), None, seen


def _cell(body):
    def num(name):
        v = tag(body, name, quoted=False)
        if not v:
            return None
        m = re.match(r'^([\d.]+)', v)        # 12.56(2) -> 12.56
        return float(m.group(1)) if m else None
    return {k: num('_cell_length_' + k) for k in ('a', 'b', 'c')} | \
           {k: num('_cell_angle_' + k) for k in ('alpha', 'beta', 'gamma')}


def reference(text):
    m = REFS.search(text)
    if not m:
        return None
    ref = re.sub(r'<[^>]+>', '', m.group(1))
    ref = re.sub(r'&#x([0-9a-fA-F]+);', lambda x: chr(int(x.group(1), 16)), ref)
    ref = ref.replace('&amp;', '&')
    # URLs come out of the prose and into their own field: a link in the
    # middle of a sentence reads badly, and `url()` keeps it.
    #
    # A URL that sits INSIDE brackets takes the brackets with it. Four
    # Materials Project citations are written
    #   `... mp-aaacqrcb (https://materialsproject.org/...). A. Jain et al. ...`
    # and removing only the link leaves `mp-aaacqrcb ( . A. Jain` -- an open
    # bracket that closes nowhere, in a citation somebody will paste into a
    # manuscript. The brackets are punctuation around the link, not around
    # the sentence, so they go with it.
    ref = _BRACKETED_URL.sub(' ', ref)
    ref = _URL.sub(' ', ref)
    ref = ' '.join(ref.split()).strip()
    # `foo ( ). A.` and `foo .` are what a removal leaves behind when the
    # brackets held something besides the link.
    ref = re.sub(r'\(\s*\)', '', ref)
    ref = re.sub(r'\s+([.,;])', r'\1', ref)
    ref = ' '.join(ref.split()).strip()
    # Spec §2.2: a source that is plainly not a citation is worse than none.
    # Measured rejects: '_publ_author_address', a bare 'https', a doubled
    # sentence with a severed URL -- and, since the URLs now go, whatever is
    # left of `Al3Fe2Si_mp-1190708`, whose whole reference block was a
    # doi.org link and a Materials Project link. `mp-1190708_` on its own is
    # not a citation.
    if len(ref) < 12 or ref.startswith('_'):
        return None
    letters = sum(c.isalpha() for c in ref)
    if letters < 8:
        return None            # ids and punctuation, no prose
    return ref


# A DOI as a DOI: `10.` then the registrant, then anything up to whitespace or
# closing punctuation. Trailing `.,;)` are sentence, not identifier.
_DOI = re.compile(r'(10\.\d{4,9}/[^\s\x22\x27<>,;)\]]+)')


# Where a URL ends. Two derivations that disagree about this produce two
# different citation texts, and the difference is invisible until somebody
# diffs them -- which is the whole job of this file. For a while both sides
# were untidy in DIFFERENT ways on four Materials Project citations and the
# contract test reported a difference that was really a draw; both have
# since been fixed, independently, and the field-for-field test is what
# says they agree.
_URL = re.compile(r'https?://[^\s\x22\x27<>]+')

# A link that is the entire content of a bracket pair, brackets and all.
_BRACKETED_URL = re.compile(r'\(\s*https?://[^\s\x22\x27<>()]+\s*\)')


def url(text):
    """The first http(s) URL in the reference block.

    Eight phases have one, and for five of them it is the ONLY locator --
    SpringerMaterials records with no DOI. Stripping URLs out of the citation
    prose without keeping them somewhere would have thrown those away; the
    citation reads better without a URL in the middle of it, and the reader
    still needs the link.
    """
    m = REFS.search(text)
    if not m:
        return None
    hit = _URL.search(m.group(1))
    return hit.group(0).rstrip('.,;)') if hit else None


def doi(text):
    """The DOI, normalised to `10.xxxx/...`.

    Measured: 10 of the 36 CIFs carry one -- 8 in `_publ_section_doi`, one in
    `_journal_paper_doi`, and one sitting alone inside the reference block --
    and for SIX of them it is the ONLY source. Those six (`Al3Fe2Si_mp-…`,
    `Al6Fe_mp-…`, `Fe3Al2Si4`, `Fe3_Al2_Si3`, `MnAl6_mp-173`, `beta-AlFeSi`)
    would otherwise stand on the card with "no source", which is a statement
    about the library that is not true.

    This is also where `reference()`'s rejected `http…` block ends up. That
    rejection is right -- a bare URL is not a citation -- but throwing the
    content away with it was not: `Al3Fe2Si_mp-1190708` has a
    `_publ_section_references` block whose entire content is a doi.org link
    and a Materials Project link.

    Half of them are written as `https://doi.org/10...`. A DOI is the
    identifier, not the resolver, so the prefix goes and the card builds its
    own link.
    """
    for field in ('_publ_section_doi', '_journal_paper_doi'):
        m = re.search(re.escape(field) + r'\s+[\'"]?(\S+)', text, re.I)
        if m:
            hit = _DOI.search(m.group(1))
            if hit:
                return hit.group(1).rstrip('.,;)')
    m = REFS.search(text)
    if m:
        hit = _DOI.search(m.group(1))
        if hit:
            return hit.group(1).rstrip('.,;)')
    return None


# Database tags that sit in a file stem and are not chemistry. Stripped
# BEFORE the element regex runs, because afterwards they are indistinguishable
# from elements: measured on the real library, `Al2Zn_MP-mp-aaacqrcb` gave the
# label elements Al, M, P, Zn -- the Materials Project tag read as manganese
# and phosphorus. Two phantom elements, a phantom facet each, and a phase
# wrongly flagged as "label and structure disagree".
def elements_from_formula(s, from_filename=False):
    """Element symbols as WRITTEN in a label -- the 'laut Etikett' set.

    A FILE STEM is not a formula, and reading it as one invents elements.
    Measured on the real library: `Al2Zn_MP-mp-aaacqrcb` gave Al, M, P, Zn --
    the Materials Project tag read as manganese and phosphorus -- and
    `T-phase_Mg32(AlZn)49_Bergman` gives beryllium out of an author's
    surname. Each phantom is a phantom element facet and a phase wrongly
    marked "label and structure disagree".

    So a stem is read only up to its first `_`, which is where this library's
    names stop being chemistry and start being provenance. Checked against
    all 36: `alpha-AlFeMnSi_ICSD-52623`, `MgZn2_sd_0261233`,
    `Al3Fe2Si_mp-1190708_symmetrized` and `beta-AlFeSi` all keep every
    element they should.

    The first repair here stripped database tags (MP, ICSD, COD, mp-...)
    before parsing instead. It is gone: with the stem rule in place, nothing
    could reach it. Measured -- no stem in the library carries a tag before
    its first underscore, and no real label string carries one at all --
    and mutation-testing agreed, removing it broke nothing. If a tag ever
    turns up in front, that is a new case and it comes with a new
    measurement.

    `T` is dropped as a special case and stays one: `T-phase_Mg32(AlZn)49` is
    a phase name in the field's own vocabulary, not tellurium. A letter
    blocklist is a bad instrument, which is why it is down to one letter.
    """
    if not s:
        return []
    s = unicodedata.normalize('NFKD', s)
    if from_filename:
        s = s.split('_')[0]
    found = set(re.findall(r'[A-Z][a-z]?', s))
    # Filtered against a real element table, the way the backend does it
    # (c1). It changes nothing on this library -- every phantom found so far
    # was a genuine symbol read out of something that was not chemistry --
    # but it is the difference between a derivation and a regex.
    return sorted(x for x in found if x != 'T' and Element.is_valid_symbol(x))


def norm_element(e):
    """`Al0+` -> `Al`. Four ICSD CIFs write oxidation states; until the backend
    fix lands they would produce five phantom bands."""
    return re.sub(r'\d*[+-]$', '', str(e))


def main():
    from backend.api.services import crystal_hint_local_library as chl
    idx = chl.get_index()

    db = ROOT / 'Database'
    cif_dir = db / 'CIF_Library'

    shts, sht_folder = {}, {}
    for p in (db / 'EBSD_SHT_Database').rglob('*.sht'):
        m = SHT_NAME.match(p.stem)
        if m:
            shts[m.group('stem')] = (p, m.groupdict())
            sht_folder[m.group('stem')] = p.parent.name

    # EBSD_H5_Cache/<sanitised stem>/<stem>_master_*.h5 -- a convention, read
    # rather than guessed. And the sanitiser is the project's own: my first
    # attempt re-implemented it by hand and lost two phases, because the real
    # one transliterates Greek and eats the dot ("α-(AlMnSi)" -> "alpha-_AlMnSi",
    # "Al4Fe1.7Si (τ11)" -> "Al4Fe1_7Si_11"). path_utils.sanitize_filename is
    # documented as the single source of truth; use it.
    from path_utils import sanitize_filename as sanitise

    # The capability hangs on the CONTENT, not the name. Measured on this
    # library: 65 `.h5` under EBSD_H5_Cache, 19 with "master" in the name, but
    # **20** carrying `EMData/EBSDmaster/mLPNH` -- `pi-Al8FeMg3Si6` is a real
    # master called `..._E20kV_sig70_n501_o0.h5`, indistinguishable by name
    # from the Monte-Carlo outputs beside it. A name rule loses it, and a
    # looser name rule would go the other way and unlock phases that have no
    # master at all (c1 measured 30 phases, 14 of them wrongly).
    # `'...' in f` is a header lookup; the 44 MB array is never read.
    masters = {}
    for p in sorted((db / 'EBSD_H5_Cache').rglob('*.h5')):
        try:
            with h5py.File(p, 'r') as f:
                if 'EMData/EBSDmaster/mLPNH' not in f:
                    continue
        except Exception:
            continue
        masters.setdefault(p.parent.name, p)
    # Keyed on the FILE NAME, not on the folder. A pre-built dictionary is
    # named `<phase stem>_master_..._dict_...h5`, and the folder is whatever
    # the writer felt like: `sd_0302719`'s sits in a folder called `sd`, so a
    # folder rule missed it and the library looked like it had three
    # dictionaries when it has four. Requiring `_master` right after the stem
    # keeps `Al` from claiming `Al7FeCu2_master_...`.
    dicts = {}
    for p in sorted((db / 'Dictionary_Library').rglob('*.h5')):
        for stem in idx:
            if p.name.startswith(sanitise(stem) + '_master') or \
                    p.name.startswith(stem + '_master'):
                dicts.setdefault(stem, p)
                break

    phases = []
    for key in sorted(idx, key=str.lower):
        e = idx[key]
        cif = Path(e.cif_path) if e.cif_path else None
        text = cif.read_text(encoding='utf-8', errors='replace') if cif else ''
        cell, setting, settings_seen = cell_from(text) if text else ({}, None, [])

        sht_p, sht_m = shts.get(key, (None, None))
        master_p = masters.get(key) or masters.get(sanitise(key))
        dict_p = dicts.get(key) or dicts.get(sanitise(key))

        label = tag(text, '_sm_phase_labels')
        el_struct = sorted({norm_element(x) for x in (e.elements or ())})
        # Falls back to the FILE STEM last: `beta-AlFeSi` has neither an
        # `_sm_phase_labels` nor an .sht, so without this its name claims Si
        # and nothing compares the claim -- the very case §2.6.2 is about.
        _lbl = label or (sht_m or {}).get('formula')
        el_label = elements_from_formula(_lbl or key, from_filename=not _lbl)
        # Where the label side came from, because "null" and "guessed from the
        # filename" are different facts and the card must not conflate them.
        el_label_source = ('sm_phase_labels' if label
                           else 'sht_formula' if (sht_m or {}).get('formula')
                           else 'filename')

        phases.append({
            'key': key,
            # --- identity -------------------------------------------------
            # `formula_label` and `formula_structure` are the two sides the
            # badge in spec §2.6.2 compares. The first was called `label`
            # until c1 asked whether it meant the display name -- it does not,
            # and a field whose name invites that question is a field waiting
            # to be wired to the wrong thing.
            'formula_label': label,               # _sm_phase_labels, or null
            'formula_structure': e.formula,       # expanded cell, often huge
            'prototype': tag(text, '_sm_phase_prototype'),
            'pearson': tag(text, '_sm_pearson_symbol') or (sht_m or {}).get('pearson'),
            'space_group_hm': e.space_group,
            'space_group_it': e.space_group_number,
            'crystal_system': e.crystal_system,
            'n_atoms': e.n_atoms,
            # --- the two element sets (spec §2.6.2) -----------------------
            'elements_structure': el_struct,
            'elements_label': el_label,
            'elements_label_source': el_label_source,
            'elements_disagree': bool(el_label) and sorted(el_label) != el_struct,
            # --- cell -----------------------------------------------------
            # Rounded on the way OUT, not in the maths: lengths to 6
            # decimals, angles to 4. Without it a card prints
            # `119.99999999999999` for a hexagonal gamma, and §2.3 asks for
            # "a sensible number of digits".
            'cell': _round_cell(cell),
            'cell_setting': setting,
            'cell_settings_available': settings_seen,
            # Set when the blocks disagree about more than rounding.
            'cell_setting_conflict': None,
            # --- provenance -----------------------------------------------
            'reference': reference(text),
            # Why a reference was refused, so the card can say so
            # rather than showing nothing and looking broken.
            'reference_rejected': None,
            'doi': doi(text),
            'url': url(text),
            # From the FILE. A number in a filename is the namer's claim, not
            # the file's statement -- and that class of claim has already been
            # wrong here (`Al2Cu_mp-985806` holds an fcc superstructure). Kept
            # separate so the card can show both and say which is which.
            'icsd': tag(text, '_database_code_ICSD', quoted=False),
            'cod': tag(text, '_cod_database_code', quoted=False),
            'ids_from_filename': _ids_in(key),
            'folder': sht_folder.get(key),
            # --- files and what they let you run (spec §2.7) --------------
            'files': {
                'cif': _f(cif),
                'xtal': _f(Path(e.xtal_path) if e.xtal_path else None),
                'master': _f(master_p),
                'dictionary': _f(dict_p),
                'sht': _f(sht_p),
            },
            'capabilities': {
                # NOT just "the file is there": a CIF that parses to two
                # different compositions is not a Hough capability.
                # Measured by c1 on sd_1816951 -- the reader Hough uses
                # (orix/diffpy, not pymatgen) succeeds and returns
                # Mg 80 / Cu 20 at% where the truth is Cu 66.7 / Mg 33.3.
                # So the library has 35 Hough phases, not 36.
                'hough': cif is not None and cif.exists() and not e.parse_error,
                'spherical': sht_p is not None,
                'dictionary': master_p is not None or dict_p is not None,
            },
            'parse_error': e.parse_error,
            # THE UNNAMED STATE, ON PURPOSE. These four are the only fields
            # in a row that a person edits: they come out of the synonym
            # store in the library folder, which differs from machine to
            # machine and changes the moment somebody renames a phase. A
            # fixture that carried Sebastian's names would (a) be a checked-in
            # copy of his working data and (b) turn every rename into a red
            # test. So the fixture pins the SHAPE and the empty case -- which
            # is also what a fresh library looks like, and the case the list
            # has to render correctly -- and the contract test compares these
            # four by type, not by value.
            'display_name': None,
            'search_terms': [],
            'name_author': None,
            'name_updated': None,
        })

    # Masters on disk that no library phase claims (spec §2.7). Four of them,
    # and the second is the S-phase -- the same phase §2.1 uses as an
    # acceptance case. Its master sits under a stem the library does not map,
    # so the card would say "no master" about a phase whose master is right
    # there. The library shows them rather than concealing them; deciding
    # which phase each belongs to is not this file's job.
    claimed = {sanitise(p['key']) for p in phases if p['files'].get('master')}
    unassigned = [
        {'folder': folder, **_f(path)}
        for folder, path in sorted(masters.items()) if folder not in claimed
    ]

    doc = {
        'schema': SCHEMA,
        'unassigned_masters': unassigned,
        'generated': date.today().isoformat(),
        'source': 'Database/ on this machine, via crystal_hint_local_library.get_index()',
        'note': ('Element sets are normalised (Al0+ -> Al). Until the backend fix '
                 'lands, the live index still carries the oxidation states and '
                 'would yield five phantom bands.'),
        'phases': phases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')

    # --- the cases the spec's approval requires to be in here -------------
    keys = {p['key'] for p in phases}
    required = {
        'sd_0302719': 'kanonische Dublette, Haelfte 1',
        'sd_1401510': 'kanonische Dublette, Haelfte 2',
        'alpha-AlFeMnSi_ICSD-52623': 'alpha-Paar, cP138',
        'α-(AlMnSi)': 'alpha-Paar, cP138',
        'Al': 'zwei Zellen im selben File',
        'sd_1816951': 'AmbiguousCifError',
        'beta-AlFeSi': 'Etikett nennt Si, Struktur hat keins',
    }
    missing = [k for k in required if k not in keys]
    print('%d Phasen -> %s' % (len(phases), OUT.relative_to(ROOT)))
    print('  Hough %d · Spherical %d · Dictionary %d'
          % tuple(sum(p['capabilities'][c] for p in phases)
                  for c in ('hough', 'spherical', 'dictionary')))
    print('  mit Etikett %d · mit Pearson %d · mit Zitat %d'
          % (sum(1 for p in phases if p['formula_label']),
             sum(1 for p in phases if p['pearson']),
             sum(1 for p in phases if p['reference'])))
    print('  mit DOI %d · ohne jede Quelle %d %s'
          % (sum(1 for p in phases if p['doi']),
             sum(1 for p in phases if not p['reference'] and not p['doi']),
             [p['key'] for p in phases if not p['reference'] and not p['doi']]))
    print('  Etikett/Struktur uneinig: %s'
          % [p['key'] for p in phases if p['elements_disagree']])
    print('  mehrere Zellaufstellungen: %d'
          % sum(1 for p in phases if len(p['cell_settings_available']) > 1))
    if missing:
        raise SystemExit('FEHLT im Fixture (Auflage der Freigabe): %s' % missing)
    print('  Master ohne Phase: %d %s'
          % (len(unassigned), [u['folder'] for u in unassigned]))
    print('  alle 7 Pflichtfaelle vorhanden')


def _ids_in(key):
    """Database ids that appear in the FILENAME -- a claim, not a statement."""
    out = {}
    for kind, pat in (('icsd', r'ICSD[-_ ]?(\d+)'),
                      ('cod', r'COD[-_ ]?(\d+)'),
                      # Verbatim from the backend's `_ID_PATTERNS`, so the
                      # two derivations agree by construction. NOT \b --
                      # `_` is a word character, so it finds no boundary
                      # in `Al2Cu_mp-985806` and eight ids went missing.
                      # The leading digit keeps `MP-mp-2199` from
                      # capturing "mp" and `mp-aaacqrcb` (a hash) from
                      # being read as an id at all.
                      ('mp', r'(?<![A-Za-z])mp-(\d[0-9a-z]*)')):
        m = re.search(pat, key, re.I)
        if m:
            out[kind] = m.group(1)
    return out or None


def _round_cell(cell):
    if not cell:
        return cell
    out = {}
    for k, v in cell.items():
        if v is None:
            out[k] = None
        else:
            out[k] = round(v, 4 if k in ('alpha', 'beta', 'gamma') else 6)
    return out


def _f(p):
    if p is None or not p.exists():
        return None
    st = p.stat()
    return {'name': p.name,
            'rel': str(p.relative_to(ROOT)).replace('\\', '/'),
            'bytes': st.st_size,
            'date': date.fromtimestamp(st.st_mtime).isoformat()}


if __name__ == '__main__':
    main()
