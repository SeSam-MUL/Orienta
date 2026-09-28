// Element names for the search, because the spec pins `aluminium -> pure Al`
// as an acceptance criterion and `Al.cif` contains the string "alumin"
// NOWHERE -- measured. Without this the one thing a supervisor says out loud
// ("use the Al phases") cannot be found by name.
//
// English comes from pymatgen's periodic table, so it is the US spelling:
// `aluminum`. The spelling a European actually types is added by hand, which
// is exactly the sort of gap a generated table hides. German is hand-written
// for the elements whose name differs. Adding a language is one entry per
// element, not a project.
export const ELEMENT_NAMES = {
"H": [
"hydrogen",
"wasserstoff"
],
"He": [
"helium"
],
"Li": [
"lithium"
],
"Be": [
"beryllium"
],
"B": [
"boron",
"bor"
],
"C": [
"carbon",
"kohlenstoff"
],
"N": [
"nitrogen",
"stickstoff"
],
"O": [
"oxygen",
"sauerstoff"
],
"F": [
"fluorine",
"fluor"
],
"Ne": [
"neon"
],
"Na": [
"sodium",
"natrium"
],
"Mg": [
"magnesium"
],
"Al": [
"aluminum",
"aluminium"
],
"Si": [
"silicon",
"silizium"
],
"P": [
"phosphorus",
"phosphor"
],
"S": [
"sulfur",
"schwefel",
"sulphur"
],
"Cl": [
"chlorine",
"chlor"
],
"Ar": [
"argon"
],
"K": [
"potassium",
"kalium"
],
"Ca": [
"calcium"
],
"Sc": [
"scandium"
],
"Ti": [
"titanium",
"titan"
],
"V": [
"vanadium"
],
"Cr": [
"chromium",
"chrom"
],
"Mn": [
"manganese",
"mangan"
],
"Fe": [
"iron",
"eisen"
],
"Co": [
"cobalt"
],
"Ni": [
"nickel"
],
"Cu": [
"copper",
"kupfer"
],
"Zn": [
"zinc",
"zink"
],
"Ga": [
"gallium"
],
"Ge": [
"germanium"
],
"As": [
"arsenic",
"arsen"
],
"Se": [
"selenium",
"selen"
],
"Br": [
"bromine",
"brom"
],
"Kr": [
"krypton"
],
"Rb": [
"rubidium"
],
"Sr": [
"strontium"
],
"Y": [
"yttrium"
],
"Zr": [
"zirconium",
"zirkonium"
],
"Nb": [
"niobium",
"niob"
],
"Mo": [
"molybdenum",
"molybdaen"
],
"Tc": [
"technetium"
],
"Ru": [
"ruthenium"
],
"Rh": [
"rhodium"
],
"Pd": [
"palladium"
],
"Ag": [
"silver",
"silber"
],
"Cd": [
"cadmium"
],
"In": [
"indium"
],
"Sn": [
"tin",
"zinn"
],
"Sb": [
"antimony",
"antimon"
],
"Te": [
"tellurium",
"tellur"
],
"I": [
"iodine",
"iod"
],
"Xe": [
"xenon"
],
"Cs": [
"cesium",
"caesium"
],
"Ba": [
"barium"
],
"La": [
"lanthanum",
"lanthan"
],
"Ce": [
"cerium",
"cer"
],
"Pr": [
"praseodymium"
],
"Nd": [
"neodymium",
"neodym"
],
"Pm": [
"promethium"
],
"Sm": [
"samarium"
],
"Eu": [
"europium"
],
"Gd": [
"gadolinium"
],
"Tb": [
"terbium"
],
"Dy": [
"dysprosium"
],
"Ho": [
"holmium"
],
"Er": [
"erbium"
],
"Tm": [
"thulium"
],
"Yb": [
"ytterbium"
],
"Lu": [
"lutetium"
],
"Hf": [
"hafnium"
],
"Ta": [
"tantalum",
"tantal"
],
"W": [
"tungsten",
"wolfram"
],
"Re": [
"rhenium"
],
"Os": [
"osmium"
],
"Ir": [
"iridium"
],
"Pt": [
"platinum",
"platin"
],
"Au": [
"gold"
],
"Hg": [
"mercury",
"quecksilber"
],
"Tl": [
"thallium"
],
"Pb": [
"lead",
"blei"
],
"Bi": [
"bismuth",
"wismut"
],
"Po": [
"polonium"
],
"At": [
"astatine"
],
"Rn": [
"radon"
],
"Fr": [
"francium"
],
"Ra": [
"radium"
],
"Ac": [
"actinium"
],
"Th": [
"thorium"
],
"Pa": [
"protactinium"
],
"U": [
"uranium",
"uran"
],
"Np": [
"neptunium"
],
"Pu": [
"plutonium"
],
"Am": [
"americium"
],
"Cm": [
"curium"
],
"Bk": [
"berkelium"
],
"Cf": [
"californium"
],
"Es": [
"einsteinium"
],
"Fm": [
"fermium"
],
"Md": [
"mendelevium"
],
"No": [
"nobelium"
],
"Lr": [
"lawrencium"
],
"Rf": [
"rutherfordium"
],
"Db": [
"dubnium"
],
"Sg": [
"seaborgium"
],
"Bh": [
"bohrium"
],
"Hs": [
"hassium"
],
"Mt": [
"meitnerium"
],
"Ds": [
"darmstadtium"
],
"Rg": [
"roentgenium"
],
"Cn": [
"copernicium"
],
"Nh": [
"nihonium"
],
"Fl": [
"flerovium"
],
"Mc": [
"moscovium"
],
"Lv": [
"livermorium"
],
"Ts": [
"tennessine"
],
"Og": [
"oganesson"
]
};
