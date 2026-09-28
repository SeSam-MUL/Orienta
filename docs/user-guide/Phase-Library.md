# Phase Library

## What it is

The Phase Library is the page that answers three questions about the crystal
structures on this machine:

- **Which phase is this?** — not just its filename, but its cell, its
  composition, the paper it came from.
- **Can I index with it?** — Hough, Spherical, Dictionary: which of the three
  this phase actually has the files for.
- **Where is its file?** — one click to the Database Browser, with the file
  already picked out.

It exists because the answer used to be "open the Database Browser and read
filenames". A first-time user asked to "use the Al phases" tried several
search terms, found none of them worked, and said he would sooner ask a
colleague for a list of filenames. The library on this machine holds 36
phases under names like `sd_0302719` and
`Al2Cu_mp-985806_conventional_standard`.

**Nothing on this page changes a crystal file.** Naming a phase, filing it
into a group, deleting a group — none of it touches a CIF, XTAL, SHT, master
or Monte-Carlo file. Those stay exactly where they are; the library only
changes what you see and what a picker offers you.

---

## Finding a phase

Type anything you would say out loud. The search understands four different
kinds of question, and tells you which one it answered:

| You type | It reads this as |
|---|---|
| `Al Fe Si` | a combination of elements — phases containing all three |
| `Al2CuMg`, `Al4Cu2Mg2` | a formula — and these two are the same compound |
| `Fm-3m`, `Im-3`, `225` | a space group, in Hermann-Mauguin or by its number |
| `Barlock`, `alpha`, `sd_03` | free text — names, citations, filenames |

Every row says **why** it came back — *Element*, *System* (more than one
element), *Formula*, *Space group*, *Pearson*, *Structure type*, *Synonym*,
*Name*, *File name*, *Reference*, or *by name only*. A hit with no visible
reason reads like a bug, and a tester reported it as one. Typing
`Al Fe Si` gives *System*; typing `Barlock` gives *Reference*.

Every row also carries three letters — **H S D** — for Hough, Spherical and
Dictionary. A capital means this phase has the files that method needs; a
small letter means it does not. `H s d` is a phase you can only index with
Hough. It is the fact that most often decides which of two similar phases
you use, so it is on the row rather than two clicks away.

Hits are split into three blocks that never mix:

- **what the phase IS** — the identity hits, shown first and in bands;
- **what it is LIKE** — a structure-type (prototype) match. `Cu` is the
  prototype of both pure aluminium and pure nickel, so a prototype hit is
  not a phase called `Cu`;
- **where it is MENTIONED** — a hit in the citation or the file identifiers.

Short queries match whole words. Typing `Al` does not return a phase called
"Almandin-Typ" that contains no aluminium.

### Two views

**Systems** groups the phases into a band per element: `Al`, `Fe`, `Si`, and
so on. A phase stands in one band per element it contains, so the header
reads *36 of 36 entries · 93 placements* — the same 36 phases, seen 93
times.
`Al2CuMg` is in three bands, and ticking it in one ticks it in all three,
because it is one phase.

**List** is the same hits without the bands.

### Narrowing it down

The chips on the left are filters, and **a chip's number is what clicking it
gives you**, measured against what is on screen right now — not against the
whole library. Elements combine with AND: `Al` + `Fe` means both.

Under *Can be indexed with* are the three methods. These count phases that
have the files that method needs:

- **Hough** needs a readable structure file (CIF).
- **Spherical** needs a simulated `.sht`.
- **Dictionary** needs a master pattern or a pre-built dictionary.

The line underneath — *N simulated masters in the folder are not matched to
any phase here* — is not an error. It means the folder holds masters the
library cannot attach to a phase; nothing is missing from the library because
of them.

When a filter leaves nothing, the page says **which** filter is biting and
offers to clear it. From a screen with nothing on it you otherwise cannot
see which one to undo.

---

## The phase card

Click a phase's name and the card opens. It lives on this page; Indexing
and Crystal Hint have their own, smaller summaries.

- **The identity line** — the phase's own name if it has been given one,
  the formula, the Pearson symbol, the space group with its International
  Tables number, the citation or the DOI, and the key (the library's own
  name for the phase). The key is always there, whatever the phase is
  called. The lattice parameters are **not** on this line; they are in
  *Lattice* below, all six of them.
- **Composition, twice**: what the structure contains, and what the label
  claims. They disagree for two phases in this library, and the card marks
  it rather than picking a side. One CIF called `beta-AlFeSi` has no silicon
  site in it at all.
- **All six lattice constants**, plus the setting the file uses and how many
  atoms are in the cell.
- **Five file slots** — CIF, XTAL, master, dictionary, SHT — each with a tick
  or a cross, the file's name, its date, and where it sits relative to the
  project folder — never an absolute path, because one of those once
  reached a manuscript. **"Show in the database browser"** opens that file in the
  browser with the right tab already selected and the name in the search box.
- **The parameters the master and the `.sht` were simulated with** — beam
  energy, dmin, and the electron count where the file records one (14 of
  the 16 masters in this library do not) — because a master simulated at a
  different energy is a different master.
- **The citation**, where the file carries one, with its DOI.

`sd_0302719` and `sd_1401510` are a good example of why the card exists. In
a list they are indistinguishable: the same composition on the label
(`Mn0.5Fe0.5Al5Si0.68`), the same space group `Im-3`, the same Pearson
symbol `cI168`, lattice parameters 12.50 Å and 12.56 Å. The cards tell them
apart, and it is the **citation** that does it — Barlock & Mondolfo 1975
against Cooper 1967. Two structure models of one phase, measured twice,
thirty years apart. Which one you index with is a decision, and the card is
where you can make it — and there is a second discriminator on it, one the
label hides: both labels read `…Si0.68`, but Cooper's structure has **no
silicon site**, only Al, Fe and Mn. The card marks that; the list cannot.

When another entry in the library has **the same space group and the same
compound**, the card says so, under the name, and lists what differs —
which methods it can be used with, what its structure contains, which paper
it comes from, its lattice parameter. It does not tell you which to pick:
that is a crystallographic judgement, and a citation year is not one. On
this library it fires twice, on the pair above and on two `Mg17Al12`
entries from the same Materials Project id.

If the phase is in any of your groups, the card names them. With tags a
phase can be in three places at once, and this is the only screen that
answers "which ones".

*What these words mean* at the bottom of the filter column explains Pearson
symbols, prototypes, IT numbers, masters and `.sht` files in plain
language.

---

## Naming a phase

Your group has its own names for these phases, and the files do not. Click
**Give this phase a name…** on the card.

- **Name** is what the list should read — `α-Al(Fe,Mn)Si`, `die Alpha`.
- **Also find by** is for spellings nobody wants to *see* but everybody
  types: `s phase, s-phase, Al2CuMg`. Separate them with commas.
- **Your name** is recorded with the change, because the library is shared.
  You type it once per session.

**A name stands in front of the file name, never instead of it.** After you
name `sd_1814127` "S-Phase", searching `sd_1814127` still finds it. That
matters: the reason this page exists is that people could not find the CIF.

Names are stored with the library, so a colleague opening the same folder
sees them. Clear the name field and save to take a name away again.

---

## Groups

A group is your own arrangement of the library: *Al-Fe intermetallics*, *the
five I care about this week*, *what we published in March*. Groups live with
the library in `Database/Collections`, so everyone who opens that folder sees
them.

### Letting the library propose an arrangement

**Suggest groups from the library…**, under the group list, looks at what
is in the library and proposes a set of groups — on a typical Al library:
*Matrix and pure metals*, *Intermetallics in Al*, *Mg phases*, *Zn phases*.

**Nothing is created until you accept.** Everything starts ticked; untick
what you do not want and press *Create N groups*. Press the phase count
beside a proposal to see exactly which phases it would hold — one of them
is often most of the library, and that is worth looking at before you
agree. Underneath, the box says how many phases the whole proposal leaves
out; those simply stay unfiled.

A name that is already taken is left alone and named, rather than silently
skipped.

### Filing a phase, four ways

1. **Drag a row onto a group.** This is the main way.
2. **Alt-drag** — file it *here and nowhere else*. (Alt, and not Ctrl or
   Shift, because Windows already uses those two for copy and move while
   dragging.)
3. **Tick some phases, then press the `+N` button** on a group. No dragging,
   and it works from the keyboard.
4. **The ⋯ menu** on a group does the same, by name, with *move* beside it.

**Dragging ADDS.** A phase can be in as many groups as you like — `Al2CuMg`
can be in *Al systems* and in *this week* at the same time. Taking it out of
one is something you say explicitly, and it never removes it from the others.
Only Alt-drag, and the *move* entry in the menu, take it out of everywhere
else.

> **If you used Phase Collections before, this is the one thing that
> changed.** A collection used to be a folder: filing a phase into one took
> it out of its previous one, and only the starred "working set" was
> additive. A group is a tag. Everything is additive unless you ask for a
> move.

### Seeing and changing what is in a group

**Click a group's name** and it opens, listing its phases. Each one has an
**×** that takes it out of that group and out of no other. The ⋯ menu does
the same for a whole selection.

Taking a phase out of a group never touches the phase, its files, or any
other group it is in.

### Subgroups

A group can sit **under** another one, one level deep: *Al-Fe-Si* under *Al
systems*. Drag a group onto another group, or use ⋯ → **Subgroup of…**
and pick from the list that appears; **Top level** is the first choice in
that list, and brings a child back out. A child is drawn indented under
its parent.

One level only: a group that already has children cannot itself become a
child, and the menu is disabled rather than offering something that will
be refused. (It reads *Subgroup of… (nowhere to put it)*, which is also
what it says when there is simply no other group.)

Alt is read **when you let go**, not when you pick the row up. So you can
start an ordinary drag, change your mind halfway, and press Alt before
releasing — the hint under the cursor changes with it.

Deleting a group **keeps its children** and moves them to the top level.
Deleting a container is not a statement about what was in it.

### What the counts mean

A group's count is measured against **what is on screen**. With no filter it
reads *12 phases*; with a filter on it reads *3 of 12 shown*. A group that a
filter has emptied reads *0 of 12* and an actually empty group reads
*0 phases* — those are different sentences on purpose.

While you are dragging, the count is replaced by what the drop will do:
*add 3*, or *move 3 here*. A drop that would change nothing is outlined
differently from one that would.

### Undo

**Undo** appears next to *+ New group* as soon as there is something to
take back, and it goes back a step at a time. A change that the library
refused is taken back on your screen and nowhere else — undoing something
that never happened would act on whatever is there now.

One thing undo does not put back everywhere is a phase's *position*
inside a group. On screen the order is restored exactly; in the library
file the phase is appended, so after the next read it sits at the end.

### When a change does not reach the library

The library folder can be read-only, on a share that is not mounted, or
being edited by somebody else. When a change cannot be written, it stays on
your screen and the panel lists it, with the group and the reason, and
keeps listing it — a later change that works does not make the earlier
failure go away; if more than three are waiting, the panel names the last
three and says how many it is not showing. **Read the library again**
re-reads the folder and is the only thing that clears the list, because it
is the only thing that makes the screen true again — and it also empties
the undo history, since those steps describe a screen that no longer
exists.

### Using a group for a run

The ⋯ menu's **Use this group for the next run** makes that group the
active one and takes you to Indexing. The phase tester reads the same
choice. Which of the three methods each phase can actually be used with is
worked out there, so a phase with no master says so rather than failing
later.

While a group is in force, the toolbar reads **Group: <name>** on every
page. That line is the only thing telling you a run is narrowed, so it is
always there — and if the group it names cannot be found (renamed or
deleted, perhaps on another machine) it turns red, no phases are offered,
and Start is blocked with that reason rather than quietly running over the
whole library.

**Use the whole library**, beside *Undo*, ends the narrowing. The group's
row is also marked in the list, so you can see which one is in force
without looking at the toolbar.

---

## Where everything lives

| What | Where |
|---|---|
| Groups | `Database/Collections/*.json`, one file per group |
| Names and search terms | `Database/Synonyms/synonyms.json` |
| Which group is active, and which are hidden | `Database/Collections/_local.json` |

The **groups and the names** travel with the library — onto a network
share, into a backup, onto a colleague's machine. Copy the folder and they
come with it.

`_local.json` is the exception, on purpose: which group *you* have active
and which *you* have hidden are statements about your sitting, not about
the library. Mail somebody a group and it must not arrive hidden at their
end.

If Orienta cannot write to that folder — a read-only share, a disconnected
drive — the change stays on your screen and a line underneath says it was not
saved. It is not silently dropped, but it is also not kept: reload and it is
gone.

---

## Known limits

- **Undo does not restore a phase's position** in the library file; it is
  appended, so it moves to the end after the next read. On screen the
  order is restored.
- **A move takes the phase out of every other group**, including one a
  colleague made since you last read the library — which this screen
  cannot know about, and cannot put back if you undo.
- **Groups are addressed by name where one route is concerned.** If
  somebody renames a group between your last read and your next change,
  a change to nesting can land on the wrong group or on none.
- **The database browser shows one row per file**, so a phase in several
  groups appears under the first of them. Its card lists all of them.
- **The change history is not read back** after a reload. It is kept for the
  session so undo can say what it is undoing.
- **The library index is slow the first time** after Orienta starts — up to
  a minute on a big library, because it reads every structure file and every
  master header. The page says what it is doing while it waits, and it is
  fast afterwards.
