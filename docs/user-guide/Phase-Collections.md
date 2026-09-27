# Phase Collections

## What it is

A phase collection is a named list of phases from your crystal library —
"Matrix", "Impurities", "Al-Si system", whatever grouping makes sense for your
work. Once you have collections, every phase picker in the app (the Phase
Tester, Indexing, the EDS phase map) can be narrowed to one collection instead
of showing the whole library, which on a shared machine can easily be thirty
or more entries, several named things like `sd_1802610`.

A collection is bookkeeping, nothing else. It is a small JSON file that lists
which phases belong together and in what order. **Filing a phase into a
collection, moving it, hiding a collection, or deleting one never touches the
CIF, XTAL, SHT, master, or Monte-Carlo files themselves.** Those live in the
library exactly as before; a collection only changes which of them a picker
currently offers you. "Remove from this collection" and "delete this file"
are two different actions with two different buttons, and the collection
tools only ever do the first.

## One home each — with one deliberate exception

An ordinary collection is a folder: a phase you file into it leaves whatever
other ordinary collection it was in. Moving `Al` from "Matrix" into
"Impurities" takes it *out of* "Matrix" — there is no phase filed under two
ordinary collections at once. That is by design: the whole point is that a
collection answers "what belongs together", and a phase belongs to one group
at a time.

There is exactly one collection that breaks this rule: the **working set**
(shown with a star, created once from the "Manage collections…" dialog).
Filing a phase into the working set is purely additive — it never removes the
phase from its ordinary home, and filing that phase into an ordinary
collection afterwards never removes it from the working set either. The
working set is meant to sit across your folders: "the handful of phases I'm
actually testing today", independent of which folder each one normally lives
in.

**This is the one thing people get wrong**, so to say it plainly a second
time: filing into a named folder is a *move*; starring into the working set
is a *copy of the membership*, not a move. If you star a phase and then
wonder why it disappeared from "Matrix" — it didn't; starring never does
that. If you drag a phase into "Impurities" and then wonder why it is no
longer in "Matrix" — it moved, because that is what a folder does.

Collections nest one level deep: a top-level collection can have children,
shown indented under it everywhere, but a child cannot have children of its
own.

## Filing phases

Filing happens in the **Database Browser**. Select one or more rows and use
the "Move to collection…" control that appears once something is selected.

Only rows from the **CIF** tab (or the **XTAL** tab, when a phase's XTAL was
converted from a CIF of the same name) can be filed. That is deliberate: a
phase *is* its crystal structure, and the CIF (or its XTAL) is that
structure. An `.sht`, a master `.h5`, a Monte-Carlo `.h5`, and a dictionary
`.h5` are all *derived* from a phase, not phases in their own right, and their
filenames carry simulation parameters that do not match any library key. If
you select one of those and nothing selected can be filed, the control
explains why instead of silently doing nothing: *"Not a phase — file the CIF
(or XTAL); derived files (SHT, master, MC, dictionary) follow automatically."*
Filing the CIF is enough — a picker that needs the `.sht` or the master finds
it by matching against the phase, not against a second filing step.

Moving a selection into an *ordinary* collection is a real move (see above —
it leaves any other ordinary collection the phase was filed in). Moving it
into the working set only adds it.

## Hiding a collection

A collection is hidden from the eye icon on its group header in the Database
Browser (the same header that shows the collection's name and member count
above its files). **Hiding only stops a collection from being offered** in
the pickers (the toolbar, the Phase Tester, Indexing, the EDS phase map); it
does not delete the collection or touch a single file, and the Database
Browser itself keeps showing the hidden collection's group, just dimmed.

Hiding a parent hides its children too, even if a child's own name was never
added to the hidden list — a hidden folder takes everything filed under it
out of view with it.

There is always a way back: open the toolbar's collection picker, and as long
as anything is hidden its list starts with a standing notice — *"N
collection(s) hidden — show them"* — that stays there rather than fading like
a toast. Clicking it un-hides everything at once. To un-hide one collection
on its own without touching the rest, click the same eye icon again in the
Database Browser.

## Why the same collection shows a different count in different places

Open a collection with, say, 18 members in three different tools and you can
get three different "usable" counts, and that is normal, not a bug:

- The **Phase Tester** renders a simulated pattern for comparison, so it can
  only test a phase that has an **`.sht` master**. Eighteen members with only
  twelve simulated masters means "12 of 18 phases · Collection Name" and a
  note that six cannot be used here.
- **Indexing** needs a different file depending on the method you picked: a
  **CIF** for Hough, an **`.sht`** for Spherical, a master **`.h5`** for
  Dictionary. The same collection can therefore show a different usable count
  just from switching the method dropdown, with nothing else changed. The
  "Use collection: *name*" button fills the phase list from whichever files
  the chosen method can actually use, and logs which members it had to skip
  and why.
- The **EDS phase map** matches phases by chemistry, so all it needs is the
  phase's own **CIF** — the same requirement Hough has. A member missing
  from the library entirely (not just missing a derived file) is the only
  thing that shrinks the count there.

Every one of these views also has an escape hatch: a link back to "show all
phases" that is never more than one click away, so narrowing to a collection
never becomes a filter you cannot get out of.

A count is an offer, not a promise of a result. "18 of 18 phases usable
here" on the EDS panel means all eighteen have a CIF and are on the table for
classification — it does not mean any of them can win. Classification scores
every offered phase against the measured chemistry at each pixel, and a
phase whose defining elements were never acquired in this scan is vetoed or
scored down there, per pixel, regardless of how many phases the count above
said were available. A full count next to zero pixels of a given phase is
not a contradiction; it means that phase's elements were not in what you
measured.

## Order is stored, and it can change the answer

A collection's member order is saved, and it is not just cosmetic:

- **Hough** stops checking further phases once a phase's pattern match is
  good enough — `nband_earlyexit` in PyEBSDIndex. A phase earlier in your
  list can therefore "win" and prevent a later phase from ever being tried on
  a pixel where both would have matched.
- **Spherical and Dictionary** number the phases in their results by the
  position of each file in the list you handed over — phase 1 is whichever
  phase came first. Reorder the collection and the *same run* can label its
  phases differently.

Reorder members with the up/down arrows in "Manage collections…". If a run's
phase numbering or its pixel-by-pixel outcome looks different from a previous
run on the same phases, check whether the collection's order changed before
looking anywhere else.

## Suggested collections

"Suggest collections" (in "Manage collections…") looks at the phases actually
present in *your* library right now and proposes a few groupings — pure
metals and the matrix phase, Al intermetallics, Mg phases, Zn phases,
carbides/nitrides — built from simple element rules. Nothing is written to
disk until you tick the ones you want and confirm; a name that already exists
is skipped and reported, never overwritten.

These suggestions ship as *code*, not as files — an empty `Database/` has
nothing to suggest from, so there is nothing to point at phases you don't
have. Once you accept a suggestion it becomes an ordinary collection like any
other: rename it, delete it, move phases in and out, or nest another
collection under it, exactly as if you had typed the name yourself.

## Giving a collection to a colleague

A collection is one file: `Database/Collections/<name>.json`. To share it,
copy that file to the same folder on the other machine (by email, a shared
drive, whatever you'd use for any other file). No restart needed — the folder
is rescanned on every request, so a dropped-in file shows up the next time a
picker refreshes. Don't copy `Database/Collections/_local.json` — that one
holds which collection is *active* and which are *hidden* on this particular
installation, and is meant to stay behind.

On the other end, a member key the receiving library doesn't have is **never
silently dropped**. It stays in the collection, shown struck through in
"Manage collections…" with a small "not in library" label, and the
collection's header counts how many members are missing. The collection is
still fully usable for whatever phases the other person *does* have; it just
tells them honestly what it expected and couldn't find, so a copied
collection can never quietly shrink a run without saying so.

## Tips & notes

- **Removing a phase from a collection never deletes a file.** The phase
  stays in the library, and stays in any other collection (including the
  working set) it was independently filed into.
- **Deleting a collection never deletes a file either.** Its phases simply
  become unfiled again; a collection with children promotes them to the top
  level rather than deleting them too.
- **The working set is the only collection you cannot delete** from "Manage
  collections…" — it has no special status in the data itself (it is a plain
  `exclusive: false` collection), the tool simply never offers to remove it.
- **Renaming a CIF file breaks the pointer.** A collection remembers a
  library filename stem, not file content, so renaming the underlying CIF
  makes that member show up as "not in library" the same way a missing one
  from a colleague's machine does — the fix is to re-file the phase under its
  new name.
