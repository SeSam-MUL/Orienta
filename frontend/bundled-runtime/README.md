Staging for the runtime package that ships INSIDE the installer.

`scripts/build_release.py` copies `orienta-runtime-<tag>.zip` and its
`.sha256` here before electron-builder runs, and electron-builder copies them
into the application's `resources/` folder.

Why the installer carries its own runtime rather than fetching it:

The original design was a small bootstrap that downloads everything, on the
assumption that the runtime would be large. It is 4.5 MB against a 106 MB
installer — four percent. Paying that gets rid of an entire category of
failure: a second file to copy, a checksum file beside it, a search across
folders that Windows may have moved, and a first screen that talks about
GitHub to someone whose whole task was to put a file somewhere.

The installer still asks GitHub for the NEWEST release. The bundled copy is
what it falls back to, which is also what makes it work on a machine that
cannot reach GitHub at all.

The zips themselves are build products and are not committed.
