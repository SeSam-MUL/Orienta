# Reporting a problem

## What it does

When something goes wrong — a module fails to load, a run crashes, a map looks
wrong — Orienta can package everything needed to diagnose it into a single
**problem report**: a `.zip` file you attach to your message.

A screenshot shows the symptom. The report additionally contains the cause:
which version you run, what the program did in the minutes before, the exact
error text (which usually never reaches the screen), and your own description of
what you expected instead.

The report itself contains **no measurement data** — no patterns, no maps, no
EDS spectra. The one exception is the optional screenshot: it shows whatever was
on screen, which may include a map or a pattern, so you can leave it out.
Nothing is sent anywhere automatically: you get a file, and you decide who
receives it.

## When to use it

Use it whenever you would otherwise send a screenshot:

- a page shows **"This module crashed"**, or the window stays blank,
- an operation stops with a red error message,
- indexing, simulation or export fails or never finishes,
- a result is **wrong rather than broken** — the map looks implausible, colours
  are off, phases are misassigned. These are the cases where the report helps
  most, because there is no error message to screenshot.

## How to use it — step by step

1. Open the dialog — **"⚠ Report a problem"** sits in the toolbar at the top
   of every page, so you can report from where the problem happened. It is
   also in **Settings → About Orienta**, and on the crash screen next to
   *Retry* and *Reload App*.
2. **Describe what happened** in the text box. Two sentences are enough, but
   these two make the difference:
   - what you did (which file, which page, which action), and
   - what you expected instead of what you got.

   Example: *"Loaded SampleB, started Hough indexing on the whole map — it
   stopped at 40 % and the phase map stayed empty. I expected a finished map."*
3. If a picture of the window was captured, decide whether to include it. It
   helps a lot, but check that nothing confidential is on screen first.
4. Click **Create report**. Your browser or a save dialog asks where to put the
   `.zip`.
5. Send it. If the project has an issue tracker, **"Open a GitHub issue"**
   opens a new issue with your description, version and the preceding events
   already filled in — drag the `.zip` into the issue before submitting.
   Otherwise attach the file to your e-mail or ticket.

If the app is so broken that the dialog will not open, the log files are still
on disk — see *Where the logs live* below — and you can attach them manually.

## Inputs & outputs

- **Input:** your description (optional — an empty report still carries the logs).
- **Output:** `orienta-problem-report-YYYY-MM-DD.zip`, containing:

  | File | Content |
  |---|---|
  | `report.txt` | Your description, the version, the error id, plus what the program did in the minutes before (pages visited, requests made, files loaded, errors shown). |
  | `screenshot.png` | A picture of the window at the moment you opened the dialog (only if you left it included, and only in the desktop app). |
  | `info.json` | App version and branch, Python and platform, versions of the scientific packages, GPU/VRAM, and which files were loaded. |
  | `logs/orienta.log` | The application log, including previous sessions (rotated). |
  | `logs/backend-console.log` | Raw output of the analysis backend — this is where startup failures appear that never reach the interface. |
  | `sim_logs/*.log` | The most recent simulation job logs. |

## Where the logs live

Orienta writes its logs into a `logs/` folder inside the program directory:

- **`logs/orienta.log`** — the application log. Each session starts with a banner
  naming the version, so you can tell sessions apart. Older logs are kept as
  `orienta.log.1`, `.2`, `.3`.
- **`logs/backend-console.log`** — the raw console output of the backend process.

The files are plain text and can be opened in any editor. They contain file
paths and phase names, but no measurement data.

## The error id

Reports carry a short **error id** such as `a1b2c3d4`, shown in the dialog and
in `report.txt`. It is derived from the fault itself, so the *same* problem
always produces the *same* id — on your machine and on everyone else's. If two
people report an id that matches, it is one problem, not two. Quote it when you
follow up.

An id appears only when the app actually caught an error. A report about a
result that merely looks wrong has none, which is normal.

## Finding your version

**Settings → About Orienta** shows the version, for example `v0.2.1`, or
`v0.2.1+7 (a1b2c3d)` if your copy is a few changes past that release. Always
include it in a bug report: it identifies exactly which build you are running.
The same line is in `report.txt`, in `info.json` and at the top of every log
session.

## Tips & notes

- **Report as soon as it happens.** The trail of preceding events is kept for the
  current session only; restarting the app or reloading the window clears it.
  The log files survive restarts, but the "what happened just before" part is
  most complete right after the problem.
- **"It's wrong" beats "it's broken" for detail.** When a result is merely
  implausible, describe what you expected — the logs cannot know what the map
  *should* have looked like.
- **One problem per report.** Two unrelated issues in one description are much
  harder to act on than two reports.
- **Nothing is transmitted automatically.** Orienta never sends reports, logs or
  data by itself. The zip is written to your disk and stays there until you
  send it.
- **Reports are safe to share, but check anyway.** They contain no measurement
  data, though file paths (and therefore folder and sample names) do appear, and
  an included screenshot shows whatever was on screen. If that matters to you,
  open `report.txt` and `info.json` before sending — they are readable text —
  and untick the screenshot.
