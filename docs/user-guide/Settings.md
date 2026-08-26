# Settings

## What it does

The Settings page is where you configure the system dependencies and preferences
Orienta needs for its more advanced features. It groups seven sections:

- **Dashboard Background** — pick the visual style of the start page.
- **System Status** — a read-only health check of WSL, EMsoft, EMSphInx, OpenCL,
  GPU memory, and CPU cores, with the recommended compute mode.
- **Install Wizard** — a guided, three-step installer for WSL + a Linux user +
  EMsoft (needed for simulation and spherical indexing on Windows).
- **API Keys** — store and test third-party keys (e.g. Materials Project) used by
  features like Crystal Hint.
- **Manual Paths** — override auto-detection of the EMsoft binary directory and the
  EMSphInx directory.
- **Server Mode** — point Orienta at a shared network database root for phase
  files and verify/create its standard subfolders.
- **About Orienta** — licence notice, the **version** you are running, and the
  **"Report a problem…"** button.

Settings are stored **per machine** in a user-config file outside the project
tree (so they travel with the user, not the repository). The page talks to the
backend `/api/settings/*`, `/api/install/*`, and system-status routes.

## When to use it

Open Settings:

- on **first run / on a new machine**, to install or point Orienta at WSL +
  EMsoft/EMSphInx so simulation and spherical indexing work,
- when a dependency check fails elsewhere and you need to see *what* is missing,
- to add an **API key** before using features that query external databases, and
- to connect to a **shared phase database** on a network drive (Server Mode).

Day-to-day indexing of an existing scan with the Hough or Dictionary methods does
not require any Settings changes.

## How to use it — step by step

### Dashboard Background

1. Choose **Clean**, **Crystal**, or **Fog**. The change applies immediately and
   is remembered locally.

### System Status

2. Read the table: WSL distro, EMsoft path/availability, EMSphInx path/
   availability, OpenCL/GPU, GPU memory, CPU cores, and the recommended mode. Green
   = OK, yellow = warning, red = missing. Click **Re-Check** to refresh (this
   bypasses the cached result so you see the current state after fixing something).

### Install Wizard (WSL + EMsoft)

3. **Step 0 / 1 — WSL:** the wizard reports your platform and WSL state. If WSL is
   missing or needs repair, use the install/repair action and pick a distro
   (Ubuntu 22.04/20.04/24.04 or Debian).
4. **Step 2 — User:** create a Linux user (username + password) inside WSL, or
   reset an existing user's password. EMsoft runs under this user.
5. **Step 3 — EMsoft:** start the EMsoft install. The build log **streams live** to
   the page over a WebSocket so you can watch progress and catch errors.

### API Keys

6. For a provider (e.g. **Materials Project**), paste the key into the masked field
   (use the show/hide button to reveal it), then **Save**. Use **Test** to run a
   live connectivity check against the stored key, or **Clear** to remove it. The
   page shows where the key is stored and only ever displays a masked preview — the
   raw key is never returned by the server.

### Manual Paths

7. If auto-detection fails, set the **EMsoft binary directory** and **EMSphInx
   directory** explicitly and **Save Paths**. These override the detected
   locations. The Save button enables only when you have changed something.

### Server Mode

8. Toggle **Server Mode** on and enter the **Database Root** (a network/shared
   path). Click **Test Connection** to check reachability and report the status of
   the standard subfolders (SHT database, H5 cache, CIF library, XTAL library) with
   file counts. If some subfolders are missing, **Create Missing** creates them.
   Click **Save** to persist.

### About Orienta

9. The section shows the **version** you are running — a date, a short code and
   the branch, e.g. `2026-08-26 (a1b2c3d) · main`. Quote it in any bug report.
10. **"Report a problem…"** opens a dialog that packages your description
    together with the version, environment details and log files into one zip.
    See [Reporting a problem](ReportingProblems.md).

## Inputs & outputs

- **Inputs:** text you type (binary/EMSphInx paths, database root, API keys),
  toggles (server mode), and the dashboard-background choice. The Install Wizard
  additionally needs a WSL username/password.
- **Outputs:**
  - A per-machine **user-config file** holding server config, manual paths, and API
    keys (keys redacted on read). The path is shown in the API Keys section.
  - The dashboard-background preference (stored locally in the browser).
  - Side effects from the Install Wizard: an installed/repaired **WSL** distro, a
    **Linux user**, and a built **EMsoft** toolchain.
  - Newly created server subdirectories (when you use Create Missing).

## Tips & notes

- **Spherical indexing & simulation need WSL + EMsoft/EMSphInx.** On Windows these
  run inside WSL; the Install Wizard exists specifically to set that up. Hough and
  Dictionary indexing do not require it.
- **Settings are per machine, not in the project.** Moving the project folder to
  another PC does not carry your keys/paths/server config — reconfigure on the new
  machine.
- **API keys are write-once over the wire.** The raw key is sent only when you
  click Save; afterwards the UI shows just a masked preview plus a "configured"
  flag. Test never reveals the key.
- **Re-Check bypasses the cache.** System status is cached for a few minutes for
  speed; use Re-Check after installing/fixing a dependency to force a fresh probe.
- **OpenCL/GPU shows as a warning, not an error, when absent.** Orienta still works
  CPU-only; the GPU rows just indicate whether GPU-accelerated paths are available.
- **EMsoft install streams its log.** Step 3 can take a while; watch the live log
  for build errors rather than assuming it hung. The same log is also written to
  `logs/orienta.log`, so a dropped connection no longer loses it.
- **Something went wrong? Send a report, not a screenshot.** "Report a problem…"
  in the About section (and on the crash screen) collects the error text, the
  preceding events and your version into one file —
  see [Reporting a problem](ReportingProblems.md).
