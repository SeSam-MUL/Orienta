# Updating Orienta

## What it does

Orienta checks for a newer **release** a few seconds after it starts. If one
exists, a dialog shows what is new and offers to install it: the new version is
fetched, any changed packages are installed, the interface is rebuilt, and the
app restarts. One click, a few minutes, and you are on the new version.

Only **released versions** are offered — never work in progress — so what you
get has been deliberately published.

## When it appears

- **At start-up**, once, a few seconds in. It never interrupts what you are
  doing and never appears twice for a version you skipped.
- **On demand**, via **Settings → About Orienta → "Check for updates"**.

If the check cannot run — no internet, not signed in — it stays quiet at
start-up. The button in Settings tells you what went wrong.

## How to use it — step by step

1. When the dialog appears, read what is new. It lists the changes for the new
   version straight from its changelog.
2. Choose one:
   - **Install update** — do it now.
   - **Later** — the dialog comes back the next time you start Orienta.
   - **Skip this version** — you will not be asked about *this* version again;
     the next release will still be offered.
3. During the installation, leave the window open. It names each step
   (downloading, installing packages, building the interface) and shows the
   output, so a long step never looks like a hang.
4. When it finishes, click **Restart now**. Orienta closes and comes back on
   the new version.

## If something goes wrong

**Your installation is left working.** The update runs against a snapshot of
both the program files and the built interface, and if any step fails, both are
put back together — you can never end up with new program files and an old
interface. The dialog then shows the error and says the previous version is
still installed.

Common messages:

| Message | What it means |
|---|---|
| Could not sign in to the update server | Your credentials for the repository are not stored. Open a terminal in the Orienta folder and run `git fetch` once, enter your login, then try again. |
| Could not reach the update server | No internet connection, or the server is down. Nothing was changed. |
| Files have local changes | You (or something else) edited files inside the Orienta folder. Updating would overwrite them, so it stopped. Move the changed files elsewhere, or undo the changes, then retry. |
| This installation cannot update itself | See below. |

## Installations that cannot update themselves

If you set Orienta up by **unzipping a downloaded archive**, there is no link
back to the source it came from, so it cannot update in place. Orienta detects
this and tells you instead of offering a button that cannot work.

To get automatic updates in future, install by **cloning** the repository
instead of downloading an archive — see INSTALL.md. Otherwise, download the new
version and repeat the install steps.

## Turning it off

Untick **"Check for updates when Orienta starts"** under Settings → About
Orienta. You can still check manually at any time with the button next to it.

## Tips & notes

- **Nothing is installed without you asking.** The check only reads which
  versions exist; nothing is downloaded or changed until you click *Install
  update*.
- **Rebuilding takes the longest.** Installing packages only happens when they
  actually changed between your version and the new one; the interface is
  always rebuilt, which is usually the bulk of the wait.
- **Long-running work first.** The restart closes the app. If an indexing run
  or a simulation is going, finish it before updating.
- **Your data is untouched.** Updates replace program files only. Measurement
  files, results, the crystal database and your settings are not part of the
  update.
