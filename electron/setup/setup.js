/**
 * The setup wizard's renderer.
 *
 * Four screens: what we found -> installing -> (failed | done). It talks to the
 * main process through `window.setupAPI`, exposed by the preload; there is no
 * node integration on this page, because it is the page that runs before
 * anything is trusted.
 *
 * Every string comes from locales.json. Nothing user-visible is written in
 * English here: an installer that fails in a language the user does not read is
 * a dead end with extra steps, and the failure path is exactly where the
 * English-only shortcut is most tempting and least affordable.
 */

const $ = (id) => document.getElementById(id);

const GB = 1024 ** 3;

/** Every language, kept so a switch does not need another round trip. */
let ALL_LOCALES = null;
/** The chosen one; every read goes through `t`. */
let L = null;
let lang = 'en';
/** What the main process told us about why we are here. */
let ctx = { reason: 'first-run', message: '', locale: 'en', home: '' };
/** The last probe, kept so the failure screen can offer a sane fallback. */
let probed = null;
let running = false;

// --------------------------------------------------------------------------
// text
// --------------------------------------------------------------------------

/** Dotted key lookup with {{placeholder}} filling.
 *
 * Returns the KEY when a string is missing, rather than an empty element: a
 * blank line in a wizard looks like a rendering bug, while a visible
 * `errors.pip` is a bug report.
 */
function t(key, vars) {
  let value = L;
  for (const part of String(key).split('.')) {
    value = value && typeof value === 'object' ? value[part] : undefined;
  }
  if (typeof value !== 'string') return key;
  return value.replace(/\{\{(\w+)\}\}/g, (_, name) => (
    vars && name in vars ? String(vars[name]) : `{{${name}}}`));
}

function setText(id, key, vars) { $(id).textContent = t(key, vars); }

function show(section) {
  for (const name of ['choose', 'working', 'failed', 'done']) {
    $(name).hidden = name !== section;
  }
}

// --------------------------------------------------------------------------
// screen 1: what this computer is, and what it should install
// --------------------------------------------------------------------------

const STEPS = ['probe', 'resolve', 'python', 'runtime', 'packages', 'verify', 'done'];

/**
 * The steps THIS install will actually take.
 *
 * `verify` runs only for a GPU install (installer.js runs verifyCuda behind
 * `if (mode === 'gpu')`), but the list was a fixed array, so a Mac -- where a
 * GPU install is impossible -- watched a step called "checking the graphics
 * card" that was never going to happen.
 */
function stepsFor(mode) {
  return mode === 'gpu' ? STEPS : STEPS.filter((s) => s !== 'verify');
}

/**
 * The installer's sentence, in the user's language where it has a code.
 *
 * `reason` is English prose built in the main process; `reasonCode` is the
 * same thing as a key, following the `warningCode` the installer already used
 * for cudaUnverified. The prose stays as the fallback, so an older main
 * process still says something.
 */
function reasonFrom(probe) {
  if (probe && probe.reasonCode) {
    return t(`reasons.${probe.reasonCode}`, probe.reasonValues || {});
  }
  return (probe && probe.reason) || '';
}

function renderFacts(probe) {
  setText('factsTitle', 'yourComputer');
  const list = $('factsList');
  list.replaceChildren();

  const row = (labelKey, value) => {
    const dt = document.createElement('dt');
    dt.textContent = t(labelKey);
    const dd = document.createElement('dd');
    dd.textContent = value;
    list.append(dt, dd);
  };

  const gpu = probe.gpu || {};
  row('graphics', gpu.name || t('noGraphics'));
  if (gpu.driver) row('driver', String(gpu.driver));
  row('freeSpace', probe.freeKnown === false
    ? t('unknownSpace')
    : `${(probe.freeBytes / GB).toFixed(1)} GB`);
}

function renderChoice(probe) {
  setText('chooseTitle', 'chooseVersion');

  // `blocked` is the installer's own verdict that NEITHER version can be
  // installed, and it carries the sentence explaining why. Ignoring it and
  // re-deriving a cheerier answer here is how a user whose free space could
  // not be READ -- a redirected profile on a university network share, which
  // installer.js calls the default outcome there -- was offered the install
  // and then told their disk was full.
  if (probe.blocked) {
    renderBlocked(probe);
    return false;
  }

  const free = probe.freeBytes;
  const gpuFits = free >= probe.gpuBytesNeeded;
  const hasCard = Boolean(probe.gpu && probe.gpu.cudaCapable);
  // The installer already worked out WHY the graphics card version is not on
  // offer, in one sentence naming the actual number: a driver too old, a
  // driver not answering, no card at all, or not enough room. The renderer
  // cannot tell those apart from `cudaCapable` alone, and guessing produced
  // "Needs an NVIDIA graphics card" on a machine whose facts list, two lines
  // above, read "NVIDIA GeForce RTX 4090".
  const gpuUnavailableReason = probe.recommendation !== 'gpu' ? reasonFrom(probe) : null;

  const configure = (which, fits, extraBlockKey) => {
    const label = $(which === 'gpu' ? 'optGpuLabel' : 'optCpuLabel');
    const input = $(which === 'gpu' ? 'optGpu' : 'optCpu');
    const block = $(which === 'gpu' ? 'optGpuBlock' : 'optCpuBlock');
    const title = $(which === 'gpu' ? 'optGpuTitle' : 'optCpuTitle');
    const detail = $(which === 'gpu' ? 'optGpuDetail' : 'optCpuDetail');

    const size = sizeOf(which, probe);
    title.replaceChildren();
    title.textContent = t(which === 'gpu' ? 'gpuOption' : 'cpuOption');
    detail.textContent = t(which === 'gpu' ? 'gpuDetail' : 'cpuDetail', { size });

    const blocked = !fits || Boolean(extraBlockKey);
    input.disabled = blocked;
    label.classList.toggle('disabled', blocked);
    // The reason is shown, not implied: "there is a faster version, and here is
    // exactly why you cannot have it" is actionable; a greyed row is not.
    // The installer's sentence wins where it has one, because it names the
    // number the user has to change.
    let reasonText = '';
    if (blocked && which === 'gpu' && gpuUnavailableReason) reasonText = gpuUnavailableReason;
    else if (blocked && extraBlockKey) reasonText = t(extraBlockKey);
    else if (blocked) reasonText = t('notEnoughSpace', { needed: sizeOf(which, probe) });
    block.hidden = !reasonText;
    block.textContent = reasonText;

    if (probe.recommendation === which && !blocked) {
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = t('recommended');
      title.append(badge);
    }
    return !blocked;
  };

  const gpuOk = configure('gpu', gpuFits, hasCard ? null : 'needsCard');
  const cpuOk = configure('cpu', free >= probe.cpuBytesNeeded, null);

  const preferred = probe.recommendation === 'gpu' && gpuOk ? 'gpu' : 'cpu';
  $(preferred === 'gpu' ? 'optGpu' : 'optCpu').checked = true;

  // Where the machine cannot have the other version at all, say so in one
  // sentence instead of showing a choice. The Mac tester on 2026-09-25 was
  // offered "Version for the graphics card" (greyed) against "Version for the
  // processor" on a computer that cannot have an NVIDIA card -- with the
  // explanation in English under a German heading, because the installer built
  // that sentence as prose. The radios stay in the DOM, still checked, so
  // chosenMode() keeps working.
  const noChoice = probe.reasonCode === 'macosCpuOnly';
  const group = $('optionGroup');
  const single = $('singleOption');
  if (group) group.hidden = noChoice;
  if (single) {
    single.hidden = !noChoice;
    single.textContent = noChoice ? reasonFrom(probe) : '';
  }

  $('installBtn').disabled = !gpuOk && !cpuOk;
  setText('installBtn', 'install');
  return true;
}

/** What the version row currently says, kept so a language switch redraws it
 *  without asking GitHub again. */
let releases = null;        // null = not fetched yet
let chosenTag = null;       // null = newest

function renderHome() {
  setText('homeLabel', 'homeLabel');
  $('homeValue').textContent = ctx.home || '';
  setText('pickHomeBtn', 'pickHome');
}

function renderVersion() {
  setText('versionLabel', 'versionLabel');
  $('versionValue').textContent = chosenTag || t('versionNewest');
  setText('pickVersionBtn', $('versionRow').hidden ? 'pickVersion' : 'pickVersionHide');
  setText('versionPickLabel', 'versionPickLabel');
}

/**
 * Fill the version list, once.
 *
 * Fetched only when asked for, not on startup: it is an extra, and a
 * repository that cannot be reached must still let someone install from a
 * package they already have.
 */
async function loadVersions() {
  if (releases) return;
  const answer = await window.setupAPI.releases();
  releases = answer.releases || [];
  const select = $('versionSelect');
  select.replaceChildren();

  const newest = document.createElement('option');
  newest.value = '';
  newest.textContent = t('versionNewest');
  select.append(newest);

  for (const release of releases) {
    const option = document.createElement('option');
    option.value = release.tag;
    // A release with no Orienta package cannot be installed, and offering it
    // would hand the user a choice that always fails.
    option.disabled = !release.installable;
    option.textContent = release.tag
      + (release.prerelease ? ` (${t('versionPrerelease')})` : '')
      + (release.installable ? '' : ` — ${t('versionNoPackage')}`);
    select.append(option);
  }
  select.value = chosenTag || '';
  if (!releases.length) {
    newest.textContent = answer.ok ? t('versionNone') : t('versionListFailed');
  }
}

function sizeOf(which, probe) {
  return ((which === 'gpu' ? probe.gpuBytesNeeded : probe.cpuBytesNeeded) / GB).toFixed(1);
}

/**
 * Neither version can be installed.
 *
 * Shown on the FAILURE screen rather than as a greyed-out choose screen with a
 * dead Install button, because it is a failure -- and because that screen is
 * the only one with a way out. The static page this wizard replaced offered a
 * releases URL and a log path in the same situation; a wizard that offers
 * less than the placeholder it replaced is a regression.
 */
function renderBlocked(probe) {
  // Its own code. `space` says "free some room", which is wrong and
  // unactionable for a drive that is not connected at all.
  const code = probe.freeKnown === false ? 'homeUnusable' : 'space';
  renderFailure({ code, error: probe.reason, blockedBeforeStart: true }, null);
  show('failed');
}

// --------------------------------------------------------------------------
// screen 2: installing
// --------------------------------------------------------------------------

/** Rough minutes, so "this is the long one" can be said before the silence. */
const LONG_STEPS = { packages: { gpu: 20, cpu: 6 } };

/** The step the progress screen is showing, so a language switch mid-install
 *  can redraw the list without waiting for the next progress event -- which,
 *  during pip, can be minutes away. */
let currentStepOnScreen = null;

function renderSteps(current, failedAt) {
  currentStepOnScreen = current;
  const list = $('stepList');
  list.replaceChildren();
  const steps = stepsFor(lastMode);
  const at = steps.indexOf(current);
  steps.forEach((step, index) => {
    const li = document.createElement('li');
    const mark = document.createElement('span');
    mark.className = 'mark';
    const passed = at >= 0 && index < at;
    mark.textContent = failedAt === step ? '×' : (passed ? '✓' : '·');
    mark.setAttribute('aria-hidden', 'true');
    const text = document.createElement('span');
    text.textContent = t(`steps.${step}`);
    li.append(mark, text);
    if (step === current) li.classList.add('active');
    if (passed) li.classList.add('done');
    if (failedAt === step) li.classList.add('failed');
    list.append(li);
  });
}

function formatElapsed(seconds) {
  return t('elapsed', {
    minutes: Math.floor(seconds / 60),
    seconds: String(seconds % 60).padStart(2, '0'),
  });
}

function onProgress(event, mode) {
  const { step, message, done, total, elapsedS, stepElapsedS, tag } = event;
  // The resolve step says which release it settled on. Recording it means the
  // done screen can name the version that was actually installed rather than
  // the one that was asked for, which are different whenever "newest" is used.
  if (tag) installedTag = tag;
  renderSteps(step, null);

  const bar = $('bar');
  if (total > 0) {
    bar.classList.remove('indeterminate');
    bar.style.width = `${Math.min(100, (done / total) * 100).toFixed(1)}%`;
    $('workingDetail').textContent = t('downloading', {
      done: (done / 1e6).toFixed(0),
      total: (total / 1e6).toFixed(0),
    });
  } else {
    // pip emits thousands of lines and no total. Showing its output is the
    // honest signal that work is happening; the bar only paces.
    bar.classList.add('indeterminate');
    bar.style.width = '';
    if (message) {
      $('workingDetail').textContent = message;
    } else if (elapsedS !== undefined) {
      $('workingDetail').textContent = t('stillWorking', {
        elapsed: formatElapsed(stepElapsedS ?? elapsedS),
      });
    }
  }

  const long = LONG_STEPS[step];
  $('workingHint').textContent = long ? t('longStep', { minutes: long[mode] ?? long.cpu }) : '';
}

// --------------------------------------------------------------------------
// screen 3: it did not work
// --------------------------------------------------------------------------

function renderFailure(result, mode) {
  renderSteps(null, result.step || null);
  setText('failedTitle', 'failedTitle');

  // The translated sentence for the code, and the installer's own English
  // detail underneath it -- the first is what the user acts on, the second is
  // what they paste into a message to us.
  const translated = t(`errors.${result.code}`);
  const known = translated !== `errors.${result.code}`;
  // Two elements, not one string with \n\n in it: #failedMessage is a <p>,
  // and a newline in a <p> collapses to a space -- so the sentence the user
  // acts on and the English detail they paste to us ran together into one
  // bilingual line.
  $('failedMessage').textContent = known ? translated : String(result.error || '');
  const detail = known && result.error ? String(result.error) : '';
  $('failedDetail').textContent = detail;
  $('failedDetail').hidden = !detail;

  // There is nothing to retry when the machine was refused before the install
  // started, and nowhere else to go -- so offer the one thing that is true:
  // where Orienta comes from, and how to leave.
  const beforeStart = Boolean(result.blockedBeforeStart);
  $('releasesLine').hidden = !(beforeStart || result.code === 'notFound');
  // When we could not get the package over the network, the file may already
  // be on this machine -- and pointing at it is a great deal more likely to
  // succeed than retrying the request that just failed.
  const offerFile = ['notFound', 'network', 'timeout', 'rateLimited', 'checksum']
    .includes(result.code);
  $('pickPackageBtn').hidden = !offerFile;
  if (offerFile) setText('pickPackageBtn', 'pickPackage');
  $('closeBtn').hidden = !beforeStart;
  $('retryBtn').hidden = beforeStart;

  $('failedLog').hidden = !result.log;
  if (result.log) $('failedLog').textContent = result.log;

  $('failedLogHint').textContent = result.logFile
    ? t('logHint', { path: result.logFile })
    : '';
  $('failedLogHint').hidden = !result.logFile;

  setText('retryBtn', 'retry');
  setText('openLogBtn', 'openLog');
  setText('closeBtn', 'close');
  setText('releasesLabel', 'releasesLabel');
  $('releasesLink').textContent = ctx.releasesUrl || '';
  $('releasesLink').href = ctx.releasesUrl || '#';
  $('openLogBtn').hidden = !result.logFile;

  // Offered only when the installer itself said it is possible: a CUDA failure
  // or a GPU-sized disk shortfall. Guessing here would offer a 2 GB install to
  // someone with no room for it.
  const canFallBack = result.canFallBackToCpu && mode === 'gpu';
  // After a CUDA failure the one action that resolves the situation is the
  // fallback, and Retry -- which re-downloads 8 GB to reach the identical
  // failure -- was the primary button sitting to its left.
  $('fallbackBtn').classList.toggle('primary', canFallBack);
  $('retryBtn').classList.toggle('primary', !canFallBack);
  $('fallbackBtn').hidden = !canFallBack;
  if (canFallBack) setText('fallbackBtn', 'installCpuInstead');
}

// --------------------------------------------------------------------------
// running
// --------------------------------------------------------------------------

/** The run that is on screen. `chosenMode()` reads a radio on a HIDDEN
 *  screen, so after a CPU fallback it still answers "gpu" -- and Retry then
 *  restarted the 8 GB install the user had just been told to abandon. */
let lastMode = null;
let lastResult = null;
let installedTag = null;

async function install(mode, { resume = false } = {}) {
  lastMode = mode;
  running = true;
  show('working');
  $('cancelBtn').disabled = false;
  setText('cancelBtn', 'cancel');
  renderSteps('probe', null);
  $('workingDetail').textContent = '';
  $('bar').classList.add('indeterminate');

  const stop = window.setupAPI.onProgress((event) => onProgress(event, mode));
  let result;
  try {
    result = await window.setupAPI.run({ mode, resume });
  } catch (err) {
    result = { ok: false, code: 'unexpected', error: String((err && err.message) || err) };
  } finally {
    stop();
    running = false;
  }

  lastResult = result;
  if (result.ok) {
    setText('doneTitle', result.warningCode ? 'warningTitle' : 'succeededTitle');
    // A code, translated like everything else. Rendering the installer's own
    // English sentence here put an English paragraph under a German heading.
    const version = result.tag || installedTag;
    $('doneBody').textContent = result.warningCode
      ? t(`warnings.${result.warningCode}`)
      : (version ? t('succeededBodyVersion', { version }) : t('succeededBody'));
    setText('startBtn', 'start');
    show('done');
    focusSection('done');
    return;
  }
  renderFailure(result, mode);
  show('failed');
  focusSection('failed');
}

/**
 * Move focus to the screen that just appeared.
 *
 * `show()` puts the focused button inside a display:none subtree, so focus
 * falls back to <body> and a keyboard user has to Tab from the top of the
 * document on every transition -- while a screen-reader user hears nothing at
 * all, four times, as the page replaces its entire contents.
 */
function focusSection(name) {
  const section = $(name);
  section.setAttribute('tabindex', '-1');
  section.focus({ preventScroll: false });
}

function chosenMode() {
  return $('optGpu').checked ? 'gpu' : 'cpu';
}

// --------------------------------------------------------------------------
// start
// --------------------------------------------------------------------------

/**
 * Apply a language to everything currently on screen.
 *
 * Re-rendering rather than reloading, because a switch during a twenty-minute
 * install must not restart the install -- and because the choose screen is
 * built from a probe we would otherwise have to run again.
 */
function applyLanguage(next, { explicit = false } = {}) {
  lang = ALL_LOCALES[next] ? next : 'en';
  // Handed on to Orienta itself, which would otherwise start in English
  // whatever this page spoke. `explicit` marks a choice the user made here,
  // which is allowed to override one they made earlier inside the app.
  window.setupAPI.setLanguage(lang, explicit).catch(() => {});
  L = ALL_LOCALES[lang];
  document.documentElement.lang = lang;
  $('langSelect').value = lang;
  setText('langLabel', 'langLabel');

  setText('title', 'title');
  renderVersion();
  renderHome();
  // The repair message comes from the main process in ITS language, so it is
  // only kept while that language is the one on screen; otherwise the
  // translated general sentence is the honest choice.
  $('intro').textContent = ctx.reason === 'repair'
    ? ((lang === ctx.locale && ctx.message) || t('introRepair'))
    : t('introFirst');

  if (probed && !$('choose').hidden) {
    renderFacts(probed);
    renderChoice(probed);
  }
  if (!$('failed').hidden && lastResult) renderFailure(lastResult, lastMode);
  if (!$('working').hidden) {
    renderSteps(currentStepOnScreen, null);
    setText('cancelBtn', running ? 'cancel' : 'cancel');
  }
  if (!$('done').hidden && lastResult) {
    setText('doneTitle', lastResult.warningCode ? 'warningTitle' : 'succeededTitle');
    $('doneBody').textContent = lastResult.warningCode
      ? t(`warnings.${lastResult.warningCode}`) : t('succeededBody');
    setText('startBtn', 'start');
  }
}

async function main() {
  ctx = await window.setupAPI.context();
  ALL_LOCALES = await window.setupAPI.locales();
  applyLanguage(ctx.locale);
  $('langSelect').addEventListener('change',
    (event) => applyLanguage(event.target.value, { explicit: true }));

  $('installBtn').addEventListener('click', () => install(chosenMode()));
  $('pickHomeBtn').addEventListener('click', async () => {
    const picked = await window.setupAPI.pickHome();
    if (!picked) return;
    const note = $('homeNote');
    note.hidden = false;
    if (picked.error) {
      note.textContent = t('homeUnwritable', { dir: picked.dir, detail: picked.error });
      return;
    }
    ctx.home = picked.dir;
    renderHome();
    // Chromium's storage was pointed at the old folder before this window
    // existed, so the move takes effect on the next launch. Saying that is
    // better than letting someone install 8 GB and wonder why it went to the
    // folder they just changed away from.
    note.textContent = picked.restartNeeded ? t('homeRestart') : t('homeChanged');
  });

  $('pickVersionBtn').addEventListener('click', async () => {
    const row = $('versionRow');
    row.hidden = !row.hidden;
    renderVersion();
    if (!row.hidden) {
      $('versionSelect').disabled = true;
      try {
        await loadVersions();
      } finally {
        $('versionSelect').disabled = false;
      }
    }
  });
  $('versionSelect').addEventListener('change', async (event) => {
    chosenTag = event.target.value || null;
    await window.setupAPI.selectRelease(chosenTag);
    renderVersion();
  });
  // `resumable` comes from the FAILURE, not from the call site. Assuming it
  // here sent the fallback looking for a lock file in a tree that had never
  // been downloaded, and sent Retry back to re-fetch 48 MB it already had.
  $('fallbackBtn').addEventListener('click', () => install('cpu', {
    resume: Boolean(lastResult && lastResult.resumable),
  }));
  $('retryBtn').addEventListener('click', () => install(lastMode || chosenMode(), {
    resume: Boolean(lastResult && lastResult.resumable),
  }));
  $('closeBtn').addEventListener('click', () => window.setupAPI.quit());
  $('pickPackageBtn').addEventListener('click', async () => {
    const picked = await window.setupAPI.pickPackage();
    if (!picked) return;
    if (!picked.hasChecksum) {
      // Said plainly rather than worked around: the digest is required for a
      // local file exactly as for a downloaded one, and a missing .sha256 is
      // something the user can fix in one copy.
      $('failedDetail').hidden = false;
      $('failedDetail').textContent = t('packageNoChecksum', { file: picked.file });
      return;
    }
    install(lastMode || chosenMode());
  });
  $('openLogBtn').addEventListener('click', () => window.setupAPI.openLog());
  $('startBtn').addEventListener('click', () => window.setupAPI.relaunch());
  $('cancelBtn').addEventListener('click', async () => {
    if (!running) return;
    if (!window.confirm(t('confirmCancel'))) return;
    $('cancelBtn').disabled = true;
    setText('cancelBtn', 'cancelling');
    await window.setupAPI.cancel();
    // Re-enabled after a moment: cancelling kills the CHILD PROCESSES, and
    // between two of them -- during a download, which runs inside the main
    // process -- there is nothing to kill, so the install carries on. Leaving
    // the button dead there left the user watching a cancelled install run for
    // twenty-five minutes with no second press available.
    setTimeout(() => {
      if (!running) return;
      $('cancelBtn').disabled = false;
      setText('cancelBtn', 'cancel');
    }, 4000);
  });

  $('intro').after(Object.assign(document.createElement('p'), {
    className: 'muted', id: 'probing', textContent: t('detecting'),
  }));

  probed = await window.setupAPI.probe();
  $('probing').remove();
  renderFacts(probed);
  // renderChoice answers whether there is a choice to show. It draws the
  // blocked screen itself when there is not, so showing 'choose' regardless
  // covers it over with a greyed-out, unusable page.
  if (renderChoice(probed)) {
    show('choose');
    focusSection('choose');
  }
}

main().catch((err) => {
  // A wizard that fails while starting must still say something: this screen is
  // the only one the user has.
  document.body.textContent = `Orienta setup could not start: ${err && err.message}`;
});
