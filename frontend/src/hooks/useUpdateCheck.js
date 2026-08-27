/**
 * Check for a newer release once, shortly after start-up.
 *
 * Rules that keep this from being an annoyance:
 *  - it never blocks the interface: the check runs after a short delay and a
 *    failure is silent (offline, no credentials, zip install — all normal),
 *  - a version the user dismissed with "skip" is not offered again,
 *  - the user can switch the check off entirely.
 */

import { useCallback, useEffect, useState } from 'react';
import { checkForUpdate } from '../services/api';

const START_DELAY_MS = 4000;   // let the app finish loading first
const SKIP_KEY = 'orienta.update.skippedVersion';
const ENABLED_KEY = 'orienta.update.checkOnStart';

export function isCheckEnabled() {
  try {
    return localStorage.getItem(ENABLED_KEY) !== 'false';
  } catch {
    return true;
  }
}

export function setCheckEnabled(enabled) {
  try {
    localStorage.setItem(ENABLED_KEY, enabled ? 'true' : 'false');
  } catch {
    // Private window / storage blocked — the default (on) still applies.
  }
}

function skippedVersion() {
  try {
    return localStorage.getItem(SKIP_KEY);
  } catch {
    return null;
  }
}

export default function useUpdateCheck() {
  const [info, setInfo] = useState(null);   // set only when we want to show it
  const [dismissed, setDismissed] = useState(false);

  const run = useCallback(async (force = false) => {
    try {
      const result = await checkForUpdate({ force });
      if (!result?.available) return result;
      if (!force && result.latest && result.latest === skippedVersion()) {
        return result;   // user asked not to see this one again
      }
      setInfo(result);
      setDismissed(false);
      return result;
    } catch {
      return null;       // never surface a failed check on start-up
    }
  }, []);

  useEffect(() => {
    if (!isCheckEnabled()) return undefined;
    const timer = setTimeout(() => { run(false); }, START_DELAY_MS);
    return () => clearTimeout(timer);
  }, [run]);

  const close = useCallback(() => setDismissed(true), []);

  const skip = useCallback(() => {
    try {
      if (info?.latest) localStorage.setItem(SKIP_KEY, info.latest);
    } catch {
      // ignore
    }
    setDismissed(true);
  }, [info]);

  return {
    updateInfo: dismissed ? null : info,
    checkNow: () => run(true),
    close,
    skip,
  };
}
