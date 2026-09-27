/**
 * The example path under "Database root" has to exist on the machine reading it.
 *
 * The field offered `Z:\SharedDrive\EBSD_Database` everywhere. On a Mac that is
 * not a path at all, and the M5 tester met it as the ghost text of an empty
 * field next to a switch that was ON — which reads as a share someone
 * configured, rather than as a switch nobody ever touched.
 */
import { describe, it, expect } from 'vitest';
import { serverPathPlaceholderKey } from './SettingsPage';

describe('the database-root example path', () => {
  it('is a POSIX volume on macOS', () => {
    expect(serverPathPlaceholderKey('macos'))
      .toBe('settings:serverMode.databaseRootPlaceholderMac');
  });

  it('is a mount point on Linux', () => {
    expect(serverPathPlaceholderKey('linux'))
      .toBe('settings:serverMode.databaseRootPlaceholderLinux');
  });

  it('keeps the drive letter on Windows', () => {
    expect(serverPathPlaceholderKey('windows'))
      .toBe('settings:serverMode.databaseRootPlaceholder');
  });

  it('assumes Windows when the backend did not say', () => {
    // Every installation before v0.4.5 was Windows, and an older backend
    // does not send platform_os at all.
    expect(serverPathPlaceholderKey(undefined))
      .toBe('settings:serverMode.databaseRootPlaceholder');
  });
});
