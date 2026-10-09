/**
 * User-visible text for the Electron shell, in the four languages the app
 * already ships.
 *
 * The renderer has `frontend/src/locales/{en,de,ja,zh}/` with 23 namespaces,
 * and even the installer UI is translated. But the shell's own messages appear
 * at exactly the moments the renderer cannot run — nothing installed, a broken
 * installation, a backend that never answered — so a German or Japanese user
 * would meet raw English precisely when something has gone wrong and the words
 * matter most.
 *
 * The renderer's own choice lives in localStorage on the backend's origin, which
 * a file:// page cannot read, so this falls back to the OS locale. When the
 * wizard lands it should persist the explicit choice to a file under
 * orientaHome() and this module should prefer that.
 *
 * Wording deliberately mirrors `locales/*\/shell.json` where the same thing is
 * already said there, so the sentence a user sees before the app starts matches
 * the one they see inside it.
 */

const STRINGS = {
  en: {
    startingTitle: 'Orienta is starting',
    startingBody: 'Loading the analysis engine…',
    startingHint: 'The first start after installing or updating loads the scientific '
      + 'libraries and can take a minute or two.',
    elapsed: '{{seconds}} s',

    updatingBody: 'Updating the program files…',
    updatingHint: 'This runs once after an update and takes a few seconds. Your '
      + 'crystal library and your settings are not touched.',
    updateFailedTitle: 'Orienta could not finish updating',
    updateFailedBody: 'The program files were left part-way between two versions, so '
      + 'Orienta has not started the analysis engine. Start Orienta again and it will '
      + 'finish the update. Your data and crystal library are not touched.',
    updateSkippedBody: 'Orienta could not update its program files and has started the '
      + 'previous version instead, so this window may not show the newest changes. '
      + 'Reinstalling Orienta repairs it; your data and crystal library are not touched.',

    syncingBody: 'Updating the Python packages (kikuchipy, orix, PyEBSDIndex)…',
    syncingHint: 'This runs once after an update. It needs an internet connection and '
      + 'takes about a minute. Your data is not touched. If it cannot finish, Orienta '
      + 'starts with the current packages.',
    syncSkippedBody: 'Orienta could not update its Python packages and is running with '
      + 'the current ones, which this version also supports. Your data and crystal '
      + 'library are not touched. Details are in the log file:\n{{path}}',
    syncRepairBody: 'The updated Python packages did not pass the check, so Orienta has '
      + 'not started the analysis engine. Reinstalling Orienta repairs it; your data '
      + 'and crystal library are not touched.',

    notSetUpTitle: 'Orienta is not set up yet',
    needsRepairTitle: 'Orienta needs repairing',
    setupMissing: 'Orienta could not open the setup assistant: part of this '
      + 'installation is missing. Reinstall Orienta from the releases page, or '
      + 'follow INSTALL.md in the project.',
    logLocation: 'Details are written to: {{path}}',
    downloadHere: 'Download Orienta here:',
    reinstallHint: 'Reinstall Orienta; your data and crystal library are not touched.',
    repairCannotInspect: 'Orienta could not inspect its installation: {{detail}}',
    openReleases: 'Open the releases page',

    repairPythonMissing: 'Orienta\'s Python environment is missing — antivirus software '
      + 'sometimes removes it. Reinstall Orienta; your data and crystal library are not '
      + 'touched.',
    repairPythonUnreadable: '{{detail}}. Reinstall Orienta; your data and crystal library '
      + 'are not touched.',
    repairRuntimeMissing: 'Orienta\'s program files are missing or incomplete, which '
      + 'usually means an update was interrupted. Reinstall Orienta; your data and crystal '
      + 'library are not touched.',

    backendFailedTitle: 'Orienta could not start',
    backendFailedMessage: 'The analysis engine did not answer within three minutes.',
    backendFailedDetail: 'This is usually a missing package, a port already in use, or an '
      + 'import error.\n\nThe details are in:\n{{shellLog}}\n{{backendLog}}',
    showLogFolder: 'Show the log folder',
    tryAgain: 'Try again',
    close: 'Close',

    noInterpreterTitle: 'Orienta could not start',
    noInterpreterBody: 'No Python environment is installed for Orienta. Reinstall Orienta '
      + 'to set one up; your data and crystal library are not touched.',
  },

  de: {
    startingTitle: 'Orienta startet',
    startingBody: 'Die Auswertung wird geladen…',
    startingHint: 'Der erste Start nach einer Installation oder einem Update lädt die '
      + 'wissenschaftlichen Bibliotheken und kann ein bis zwei Minuten dauern.',
    elapsed: '{{seconds}} s',

    updatingBody: 'Die Programmdateien werden aktualisiert…',
    updatingHint: 'Das läuft einmal nach einem Update und dauert wenige Sekunden. '
      + 'Ihre Kristallbibliothek und Ihre Einstellungen werden nicht angefasst.',
    updateFailedTitle: 'Orienta konnte das Update nicht abschließen',
    updateFailedBody: 'Die Programmdateien stehen zwischen zwei Versionen, deshalb hat '
      + 'Orienta die Auswertung nicht gestartet. Starten Sie Orienta erneut, dann wird '
      + 'das Update beendet. Ihre Daten und Ihre Kristallbibliothek werden nicht '
      + 'angefasst.',
    updateSkippedBody: 'Orienta konnte seine Programmdateien nicht aktualisieren und hat '
      + 'die vorherige Fassung gestartet; dieses Fenster zeigt also möglicherweise '
      + 'nicht die neuesten Änderungen. Eine Neuinstallation behebt das; Ihre Daten '
      + 'und Ihre Kristallbibliothek werden nicht angefasst.',

    syncingBody: 'Die Python-Pakete (kikuchipy, orix, PyEBSDIndex) werden aktualisiert…',
    syncingHint: 'Das läuft einmal nach einem Update. Dafür ist eine Internetverbindung '
      + 'nötig, und es dauert etwa eine Minute. Ihre Daten werden nicht angefasst. '
      + 'Falls die Aktualisierung nicht abgeschlossen werden kann, startet Orienta mit '
      + 'den vorhandenen Paketen.',
    syncSkippedBody: 'Orienta konnte seine Python-Pakete nicht aktualisieren und läuft '
      + 'mit den vorhandenen, die auch diese Version unterstützt. Ihre Daten und Ihre '
      + 'Kristallbibliothek werden nicht angefasst. Einzelheiten stehen in der '
      + 'Log-Datei:\n{{path}}',
    syncRepairBody: 'Die aktualisierten Python-Pakete haben die Prüfung nicht bestanden, '
      + 'deshalb hat Orienta die Auswertung nicht gestartet. Eine Neuinstallation '
      + 'behebt das; Ihre Daten und Ihre Kristallbibliothek werden nicht angefasst.',

    notSetUpTitle: 'Orienta ist noch nicht eingerichtet',
    needsRepairTitle: 'Orienta muss repariert werden',
    setupMissing: 'Orienta konnte den Einrichtungsassistenten nicht öffnen: ein Teil '
      + 'dieser Installation fehlt. Installieren Sie Orienta über die Releases-Seite neu, '
      + 'oder folgen Sie INSTALL.md.',
    logLocation: 'Einzelheiten stehen in: {{path}}',
    downloadHere: 'Orienta hier herunterladen:',
    reinstallHint: 'Installieren Sie Orienta neu; Ihre Daten und Ihre Kristallbibliothek bleiben unberührt.',
    repairCannotInspect: 'Orienta konnte die Installation nicht prüfen: {{detail}}',
    openReleases: 'Releases-Seite öffnen',

    repairPythonMissing: 'Die Python-Umgebung von Orienta fehlt — manche Virenscanner '
      + 'entfernen sie. Installieren Sie Orienta neu; Ihre Daten und Ihre '
      + 'Kristall-Datenbank bleiben unberührt.',
    repairPythonUnreadable: '{{detail}}. Installieren Sie Orienta neu; Ihre Daten und Ihre '
      + 'Kristall-Datenbank bleiben unberührt.',
    repairRuntimeMissing: 'Die Programmdateien von Orienta fehlen oder sind unvollständig '
      + '— meist wurde ein Update unterbrochen. Installieren Sie Orienta neu; Ihre Daten '
      + 'und Ihre Kristall-Datenbank bleiben unberührt.',

    backendFailedTitle: 'Orienta konnte nicht starten',
    backendFailedMessage: 'Die Auswertung hat innerhalb von drei Minuten nicht geantwortet.',
    backendFailedDetail: 'Meist fehlt ein Paket, der Port ist belegt, oder ein Import '
      + 'schlägt fehl.\n\nEinzelheiten stehen in:\n{{shellLog}}\n{{backendLog}}',
    showLogFolder: 'Log-Ordner anzeigen',
    tryAgain: 'Erneut versuchen',
    close: 'Schließen',

    noInterpreterTitle: 'Orienta konnte nicht starten',
    noInterpreterBody: 'Für Orienta ist keine Python-Umgebung installiert. Installieren '
      + 'Sie Orienta neu; Ihre Daten und Ihre Kristall-Datenbank bleiben unberührt.',
  },

  ja: {
    startingTitle: 'Orienta を起動しています',
    startingBody: '解析エンジンを読み込んでいます…',
    startingHint: 'インストールまたは更新後の初回起動では科学計算ライブラリを読み込むため、'
      + '1〜2 分かかることがあります。',
    elapsed: '{{seconds}} 秒',

    updatingBody: 'プログラムファイルを更新しています…',
    updatingHint: '更新後に一度だけ実行され、数秒で終わります。'
      + '結晶ライブラリと設定はそのままです。',
    updateFailedTitle: 'Orienta の更新を完了できませんでした',
    updateFailedBody: 'プログラムファイルが 2 つのバージョンの途中にあるため、'
      + '解析エンジンを起動していません。Orienta をもう一度起動すると更新が完了します。'
      + 'データと結晶ライブラリはそのままです。',
    updateSkippedBody: 'プログラムファイルを更新できず、以前のバージョンを起動しました。'
      + 'この画面には最新の変更が反映されていない可能性があります。'
      + '再インストールで修復します。データと結晶ライブラリはそのままです。',

    syncingBody: 'Python パッケージ（kikuchipy、orix、PyEBSDIndex）を更新しています…',
    syncingHint: '更新後に一度だけ実行されます。インターネット接続が必要で、1 分ほどかかります。'
      + 'データはそのままです。完了できない場合は、現在のパッケージで Orienta を起動します。',
    syncSkippedBody: 'Python パッケージを更新できなかったため、現在のパッケージのまま'
      + '起動しました。このバージョンは現在のパッケージにも対応しています。'
      + 'データと結晶ライブラリはそのままです。詳細は次のログファイルに記録されています:\n{{path}}',
    syncRepairBody: '更新した Python パッケージが確認に合格しなかったため、'
      + '解析エンジンを起動していません。Orienta を再インストールすると修復できます。'
      + 'データと結晶ライブラリはそのままです。',

    notSetUpTitle: 'Orienta はまだセットアップされていません',
    needsRepairTitle: 'Orienta の修復が必要です',
    setupMissing: 'セットアップアシスタントを開けませんでした。このインストールの一部が'
      + '見つかりません。リリースページから Orienta を再インストールするか、'
      + 'INSTALL.md に従ってください。',
    logLocation: '詳細は次の場所に記録されています: {{path}}',
    downloadHere: 'Orienta はこちらからダウンロードできます:',
    reinstallHint: 'Orienta を再インストールしてください。データと結晶ライブラリはそのまま残ります。',
    repairCannotInspect: 'Orienta はインストール状態を確認できませんでした: {{detail}}',
    openReleases: 'リリースページを開く',

    repairPythonMissing: 'Orienta の Python 環境が見つかりません。ウイルス対策ソフトが'
      + '削除することがあります。Orienta を再インストールしてください。データと結晶'
      + 'データベースはそのまま残ります。',
    repairPythonUnreadable: '{{detail}}。Orienta を再インストールしてください。データと'
      + '結晶データベースはそのまま残ります。',
    repairRuntimeMissing: 'Orienta のプログラムファイルが見つからないか不完全です。'
      + '多くの場合、更新が中断されたことが原因です。Orienta を再インストールしてください。'
      + 'データと結晶データベースはそのまま残ります。',

    backendFailedTitle: 'Orienta を起動できませんでした',
    backendFailedMessage: '解析エンジンが 3 分以内に応答しませんでした。',
    backendFailedDetail: '通常はパッケージの不足、ポートの使用中、または読み込みエラーです。'
      + '\n\n詳細は次の場所にあります:\n{{shellLog}}\n{{backendLog}}',
    showLogFolder: 'ログフォルダーを表示',
    tryAgain: '再試行',
    close: '閉じる',

    noInterpreterTitle: 'Orienta を起動できませんでした',
    noInterpreterBody: 'Orienta 用の Python 環境がインストールされていません。Orienta を'
      + '再インストールしてください。データと結晶データベースはそのまま残ります。',
  },

  zh: {
    startingTitle: 'Orienta 正在启动',
    startingBody: '正在加载分析引擎…',
    startingHint: '安装或更新后的首次启动需要加载科学计算库，可能需要一到两分钟。',
    elapsed: '{{seconds}} 秒',

    updatingBody: '正在更新程序文件…',
    updatingHint: '更新后只运行一次，需要几秒钟。晶体库和设置不会被改动。',
    updateFailedTitle: 'Orienta 无法完成更新',
    updateFailedBody: '程序文件停在两个版本之间，因此 Orienta 没有启动分析引擎。'
      + '请再次启动 Orienta，它会完成更新。您的数据和晶体库不会被改动。',
    updateSkippedBody: 'Orienta 无法更新程序文件，已启动之前的版本，'
      + '所以此窗口可能不显示最新的改动。重新安装可以修复；'
      + '您的数据和晶体库不会被改动。',

    syncingBody: '正在更新 Python 软件包（kikuchipy、orix、PyEBSDIndex）…',
    syncingHint: '更新后只运行一次，需要联网，大约需要一分钟。您的数据不会被改动。'
      + '如果无法完成，Orienta 会使用当前的软件包启动。',
    syncSkippedBody: 'Orienta 无法更新其 Python 软件包，将继续使用当前的软件包，'
      + '此版本同样支持它们。您的数据和晶体库不会被改动。详细信息见日志文件：\n{{path}}',
    syncRepairBody: '更新后的 Python 软件包未通过检查，因此 Orienta 没有启动分析引擎。'
      + '重新安装 Orienta 可以修复；您的数据和晶体库不会被改动。',

    notSetUpTitle: 'Orienta 尚未完成安装',
    needsRepairTitle: 'Orienta 需要修复',
    setupMissing: '无法打开安装向导：此安装的一部分文件缺失。'
      + '请从发布页面重新安装 Orienta，或按照 INSTALL.md 操作。',
    logLocation: '详细信息记录在：{{path}}',
    downloadHere: '在此下载 Orienta：',
    reinstallHint: '请重新安装 Orienta；您的数据和晶体库不会受到影响。',
    repairCannotInspect: 'Orienta 无法检查其安装状态：{{detail}}',
    openReleases: '打开发布页面',

    repairPythonMissing: '找不到 Orienta 的 Python 环境——某些杀毒软件会将其删除。'
      + '请重新安装 Orienta；您的数据和晶体数据库不会受到影响。',
    repairPythonUnreadable: '{{detail}}。请重新安装 Orienta；您的数据和晶体数据库'
      + '不会受到影响。',
    repairRuntimeMissing: 'Orienta 的程序文件缺失或不完整，通常是更新被中断所致。'
      + '请重新安装 Orienta；您的数据和晶体数据库不会受到影响。',

    backendFailedTitle: 'Orienta 无法启动',
    backendFailedMessage: '分析引擎在三分钟内没有响应。',
    backendFailedDetail: '通常是缺少软件包、端口被占用或导入错误。'
      + '\n\n详细信息位于：\n{{shellLog}}\n{{backendLog}}',
    showLogFolder: '显示日志文件夹',
    tryAgain: '重试',
    close: '关闭',

    noInterpreterTitle: 'Orienta 无法启动',
    noInterpreterBody: '尚未为 Orienta 安装 Python 环境。请重新安装 Orienta；'
      + '您的数据和晶体数据库不会受到影响。',
  },
};

/** The two-letter language this shell should speak. */
function shellLanguage(locale) {
  const code = String(locale || 'en').slice(0, 2).toLowerCase();
  return Object.prototype.hasOwnProperty.call(STRINGS, code) ? code : 'en';
}

/**
 * Which locale to believe, given what Electron offers.
 *
 * `app.getLocale()` is negotiated against the localizations the app BUNDLE
 * declares. Orienta declares none — there is no CFBundleLocalizations and no
 * electronLanguages anywhere in the build config — so on macOS it answers
 * "en" whatever the user's language is. That is why the M5 tester's setup
 * wizard opened in English on a German Mac on 2026-09-25, and why switching
 * it by hand worked: the strings were there all along.
 *
 * `app.getPreferredSystemLanguages()` asks the system instead of the bundle,
 * so it is the one to prefer. It can return an empty list, and it does not
 * exist on very old Electron, hence the fallback — Windows and Linux agree
 * with getLocale() anyway.
 *
 * Pure, so it can be tested without an Electron app object.
 */
function preferredLocale(preferredList, fallbackLocale) {
  const first = Array.isArray(preferredList) ? preferredList.find(Boolean) : null;
  return first || fallbackLocale || 'en';
}

/**
 * One translated string, with {{placeholders}} substituted.
 *
 * Falls back to English for a key a translation has not caught up with, and
 * returns the key itself if it exists nowhere — visible in testing, rather than
 * an empty label in front of a user.
 */
function t(locale, key, vars = {}) {
  const lang = shellLanguage(locale);
  const template = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return String(template).replace(
    /\{\{(\w+)\}\}/g,
    (_match, name) => (name in vars ? String(vars[name]) : `{{${name}}}`),
  );
}

module.exports = { STRINGS, shellLanguage, preferredLocale, t };
