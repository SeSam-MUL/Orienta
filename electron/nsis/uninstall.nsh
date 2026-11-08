; Orienta's own uninstall step: what happens to the DATA folder.
;
; electron-builder's uninstaller removes the application and its shortcuts and
; nothing else. The data folder -- Python (2.5 GB, or 8 for the graphics-card
; build), the program files, the logs and the user's crystal library -- stayed
; behind, so someone who uninstalled through Windows' "Installed apps" believed
; the machine was clean while gigabytes remained.
;
; Two questions, asked only for a REAL uninstall:
;   1. remove the downloaded components?           default: Yes
;   2. ALSO delete the crystal library?             default: No, and says it
;                                                   cannot be undone
;
; WHY THIS FILE IS SO CAREFUL: an update runs the uninstaller of the OLD
; version. Whatever ships here keeps running for that version for as long as
; anyone has it installed, so a flaw cannot be fixed by the next release.
;
; The rules, each with the measurement or finding behind it:
;
;   * Nothing at all on an UPDATE (${isUpdated}, set by electron-builder for
;     every old-uninstaller run, downgrades and same-version reinstalls
;     included).
;
;   * Never `RMDir /r`. Measured on NSIS 3.0.4: it FOLLOWS a junction and
;     deletes what is behind it. Directories go through Windows' own
;     `rd /s /q`, measured NOT to follow junctions, nested or not.
;
;   * A link is removed as a link -- checked on EVERY level this code walks
;     into, not only on the last path component. A linked `runtime\` pointing
;     at a developer checkout would otherwise have been emptied (review
;     finding). Links are recognised by their reparse TAG, junction or
;     symlink only: OneDrive's cloud placeholders are reparse points too, and
;     treating them as links would report "only the link was removed" while
;     leaving gigabytes behind.
;
;   * Names ending in '.' or ' ' are never passed on. Measured: `rd` on an
;     enumerated `Database.` -- creatable only through a \\?\ path -- deleted
;     the REAL `Database` next to it, because Win32 strips the trailing dot.
;
;   * Only a folder that is recognisably Orienta's: the marker file setup
;     writes (.orienta-home), or for older installs runtime\VERSION PLUS a
;     second marker -- and never a folder holding .git, never a drive root or
;     share root after normalisation, never the user profile.
;
;   * cmd.exe runs with /d /v:off: no AutoRun commands, and no delayed
;     expansion, under which `!name!` in a path is expanded even inside
;     quotes. A '%' is refused outright, because it is expanded regardless.
;
;   * An unreachable folder (an unplugged drive, an offline share) changes
;     nothing, and the pointer to it is KEPT: it is the only record of where
;     the library is.
;
; This file is UTF-8 WITH a byte-order mark: NSIS reads an include without one
; through the ANSI code page, and the dialogs would come out as mojibake.

!include "LogicLib.nsh"

; ---------------------------------------------------------------------------
; What is at PATH?  Sets $oKind to one of:
;   missing | dir | file | linkdir | linkfile
; A reparse point that is not a junction or symlink (a cloud placeholder)
; counts as the dir or file it presents itself as.
; ---------------------------------------------------------------------------
!macro orientaKindOf PATH
  Push $0
  Push $1
  Push $2
  Push $3
  StrCpy $oKind "missing"
  ; WIN32_FIND_DATAW: attributes at offset 0, dwReserved0 (the reparse tag)
  ; at offset 36 -- after three FILETIMEs and the two size words.
  System::Alloc 600
  Pop $1
  System::Call 'kernel32::FindFirstFileW(w "${PATH}", p r1) p .r0'
  ${If} $0 <> -1
    System::Call 'kernel32::FindClose(p r0)'
    System::Call '*$1(i .r2, i, i, i, i, i, i, i, i, i .r3)'
    IntOp $0 $2 & 0x10
    IntOp $2 $2 & 0x400
    ${If} $2 <> 0
    ${AndIf} $3 = 0xA0000003
      StrCpy $oKind "link"
    ${ElseIf} $2 <> 0
    ${AndIf} $3 = 0xA000000C
      StrCpy $oKind "link"
    ${EndIf}
    ${If} $oKind == "link"
      ${If} $0 <> 0
        StrCpy $oKind "linkdir"
      ${Else}
        StrCpy $oKind "linkfile"
      ${EndIf}
    ${ElseIf} $0 <> 0
      StrCpy $oKind "dir"
    ${Else}
      StrCpy $oKind "file"
    ${EndIf}
  ${EndIf}
  System::Free $1
  Pop $3
  Pop $2
  Pop $1
  Pop $0
!macroend

; ---------------------------------------------------------------------------
; Remove ONE path, never following a link. Sets $oLeftBehind to "yes" when
; something could not safely be removed, so the user is told.
; ---------------------------------------------------------------------------
!macro orientaRemovePath PATH
  Push $R6
  Push $R7
  Push $R8
  Push $R9

  StrCpy $R9 "${PATH}"
  StrCpy $R8 "ok"
  ; A '%' is expanded by cmd.exe even inside quotes, handing `rd` a different
  ; folder. Refused, not escaped: nothing Orienta creates contains one.
  StrCpy $R6 0
  ${Do}
    StrCpy $R7 $R9 1 $R6
    ${If} $R7 == ""
      ${ExitDo}
    ${EndIf}
    ${If} $R7 == "%"
      StrCpy $R8 "refuse"
      ${ExitDo}
    ${EndIf}
    IntOp $R6 $R6 + 1
  ${Loop}
  ; A trailing '.' or ' ' is stripped by Win32, so the name reaches a
  ; DIFFERENT entry -- measured to delete the real `Database` beside a
  ; `Database.`.
  StrCpy $R7 $R9 1 -1
  ${If} $R7 == "."
  ${OrIf} $R7 == " "
    StrCpy $R8 "refuse"
  ${EndIf}

  ${If} $R8 == "refuse"
    DetailPrint "Orienta: left in place (unsafe name): ${PATH}"
    StrCpy $oLeftBehind "yes"
  ${Else}
    !insertmacro orientaKindOf "${PATH}"
    ${If} $oKind == "linkdir"
      RMDir "${PATH}"
      DetailPrint "Orienta: removed link, target untouched: ${PATH}"
    ${ElseIf} $oKind == "linkfile"
      Delete "${PATH}"
      DetailPrint "Orienta: removed link, target untouched: ${PATH}"
    ${ElseIf} $oKind == "dir"
      nsExec::Exec '"$SYSDIR\cmd.exe" /d /v:off /c rd /s /q "${PATH}"'
      Pop $R6
      DetailPrint "Orienta: removed ${PATH}"
    ${ElseIf} $oKind == "file"
      Delete "${PATH}"
    ${EndIf}
    ; Check the result rather than trusting it.
    ${If} $oKind != "missing"
      !insertmacro orientaKindOf "${PATH}"
      ${If} $oKind != "missing"
        DetailPrint "Orienta: could not remove ${PATH}"
        StrCpy $oLeftBehind "yes"
      ${EndIf}
    ${EndIf}
  ${EndIf}

  Pop $R9
  Pop $R8
  Pop $R7
  Pop $R6
!macroend

; ---------------------------------------------------------------------------
; What counts as Orienta's own LEFTOVER -- by SHAPE, never by name.
;
; Names such as logs, pending, electron, python and runtime are everyday folder
; names. A level holding "only names from a list" would take a user's own
; D:\Work\Orienta with logs\mydata.csv for Orienta's, and the data would go with it.
; So an entry is a leftover only if it has
; the shape of what Orienta itself writes:
;
;   electron\    only userData, sessionData, temp, .writable
;   logs\        only FILES named orienta*.log / backend-console*.log (+ .log.N)
;   pending\     only orienta-runtime-* files;  pending.json only beside pending\
;   setup-tmp\   only cpython-*, orienta-*, python-unpacked*, micromamba*,
;                extract-result*, apply-result*, package-sync*, *.tmp
;   state files  .python_path .install_mode .install_incomplete
;                .packages_lock.json .packages_sync.json .packages_sync_failed.json
;
; python\ and runtime\ are NEVER leftovers by name: a folder holding them has an
; installation in it (it is recognised through its marker, not through this).
; Operating-system litter FILES (Thumbs.db, desktop.ini, .DS_Store) count as
; nothing, like in electron/leftovers.js -- which carries the same rules for the
; setup, and tests/leftover_shapes.json is ONE table of cases run against both.
;
; A "level" is a folder in a chain Orienta\Orienta\... (0.4.6 added one level per
; click of "change..."); every level may hold leftovers and the next Orienta.
; Each macro answers only in the cautious direction and removes nothing but
; what it has just recognised, by fixed name, or an EMPTY directory (RMDir
; without /r).
; ---------------------------------------------------------------------------

; $oOk = "yes" if the string STR starts with PREFIX (case-insensitive).
!macro orientaStarts STR PREFIX
  StrLen $oLen "${PREFIX}"
  StrCpy $oCmp "${STR}" $oLen
  ${If} $oCmp == "${PREFIX}"
    StrCpy $oOk "yes"
  ${Else}
    StrCpy $oOk "no"
  ${EndIf}
!macroend

; $oOk = "yes" if the string STR ends with SUFFIX (case-insensitive).
!macro orientaEnds STR SUFFIX
  StrLen $oLen "${SUFFIX}"
  IntOp $oLen 0 - $oLen
  StrCpy $oCmp "${STR}" "" $oLen
  ${If} $oCmp == "${SUFFIX}"
    StrCpy $oOk "yes"
  ${Else}
    StrCpy $oOk "no"
  ${EndIf}
!macroend

; $oOk = "yes" if every child of DIR has the shape KIND (electron|logs|pending|
; setup-tmp) allows. An empty folder passes.
!macro orientaChildrenOk DIR KIND
  Push $R4
  ; The running verdict has its OWN variable: the helper macros below answer in
  ; $oOk, and one child's answer must not overwrite another's refusal.
  StrCpy $oAll "yes"
  FindFirst $oFind4 $oName4 "${DIR}\*"
  ${DoWhile} $oName4 != ""
    ${If} $oName4 != "."
    ${AndIf} $oName4 != ".."
      StrCpy $R4 "no"
      ${If} "${KIND}" == "electron"
        ${If} $oName4 == "userData"
        ${OrIf} $oName4 == "sessionData"
        ${OrIf} $oName4 == "temp"
        ${OrIf} $oName4 == ".writable"
          StrCpy $R4 "yes"
        ${EndIf}
      ${ElseIf} "${KIND}" == "pending"
        !insertmacro orientaStarts "$oName4" "orienta-runtime-"
        StrCpy $R4 $oOk
      ${ElseIf} "${KIND}" == "setup-tmp"
        !insertmacro orientaEnds "$oName4" ".tmp"
        StrCpy $R4 $oOk
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "cpython-"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "orienta-"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "python-unpacked"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "micromamba"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "extract-result"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "apply-result"
          StrCpy $R4 $oOk
        ${EndIf}
        ${If} $R4 == "no"
          !insertmacro orientaStarts "$oName4" "package-sync"
          StrCpy $R4 $oOk
        ${EndIf}
      ${ElseIf} "${KIND}" == "logs"
        ; Orienta's own log FILES: orienta*.log, backend-console*.log, and the
        ; rotations (.log.1 ...). A folder, or any other name, is not ours.
        !insertmacro orientaKindOf "${DIR}\$oName4"
        ${If} $oKind == "file"
          !insertmacro orientaStarts "$oName4" "orienta"
          StrCpy $R4 $oOk
          ${If} $R4 == "no"
            !insertmacro orientaStarts "$oName4" "backend-console"
            StrCpy $R4 $oOk
          ${EndIf}
          ${If} $R4 == "yes"
            !insertmacro orientaEnds "$oName4" ".log"
            StrCpy $R4 $oOk
            ${If} $R4 == "no"
              StrLen $oLen "$oName4"
              ${If} $oLen > 5
                StrCpy $oCmp "$oName4" 5 -6
                ${If} $oCmp == ".log."
                  ; ... and ONE digit after it: .log.1 ... .log.9, as file_log.py and
                  ; the shell log rotate -- not ".log.~" or ".log.12".
                  StrCpy $oCmp "$oName4" 1 -1
                  ${If} $oCmp == "0"
                  ${OrIf} $oCmp == "1"
                  ${OrIf} $oCmp == "2"
                  ${OrIf} $oCmp == "3"
                  ${OrIf} $oCmp == "4"
                  ${OrIf} $oCmp == "5"
                  ${OrIf} $oCmp == "6"
                  ${OrIf} $oCmp == "7"
                  ${OrIf} $oCmp == "8"
                  ${OrIf} $oCmp == "9"
                    StrCpy $R4 "yes"
                  ${EndIf}
                ${EndIf}
              ${EndIf}
            ${EndIf}
          ${EndIf}
        ${EndIf}
      ${EndIf}
      ${If} $R4 == "no"
        StrCpy $oAll "no"
      ${EndIf}
    ${EndIf}
    FindNext $oFind4 $oName4
  ${Loop}
  FindClose $oFind4
  StrCpy $oOk $oAll
  Pop $R4
!macroend

; $oOk = "yes" if the entry NAME of LEVEL has the shape of something Orienta
; wrote there (see the table above); never python, runtime or the conda folders.
!macro orientaEntryLeftover LEVEL NAME
  StrCpy $oOk "no"
  ${If} "${NAME}" == ".orienta-home"
  ${AndIf} $oMarkerOk == "yes"
    ; Only while the caller is judging the chain BELOW a recognised data folder,
    ; whose own marker is still there (it is removed last).
    StrCpy $oOk "yes"
  ${ElseIf} "${NAME}" == ".python_path"
  ${OrIf} "${NAME}" == ".install_mode"
  ${OrIf} "${NAME}" == ".install_incomplete"
  ${OrIf} "${NAME}" == ".packages_lock.json"
  ${OrIf} "${NAME}" == ".packages_sync.json"
  ${OrIf} "${NAME}" == ".packages_sync_failed.json"
    StrCpy $oOk "yes"
  ${ElseIf} "${NAME}" == "pending.json"
    !insertmacro orientaKindOf "${LEVEL}\pending"
    ${If} $oKind == "dir"
      StrCpy $oOk "yes"
    ${EndIf}
  ${ElseIf} "${NAME}" == "electron"
  ${OrIf} "${NAME}" == "logs"
  ${OrIf} "${NAME}" == "pending"
  ${OrIf} "${NAME}" == "setup-tmp"
    !insertmacro orientaKindOf "${LEVEL}\${NAME}"
    ${If} $oKind == "dir"
      !insertmacro orientaChildrenOk "${LEVEL}\${NAME}" "${NAME}"
    ${EndIf}
  ${EndIf}
!macroend

; Is PATH -- and the chain Orienta\Orienta\... below it, to 16 levels -- made of
; nothing but leftovers (shapes above), operating-system litter files and the
; next Orienta? Sets $oLight, and $oEntries = the top-level entries that are
; not litter. A link named Orienta, an unknown entry, or a chain deeper than 16
; answers no.
!macro orientaLevelsLight PATH
  Push $R5
  Push $R6
  Push $R7
  Push $R8
  StrCpy $oLight "yes"
  StrCpy $oEntries 0
  StrCpy $R6 "${PATH}"
  StrCpy $R8 ""
  ${For} $R7 1 17
    !insertmacro orientaKindOf "$R6"
    ${If} $oKind != "dir"
      StrCpy $oLight "no"
      StrCpy $R8 ""
      ${ExitFor}
    ${EndIf}
    StrCpy $R8 ""
    FindFirst $oFind2 $oName2 "$R6\*"
    ${DoWhile} $oName2 != ""
      ${If} $oName2 != "."
      ${AndIf} $oName2 != ".."
        StrCpy $R5 "no"
        ${If} $oName2 == "Thumbs.db"
        ${OrIf} $oName2 == "desktop.ini"
        ${OrIf} $oName2 == ".DS_Store"
          !insertmacro orientaKindOf "$R6\$oName2"
          ${If} $oKind == "file"
            StrCpy $R5 "litter"
          ${EndIf}
        ${EndIf}
        ${If} $R5 == "no"
        ${AndIf} $oName2 == "Orienta"
          !insertmacro orientaKindOf "$R6\$oName2"
          ${If} $oKind == "dir"
            StrCpy $R5 "chain"
          ${EndIf}
        ${EndIf}
        ${If} $R5 == "no"
          !insertmacro orientaEntryLeftover "$R6" "$oName2"
          ${If} $oOk == "yes"
            StrCpy $R5 "leftover"
          ${EndIf}
        ${EndIf}
        ${If} $R5 == "no"
          StrCpy $oLight "no"
        ${ElseIf} $R5 == "chain"
          ; Two folders that NSIS's case-insensitive == both takes for "Orienta"
          ; (Orienta and orienta on a case-sensitive directory): only one of them
          ; would be followed and checked, and the cleaning below follows the
          ; literal name -- which may be the other one. Never guess which.
          ${If} $R8 != ""
            StrCpy $oLight "no"
          ${Else}
            StrCpy $R8 "$R6\$oName2"
          ${EndIf}
        ${EndIf}
        ${If} $R7 = 1
        ${AndIf} $R5 != "litter"
          IntOp $oEntries $oEntries + 1
        ${EndIf}
      ${EndIf}
      FindNext $oFind2 $oName2
    ${Loop}
    FindClose $oFind2
    ${If} $oLight == "no"
      ${ExitFor}
    ${EndIf}
    ${If} $R8 == ""
      ${ExitFor}
    ${EndIf}
    StrCpy $R6 "$R8"
  ${Next}
  ; Still a next level after 16: too deep to judge.
  ${If} $R8 != ""
    StrCpy $oLight "no"
  ${EndIf}
  Pop $R8
  Pop $R7
  Pop $R6
  Pop $R5
!macroend

; Kept under its old name for the callers that only want the verdict.
!macro orientaOnlyLight PATH
  !insertmacro orientaLevelsLight "${PATH}"
!macroend

!macro orientaPruneChain PATH
  Push $R6
  Push $R7
  Push $R8
  StrCpy $R6 "${PATH}"
  StrCpy $R7 0
  ${For} $R8 1 16
    !insertmacro orientaKindOf "$R6\Orienta"
    ${If} $oKind == "dir"
      StrCpy $R6 "$R6\Orienta"
      IntOp $R7 $R7 + 1
    ${Else}
      ${ExitFor}
    ${EndIf}
  ${Next}
  ${While} $R7 > 0
    Delete "$R6\Thumbs.db"
    Delete "$R6\desktop.ini"
    Delete "$R6\.DS_Store"
    RMDir "$R6"
    StrCpy $R6 $R6 -8
    IntOp $R7 $R7 - 1
  ${EndWhile}
  Pop $R8
  Pop $R7
  Pop $R6
!macroend

; ---------------------------------------------------------------------------
; Split PATH at its last backslash: $oParent, and $oBase (what follows). Both
; empty for a path with no backslash.
; ---------------------------------------------------------------------------
!macro orientaSplit PATH
  Push $R6
  Push $R7
  Push $R8
  StrCpy $R6 "${PATH}"
  StrLen $R7 $R6
  StrCpy $oParent ""
  StrCpy $oBase ""
  ${Do}
    IntOp $R7 $R7 - 1
    ${If} $R7 < 0
      ${ExitDo}
    ${EndIf}
    StrCpy $R8 $R6 1 $R7
    ${If} $R8 == "\"
      StrCpy $oParent $R6 $R7
      IntOp $R7 $R7 + 1
      StrCpy $oBase $R6 "" $R7
      ${ExitDo}
    ${EndIf}
  ${Loop}
  Pop $R8
  Pop $R7
  Pop $R6
!macroend

; ---------------------------------------------------------------------------
; Remove the leftovers of ONE level, by fixed name. Called only for a level that
; orientaLevelsLight has just found to hold nothing else, so every name here has
; been checked for its shape; the marker and python\ and runtime\ are not on it.
; ---------------------------------------------------------------------------
!macro orientaWipeLevel LEVEL
  ; Copied to a GLOBAL first: orientaRemovePath re-expands its argument after it
  ; has reused R6..R9, so a register-based path would point somewhere else.
  StrCpy $oWipe "${LEVEL}"
  !insertmacro orientaRemovePath "$oWipe\electron"
  !insertmacro orientaRemovePath "$oWipe\logs"
  !insertmacro orientaRemovePath "$oWipe\setup-tmp"
  !insertmacro orientaRemovePath "$oWipe\pending"
  !insertmacro orientaRemovePath "$oWipe\pending.json"
  !insertmacro orientaRemovePath "$oWipe\.python_path"
  !insertmacro orientaRemovePath "$oWipe\.install_mode"
  !insertmacro orientaRemovePath "$oWipe\.install_incomplete"
  !insertmacro orientaRemovePath "$oWipe\.packages_lock.json"
  !insertmacro orientaRemovePath "$oWipe\.packages_sync.json"
  !insertmacro orientaRemovePath "$oWipe\.packages_sync_failed.json"
  ; Litter FILES only: Delete does not touch a folder of that name.
  Delete "$oWipe\Thumbs.db"
  Delete "$oWipe\desktop.ini"
  Delete "$oWipe\.DS_Store"
!macroend

; ---------------------------------------------------------------------------
; The chain levels BELOW PATH, wiped and removed from the deepest up. Only after
; orientaLevelsLight has just answered "yes" for PATH: every level was checked.
; PATH itself is not touched here.
; ---------------------------------------------------------------------------
!macro orientaCleanBelow PATH
  Push $R6
  Push $R7
  Push $R8
  StrCpy $R6 "${PATH}"
  StrCpy $R7 0
  ${For} $R8 1 16
    !insertmacro orientaKindOf "$R6\Orienta"
    ${If} $oKind == "dir"
      StrCpy $R6 "$R6\Orienta"
      IntOp $R7 $R7 + 1
    ${Else}
      ${ExitFor}
    ${EndIf}
  ${Next}
  ${While} $R7 > 0
    !insertmacro orientaWipeLevel "$R6"
    RMDir "$R6"
    StrCpy $R6 $R6 -8
    IntOp $R7 $R7 - 1
  ${EndWhile}
  Pop $R8
  Pop $R7
  Pop $R6
!macroend

; ---------------------------------------------------------------------------
; A folder that holds only leftovers (see orientaLevelsLight) is cleaned, level
; by level from the deepest chain level up, and removed if that leaves it empty.
; Anything else in it -- in ANY level -- and nothing is touched. Used for the
; default folder and for the levels above a nested data folder.
; ---------------------------------------------------------------------------
!macro orientaCleanLight PATH
  !insertmacro orientaLevelsLight "${PATH}"
  ${If} $oLight == "yes"
    !insertmacro orientaCleanBelow "${PATH}"
    !insertmacro orientaWipeLevel "${PATH}"
    RMDir "${PATH}"
    DetailPrint "Orienta: cleaned the leftover folder ${PATH}"
  ${EndIf}
!macroend

; ---------------------------------------------------------------------------
; The whole decision.
; ---------------------------------------------------------------------------
!macro orientaRemoveData
  Var /GLOBAL oHome
  Var /GLOBAL oPointer
  Var /GLOBAL oSource
  Var /GLOBAL oAppData
  Var /GLOBAL oLocalAppData
  Var /GLOBAL oKind
  Var /GLOBAL oVerdict
  Var /GLOBAL oHomeIsLink
  Var /GLOBAL oCount
  Var /GLOBAL oRemove
  Var /GLOBAL oRemoveLib
  Var /GLOBAL oLeftBehind
  Var /GLOBAL oLibWasLink
  Var /GLOBAL oFind
  Var /GLOBAL oName
  Var /GLOBAL oMsgAsk
  Var /GLOBAL oMsgLib
  Var /GLOBAL oMsgKept
  Var /GLOBAL oMsgLinked
  Var /GLOBAL oMsgUnsure
  Var /GLOBAL oMsgLeft
  Var /GLOBAL oMsgMissing
  Var /GLOBAL oOk
  Var /GLOBAL oAll
  Var /GLOBAL oMarkerOk
  Var /GLOBAL oWipe
  Var /GLOBAL oCmp
  Var /GLOBAL oLen
  Var /GLOBAL oLight
  Var /GLOBAL oEntries
  Var /GLOBAL oFind2
  Var /GLOBAL oName2
  Var /GLOBAL oFind4
  Var /GLOBAL oName4
  Var /GLOBAL oParent
  Var /GLOBAL oBase
  Var /GLOBAL oDefault

  Push $R0
  Push $R1
  Push $R2

  StrCpy $oLeftBehind "no"
  StrCpy $oLibWasLink "no"
  StrCpy $oHomeIsLink "no"
  StrCpy $oRemove "no"
  StrCpy $oRemoveLib "no"
  StrCpy $oVerdict "ours"

  ; The data belongs to the USER who ran Orienta. After an "install for all
  ; users" NSIS's context is `all`, where $LOCALAPPDATA means C:\ProgramData.
  ; Switched to the current user and back, as electron-builder's own
  ; app-data option does. Nothing below returns early, so the restore runs.
  ${If} $installMode == "all"
    SetShellVarContext current
  ${EndIf}

  ; Overridable ONLY in a test build: NSIS reads these from Windows, so a
  ; test would otherwise touch the real preference and data folder.
  StrCpy $oAppData "$APPDATA"
  StrCpy $oLocalAppData "$LOCALAPPDATA"
  !ifdef ORIENTA_TEST_APPDATA
    StrCpy $oAppData "${ORIENTA_TEST_APPDATA}"
  !endif
  !ifdef ORIENTA_TEST_LOCALAPPDATA
    StrCpy $oLocalAppData "${ORIENTA_TEST_LOCALAPPDATA}"
  !endif

  ; ---- where is it? Same precedence as orientaHome() in the app:
  ;      ORIENTA_HOME, then the recorded choice, then the default.
  StrCpy $oPointer "$oAppData\Orienta\home.txt"
  ReadEnvStr $oHome "ORIENTA_HOME"
  StrCpy $oSource "env"
  ${If} $oHome == ""
  ${AndIf} ${FileExists} "$oPointer"
    StrCpy $oSource "pointer"
    ClearErrors
    FileOpen $R0 "$oPointer" r
    ${IfNot} ${Errors}
      FileReadByte $R0 $R1
      FileReadByte $R0 $R2
      ${If} $R1 = 255
      ${AndIf} $R2 = 254
        ; UTF-16LE, as the app writes it -- the one encoding NSIS reads
        ; faithfully, so a path with non-ASCII characters survives.
        FileReadUTF16LE $R0 $oHome
      ${Else}
        ; Hand-edited. Read through the ANSI code page: correct for ASCII,
        ; and a non-ASCII path simply fails to resolve below -- which then
        ; changes nothing and says so.
        FileSeek $R0 0 SET
        FileRead $R0 $oHome
        StrCpy $R1 $oHome 3
        ${If} $R1 == "ï»¿"
          StrCpy $oHome $oHome "" 3
        ${EndIf}
      ${EndIf}
      FileClose $R0
    ${EndIf}
  ${EndIf}
  ${If} $oHome == ""
    StrCpy $oHome "$oLocalAppData\Orienta"
    StrCpy $oSource "default"
  ${EndIf}

  ; Trim both ends, as the app does.
  ${Do}
    StrCpy $R1 $oHome 1 -1
    ${If} $R1 == "$\r"
    ${OrIf} $R1 == "$\n"
    ${OrIf} $R1 == " "
      StrCpy $oHome $oHome -1
    ${Else}
      ${ExitDo}
    ${EndIf}
  ${Loop}
  ${Do}
    StrCpy $R1 $oHome 1
    ${If} $R1 == " "
      StrCpy $oHome $oHome "" 1
    ${Else}
      ${ExitDo}
    ${EndIf}
  ${Loop}

  ; Absolute only (X:\... or \\server\...), as the app requires. A relative
  ; path would resolve against the uninstaller's own folder.
  ; `X:x` is drive-RELATIVE, not absolute, so the third character must be a
  ; backslash too. And never \\?\ or \\.\: GetFullPathNameW leaves those
  ; untouched, so `\\?\C:\.` would reach the checks below un-normalised, and
  ; the app never produces one unless the user types it.
  StrCpy $R1 $oHome 4
  ${If} $R1 == "\\?\"
  ${OrIf} $R1 == "\\.\"
    StrCpy $oVerdict "unsure"
  ${EndIf}
  StrCpy $R1 $oHome 2
  StrCpy $R2 $oHome 2 1
  ${If} $R1 != "\\"
  ${AndIf} $R2 != ":\"
    StrCpy $oVerdict "unsure"
  ${EndIf}

  ; Normalised, so `C:\.`, `C:\\` or `D:\x\..` cannot pass for something else.
  ${If} $oVerdict == "ours"
    System::Call 'kernel32::GetFullPathNameW(w "$oHome", i ${NSIS_MAX_STRLEN}, w .R1, p 0) i .R2'
    ; A result that does not fit returns the size it NEEDS and leaves the
    ; buffer empty -- which read as "" would be the root of the current drive.
    ${If} $R2 > 0
    ${AndIf} $R2 < ${NSIS_MAX_STRLEN}
      StrCpy $oHome $R1
    ${Else}
      StrCpy $oVerdict "unsure"
    ${EndIf}
    ${Do}
      StrCpy $R1 $oHome 1 -1
      ${If} $R1 == "\"
        StrCpy $oHome $oHome -1
      ${Else}
        ${ExitDo}
      ${EndIf}
    ${Loop}
  ${EndIf}

  ; ---- the sentences, in the language the installer was run in ----
  ${If} $LANGUAGE == 1031
    StrCpy $oMsgAsk "Auch die Bestandteile entfernen, die Orienta heruntergeladen hat (Python und die Programmdateien, 2,5 bis 8 GB)?$\r$\n$\r$\nOrdner: $oHome$\r$\n$\r$\nIhre Kristallbibliothek bleibt erhalten, sofern Sie im nächsten Schritt nichts anderes wählen."
    StrCpy $oMsgLib "Auch Ihre Kristallbibliothek LÖSCHEN?$\r$\n$\r$\n$oHome\runtime\Database$\r$\n$\r$\nDas lässt sich nicht rückgängig machen. Wählen Sie „Nein“, um sie zu behalten — bei einer Neuinstallation wird sie wiedergefunden."
    StrCpy $oMsgKept "Ihre Kristallbibliothek wurde behalten:$\r$\n$oHome\runtime\Database"
    StrCpy $oMsgLinked "Ihre Kristallbibliothek ist eine Verknüpfung auf einen anderen Ordner. Entfernt wurde nur die Verknüpfung; die Bibliothek selbst wurde nicht angefasst."
    StrCpy $oMsgUnsure "Der Datenordner von Orienta wurde nicht angetastet, weil er sich nicht sicher als solcher erkennen ließ:$\r$\n$oHome$\r$\n$\r$\nSie können ihn von Hand löschen."
    StrCpy $oMsgLeft "Einiges im Datenordner ließ sich nicht sicher entfernen und liegt noch da:$\r$\n$oHome"
    StrCpy $oMsgMissing "Der Datenordner von Orienta wurde nicht gefunden, es wurde nichts entfernt:$\r$\n$oHome$\r$\n$\r$\nLiegt er auf einem Laufwerk, das gerade nicht verbunden ist, bleibt er dort unverändert."
  ${ElseIf} $LANGUAGE == 1041
    StrCpy $oMsgAsk "Orienta がダウンロードした構成要素（Python とプログラムファイル、2.5〜8 GB）も削除しますか？$\r$\n$\r$\nフォルダー: $oHome$\r$\n$\r$\n次の手順で別の選択をしない限り、結晶ライブラリは残ります。"
    StrCpy $oMsgLib "結晶ライブラリも削除しますか？$\r$\n$\r$\n$oHome\runtime\Database$\r$\n$\r$\nこの操作は元に戻せません。残す場合は「いいえ」を選んでください。Orienta を再インストールすると再び使用されます。"
    StrCpy $oMsgKept "結晶ライブラリは残しました:$\r$\n$oHome\runtime\Database"
    StrCpy $oMsgLinked "結晶ライブラリは別のフォルダーへのリンクです。削除したのはリンクだけで、ライブラリ本体には触れていません。"
    StrCpy $oMsgUnsure "Orienta のデータフォルダーを安全に特定できなかったため、そのまま残しました:$\r$\n$oHome$\r$\n$\r$\n手動で削除できます。"
    StrCpy $oMsgLeft "データフォルダーの一部を安全に削除できず、残っています:$\r$\n$oHome"
    StrCpy $oMsgMissing "Orienta のデータフォルダーが見つからなかったため、何も削除していません:$\r$\n$oHome$\r$\n$\r$\n現在接続されていないドライブにある場合は、そのまま残ります。"
  ${ElseIf} $LANGUAGE == 2052
    StrCpy $oMsgAsk "是否同时删除 Orienta 下载的组件（Python 和程序文件，2.5–8 GB）？$\r$\n$\r$\n文件夹：$oHome$\r$\n$\r$\n除非您在下一步另行选择，否则晶体库将被保留。"
    StrCpy $oMsgLib "是否同时删除您的晶体库？$\r$\n$\r$\n$oHome\runtime\Database$\r$\n$\r$\n此操作无法撤销。选择“否”即可保留 — 重新安装 Orienta 时会再次找到它。"
    StrCpy $oMsgKept "您的晶体库已保留：$\r$\n$oHome\runtime\Database"
    StrCpy $oMsgLinked "您的晶体库是指向其他文件夹的链接。仅删除了该链接，晶体库本身未被改动。"
    StrCpy $oMsgUnsure "无法安全确认 Orienta 的数据文件夹，因此未作改动：$\r$\n$oHome$\r$\n$\r$\n您可以手动删除。"
    StrCpy $oMsgLeft "数据文件夹中的部分内容无法安全删除，仍然保留：$\r$\n$oHome"
    StrCpy $oMsgMissing "未找到 Orienta 的数据文件夹，未删除任何内容：$\r$\n$oHome$\r$\n$\r$\n如果它位于当前未连接的驱动器上，将保持不变。"
  ${Else}
    StrCpy $oMsgAsk "Also remove the components Orienta downloaded (Python and the program files, 2.5 to 8 GB)?$\r$\n$\r$\nFolder: $oHome$\r$\n$\r$\nYour crystal library is kept unless you choose otherwise in the next step."
    StrCpy $oMsgLib "Also DELETE your crystal library?$\r$\n$\r$\n$oHome\runtime\Database$\r$\n$\r$\nThis cannot be undone. Choose No to keep it — it will be found again if you reinstall Orienta."
    StrCpy $oMsgKept "Your crystal library was kept:$\r$\n$oHome\runtime\Database"
    StrCpy $oMsgLinked "Your crystal library is a link to another folder. Only the link was removed; the library itself was not touched."
    StrCpy $oMsgUnsure "Orienta's data folder was left in place because it could not be identified safely:$\r$\n$oHome$\r$\n$\r$\nYou can delete it by hand."
    StrCpy $oMsgLeft "Some of the data folder could not be removed safely and is still there:$\r$\n$oHome"
    StrCpy $oMsgMissing "Orienta's data folder was not found, so nothing was removed:$\r$\n$oHome$\r$\n$\r$\nIf it is on a drive that is not connected right now, it stays there unchanged."
  ${EndIf}

  ${If} $oVerdict == "ours"
  ${AndIfNot} ${FileExists} "$oHome\*.*"
    ; Unreachable is NOT gone: an unplugged drive, an offline share. The
    ; pointer is the only record of where the library is, so it stays, and
    ; the user is told. (Silent for the default location, where "not there"
    ; simply means nothing was ever installed.)
    StrCpy $oVerdict "missing"
    ${If} $oSource != "default"
      MessageBox MB_OK|MB_ICONINFORMATION "$oMsgMissing" /SD IDOK
    ${EndIf}
  ${EndIf}

  ; ---- is this recognisably Orienta's? ----
  ${If} $oVerdict == "ours"
    StrCpy $R1 "no"
    ${If} ${FileExists} "$oHome\.orienta-home"
      StrCpy $R1 "yes"
    ${ElseIf} ${FileExists} "$oHome\runtime\VERSION"
      ${If} ${FileExists} "$oHome\.python_path"
        StrCpy $R1 "yes"
      ${ElseIf} ${FileExists} "$oHome\python\python.exe"
        StrCpy $R1 "yes"
      ${EndIf}
    ${EndIf}
    ; The DEFAULT folder that only ever held what the application writes before
    ; any setup (electron\, logs\) and, after repeated "change..." clicks of
    ; 0.4.6, an empty Orienta\Orienta chain: no marker, nothing of anyone
    ; else's, and nothing to lose. Only for the default location -- a folder the
    ; user or ORIENTA_HOME named is judged by its markers alone.
    ${If} $R1 != "yes"
    ${AndIf} $oSource == "default"
      !insertmacro orientaOnlyLight "$oHome"
      ${If} $oLight == "yes"
      ${AndIf} $oEntries > 0
        StrCpy $R1 "yes"
      ${EndIf}
    ${EndIf}
    ; A repository is never a data folder, however many markers it carries.
    ${If} ${FileExists} "$oHome\.git"
      StrCpy $R1 "no"
    ${EndIf}
    ; Never a drive root: after normalisation and trimming, "X:" is 2 long.
    StrLen $R2 $oHome
    ${If} $R2 < 4
      StrCpy $R1 "no"
    ${EndIf}
    ; Never a share root: \\server\share has three backslashes, a folder
    ; inside it at least four.
    StrCpy $R2 $oHome 2
    ${If} $R2 == "\\"
      StrCpy $oCount 0
      StrCpy $R2 0
      ${Do}
        StrCpy $R0 $oHome 1 $R2
        ${If} $R0 == ""
          ${ExitDo}
        ${EndIf}
        ${If} $R0 == "\"
          IntOp $oCount $oCount + 1
        ${EndIf}
        IntOp $R2 $R2 + 1
      ${Loop}
      ${If} $oCount < 4
        StrCpy $R1 "no"
      ${EndIf}
    ${EndIf}
    ; Never the profile or the system.
    ${If} $oHome == "$PROFILE"
    ${OrIf} $oHome == "$WINDIR"
    ${OrIf} $oHome == "$DESKTOP"
    ${OrIf} $oHome == "$DOCUMENTS"
      StrCpy $R1 "no"
    ${EndIf}
    ${If} $R1 != "yes"
      StrCpy $oVerdict "unsure"
    ${EndIf}
  ${EndIf}

  ${If} $oVerdict == "unsure"
    DetailPrint "Orienta: data folder not recognised, left in place: $oHome"
    MessageBox MB_OK|MB_ICONINFORMATION "$oMsgUnsure" /SD IDOK
  ${EndIf}

  ${If} $oVerdict == "ours"
    ; A data folder that is itself a link was moved there deliberately. Its
    ; contents are the data and are handled normally -- but the LINK is never
    ; removed, or a kept library would be orphaned behind a vanished path.
    !insertmacro orientaKindOf "$oHome"
    ${If} $oKind == "linkdir"
      StrCpy $oHomeIsLink "yes"
    ${EndIf}

    ; ---- question 1 ----
    !ifdef ORIENTA_TEST_ANSWER_COMPONENTS
      StrCpy $oRemove "${ORIENTA_TEST_ANSWER_COMPONENTS}"
    !else
      ${If} ${Cmd} `MessageBox MB_YESNO|MB_ICONQUESTION "$oMsgAsk" /SD IDNO IDYES`
        StrCpy $oRemove "yes"
      ${EndIf}
    !endif
  ${EndIf}

  ${If} $oRemove == "yes"
    ; ---- question 2, only if there is a library to ask about ----
    ${If} ${FileExists} "$oHome\runtime\Database\*.*"
      !ifdef ORIENTA_TEST_ANSWER_LIBRARY
        StrCpy $oRemoveLib "${ORIENTA_TEST_ANSWER_LIBRARY}"
      !else
        ; MB_DEFBUTTON2: "No" is the button Enter presses.
        ${If} ${Cmd} `MessageBox MB_YESNO|MB_ICONEXCLAMATION|MB_DEFBUTTON2 "$oMsgLib" /SD IDNO IDYES`
          StrCpy $oRemoveLib "yes"
        ${EndIf}
      !endif
    ${EndIf}

    ; ---- the top level, by name ----
    !insertmacro orientaRemovePath "$oHome\python"
    !insertmacro orientaRemovePath "$oHome\electron"
    !insertmacro orientaRemovePath "$oHome\logs"
    !insertmacro orientaRemovePath "$oHome\setup-tmp"
    !insertmacro orientaRemovePath "$oHome\pending"
    !insertmacro orientaRemovePath "$oHome\pending.json"
    !insertmacro orientaRemovePath "$oHome\.python_path"
    !insertmacro orientaRemovePath "$oHome\.install_mode"
    !insertmacro orientaRemovePath "$oHome\.install_incomplete"
    !insertmacro orientaRemovePath "$oHome\.packages_lock.json"
    !insertmacro orientaRemovePath "$oHome\.packages_sync.json"
    !insertmacro orientaRemovePath "$oHome\.packages_sync_failed.json"
    ; A repair renames the old interpreter aside as python.old-<time>.
    FindFirst $oFind $oName "$oHome\python.old-*"
    ${DoWhile} $oName != ""
      !insertmacro orientaRemovePath "$oHome\$oName"
      FindNext $oFind $oName
    ${Loop}
    FindClose $oFind

    ; ---- runtime\: the program folder, emptied except Database ----
    !insertmacro orientaKindOf "$oHome\runtime"
    ${If} $oKind == "linkdir"
      ; Linked elsewhere -- a developer checkout, perhaps. Its contents are
      ; not Orienta's to judge: only the link goes. The library behind it is
      ; untouched, and a user who asked for it to be deleted must be told so
      ; rather than left believing it is gone.
      !insertmacro orientaRemovePath "$oHome\runtime"
      ${If} $oRemoveLib == "yes"
        StrCpy $oLibWasLink "yes"
      ${EndIf}
    ${ElseIf} $oKind == "dir"
      ; Two passes: deleting while enumerating may skip an entry.
      ${For} $R0 1 2
        FindFirst $oFind $oName "$oHome\runtime\*"
        ${DoWhile} $oName != ""
          ${If} $oName != "."
          ${AndIf} $oName != ".."
          ${AndIf} $oName != "Database"
            !insertmacro orientaRemovePath "$oHome\runtime\$oName"
          ${EndIf}
          FindNext $oFind $oName
        ${Loop}
        FindClose $oFind
      ${Next}

      ${If} $oRemoveLib == "yes"
        !insertmacro orientaKindOf "$oHome\runtime\Database"
        ${If} $oKind == "linkdir"
          StrCpy $oLibWasLink "yes"
        ${EndIf}
        !insertmacro orientaRemovePath "$oHome\runtime\Database"
      ${EndIf}
      ; Only if now empty: RMDir without /r refuses a non-empty folder.
      RMDir "$oHome\runtime"
    ${EndIf}

    ; An empty Orienta\Orienta chain inside the folder (0.4.6's repeated
    ; "change..." clicks) would keep the folder from being removed below.
    ; Empty directories only: RMDir refuses anything with a file in it.
    ; Levels of the chain that hold only Orienta's leftovers (0.4.6 told the user
    ; to restart after every "change...", so each level holds electron\ and
    ; logs\): checked level by level, and only then wiped.
    ; The home's own marker is still here (it goes last), and would make the
    ; level "not only leftovers" for ever: it is allowed for this one call.
    StrCpy $oMarkerOk "yes"
    !insertmacro orientaLevelsLight "$oHome"
    StrCpy $oMarkerOk "no"
    ${If} $oLight == "yes"
      !insertmacro orientaCleanBelow "$oHome"
    ${EndIf}
    ; Operating-system litter FILES at the top do not keep the folder alive.
    Delete "$oHome\Thumbs.db"
    Delete "$oHome\desktop.ini"
    Delete "$oHome\.DS_Store"
    !insertmacro orientaPruneChain "$oHome"

    ; The marker goes last, and only once nothing is left to identify.
    ${IfNot} ${FileExists} "$oHome\runtime\*.*"
      !insertmacro orientaRemovePath "$oHome\.orienta-home"
    ${EndIf}

    ; The pointer goes only if THIS run removed the folder. "The folder is not
    ; there any more" is also what an unplugged drive or a dropped share looks
    ; like, and deleting the pointer then loses the only record of where a
    ; kept library is.
    StrCpy $R2 "kept"
    ${If} $oHomeIsLink == "no"
      ClearErrors
      RMDir "$oHome"
      ${IfNot} ${Errors}
        StrCpy $R2 "removed"
      ${EndIf}
    ${EndIf}

    ${If} $R2 == "removed"
    ${AndIf} $oSource == "pointer"
      ; Removed by this run, and the recorded choice was what pointed here.
      Delete "$oPointer"
      ; The record of the volume the folder was on (electron/volume_id.js) goes with
      ; it, or it would keep the settings folder alive.
      Delete "$oAppData\Orienta\home.volume"
      RMDir "$oAppData\Orienta"

      ; The levels ABOVE a data folder that sat inside a chain of Orienta
      ; folders. Each goes only if it is named Orienta, is a plain directory
      ; (never a link) and is empty -- RMDir refuses otherwise, and the walk
      ; stops at the first level that refuses.
      StrCpy $R0 "$oHome"
      ${For} $R1 1 16
        !insertmacro orientaSplit "$R0"
        ${If} $oParent == ""
          ${ExitFor}
        ${EndIf}
        StrCpy $R0 "$oParent"
        !insertmacro orientaSplit "$R0"
        ${If} $oBase != "Orienta"
          ${ExitFor}
        ${EndIf}
        !insertmacro orientaKindOf "$R0"
        ${If} $oKind != "dir"
          ${ExitFor}
        ${EndIf}
        ; Leftovers the application wrote into that level before the next click
        ; (electron\, logs\) go first -- the same rule as for the default folder,
        ; and again only if that is ALL the level holds.
        !insertmacro orientaCleanLight "$R0"
        ClearErrors
        RMDir "$R0"
        ${If} ${Errors}
          ${ExitFor}
        ${EndIf}
        DetailPrint "Orienta: removed empty folder $R0"
      ${Next}
    ${EndIf}

    ; The DEFAULT folder, when the data was recorded somewhere else. Before the
    ; user's choice the application had already written electron\ and logs\
    ; there, and the repeated clicks left an empty chain beside them: all of it
    ; Orienta's, none of it worth keeping -- but only if that is ALL there is.
    ; One file of anybody's, or a python\ or runtime\ (a real installation),
    ; and the whole folder is left exactly as it is.
    StrCpy $oDefault "$oLocalAppData\Orienta"
    ${If} $oSource == "pointer"
    ${AndIf} $oDefault != $oHome
      !insertmacro orientaCleanLight "$oDefault"
    ${EndIf}

    ${If} $oLibWasLink == "yes"
      MessageBox MB_OK|MB_ICONINFORMATION "$oMsgLinked" /SD IDOK
    ${ElseIf} ${FileExists} "$oHome\runtime\Database\*.*"
      MessageBox MB_OK|MB_ICONINFORMATION "$oMsgKept" /SD IDOK
    ${EndIf}
    ${If} $oLeftBehind == "yes"
      MessageBox MB_OK|MB_ICONEXCLAMATION "$oMsgLeft" /SD IDOK
    ${EndIf}
  ${EndIf}

  !ifdef ORIENTA_TEST_DUMP
    ; Test builds only: what was decided, for the test to read.
    FileOpen $R0 "${ORIENTA_TEST_DUMP}" w
    FileWriteUTF16LE /BOM $R0 "verdict=$oVerdict$\r$\nsource=$oSource$\r$\nhome=$oHome$\r$\nlibWasLink=$oLibWasLink$\r$\nleftBehind=$oLeftBehind$\r$\nask=$oMsgAsk$\r$\nlib=$oMsgLib$\r$\n"
    FileClose $R0
  !endif

  ${If} $installMode == "all"
    SetShellVarContext all
  ${EndIf}

  Pop $R2
  Pop $R1
  Pop $R0
!macroend

; ---------------------------------------------------------------------------
; The hook electron-builder calls. Only for a real uninstall.
; ---------------------------------------------------------------------------
!macro customUnInstall
  ${ifNot} ${isUpdated}
    !insertmacro orientaRemoveData
  ${endIf}
!macroend
