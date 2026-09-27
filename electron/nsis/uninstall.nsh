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
      RMDir "$oAppData\Orienta"
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
