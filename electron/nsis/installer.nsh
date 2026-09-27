; Orienta's additions to electron-builder's NSIS installer. electron-builder
; takes exactly one include file (package.json build.nsis.include); this is it.
;
;   uninstall.nsh   what happens to the DATA folder on uninstall
;   below           a "create a desktop shortcut" checkbox on the finish page
;
; UTF-8 WITH a byte-order mark, like uninstall.nsh: NSIS reads an include
; without one through the ANSI code page.

!include "${__FILEDIR__}\uninstall.nsh"

; ---------------------------------------------------------------------------
; The desktop shortcut, as a choice.
;
; electron-builder has no checkbox for it: `createDesktopShortcut` is either
; always or never. It is kept at `true` deliberately, so electron-builder still
; CREATES the shortcut, re-points it on a rename, and REMOVES it on uninstall.
; Turning its creation off and making the shortcut here instead was the obvious
; design and the wrong one: electron-builder's uninstaller deletes the desktop
; link only while its own creation is enabled (uninstaller.nsh, guarded by
; DO_NOT_CREATE_DESKTOP_SHORTCUT), so a shortcut made here would outlive the
; application.
;
; So electron-builder creates it, and the checkbox settles the result as the
; page closes: unticked, the shortcut is taken away; ticked, it is made if it
; is not there. The second half matters on a REINSTALL, where electron-builder
; keeps shortcuts as they were -- a shortcut removed on the first install was
; not re-made, while the box said "create". Made here under the same path as
; electron-builder's own, so its uninstaller (which deletes by path) still
; removes it. The box always matches the result.
;
; A silent install never shows this page and keeps electron-builder's default,
; which is to create the shortcut -- the behaviour before this existed.
;
; This macro replaces electron-builder's own finish page, so its "Run Orienta"
; checkbox is reproduced here exactly as in templates/nsis/assistedInstaller.nsh.
; ---------------------------------------------------------------------------
!macro customFinishPage
  Var orientaShortcutText

  Function orientaFinishPre
    ; The label is chosen at run time from the installer's language. A
    ; LangString would have to be declared after the languages are loaded,
    ; and this file is included before them.
    ${If} $LANGUAGE == 1031
      StrCpy $orientaShortcutText "Verknüpfung auf dem Desktop anlegen"
    ${ElseIf} $LANGUAGE == 1041
      StrCpy $orientaShortcutText "デスクトップにショートカットを作成する"
    ${ElseIf} $LANGUAGE == 2052
      StrCpy $orientaShortcutText "在桌面上创建快捷方式"
    ${ElseIf} $LANGUAGE == 1028
      StrCpy $orientaShortcutText "在桌面上建立捷徑"
    ${Else}
      StrCpy $orientaShortcutText "Create a shortcut on the desktop"
    ${EndIf}
  FunctionEnd

  Function orientaShortcutWanted
    ; Ticked at Finish: electron-builder has already made it. Nothing to do.
  FunctionEnd


  !ifndef HIDE_RUN_AFTER_FINISH
    Function StartApp
      ${if} ${isUpdated}
        StrCpy $1 "--updated"
      ${else}
        StrCpy $1 ""
      ${endif}
      ${StdUtils.ExecShellAsUser} $0 "$launchLink" "open" "$1"
    FunctionEnd

    !define MUI_FINISHPAGE_RUN
    !define MUI_FINISHPAGE_RUN_FUNCTION "StartApp"
  !endif

  !define MUI_FINISHPAGE_SHOWREADME ""
  !define MUI_FINISHPAGE_SHOWREADME_TEXT "$orientaShortcutText"
  !define MUI_FINISHPAGE_SHOWREADME_FUNCTION orientaShortcutWanted
  !define MUI_PAGE_CUSTOMFUNCTION_PRE orientaFinishPre
  !define MUI_PAGE_CUSTOMFUNCTION_LEAVE orientaFinishLeave
  !insertmacro MUI_PAGE_FINISH

  ; Defined AFTER the page: MUI declares $mui.FinishPage.ShowReadme only when
  ; MUI_PAGE_FINISH is inserted, and a function above it would read an
  ; unknown variable -- which, without warnings-as-errors, compiles into a
  ; checkbox that does nothing.
  Function orientaFinishLeave
    Push $0
    ; BM_GETCHECK (0x00F0) on the checkbox MUI keeps in this variable;
    ; BST_CHECKED is 1.
    ; No checkbox at all when MUI shows the reboot question instead: a
    ; SendMessage to window 0 returns 0, which would read as "unticked" and
    ; delete a shortcut the user was never asked about.
    ${If} $mui.FinishPage.ShowReadme <> 0
      SendMessage $mui.FinishPage.ShowReadme 0x00F0 0 0 $0
      ${If} $0 <> 1
        WinShell::UninstShortcut "$newDesktopLink"
        Delete "$newDesktopLink"
      ${ElseIfNot} ${FileExists} "$newDesktopLink"
        ; Ticked, and still no shortcut: on a REINSTALL electron-builder keeps
        ; shortcuts as they were, so one removed earlier is not made again —
        ; and the box would say "create" over a desktop with nothing on it.
        ; Made exactly as electron-builder makes it, under the same path, so
        ; its uninstaller (which deletes by path) removes this one too.
        CreateShortCut "$newDesktopLink" "$appExe" "" "$appExe" 0 "" "" "${APP_DESCRIPTION}"
        ClearErrors
        WinShell::SetLnkAUMI "$newDesktopLink" "${APP_ID}"
      ${EndIf}
    ${EndIf}
    Pop $0
  FunctionEnd
!macroend
