# =============================================================================
#  make_shortcut.ps1  -  create a pretty "Sonari" shortcut that launches the app
# =============================================================================
#  A .bat file can't show a custom icon, but a Windows shortcut (.lnk) can. This
#  builds one that points at app.bat and wears assets\sonari.ico, so you get a
#  nice double-click icon to open Sonari — on your Desktop and in this folder.
#
#  Run it by double-clicking make_shortcut.bat, or directly:
#      powershell -ExecutionPolicy Bypass -File tools\make_shortcut.ps1
# =============================================================================
$ErrorActionPreference = 'Stop'

$root   = Split-Path -Parent $PSScriptRoot      # tools\ -> project root
$target = Join-Path $root 'app.bat'
$icon   = Join-Path $root 'assets\sonari.ico'
$wsh    = New-Object -ComObject WScript.Shell

function New-SonariShortcut([string]$path) {
    $sc = $wsh.CreateShortcut($path)
    $sc.TargetPath       = $target
    $sc.WorkingDirectory = $root
    $sc.IconLocation     = "$icon,0"
    $sc.Description       = 'Sonari - turn audio into text, lyrics, karaoke & chords'
    $sc.WindowStyle      = 1
    $sc.Save()
    Write-Host "[shortcut] created  $path"
}

# 1) next to app.bat, inside the project folder
New-SonariShortcut (Join-Path $root 'Sonari.lnk')

# 2) on the Desktop, so it's one double-click away (best-effort)
try {
    $desktop = [Environment]::GetFolderPath('Desktop')
    New-SonariShortcut (Join-Path $desktop 'Sonari.lnk')
} catch {
    Write-Host "[shortcut] desktop copy skipped: $_"
}

Write-Host ''
Write-Host 'Done. Double-click the Sonari icon to open the app.'
