# zeus_sandbox/child_start.ps1 -- runs INSIDE the scheduled task as zeus_guest.
# Starts zsession and records its PID under universe/logs/ (the jail), so the
# parent supervisor can taskkill the whole tree.
param(
    [string]$Py,
    [string]$Script,
    [string]$Cfg,
    [string]$Mode,
    [string]$Sid,
    [string]$Out,
    [string]$Err
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$p = Start-Process -FilePath $Py `
    -ArgumentList @($Script, "--config", $Cfg, "--mode", $Mode, "--sid", $Sid) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Out `
    -RedirectStandardError $Err `
    -PassThru

$pidFile = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) `
    "universe/logs/pid_$Sid.txt"
$logDir = Split-Path -Parent $pidFile
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ascii
Write-Output ("child pid {0}" -f $p.Id)