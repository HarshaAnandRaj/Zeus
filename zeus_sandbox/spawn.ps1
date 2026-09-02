# zeus_sandbox/spawn.ps1 -- spawn zsession as zeus_guest (called by supervisor.py).
param(
    [string]$User,
    [string]$CredFile,
    [string]$Python,
    [string]$Script,
    [string]$Config,
    [string]$Mode,
    [string]$Sid,
    [string]$WorkDir,
    [string]$OutLog,
    [string]$ErrLog
)

$ErrorActionPreference = "Stop"
$j = Get-Content -LiteralPath $CredFile -Raw | ConvertFrom-Json
if (-not $j.pass -or -not $j.user) {
    Write-Error "credential file malformed"
}
$sec = ConvertTo-SecureString $j.pass -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential($j.user, $sec)

$p = Start-Process -FilePath $Python `
    -ArgumentList @($Script, "--config", $Config, "--mode", $Mode, "--sid", $Sid) `
    -Credential $cred `
    -WorkingDirectory $WorkDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

$pidFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "control/pid_$Sid"
Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ascii
Write-Output ("spawned pid {0}" -f $p.Id)