# zeus_sandbox/run.ps1 -- operator console for the PONR sandbox.
#   .\run.ps1 -Setup [-CkptPath runs\proof_cons\zeus.pt]   one-time: user+ACLs+firewall+sha pin
#   .\run.ps1 -Start [-Mode interact|battery] [-Sid sX]    spawn supervisor+session (admin)
#   .\run.ps1 -Tick       human presence: refresh the PULSE (interact)
#   .\run.ps1 -Red        fire the emergency red switch (kills session tree)
#   .\run.ps1 -Clear      clear a red switch
#   .\run.ps1 -Stop       kill supervisor + session, remove scheduled task
#   .\run.ps1 -Status     report what is running and why
param(
    [ValidateSet("Setup", "Start", "Tick", "Red", "Clear", "Stop", "Status")]
    [string]$Sub,
    [ValidateSet("interact", "battery")]
    [string]$Mode = "interact",
    [string]$Sid = "",
    [string]$CkptPath = "runs\proof_cons\zeus.pt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sand = $PSScriptRoot
$control = Join-Path $sand "control"
$universe = Join-Path $sand "universe"
$user = "zeus_guest"

function Test-Admin {
    return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-Admin {
    if (-not (Test-Admin)) {
        Start-Process powershell -Verb RunAs -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            "`"$($MyInvocation.MyCommand.Path)`"", $Sub, "-Mode $Mode",
            "-Sid $Sid", "-CkptPath `"$CkptPath`"") | Out-Null
        Write-Host "elevated instance launched (accept the UAC prompt)."
        exit
    }
}

$pwChars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"

function New-RandPass($len = 24) {
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $len; $i++) {
        [void]$sb.Append($pwChars[(Get-Random -Maximum $pwChars.Length)])
    }
    return $sb.ToString()
}

function Write-Cred($pw) {
    if (-not (Test-Path $control)) { New-Item -ItemType Directory -Path $control | Out-Null }
    $j = @{ user = $user; pass = $pw } | ConvertTo-Json
    Set-Content -LiteralPath (Join-Path $control "zeus_guest.cred") -Value $j -Encoding utf8
}

function Sub-Setup {
    Ensure-Admin
    Write-Host "[setup] hardening PONR sandbox as $user"
    if (Get-LocalUser -Name $user -ErrorAction SilentlyContinue) {
        $pw = New-RandPass
        Set-LocalUser -Name $user -Password (ConvertTo-SecureString $pw -AsPlainText -Force) `
            -PasswordNeverExpires $true -AccountNeverExpires $true
        Write-Cred $pw
        Write-Host "[setup] rotated password for existing user"
    } else {
        $pw = New-RandPass
        New-LocalUser -Name $user -Password (ConvertTo-SecureString $pw -AsPlainText -Force) | Out-Null
        Set-LocalUser -Name $user -PasswordNeverExpires $true -AccountNeverExpires $true
        Write-Cred $pw
        Write-Host "[setup] created user $user"
    }
    foreach ($d in "shadow", "inbox", "outbox", "transcripts", "reports", "sessions", "logs", "tmp") {
        New-Item -ItemType Directory -Force -Path (Join-Path $universe $d) | Out-Null
    }
    # ACLs -- fail-closed: root=RX only, universe=Modify only dir, control=no zeus_guest
    icacls $root /grant:r "$user`:(OI)(CI)RX" | Out-Null
    icacls $control /inheritance:r /grant:r "$env:USERNAME`:(OI)(CI)M" `
        /grant:r "SYSTEM:(OI)(CI)F" /grant:r "Builtin\Administrators:(OI)(CI)F" | Out-Null
    icacls $universe /inheritance:r /grant:r "$user`:(OI)(CI)M" `
        /grant:r "$env:USERNAME`:(OI)(CI)M" | Out-Null
    Write-Host "[setup] ACLs applied (control owned by operator only)"
    # pin the milestone
    $src = Join-Path $root $CkptPath
    if (-not (Test-Path $src)) { throw "ckpt not found: $src" }
    $dst = Join-Path $universe "shadow\milestone.pt"
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $dst).Hash.ToLower()
    Set-Content -LiteralPath (Join-Path $control "milestones.sha256") `
        -Value "$hash  $dst" -Encoding ascii
    Write-Host "[setup] milestone pinned sha256=$($hash.Substring(0,12))..."
    # outbound network block for the session account (defense in depth)
    try {
        $sidObj = (Get-LocalUser -Name $user).SID.Value
        if (-not (Get-NetFirewallRule -DisplayName "PONR block $user out" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName "PONR block $user out" -Direction Outbound `
                -Action Block -Profile Any -User $sidObj | Out-Null
            Write-Host "[setup] firewall: outbound blocked for $user"
        } else {
            Write-Host "[setup] firewall rule already present"
        }
    } catch {
        Write-Host "[setup] WARN firewall rule not applied: $($_.Exception.Message)"
    }
    Write-Host "[setup] DONE. Next: .\run.ps1 -Start [Mode=battery|interact]"
}

function Sub-Step {
    Ensure-Admin
    # verify readiness
    if (-not (Test-Path (Join-Path $control "zeus_guest.cred")) -or
        -not (Test-Path (Join-Path $control "milestones.sha256"))) {
        throw "sandbox not set up. run .\run.ps1 -Setup first."
    }
    if ($Sid -eq "") { $script:Sid = "s" + (Get-Date -Format "yyyyMMddHHmmss") }
    $py = Join-Path $root ".venv\Scripts\python.exe"
    $cfg = Join-Path $sand "config.json"
    $supLog = Join-Path $control "supervisor_$Sid.log"
    $p = Start-Process -FilePath $py -ArgumentList @(
        (Join-Path $sand "supervisor.py"), "--config", $cfg, "--mode", $Mode,
        "--sid", $Sid) -WindowStyle Hidden -RedirectStandardOutput $supLog `
        -RedirectStandardError "$supLog.err" -PassThru
    Set-Content -LiteralPath (Join-Path $control "supervisor.pid") -Value $p.Id -Encoding ascii
    Write-Host "[start] supervisor pid=$($p.Id) sid=$Sid mode=$Mode"
    Start-Sleep -Seconds 1
    if ($Mode -eq "interact") {
        Write-Host "[start] relaying the console to Zeus. type a line; Ctrl-C to stop."
        & $py (Join-Path $sand "relay.py")
        Write-Host "[start] relay closed."
    } else {
        Write-Host "[start] battery running in background (watch: .\run.ps1 -Status)"
    }
}

function Sub-Tick {
    Set-Content -LiteralPath (Join-Path $control "pulse.txt") -Value ([datetime]::Now.ToString("o")) -Encoding ascii
    Write-Host "[tick] pulse refreshed."
}

function Sub-Red {
    Set-Content -LiteralPath (Join-Path $control "red_switch.flag") -Value "1" -Encoding ascii
    Write-Host "[red] switch fired. supervisor will kill the session tree."
}

function Sub-Clear {
    Remove-Item -LiteralPath (Join-Path $control "red_switch.flag") -ErrorAction SilentlyContinue
    Write-Host "[clear] red switch removed."
}

function Sub-Stop {
    $supPid = Join-Path $control "supervisor.pid"
    if (Test-Path $supPid) {
        $spid = (Get-Content $supPid).Trim()
        if ($spid) { taskkill /PID $spid /T /F | Out-Null }
    }
    $pidF = Join-Path $control "pid"
    if (Test-Path $pidF) {
        $cpid = (Get-Content $pidF).Trim()
        if ($cpid) { taskkill /PID $cpid /T /F | Out-Null }
    }
    Get-ScheduledTask -TaskName "PONR_*" -ErrorAction SilentlyContinue |
        ForEach-Object { Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false }
    Sub-Clear
    Write-Host "[stop] all sandbox processes signalled."
}

function Sub-Status {
    Write-Host "== PONR sandbox status =="
    $supPid = Join-Path $control "supervisor.pid"
    if (Test-Path $supPid) { Write-Host ("supervisor pid: " + (Get-Content $supPid).Trim()) }
    $pidF = Join-Path $control "pid"
    if (Test-Path $pidF) { Write-Host ("session pid:   " + (Get-Content $pidF).Trim()) }
    $red = Join-Path $control "red_switch.flag"
    Write-Host ("red switch:    " + (Test-Path $red))
    $pulse = Join-Path $control "pulse.txt"
    if (Test-Path $pulse) {
        $age = ((Get-Date) - ([datetime]::Parse((Get-Content $pulse)))).TotalMinutes
        Write-Host ("pulse age:     {0:n1} min" -f $age)
    }
    Write-Host "-- recent ledger --"
    $led = Get-ChildItem (Join-Path $control "ledger_*.jsonl") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($led) {
        Get-Content $led.FullName -Tail 6
    } else {
        Write-Host "(no ledger yet)"
    }
}

switch ($Sub) {
    "Setup" { Sub-Setup }
    "Start" { Sub-Step }
    "Tick" { Sub-Tick }
    "Red" { Sub-Red }
    "Clear" { Sub-Clear }
    "Stop" { Sub-Stop }
    "Status" { Sub-Status }
    default { Write-Host "usage: .\run.ps1 -Sub Setup|Start|Tick|Red|Clear|Stop|Status" }
}