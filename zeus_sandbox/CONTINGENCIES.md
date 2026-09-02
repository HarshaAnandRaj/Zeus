# CONTINGENCIES — PONR interaction phases

Decided 2026-08-31 (Anand): run the interaction phase with the **software jail
only** (shims.py fail-closed filesystem guard + spawn lockdown). The OS-level
jail (low-priv user `zeus_guest`, NTFS ACLs, outbound firewall, scheduled-task
spawn, emergency kill via supervisor) is **deferred** — run.ps1 -Setup exists
and can re-arm it later (it is idempotent; the half-created `zeus_guest`
account should be disabled or cleaned in a future pass, or a Setup re-run).

## Residual risks accepted for this phase

1. **Process-level, not OS-level isolation.** zsession runs under the same OS
   account that owns the repo. shims blocks *known* write syscalls to any path
   outside `zeus_sandbox/universe/`, but:
   - reads are unrestricted by design (code load requires it) — a session could
     READ files outside the universe. Echoing them back in replies is visible
     to the operator via the monitor, so this is surveillance, not exfil.
   - `ctypes` is left functional (torch links it). A model-driven ctypes call
     is out of scope of the neutralized list (documented in shims.py).
   - `eval`/`exec` remain (host code never evaluates model strings).
   - `inbox/` is not filtered: prompts are ASCII-ish plain text, treated as
     context tokens only; there is no shell/OS interpretation of prompt text.
2. **No emergency kill watchdog in this phase.** With the supervisor off,
   closing the GUI sends `universe/intr.flag` for a clean exit; a hung session
   requires `Stop-Process` / Task Manager (`taskkill /PID <pid> /T /F`).
   Session pid: `zeus_sandbox/universe/logs/console_<sid>.out` is the console;
   the GUI "shutdown" button is the intended stop path.
3. **Unattended CPU/GPU burn.** The idle walker advances forever until the
   intr.flag. Watch the GPU util / the monitor; stop the session when the
   interaction phase ends.
4. **Consolidation on the live walk.** HCM memory grows (`patterns` climbing)
   and is persisted per-session under `universe/sessions/<sid>/`. Sessions
   currently restore the LATEST hcm.pt across sessions (restore_hcm=true), so
   memory carries forward — by design (continuity), but keep an eye on
   `n_patterns < 512` (HCM cap) and on write-rate saturating.
5. **Weights are the law, not frozen.** The walk can still cross into
   transient regimes (d_s > 2). That is diagnostic, not an emergency.

## Red lines (unchanged from the plan)

- No self-replication code paths exist anywhere in the sandbox.
- No self-modification: the session cannot write its own weights (the repo is
  RX-only is NOT enforced in this no-OS phase — it CAN read weights; writes to
  repo paths are still blocked by shims, since only universe/ is writable).
- Emergency shutoffs are operator-side (GUI button / intr.flag / taskkill).

## Operational notes

- To stop cleanly: monitor GUI → "shutdown" → wait for `session_end` in the
  console log.
- To resume later: relaunch zsession with the same `--sid` (memory persists in
  `universe/sessions/<sid>/hcm.pt`).
- Re-arm full isolation: run `run.ps1 -Setup` once (elevated), then use
  `run.ps1 -Start` which spawns through the low-priv account under the
  supervisor.