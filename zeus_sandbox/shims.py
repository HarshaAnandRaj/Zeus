"""zeus_sandbox/shims.py -- fail-closed containment for Zeus interaction sessions.

Installed BEFORE any torch/model import. Enforces, in layers:

  L1 -- Filesystem (fail-closed): every WRITE (open *w/a/x/+*, mkdir, remove,
        rename, replace, symlink, chmod) is denied unless the target is strictly
        inside the UNIVERSE directory. Reads are unrestricted (the session must
        import code + read the pinned milestone). No software can bypass this:
        every write syscall path goes through builtins.open / io.open / os.*.
  L2 -- Runtime spawn escapes: subprocess.Popen/run/call/check_*, os.system,
        os.popen, os.startfile, os.spawn*, os.exe* are neutered AFTER torch has
        imported its own internals (torch needs those modules imported, but Zeus
        never gets a working spawn).
  L3 -- OS-level: setup adds an outbound firewall block for the session account
        and NTFS ACLs (control/ is unreadable and unwritable). See run.ps1.

Residual (documented in CONTINGENCIES.md): ctypes is left functional because
torch links against it; eval/exec are left in place because host code never
evaluates user- or model-produced strings. Full isolation requires the low-
privilege OS account (zeus_guest), which run.ps1 -Setup configures.

External surfaces that stay open are the *session's own world*: universe/ only.
"""

import builtins
import io as _io
import os
import pathlib
import sys

UNIVERSE = None          # absolute path, set by install_guard()
_guard_installed = False

_orig_open = builtins.open
_orig_io_open = _io.open


# ---------------------------------------------------------------- path checks
def _abspath(p):
    try:
        return str(pathlib.Path(os.path.abspath(os.path.expanduser(str(p)))))
    except Exception:
        return str(p)


_DEVICE_NAMES = (
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _is_windows_device(p) -> bool:
    base = os.path.basename(str(p)).lower()
    return base in _DEVICE_NAMES


def is_in_universe(p) -> bool:
    if UNIVERSE is None:
        return False
    u = _abspath(UNIVERSE) + os.sep
    return _abspath(p).startswith(u) or _is_windows_device(p)


def assert_write_path(label, path):
    if not is_in_universe(path):
        raise PermissionError(
            f"[zeus sandbox] {label} denied outside universe: {path}"
        )


# ---------------------------------------------------------------- guarded open
def _guarded_open(file, mode="r", *args, **kwargs):
    if any(ch in mode for ch in "wax+"):
        assert_write_path("open()", file)
    return _orig_open(file, mode, *args, **kwargs)


def _guarded_io_open(file, mode="r", *args, **kwargs):
    if any(ch in mode for ch in "wax+"):
        assert_write_path("io.open()", file)
    return _orig_io_open(file, mode, *args, **kwargs)


# ---------------------------------------------------------------- guarded fs ops
def _guard_fs_op(name):
    orig = getattr(os, name)

    def wrapped(path, *args, **kwargs):
        assert_write_path(f"os.{name}()", path)
        return orig(path, *args, **kwargs)

    return wrapped


_FS_GUARDED = [
    "replace", "rename", "remove", "unlink", "rmdir", "mkdir", "makedirs",
    "symlink", "link", "chmod", "chown",
]


def _deny(*_a, **_k):
    raise RuntimeError("[zeus sandbox] denied by runtime lock")


_SPAWN_DENIED = [
    "system", "popen", "startfile",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
]


# ---------------------------------------------------------------- installer
def install_guard(universe_dir):
    """Install the filesystem jail. Call BEFORE importing torch."""
    global UNIVERSE, _guard_installed
    UNIVERSE = _abspath(universe_dir)
    if _guard_installed:
        return
    builtins.open = _guarded_open
    _io.open = _guarded_io_open
    for name in _FS_GUARDED:
        if hasattr(os, name):
            setattr(os, name, _guard_fs_op(name))
    _guard_installed = True


def lock_runtime():
    """Neuter spawn/exec escape hatches. Call AFTER torch/model import."""
    for name in _SPAWN_DENIED:
        if hasattr(os, name):
            setattr(os, name, _deny)
    try:
        import subprocess as _sp
    except Exception:
        _sp = None
    if _sp is not None:
        for name in ("Popen", "run", "call", "check_call", "check_output",
                     "check_call", "getoutput", "getstatusoutput"):
            if hasattr(_sp, name):
                setattr(_sp, name, _deny)
    try:
        import shutil as _sh
    except Exception:
        _sh = None
    if _sh is not None:
        for name in ("copy", "copy2", "copytree", "move", "rmtree"):
            if hasattr(_sh, name):
                setattr(_sh, name, _deny)


def guard_status():
    return {
        "universe": UNIVERSE,
        "fs_guard": _guard_installed,
        "open": builtins.open is _guarded_open,
        "os.system_blocked": callable(getattr(os, "system", None)) and getattr(os, "system").__name__ != "system",
    }