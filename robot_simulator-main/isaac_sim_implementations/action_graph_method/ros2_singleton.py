from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Optional


def _lock_root() -> Path:
    return Path(tempfile.gettempdir()) / "jz_robot_isaac_singletons"


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            )
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
            )
            if not handle:
                return False
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            if not ok:
                return False
            return int(exit_code.value) == STILL_ACTIVE
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except Exception:
        return False
    return True


class ProcessSingletonLock:
    def __init__(self, service_name: str, domain_id: Optional[str] = None) -> None:
        self.service_name = service_name
        self.domain_id = (domain_id or os.environ.get("ROS_DOMAIN_ID", "0")).strip() or "0"
        self.pid = os.getpid()
        self.path = _lock_root() / f"{self.service_name}_domain_{self.domain_id}.json"
        self._held = False
        self.owner_pid: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "service_name": self.service_name,
            "domain_id": self.domain_id,
            "pid": self.pid,
            "created_at": time.time(),
        }

        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner_pid = self._read_owner_pid()
                self.owner_pid = owner_pid
                if owner_pid is not None and _is_process_alive(owner_pid):
                    return False
                self._unlink_if_exists()
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
            except Exception:
                self._unlink_if_exists()
                raise
            self._held = True
            self.owner_pid = self.pid
            return True
        return False

    def release(self) -> None:
        if not self._held:
            return
        owner_pid = self._read_owner_pid()
        if owner_pid is not None and owner_pid != self.pid:
            self._held = False
            return
        self._unlink_if_exists()
        self._held = False

    def owner_description(self) -> str:
        if self.owner_pid is None:
            return "unknown owner"
        return f"pid={self.owner_pid} domain={self.domain_id}"

    def _read_owner_pid(self) -> Optional[int]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        raw_pid = data.get("pid")
        if raw_pid is None:
            return None
        try:
            return int(raw_pid)
        except Exception:
            return None

    def _unlink_if_exists(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
