"""A Windows Job Object that kills every compute worker when the backend ends.

One job per backend process, created on first use, with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The backend holds the only handle; when
the backend ends for any reason -- also ``taskkill /F`` -- the kernel closes
the handle and kills every process in the job. Nested jobs exist since
Windows 8, so this works even if the backend itself already runs in a job.
Membership is inherited: a process started by a job member (the real worker
behind a venv launcher) is in the job without being added.
Through ctypes because pywin32 is not in the locks.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
from ctypes import wintypes

logger = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
CREATE_SUSPENDED = 0x00000004     # Popen creationflags: start the process with its main thread suspended


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BASIC_LIMIT),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB = None
_LOCK = threading.Lock()


def _k32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.IsProcessInJob.restype = wintypes.BOOL
    k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    return k32


def _job(k32):
    global _JOB
    with _LOCK:
        if _JOB is None:
            job = k32.CreateJobObjectW(None, None)
            if not job:
                logger.warning("CreateJobObjectW failed, error %d", ctypes.get_last_error())
                return None
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                               ctypes.byref(info), ctypes.sizeof(info)):
                logger.warning("SetInformationJobObject failed, error %d", ctypes.get_last_error())
                k32.CloseHandle(job)
                return None
            _JOB = job
        return _JOB


def _resume(popen) -> None:
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.restype = wintypes.LONG       # NTSTATUS, 0 = success
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    status = ntdll.NtResumeProcess(wintypes.HANDLE(int(popen._handle)))
    if status != 0:
        logger.error("NtResumeProcess failed for PID %d, NTSTATUS 0x%08X",
                     popen.pid, status & 0xFFFFFFFF)


def kill_with_this_process(popen, resume: bool = False) -> bool:
    """Put the process ``popen`` started into the job. False if Windows refused.

    How to use it (the manager does exactly this): start the worker with
    ``creationflags=... | winjob.CREATE_SUSPENDED`` and call
    ``kill_with_this_process(proc, resume=True)``. Membership is inherited only
    by processes created *after* the assignment; the interpreter behind a venv
    launcher may be created before a running process gets assigned (measured:
    25 of 25 under a busy parent). A suspended start closes that window: the
    launcher cannot spawn anything until it is in the job. With ``resume=True``
    the process is resumed afterwards -- always, also when the assignment
    failed, otherwise the worker would stay suspended forever.
    """
    if sys.platform != "win32":
        return False
    try:
        k32 = _k32()
        job = _job(k32)
        if job is None:
            return False
        ok = bool(k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(popen._handle))))
        if not ok:
            logger.warning("AssignProcessToJobObject failed for PID %d, error %d",
                           popen.pid, ctypes.get_last_error())
        return ok
    finally:
        if resume:
            _resume(popen)


def is_in_job(pid: int) -> bool:
    if sys.platform != "win32" or _JOB is None:
        return False
    k32 = _k32()
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        result = wintypes.BOOL()
        if not k32.IsProcessInJob(h, _JOB, ctypes.byref(result)):
            return False
        return bool(result.value)
    finally:
        k32.CloseHandle(h)
