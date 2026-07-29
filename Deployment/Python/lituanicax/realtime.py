"""Real-time scheduling helper for latency-critical threads.

The ROS drive node raised its whole process to SCHED_RR so sysfs PWM writes
fired consistently under camera and inference load.  Everything now shares one
process, so the bump is applied per-thread instead: Linux accepts a thread id
in `sched_setscheduler`, which `os.sched_setscheduler` cannot express.
"""

from __future__ import annotations

import ctypes
from ctypes.util import find_library

SCHED_RR = 2
_SYS_gettid = 178 if ctypes.sizeof(ctypes.c_void_p) == 8 else 224


class _SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]


def set_thread_realtime(priority: int = 1) -> bool:
    """Put the calling thread on SCHED_RR. Returns True on success.

    Priority 1 is the lowest real-time level: enough to preempt normal threads,
    not enough to starve the system.  Needs CAP_SYS_NICE (i.e. root), and
    quietly reports failure otherwise.
    """
    try:
        libc = ctypes.CDLL(find_library("c") or "libc.so.6", use_errno=True)
        tid = libc.syscall(_SYS_gettid)
        param = _SchedParam(int(priority))
        result = libc.sched_setscheduler(
            ctypes.c_int(tid), ctypes.c_int(SCHED_RR), ctypes.byref(param)
        )
        return result == 0
    except (OSError, AttributeError, ValueError):
        return False
