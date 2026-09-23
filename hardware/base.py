"""Common base pieces shared by every subsystem class in hardware."""
import subprocess
from abc import ABC, abstractmethod


class HardwareError(RuntimeError):
    """Raised when a subsystem is missing, misconfigured or a hardware call fails."""


def run_cmd(args, timeout=30, check=True):
    """Run a command and return (returncode, stdout+stderr as text)."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise HardwareError("command not found: %s" % args[0])
    except subprocess.TimeoutExpired:
        raise HardwareError("command timed out after %ss: %s" % (timeout, " ".join(args)))
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise HardwareError("%s failed (rc=%d): %s" % (" ".join(args), p.returncode, out.strip()[:400]))
    return p.returncode, out


def read_sysfs(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        if default is not None:
            return default
        raise HardwareError("cannot read %s" % path)


def write_sysfs(path, value):
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except OSError as e:
        raise HardwareError("cannot write %r to %s: %s" % (value, path, e))


class Subsystem(ABC):
    """Every hardware interface derives from this.

    - available()  cheap check that the hardware/driver is present
    - status()     dict snapshot suitable for logging / JSON
    - close()      release resources; usable as a context manager
    """

    name = "subsystem"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def status(self) -> dict: ...

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.name)
