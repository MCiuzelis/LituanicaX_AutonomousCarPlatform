"""Run a Pi5 tool on Windows: stubs the Linux-only modules the hardware package imports (fcntl, spidev, ...)
so replay / analysis tools work on the PC. Only for offline tools: nothing touches hardware.

    cd Pi5 && uv run --project ../YoloTraining --with scipy python tools/pc_run.py tools/autonomy_eval.py recordings/<id> --plot p.png
"""
import runpy, sys, types
class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return 0
for name in ("fcntl", "termios", "spidev", "smbus", "smbus2", "gpiod", "resource"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = _Stub(name)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
