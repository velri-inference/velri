"""Check both Slang host targets for reuse of a generic interface specialization."""
import ctypes
import subprocess
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    source = here / "generic-stride.slang"
    strip = here.parents[3] / "velri-lib/cmake/strip_ll_target.cmake"
    ok = True
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for target, cc, ext in [("cpp", "g++", "cpp"), ("llvm-ir", "clang", "ll")]:
            code = work / ("probe." + ext)
            module = work / (target + ".so")
            opt = "-O3" if target == "cpp" else "-O0"
            subprocess.run(["slangc", source, "-target", target, opt, "-o", code], check=True)
            if target == "llvm-ir":
                neutral = work / "neutral.ll"
                subprocess.run(["cmake", f"-DINPUT_LL={code}", f"-DOUTPUT_LL={neutral}", "-P", strip], check=True)
                code = neutral
            subprocess.run([cc, "-O3", "-shared", "-fPIC", code, "-o", module], check=True)
            lib = ctypes.CDLL(str(module))
            for name, expected in [("raw", 40), ("packed", 48), ("control", 48)]:
                values = (ctypes.c_uint32 * 64)()
                fn = getattr(lib, name)
                fn.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
                fn.restype = None
                fn(values)
                actual = [index for index, value in enumerate(values) if value]
                print(f"{target} {name} wrote {actual}, expected [{expected}]")
                ok &= actual == [expected] and values[expected] == 17
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
