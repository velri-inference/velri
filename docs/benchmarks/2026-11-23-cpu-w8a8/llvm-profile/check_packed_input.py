"""Check unsigned packed activation bytes in the Slang C++ target."""
import ctypes as ct
import sys
from pathlib import Path


class ByteAddressBuffer(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("size", ct.c_size_t)]


library = ct.CDLL(str(Path(sys.argv[1]).resolve()))
matmul = library.packed_input
matmul.argtypes = [ct.c_uint32, ByteAddressBuffer, ct.POINTER(ct.c_int32)]
matmul.restype = None

activations = bytes([128, 1, 255, 4])
weights = (ct.c_int8 * 64)(*([1] * 64))
outputs = (ct.c_int32 * 16)()
matmul(int.from_bytes(activations, "little"), ByteAddressBuffer(ct.addressof(weights), 64), outputs)
expected = sum(activations)
print(f"Expected {expected} in each lane. Observed {list(outputs)}")
sys.exit(0 if all(value == expected for value in outputs) else 1)
