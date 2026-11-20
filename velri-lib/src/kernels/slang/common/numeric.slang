// Shared numeric storage/conversion abstractions for Velri-lib Slang kernels.
// One interface powers FP32, FP16, BF16, INT8, and F64 across CPU and Vulkan
// without duplicating kernel implementations per dtype.
module numeric;

public interface IElement {
    associatedtype Storage;
    static float toF32(Storage v);
    static Storage fromF32(float x);
    static int32_t toI32(Storage v);
    static float4 load4F32(Ptr<Storage, Access.Read> p);
}

public struct ElemF32 : IElement {
    public typealias Storage = float;
    public static float toF32(float v) { return v; }
    public static float fromF32(float x) { return x; }
    public static int32_t toI32(float v) { return int32_t(v); }
    public static float4 load4F32(Ptr<float, Access.Read> p) {
        return float4(p[0], p[1], p[2], p[3]);
    }
}

public struct ElemF16 : IElement {
    public typealias Storage = half;
    public static float toF32(half v) { return float(v); }
    public static half fromF32(float x) { return half(x); }
    public static int32_t toI32(half v) { return int32_t(float(v)); }
    public static float4 load4F32(Ptr<half, Access.Read> p) {
        half4 h = half4(p[0], p[1], p[2], p[3]);
        return float4(h);
    }
}

// Stored as uint16_t so one source compiles cleanly to both CPU (slang-llvm,
// where BFloat16 is not yet lowered natively) and Vulkan SPIR-V.
public struct ElemBF16 : IElement {
    public typealias Storage = uint16_t;
    public static float toF32(uint16_t v) { return asfloat(uint(v) << 16u); }
    public static uint16_t fromF32(float x) {
        uint bits = asuint(x);
        // Preserve NaN as canonical quiet NaN; round finite values to nearest even.
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            return uint16_t(0x7FC0u);
        }
        uint rounding = 0x7FFFu + ((bits >> 16u) & 1u);
        return uint16_t((bits + rounding) >> 16u);
    }
    public static int32_t toI32(uint16_t v) { return int32_t(toF32(v)); }
    public static float4 load4F32(Ptr<uint16_t, Access.Read> p) {
        uint4 u = uint4(uint(p[0]), uint(p[1]), uint(p[2]), uint(p[3]));
        return asfloat(u << 16u);
    }
}

public struct ElemI8 : IElement {
    public typealias Storage = int8_t;
    public static float toF32(int8_t v) { return float(int(v)); }
    public static int8_t fromF32(float x) { return int8_t(clamp(x, -127.0f, 127.0f)); }
    public static int32_t toI32(int8_t v) { return int32_t(v); }
    public static float4 load4F32(Ptr<int8_t, Access.Read> p) {
        int4 v = int4(int(p[0]), int(p[1]), int(p[2]), int(p[3]));
        return float4(v);
    }
}
