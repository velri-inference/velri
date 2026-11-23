interface IOutput {
    void store(uint row, uint value);
}

struct Output<let STRIDE : int> : IOutput {
    Ptr<uint> dst;
    [ForceInline] void store(uint row, uint value) { dst[row * uint(STRIDE)] = value; }
}

[ForceInline] void write_row<T : IOutput>(T output) { output.store(1u, 17u); }

void local_width<let PACKED : bool>(Ptr<uint> dst) {
    static const int channels = PACKED ? 6 : 5;
    static const int width = 8 * channels;
    Output<width> output = {dst};
    write_row<Output<width>>(output);
}

[DllExport] export __extern_cpp void raw(Ptr<uint> dst) { local_width<false>(dst); }
[DllExport] export __extern_cpp void packed(Ptr<uint> dst) { local_width<true>(dst); }
[DllExport] export __extern_cpp void control(Ptr<uint> dst) {
    Output<48> output = {dst};
    write_row<Output<48>>(output);
}
