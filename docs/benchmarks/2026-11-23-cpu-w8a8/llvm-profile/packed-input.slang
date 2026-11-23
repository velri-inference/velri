// Four unsigned bytes times sixteen rows of signed bytes.
[DllExport] export __extern_cpp void packed_input(uint x, ByteAddressBuffer w, Ptr<int> output)
{
    CoopVec<uint, 1> input = CoopVec<uint, 1>(x);
    let sums = coopVecMatMulPacked<int, 16>(
        input, CoopVecComponentType::UnsignedInt8Packed, 4, w, 0,
        CoopVecComponentType::SignedInt8, CoopVecMatrixLayout::RowMajor, false, 4);
    for (int i = 0; i < 16; ++i) output[i] = sums[i];
}
