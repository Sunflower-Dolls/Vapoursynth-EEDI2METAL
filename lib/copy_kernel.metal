kernel void copy_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& width_bytes [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width_bytes) {
        return;
    }
    uint src_idx = gid.y * src_stride + gid.x;
    uint dst_idx = gid.y * dst_stride + gid.x;
    dst[dst_idx] = src[src_idx];
}
