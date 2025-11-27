kernel void KERNEL_NAME(enlarge2)(constant EEDI2Param &d [[buffer(0)]],
                                  const device TYPE *src [[buffer(1)]],
                                  device TYPE *dst [[buffer(2)]],
                                  uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= d.width || pos.y >= d.height)
        return;
    uint dst_y = 2 * pos.y + (1 - d.field);
    const device TYPE *src_line =
        (const device TYPE *)((const device char *)src + pos.y * d.d_pitch);
    device TYPE *dst_line =
        (device TYPE *)((device char *)dst + dst_y * d.d_pitch);
    dst_line[pos.x] = src_line[pos.x];
}