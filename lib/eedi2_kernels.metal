#ifndef TYPE
#error "TYPE must be defined before including this header."
#endif

#ifndef SUFFIX
#error "SUFFIX must be defined before including this header."
#endif

#define GET_LINE(p, y_off)                                                     \
    ((device const TYPE *)((device const char *)p + (pos.y + (y_off)) * pitch))
#define GET_LINE_W(p, y_off)                                                   \
    ((device TYPE *)((device char *)p + (pos.y + (y_off)) * pitch))
#define GET_POINT(line, x_off) line[pos.x + (x_off)]

// MARK: - Kernels
kernel void KERNEL_NAME(buildEdgeMask)(constant EEDI2Param &d [[buffer(0)]],
                                       const device TYPE *src [[buffer(1)]],
                                       device TYPE *dst [[buffer(2)]],
                                       uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const uint shift = d.shift;

    device TYPE *dst_line = (device TYPE *)((device char *)dst + pos.y * pitch);
    dst_line[pos.x] = 0;

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1)
        return;

    device const TYPE *srcp = GET_LINE(src, 0);
    device const TYPE *srcpp = GET_LINE(src, -1);
    device const TYPE *srcpn = GET_LINE(src, 1);

    const TYPE ten = 10 << shift;

    if ((abs(srcpp[pos.x] - srcp[pos.x]) < ten &&
         abs(srcp[pos.x] - srcpn[pos.x]) < ten &&
         abs(srcpp[pos.x] - srcpn[pos.x]) < ten) ||
        (abs(srcpp[pos.x - 1] - srcp[pos.x - 1]) < ten &&
         abs(srcp[pos.x - 1] - srcpn[pos.x - 1]) < ten &&
         abs(srcpp[pos.x - 1] - srcpn[pos.x - 1]) < ten &&
         abs(srcpp[pos.x + 1] - srcp[pos.x + 1]) < ten &&
         abs(srcp[pos.x + 1] - srcpn[pos.x + 1]) < ten &&
         abs(srcpp[pos.x + 1] - srcpn[pos.x + 1]) < ten))
        return;

    const uint sum = (uint(srcpp[pos.x - 1]) + srcpp[pos.x] + srcpp[pos.x + 1] +
                      srcp[pos.x - 1] + srcp[pos.x] + srcp[pos.x + 1] +
                      srcpn[pos.x - 1] + srcpn[pos.x] + srcpn[pos.x + 1]) >>
                     shift;
    const uint sumsq =
        (uint(srcpp[pos.x - 1] >> shift) * (srcpp[pos.x - 1] >> shift)) +
        (uint(srcpp[pos.x] >> shift) * (srcpp[pos.x] >> shift)) +
        (uint(srcpp[pos.x + 1] >> shift) * (srcpp[pos.x + 1] >> shift)) +
        (uint(srcp[pos.x - 1] >> shift) * (srcp[pos.x - 1] >> shift)) +
        (uint(srcp[pos.x] >> shift) * (srcp[pos.x] >> shift)) +
        (uint(srcp[pos.x + 1] >> shift) * (srcp[pos.x + 1] >> shift)) +
        (uint(srcpn[pos.x - 1] >> shift) * (srcpn[pos.x - 1] >> shift)) +
        (uint(srcpn[pos.x] >> shift) * (srcpn[pos.x] >> shift)) +
        (uint(srcpn[pos.x + 1] >> shift) * (srcpn[pos.x + 1] >> shift));
    if (9 * sumsq - sum * sum < d.vthresh)
        return;

    const uint Ix = abs(srcp[pos.x + 1] - srcp[pos.x - 1]) >> shift;
    const uint Iy =
        mmax(abs(srcpp[pos.x] - srcpn[pos.x]), abs(srcpp[pos.x] - srcp[pos.x]),
             abs(srcp[pos.x] - srcpn[pos.x])) >>
        shift;
    if (Ix * Ix + Iy * Iy >= d.mthresh) {
        dst_line[pos.x] = numeric_limits<TYPE>::max();
    }

    const uint Ixx =
        abs(srcp[pos.x - 1] - 2 * srcp[pos.x] + srcp[pos.x + 1]) >> shift;
    const uint Iyy =
        abs(srcpp[pos.x] - 2 * srcp[pos.x] + srcpn[pos.x]) >> shift;
    if (Ixx + Iyy >= d.lthresh) {
        dst_line[pos.x] = numeric_limits<TYPE>::max();
    }
}

kernel void KERNEL_NAME(erode)(constant EEDI2Param &d [[buffer(0)]],
                               const device TYPE *msk [[buffer(1)]],
                               device TYPE *dst [[buffer(2)]],
                               uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();

    device TYPE *dst_line = GET_LINE_W(dst, 0);

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1) {
        dst_line[pos.x] = GET_LINE(msk, 0)[pos.x];
        return;
    }

    device const TYPE *mskp = GET_LINE(msk, 0);
    device const TYPE *mskpp = GET_LINE(msk, -1);
    device const TYPE *mskpn = GET_LINE(msk, 1);

    uint count = 0;
    count += mskpp[pos.x - 1] == peak;
    count += mskpp[pos.x] == peak;
    count += mskpp[pos.x + 1] == peak;
    count += mskp[pos.x - 1] == peak;
    count += mskp[pos.x + 1] == peak;
    count += mskpn[pos.x - 1] == peak;
    count += mskpn[pos.x] == peak;
    count += mskpn[pos.x + 1] == peak;

    dst_line[pos.x] = (mskp[pos.x] == peak && count < d.estr) ? 0 : mskp[pos.x];
}

kernel void KERNEL_NAME(dilate)(constant EEDI2Param &d [[buffer(0)]],
                                const device TYPE *msk [[buffer(1)]],
                                device TYPE *dst [[buffer(2)]],
                                uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();

    device TYPE *dst_line = GET_LINE_W(dst, 0);

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1) {
        dst_line[pos.x] = GET_LINE(msk, 0)[pos.x];
        return;
    }

    device const TYPE *mskp = GET_LINE(msk, 0);
    device const TYPE *mskpp = GET_LINE(msk, -1);
    device const TYPE *mskpn = GET_LINE(msk, 1);

    uint count = 0;
    count += mskpp[pos.x - 1] == peak;
    count += mskpp[pos.x] == peak;
    count += mskpp[pos.x + 1] == peak;
    count += mskp[pos.x - 1] == peak;
    count += mskp[pos.x + 1] == peak;
    count += mskpn[pos.x - 1] == peak;
    count += mskpn[pos.x] == peak;
    count += mskpn[pos.x + 1] == peak;

    dst_line[pos.x] =
        (mskp[pos.x] == 0 && count >= d.dstr) ? peak : mskp[pos.x];
}

kernel void KERNEL_NAME(removeSmallHorzGaps)(
    constant EEDI2Param &d [[buffer(0)]], const device TYPE *msk [[buffer(1)]],
    device TYPE *dst [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();

    device const TYPE *mskp = GET_LINE(msk, 0);
    device TYPE *dst_line = GET_LINE_W(dst, 0);

    TYPE orig = mskp[pos.x];
    dst_line[pos.x] = orig;

    if (pos.x < 3 || pos.x >= width - 3 || pos.y < 1 || pos.y >= height - 1)
        return;

    TYPE a = (mskp[pos.x - 3] | mskp[pos.x - 2] | mskp[pos.x - 1] |
              mskp[pos.x + 1] | mskp[pos.x + 2] | mskp[pos.x + 3])
                 ? orig
                 : 0;
    TYPE b = ((mskp[pos.x + 1] &
               (mskp[pos.x - 1] | mskp[pos.x - 2] | mskp[pos.x - 3])) |
              (mskp[pos.x + 2] & (mskp[pos.x - 1] | mskp[pos.x - 2])) |
              (mskp[pos.x + 3] & mskp[pos.x - 1]))
                 ? peak
                 : orig;

    dst_line[pos.x] = mskp[pos.x] ? a : b;
}

kernel void KERNEL_NAME(calcDirections)(constant EEDI2Param &d [[buffer(0)]],
                                        const device TYPE *src [[buffer(1)]],
                                        const device TYPE *msk [[buffer(2)]],
                                        device TYPE *dst [[buffer(3)]],
                                        uint2 pos_u [[thread_position_in_grid]],
                                        uint t_idx
                                        [[thread_index_in_threadgroup]]) {
    const int width = int(d.width);
    const int height = int(d.height);
    const int pos_x = int(pos_u.x);
    const int pos_y = int(pos_u.y);
    const uint pitch = d.d_pitch;
    const uint shift = d.shift;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift2 = shift + 2;

    if (pos_y >= height)
        return;

    device TYPE *dst_line = (device TYPE *)((device char *)dst + pos_y * pitch);
    if (pos_x < width)
        dst_line[pos_x] = peak;

    if (pos_y < 1 || pos_y >= height - 1)
        return;

    bool active = (pos_x >= 1 && pos_x < width - 1);

    if (active) {
        device const TYPE *mskp =
            (device const TYPE *)((device const char *)msk + pos_y * pitch);
        if (mskp[pos_x] != peak ||
            (mskp[pos_x - 1] != peak && mskp[pos_x + 1] != peak))
            active = false;
    }

    threadgroup int s2p[128], s1p[128], s[128], s1n[128], s2n[128];
    threadgroup int m1p[128], m1n[128];

    // Zero-initialize shared memory to prevent reading uninitialized values at
    // edges
    if (t_idx < 128) {
        s2p[t_idx] = 0;
        s1p[t_idx] = 0;
        s[t_idx] = 0;
        s1n[t_idx] = 0;
        s2n[t_idx] = 0;
        m1p[t_idx] = 0;
        m1n[t_idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr int block_w = 64;
    constexpr int off_w = block_w / 2;

#define GET_LINE_INT(p, y_off)                                                 \
    ((device const TYPE *)((device const char *)p + (pos_y + (y_off)) * pitch))

    int load_x = pos_x - off_w;
    s2p[t_idx] = GET_LINE_INT(src, -2)[clamp(load_x, 0, width - 1)];
    s1p[t_idx] = GET_LINE_INT(src, -1)[clamp(load_x, 0, width - 1)];
    s[t_idx] = GET_LINE_INT(src, 0)[clamp(load_x, 0, width - 1)];
    s1n[t_idx] = GET_LINE_INT(src, 1)[clamp(load_x, 0, width - 1)];
    s2n[t_idx] = GET_LINE_INT(src, 2)[clamp(load_x, 0, width - 1)];
    m1p[t_idx] = GET_LINE_INT(msk, -1)[clamp(load_x, 0, width - 1)];
    m1n[t_idx] = GET_LINE_INT(msk, 1)[clamp(load_x, 0, width - 1)];

    load_x = pos_x + off_w;
    s2p[t_idx + block_w] = GET_LINE_INT(src, -2)[clamp(load_x, 0, width - 1)];
    s1p[t_idx + block_w] = GET_LINE_INT(src, -1)[clamp(load_x, 0, width - 1)];
    s[t_idx + block_w] = GET_LINE_INT(src, 0)[clamp(load_x, 0, width - 1)];
    s1n[t_idx + block_w] = GET_LINE_INT(src, 1)[clamp(load_x, 0, width - 1)];
    s2n[t_idx + block_w] = GET_LINE_INT(src, 2)[clamp(load_x, 0, width - 1)];
    m1p[t_idx + block_w] = GET_LINE_INT(msk, -1)[clamp(load_x, 0, width - 1)];
    m1n[t_idx + block_w] = GET_LINE_INT(msk, 1)[clamp(load_x, 0, width - 1)];

#undef GET_LINE_INT

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active)
        return;

    const int X = t_idx + off_w;

    const int maxd = d.maxd >> d.subSampling;
    const int uStartBound = -pos_x + 1;
    const int uStopBound = width - 2 - pos_x;

    const uint min0 = abs(s[X] - s1n[X]) + abs(s[X] - s1p[X]);
    uint minA = mmin(d.nt19, min0 * 9);
    uint minB = mmin(d.nt13, min0 * 6);
    uint minC = minA;
    uint minD = minB;
    uint minE = minB;
    int dirA = -5000, dirB = -5000, dirC = -5000, dirD = -5000, dirE = -5000;

    for (int u = -maxd; u <= maxd; u++) {
        if (u < uStartBound || u > uStopBound)
            continue;

        if ((pos_y == 1 || m1p[X - 1 + u] == peak || m1p[X + u] == peak ||
             m1p[X + 1 + u] == peak) &&
            (pos_y == height - 2 || m1n[X - 1 - u] == peak ||
             m1n[X - u] == peak || m1n[X + 1 - u] == peak)) {

            const uint diffsn = abs(s[X - 1] - s1n[X - 1 - u]) +
                                abs(s[X] - s1n[X - u]) +
                                abs(s[X + 1] - s1n[X + 1 - u]);
            const uint diffsp = abs(s[X - 1] - s1p[X - 1 + u]) +
                                abs(s[X] - s1p[X + u]) +
                                abs(s[X + 1] - s1p[X + 1 + u]);
            const uint diffps = abs(s1p[X - 1] - s[X - 1 - u]) +
                                abs(s1p[X] - s[X - u]) +
                                abs(s1p[X + 1] - s[X + 1 - u]);
            const uint diffns = abs(s1n[X - 1] - s[X - 1 + u]) +
                                abs(s1n[X] - s[X + u]) +
                                abs(s1n[X + 1] - s[X + 1 + u]);
            const uint diff = diffsn + diffsp + diffps + diffns;
            uint diffD = diffsp + diffns;
            uint diffE = diffsn + diffps;

            if (diff < minB) {
                dirB = u;
                minB = diff;
            }

            if (pos_y > 1) {
                const uint diff2pp = abs(s2p[X - 1] - s1p[X - 1 - u]) +
                                     abs(s2p[X] - s1p[X - u]) +
                                     abs(s2p[X + 1] - s1p[X + 1 - u]);
                const uint diffp2p = abs(s1p[X - 1] - s2p[X - 1 + u]) +
                                     abs(s1p[X] - s2p[X + u]) +
                                     abs(s1p[X + 1] - s2p[X + 1 + u]);
                const uint diffA = diff + diff2pp + diffp2p;
                diffD += diffp2p;
                diffE += diff2pp;

                if (diffA < minA) {
                    dirA = u;
                    minA = diffA;
                }
            }

            if (pos_y < height - 2) {
                const uint diff2nn = abs(s2n[X - 1] - s1n[X - 1 + u]) +
                                     abs(s2n[X] - s1n[X + u]) +
                                     abs(s2n[X + 1] - s1n[X + 1 + u]);
                const uint diffn2n = abs(s1n[X - 1] - s2n[X - 1 - u]) +
                                     abs(s1n[X] - s2n[X - u]) +
                                     abs(s1n[X + 1] - s2n[X + 1 - u]);
                const uint diffC = diff + diff2nn + diffn2n;
                diffD += diff2nn;
                diffE += diffn2n;

                if (diffC < minC) {
                    dirC = u;
                    minC = diffC;
                }
            }

            if (diffD < minD) {
                dirD = u;
                minD = diffD;
            }

            if (diffE < minE) {
                dirE = u;
                minE = diffE;
            }
        }
    }

    bool okA = dirA != -5000, okB = dirB != -5000, okC = dirC != -5000,
         okD = dirD != -5000, okE = dirE != -5000;
    uint k = okA + okB + okC + okD + okE;

    if (k > 1) {
        thread int order[5] = {
            okA ? dirA : INT_MAX, okB ? dirB : INT_MAX, okC ? dirC : INT_MAX,
            okD ? dirD : INT_MAX, okE ? dirE : INT_MAX,
        };
        boseSortArray(order);

        const int mid = (k & 1) ? order[k / 2]
                                : (order[(k - 1) / 2] + order[k / 2] + 1) / 2;
        const int lim = mmax(limlut[abs(mid)] / 4, 2);
        int sum = 0;
        int count = 0;

        for (uint i = 0; i < 5; i++) {
            bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
            sum += cond * order[i];
            count += cond;
        }

        dst_line[pos_x] =
            (count > 1) ? neutral + ((sum / count) << shift2) : neutral;
    } else {
        dst_line[pos_x] = neutral;
    }
}

kernel void KERNEL_NAME(filterDirMap)(constant EEDI2Param &d [[buffer(0)]],
                                      const device TYPE *msk [[buffer(1)]],
                                      const device TYPE *dmsk [[buffer(2)]],
                                      device TYPE *dst [[buffer(3)]],
                                      uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device const TYPE *mskp = GET_LINE(msk, 0);
    device const TYPE *dmskp = GET_LINE(dmsk, 0);
    device TYPE *dst_line = GET_LINE_W(dst, 0);

    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1)
        return;
    if (mskp[pos.x] != peak)
        return;

    device const TYPE *dmskpp = GET_LINE(dmsk, -1);
    device const TYPE *dmskpn = GET_LINE(dmsk, 1);

    int val0 = dmskpp[pos.x - 1], val1 = dmskpp[pos.x],
        val2 = dmskpp[pos.x + 1], val3 = dmskp[pos.x - 1], val4 = dmskp[pos.x],
        val5 = dmskp[pos.x + 1], val6 = dmskpn[pos.x - 1], val7 = dmskpn[pos.x],
        val8 = dmskpn[pos.x + 1];
    bool cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak,
         cond3 = val3 != peak, cond4 = val4 != peak, cond5 = val5 != peak,
         cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
    thread int order[] = {
        cond0 ? val0 : INT_MAX, cond1 ? val1 : INT_MAX, cond2 ? val2 : INT_MAX,
        cond3 ? val3 : INT_MAX, cond4 ? val4 : INT_MAX, cond5 ? val5 : INT_MAX,
        cond6 ? val6 : INT_MAX, cond7 ? val7 : INT_MAX, cond8 ? val8 : INT_MAX,
    };
    uint u =
        cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

    if (u < 4) {
        dst_line[pos.x] = peak;
        return;
    }

    boseSortArray(order);

    const int mid =
        (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
    const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
    int sum = 0;
    int count = 0;

    for (uint i = 0; i < 9; i++) {
        bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
        sum += cond * order[i];
        count += cond;
    }

    if (count < 4 || (count < 5 && dmskp[pos.x] == peak)) {
        dst_line[pos.x] = peak;
        return;
    }

    dst_line[pos.x] = round_div(sum + mid, count + 1);
}

kernel void KERNEL_NAME(expandDirMap)(constant EEDI2Param &d [[buffer(0)]],
                                      const device TYPE *msk [[buffer(1)]],
                                      const device TYPE *dmsk [[buffer(2)]],
                                      device TYPE *dst [[buffer(3)]],
                                      uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device const TYPE *mskp = GET_LINE(msk, 0);
    device const TYPE *dmskp = GET_LINE(dmsk, 0);
    device TYPE *dst_line = GET_LINE_W(dst, 0);

    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1)
        return;
    if (dmskp[pos.x] != peak || mskp[pos.x] != peak)
        return;

    device const TYPE *dmskpp = GET_LINE(dmsk, -1);
    device const TYPE *dmskpn = GET_LINE(dmsk, 1);

    int val0 = dmskpp[pos.x - 1], val1 = dmskpp[pos.x],
        val2 = dmskpp[pos.x + 1], val3 = dmskp[pos.x - 1],
        val5 = dmskp[pos.x + 1], val6 = dmskpn[pos.x - 1], val7 = dmskpn[pos.x],
        val8 = dmskpn[pos.x + 1];
    bool cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak,
         cond3 = val3 != peak, cond5 = val5 != peak, cond6 = val6 != peak,
         cond7 = val7 != peak, cond8 = val8 != peak;
    thread int order[] = {
        cond0 ? val0 : INT_MAX, cond1 ? val1 : INT_MAX, cond2 ? val2 : INT_MAX,
        cond3 ? val3 : INT_MAX, cond5 ? val5 : INT_MAX, cond6 ? val6 : INT_MAX,
        cond7 ? val7 : INT_MAX, cond8 ? val8 : INT_MAX,
    };
    uint u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

    if (u < 5)
        return;

    boseSortArray(order);

    const int mid =
        (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
    const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
    int sum = 0;
    int count = 0;

    for (uint i = 0; i < 8; i++) {
        bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
        sum += cond * order[i];
        count += cond;
    }

    if (count < 5)
        return;

    dst_line[pos.x] = round_div(sum + mid, count + 1);
}

kernel void KERNEL_NAME(filterMap)(constant EEDI2Param &d [[buffer(0)]],
                                   const device TYPE *msk [[buffer(1)]],
                                   const device TYPE *dmsk [[buffer(2)]],
                                   device TYPE *dst [[buffer(3)]],
                                   uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device const TYPE *mskp = GET_LINE(msk, 0);
    device const TYPE *dmskp = GET_LINE(dmsk, 0);
    device TYPE *dst_line = GET_LINE_W(dst, 0);

    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || pos.y < 1 || pos.y >= height - 1)
        return;
    if (dmskp[pos.x] == peak || mskp[pos.x] != peak)
        return;

    device const TYPE *dmskpp = GET_LINE(dmsk, -1);
    device const TYPE *dmskpn = GET_LINE(dmsk, 1);

    int dir = (dmskp[pos.x] - neutral) / 4;
    const int lim = mmax(abs(dir) * 2, int(12 << shift));
    dir >>= shift;
    bool ict = false, icb = false;

    bool cond = dir < 0;
    int l = cond ? mmax(-(int)pos.x, dir) : 0;
    int r = cond ? 0 : mmin((int)width - (int)pos.x - 1, dir);
    for (int j = l; j <= r; j++) {
        if ((abs(dmskpp[pos.x + j] - dmskp[pos.x]) > lim &&
             dmskpp[pos.x + j] != peak) ||
            (dmskp[pos.x + j] == peak && dmskpp[pos.x + j] == peak) ||
            (abs(dmskp[pos.x + j] - dmskp[pos.x]) > lim &&
             dmskp[pos.x + j] != peak)) {
            ict = true;
            break;
        }
    }

    if (ict) {
        l = cond ? 0 : mmax(-(int)pos.x, -dir);
        r = cond ? mmin((int)width - (int)pos.x - 1, -dir) : 0;
        for (int j = l; j <= r; j++) {
            if ((abs(dmskpn[pos.x + j] - dmskp[pos.x]) > lim &&
                 dmskpn[pos.x + j] != peak) ||
                (dmskpn[pos.x + j] == peak && dmskp[pos.x + j] == peak) ||
                (abs(dmskp[pos.x + j] - dmskp[pos.x]) > lim &&
                 dmskp[pos.x + j] != peak)) {
                icb = true;
                break;
            }
        }
        if (icb)
            dst_line[pos.x] = peak;
    }
}

kernel void KERNEL_NAME(markDirections2X)(constant EEDI2Param &d [[buffer(0)]],
                                          const device TYPE *msk [[buffer(1)]],
                                          const device TYPE *dmsk [[buffer(2)]],
                                          device TYPE *dst [[buffer(3)]],
                                          uint2 pos
                                          [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device TYPE *dst_line = (device TYPE *)((device char *)dst + y * pitch);

    if (pos.x < 1 || pos.x >= width - 1 || y < 1 || y >= height * 2 - 1)
        return;

    device const TYPE *mskp =
        (device const TYPE *)((device const char *)msk + (y - 1) * pitch);
    device const TYPE *mskpn =
        (device const TYPE *)((device const char *)msk + (y + 1) * pitch);

    if (mskp[pos.x] != peak && mskpn[pos.x] != peak)
        return;

    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + (y - 1) * pitch);
    device const TYPE *dmskpn =
        (device const TYPE *)((device const char *)dmsk + (y + 1) * pitch);

    int val0 = dmskp[pos.x - 1], val1 = dmskp[pos.x], val2 = dmskp[pos.x + 1],
        val6 = dmskpn[pos.x - 1], val7 = dmskpn[pos.x],
        val8 = dmskpn[pos.x + 1];
    bool cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak,
         cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
    thread int order[] = {
        cond0 ? val0 : INT_MAX, cond1 ? val1 : INT_MAX, cond2 ? val2 : INT_MAX,
        cond6 ? val6 : INT_MAX, cond7 ? val7 : INT_MAX, cond8 ? val8 : INT_MAX,
    };
    uint v = cond0 + cond1 + cond2 + cond6 + cond7 + cond8;

    if (v < 3)
        return;

    boseSortArray(order);

    const int mid =
        (v & 1) ? order[v / 2] : (order[(v - 1) / 2] + order[v / 2] + 1) / 2;
    const int lim = limlut[abs(mid - neutral) >> shift2] << shift;

    uint u = 0;
    u += (abs(dmskp[pos.x - 1] - dmskpn[pos.x - 1]) <= lim ||
          dmskp[pos.x - 1] == peak || dmskpn[pos.x - 1] == peak);
    u += (abs(dmskp[pos.x] - dmskpn[pos.x]) <= lim || dmskp[pos.x] == peak ||
          dmskpn[pos.x] == peak);
    u += (abs(dmskp[pos.x + 1] - dmskpn[pos.x - 1]) <= lim ||
          dmskp[pos.x + 1] == peak || dmskpn[pos.x + 1] == peak);
    if (u < 2)
        return;

    int sum = 0;
    int new_count = 0;
    for (uint i = 0; i < 6; i++) {
        bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
        sum += cond * order[i];
        new_count += cond;
    }

    if (new_count < v - 2 || new_count < 2)
        return;

    dst_line[pos.x] = round_div(sum + mid, new_count + 1);
}

kernel void KERNEL_NAME(filterDirMap2X)(constant EEDI2Param &d [[buffer(0)]],
                                        const device TYPE *msk [[buffer(1)]],
                                        const device TYPE *dmsk [[buffer(2)]],
                                        device TYPE *dst [[buffer(3)]],
                                        uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device TYPE *dst_line = (device TYPE *)((device char *)dst + y * pitch);
    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + y * pitch);
    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || y < 1 || y >= height * 2 - 1)
        return;

    device const TYPE *mskp =
        (device const TYPE *)((device const char *)msk + (y - 1) * pitch);
    device const TYPE *mskpn =
        (device const TYPE *)((device const char *)msk + (y + 1) * pitch);

    if (mskp[pos.x] != peak && mskpn[pos.x] != peak)
        return;

    device const TYPE *dmskpp =
        (device const TYPE *)((device const char *)dmsk +
                              (y > 1 ? (y - 2) : 0) * pitch);
    device const TYPE *dmskpn =
        (device const TYPE *)((device const char *)dmsk +
                              (y < height * 2 - 2 ? (y + 2) : y) * pitch);

    int val0 = dmskpp[pos.x - 1], val1 = dmskpp[pos.x],
        val2 = dmskpp[pos.x + 1], val3 = dmskp[pos.x - 1], val4 = dmskp[pos.x],
        val5 = dmskp[pos.x + 1], val6 = dmskpn[pos.x - 1], val7 = dmskpn[pos.x],
        val8 = dmskpn[pos.x + 1];
    bool cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1,
         cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
         cond4 = val4 != peak, cond5 = val5 != peak,
         cond6 = val6 != peak && y < height * 2 - 2,
         cond7 = val7 != peak && y < height * 2 - 2,
         cond8 = val8 != peak && y < height * 2 - 2;
    thread int order[] = {
        cond0 ? val0 : INT_MAX, cond1 ? val1 : INT_MAX, cond2 ? val2 : INT_MAX,
        cond3 ? val3 : INT_MAX, cond4 ? val4 : INT_MAX, cond5 ? val5 : INT_MAX,
        cond6 ? val6 : INT_MAX, cond7 ? val7 : INT_MAX, cond8 ? val8 : INT_MAX,
    };
    uint u =
        cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

    if (u < 4) {
        dst_line[pos.x] = peak;
        return;
    }

    boseSortArray(order);

    const int mid =
        (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
    const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
    int sum = 0;
    int count = 0;

    for (uint i = 0; i < 9; i++) {
        bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
        sum += cond * order[i];
        count += cond;
    }

    if (count < 4 || (count < 5 && dmskp[pos.x] == peak)) {
        dst_line[pos.x] = peak;
        return;
    }

    dst_line[pos.x] = round_div(sum + mid, count + 1);
}

kernel void KERNEL_NAME(expandDirMap2X)(constant EEDI2Param &d [[buffer(0)]],
                                        const device TYPE *msk [[buffer(1)]],
                                        const device TYPE *dmsk [[buffer(2)]],
                                        device TYPE *dst [[buffer(3)]],
                                        uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + y * pitch);
    device TYPE *dst_line = (device TYPE *)((device char *)dst + y * pitch);
    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || y < 1 || y >= height * 2 - 1)
        return;

    device const TYPE *mskp =
        (device const TYPE *)((device const char *)msk + (y - 1) * pitch);
    device const TYPE *mskpn =
        (device const TYPE *)((device const char *)msk + (y + 1) * pitch);

    if (dmskp[pos.x] != peak || (mskp[pos.x] != peak && mskpn[pos.x] != peak))
        return;

    device const TYPE *dmskpp =
        (device const TYPE *)((device const char *)dmsk +
                              (y > 1 ? (y - 2) : 0) * pitch);
    device const TYPE *dmskpn =
        (device const TYPE *)((device const char *)dmsk +
                              (y < height * 2 - 2 ? (y + 2) : y) * pitch);

    int val0 = dmskpp[pos.x - 1], val1 = dmskpp[pos.x],
        val2 = dmskpp[pos.x + 1], val3 = dmskp[pos.x - 1],
        val5 = dmskp[pos.x + 1], val6 = dmskpn[pos.x - 1], val7 = dmskpn[pos.x],
        val8 = dmskpn[pos.x + 1];
    bool cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1,
         cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
         cond5 = val5 != peak, cond6 = val6 != peak && y < height * 2 - 2,
         cond7 = val7 != peak && y < height * 2 - 2,
         cond8 = val8 != peak && y < height * 2 - 2;
    thread int order[] = {
        cond0 ? val0 : INT_MAX, cond1 ? val1 : INT_MAX, cond2 ? val2 : INT_MAX,
        cond3 ? val3 : INT_MAX, cond5 ? val5 : INT_MAX, cond6 ? val6 : INT_MAX,
        cond7 ? val7 : INT_MAX, cond8 ? val8 : INT_MAX,
    };
    uint u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

    if (u < 5)
        return;

    boseSortArray(order);

    const int mid =
        (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
    const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
    int sum = 0;
    int count = 0;

    for (uint i = 0; i < 8; i++) {
        bool cond = order[i] != INT_MAX && abs(order[i] - mid) <= lim;
        sum += cond * order[i];
        count += cond;
    }

    if (count < 5)
        return;

    dst_line[pos.x] = round_div(sum + mid, count + 1);
}

kernel void KERNEL_NAME(fillGaps2X)(constant EEDI2Param &d [[buffer(0)]],
                                    const device TYPE *msk [[buffer(1)]],
                                    const device TYPE *dmsk [[buffer(2)]],
                                    device int *tmp [[buffer(3)]],
                                    uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;
    const TYPE fiveHundred = 500 << shift;
    const TYPE twenty = 20 << shift;
    const TYPE eight = 8 << shift;

    device int *tmp_line = (device int *)((device char *)tmp + y * pitch);
    tmp_line[pos.x] = 0;

    if (pos.x < 1 || pos.x >= width - 1 || y < 1 || y >= height * 2 - 1)
        return;

    device const TYPE *mskp =
        (device const TYPE *)((device const char *)msk + (y - 1) * pitch);
    device const TYPE *mskpn =
        (device const TYPE *)((device const char *)msk + (y + 1) * pitch);
    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + y * pitch);

    if (dmskp[pos.x] != peak || (mskp[pos.x] != peak && mskpn[pos.x] != peak))
        return;

    device const TYPE *mskpp =
        (device const TYPE *)((device const char *)msk +
                              (y > 2 ? (y - 3) : 0) * pitch);
    device const TYPE *mskpnn =
        (device const TYPE *)((device const char *)msk +
                              (y < height * 2 - 3 ? (y + 3) : y) * pitch);
    device const TYPE *dmskpp =
        (device const TYPE *)((device const char *)dmsk +
                              (y > 1 ? (y - 2) : 0) * pitch);
    device const TYPE *dmskpn =
        (device const TYPE *)((device const char *)dmsk +
                              (y < height * 2 - 2 ? (y + 2) : y) * pitch);

    uint u = pos.x - 1;
    uint v = pos.x + 1;
    int back = fiveHundred;
    int forward = -fiveHundred;

    while (u > 0 && pos.x - u < 16) {
        if (dmskp[u] != peak) {
            back = dmskp[u];
            break;
        }
        if (mskp[u] != peak && mskpn[u] != peak)
            break;
        u--;
    }

    while (v < width && v - pos.x < 16) {
        if (dmskp[v] != peak) {
            forward = dmskp[v];
            break;
        }
        if (mskp[v] != peak && mskpn[v] != peak)
            break;
        v++;
    }

    bool tc = true, bc = true;
    int mint = fiveHundred, maxt = -twenty;
    int minb = fiveHundred, maxb = -twenty;

    for (uint j = u; j <= v && tc; j++) {
        tc = !(y <= 2 || dmskpp[j] == peak ||
               (mskpp[j] != peak && mskp[j] != peak));
        mint = tc ? mmin(mint, int(dmskpp[j])) : twenty;
        maxt = tc ? mmax(maxt, int(dmskpp[j])) : twenty;
    }

    for (uint j = u; j <= v && bc; j++) {
        bc = !(y >= height * 2 - 3 || dmskpn[j] == peak ||
               (mskpn[j] != peak && mskpnn[j] != peak));
        minb = bc ? mmin(minb, int(dmskpn[j])) : twenty;
        maxb = bc ? mmax(maxb, int(dmskpn[j])) : twenty;
    }

    if (maxt == -twenty)
        maxt = mint = twenty;
    if (maxb == -twenty)
        maxb = minb = twenty;

    const int thresh =
        mmax(mmax(abs(forward - int(neutral)), abs(back - int(neutral))) / 4,
             int(eight), abs(mint - maxt), abs(minb - maxb));
    const uint lim = mmin(
        (uint)(mmax(abs(forward - int(neutral)), abs(back - int(neutral))) >>
               shift2),
        uint(6));

    if (abs(forward - back) <= thresh && (v - u - 1 <= lim || tc || bc)) {
        tmp_line[pos.x] = (pos.x - u) | ((v - pos.x) << 8);
    }
}

kernel void KERNEL_NAME(fillGaps2XStep2)(
    constant EEDI2Param &d [[buffer(0)]], const device TYPE *msk [[buffer(1)]],
    const device TYPE *dmsk [[buffer(2)]], const device int *tmp [[buffer(3)]],
    device TYPE *dst [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;
    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + y * pitch);
    device const int *tmpp =
        (device const int *)((device const char *)tmp + y * pitch);
    device TYPE *dst_line = (device TYPE *)((device char *)dst + y * pitch);

    dst_line[pos.x] = dmskp[pos.x];

    if (pos.x < 1 || pos.x >= width - 1 || y < 1 || y >= height * 2 - 1)
        return;

    uint uv = 0;
    uint pos_found = 0;

    // Search backward from current position
    for (uint i = max((int)pos.x - 16, 1); i < pos.x; ++i) {
        bool cond = i + (tmpp[i] >> 8) > pos.x;
        uv = cond ? tmpp[i] : uv;
        pos_found = cond ? i : pos_found;
    }

    // Check current position
    if (tmpp[pos.x]) {
        uv = tmpp[pos.x];
        pos_found = pos.x;
    }

    // Search forward from current position
    for (uint i = pos.x + 1; i - pos.x < 16 && i < width - 1; ++i) {
        bool cond = i - (tmpp[i] & 255u) < pos.x;
        uv = cond ? tmpp[i] : uv;
        pos_found = cond ? i : pos_found;
    }

    if (!uv)
        return;

    int u = pos_found - (uv & 255);
    int v = pos_found + (uv >> 8);
    int back = dmskp[u];
    int forward = dmskp[v];

    dst_line[pos.x] =
        back + round_div((forward - back) * (int(pos.x) - 1 - u), (v - u));
}

kernel void KERNEL_NAME(interpolateLattice)(
    constant EEDI2Param &d [[buffer(0)]], const device TYPE *omsk [[buffer(1)]],
    const device TYPE *dmsk [[buffer(2)]], device TYPE *dst [[buffer(3)]],
    device TYPE *dmsk_2 [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;

    if (pos.x >= width || pos.y >= height) {
        return;
    }

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;
    const TYPE three = 3 << shift;
    const TYPE nine = 9 << shift;

    device const TYPE *omskp =
        (device const TYPE *)((device const char *)omsk +
                              (y > 0 ? (y - 1) : 0) * pitch);
    device const TYPE *omskn =
        (device const TYPE *)((device const char *)omsk +
                              (y < height * 2 - 1 ? (y + 1) : y) * pitch);
    device const TYPE *dmskp =
        (device const TYPE *)((device const char *)dmsk + y * pitch);
    device TYPE *dstp =
        (device TYPE *)((device char *)dst + (y > 0 ? (y - 1) : 0) * pitch);
    device TYPE *dstpn = (device TYPE *)((device char *)dst + y * pitch);
    device TYPE *dstpnn =
        (device TYPE *)((device char *)dst +
                        (y < height * 2 - 1 ? (y + 1) : y) * pitch);
    device TYPE *out_line = dstpn;
    device TYPE *mout_line = (device TYPE *)((device char *)dmsk_2 + y * pitch);

    if (pos.x >= width || y < 1 || y >= height * 2 - 1)
        return;

    int dir = dmskp[pos.x];
    const int lim = limlut[abs(dir - int(neutral)) >> shift2] << shift;

    if (dir == peak || (abs(dmskp[pos.x] - dmskp[pos.x - 1]) > lim &&
                        abs(dmskp[pos.x] - dmskp[pos.x + 1]) > lim)) {
        out_line[pos.x] = (dstp[pos.x] + dstpnn[pos.x] + 1) / 2;
        if (dir != peak)
            mout_line[pos.x] = neutral;
        return;
    }

    if (lim < nine) {
        const uint sum =
            (uint(dstp[pos.x - 1]) + dstp[pos.x] + dstp[pos.x + 1] +
             dstpnn[pos.x - 1] + dstpnn[pos.x] + dstpnn[pos.x + 1]) >>
            shift;
        const uint sumsq =
            (uint(dstp[pos.x - 1] >> shift) * (dstp[pos.x - 1] >> shift)) +
            (uint(dstp[pos.x] >> shift) * (dstp[pos.x] >> shift)) +
            (uint(dstp[pos.x + 1] >> shift) * (dstp[pos.x + 1] >> shift)) +
            (uint(dstpnn[pos.x - 1] >> shift) * (dstpnn[pos.x - 1] >> shift)) +
            (uint(dstpnn[pos.x] >> shift) * (dstpnn[pos.x] >> shift)) +
            (uint(dstpnn[pos.x + 1] >> shift) * (dstpnn[pos.x + 1] >> shift));
        if (6 * sumsq - sum * sum < 576) {
            out_line[pos.x] = (dstp[pos.x] + dstpnn[pos.x] + 1) / 2;
            mout_line[pos.x] = peak;
            return;
        }
    }

    if (pos.x > 1 && pos.x < width - 2 &&
        ((dstp[pos.x] < mmax(dstp[pos.x - 2], dstp[pos.x - 1]) - three &&
          dstp[pos.x] < mmax(dstp[pos.x + 2], dstp[pos.x + 1]) - three &&
          dstpnn[pos.x] < mmax(dstpnn[pos.x - 2], dstpnn[pos.x - 1]) - three &&
          dstpnn[pos.x] < mmax(dstpnn[pos.x + 2], dstpnn[pos.x + 1]) - three) ||
         (dstp[pos.x] > mmin(dstp[pos.x - 2], dstp[pos.x - 1]) + three &&
          dstp[pos.x] > mmin(dstp[pos.x + 2], dstp[pos.x + 1]) + three &&
          dstpnn[pos.x] > mmin(dstpnn[pos.x - 2], dstpnn[pos.x - 1]) + three &&
          dstpnn[pos.x] >
              mmin(dstpnn[pos.x + 2], dstpnn[pos.x + 1]) + three))) {
        out_line[pos.x] = (dstp[pos.x] + dstpnn[pos.x] + 1) / 2;
        mout_line[pos.x] = neutral;
        return;
    }

    dir = (dir - int(neutral) + (1 << (shift2 - 1))) >> shift2;
    const int uStart =
        (dir - 2 < 0)
            ? mmax(-int(pos.x) + 1, dir - 2, -int(width) + 2 + int(pos.x))
            : mmin(int(pos.x) - 1, dir - 2, int(width) - 2 - int(pos.x));
    const int uStop =
        (dir + 2 < 0)
            ? mmax(-int(pos.x) + 1, dir + 2, -int(width) + 2 + int(pos.x))
            : mmin(int(pos.x) - 1, dir + 2, int(width) - 2 - int(pos.x));
    uint min = d.nt8;
    uint val = (dstp[pos.x] + dstpnn[pos.x] + 1) / 2;

    for (int u = uStart; u <= uStop; u++) {
        const uint diff = abs(dstp[pos.x - 1] - dstpnn[pos.x - u - 1]) +
                          abs(dstp[pos.x] - dstpnn[pos.x - u]) +
                          abs(dstp[pos.x + 1] - dstpnn[pos.x - u + 1]) +
                          abs(dstpnn[pos.x - 1] - dstp[pos.x + u - 1]) +
                          abs(dstpnn[pos.x] - dstp[pos.x + u]) +
                          abs(dstpnn[pos.x + 1] - dstp[pos.x + u + 1]);
        if (diff < min &&
            ((omskp[pos.x - 1 + u] != peak &&
              abs(omskp[pos.x - 1 + u] - dmskp[pos.x]) <= lim) ||
             (omskp[pos.x + u] != peak &&
              abs(omskp[pos.x + u] - dmskp[pos.x]) <= lim) ||
             (omskp[pos.x + 1 + u] != peak &&
              abs(omskp[pos.x + 1 + u] - dmskp[pos.x]) <= lim)) &&
            ((omskn[pos.x - 1 - u] != peak &&
              abs(omskn[pos.x - 1 - u] - dmskp[pos.x]) <= lim) ||
             (omskn[pos.x - u] != peak &&
              abs(omskn[pos.x - u] - dmskp[pos.x]) <= lim) ||
             (omskn[pos.x + 1 - u] != peak &&
              abs(omskn[pos.x + 1 - u] - dmskp[pos.x]) <= lim))) {
            const uint diff2 =
                abs(dstp[pos.x + u / 2 - 1] - dstpnn[pos.x - u / 2 - 1]) +
                abs(dstp[pos.x + u / 2] - dstpnn[pos.x - u / 2]) +
                abs(dstp[pos.x + u / 2 + 1] - dstpnn[pos.x - u / 2 + 1]);
            if (diff2 < d.nt4 &&
                (((abs(omskp[pos.x + u / 2] - omskn[pos.x - u / 2]) <= lim ||
                   abs(omskp[pos.x + u / 2] - omskn[pos.x - ((u + 1) / 2)]) <=
                       lim) &&
                  omskp[pos.x + u / 2] != peak) ||
                 ((abs(omskp[pos.x + ((u + 1) / 2)] - omskn[pos.x - u / 2]) <=
                       lim ||
                   abs(omskp[pos.x + ((u + 1) / 2)] -
                       omskn[pos.x - ((u + 1) / 2)]) <= lim) &&
                  omskp[pos.x + ((u + 1) / 2)] != peak))) {
                if ((abs(dmskp[pos.x] - omskp[pos.x + u / 2]) <= lim ||
                     abs(dmskp[pos.x] - omskp[pos.x + ((u + 1) / 2)]) <= lim) &&
                    (abs(dmskp[pos.x] - omskn[pos.x - u / 2]) <= lim ||
                     abs(dmskp[pos.x] - omskn[pos.x - ((u + 1) / 2)]) <= lim)) {
                    val = (dstp[pos.x + u / 2] + dstp[pos.x + ((u + 1) / 2)] +
                           dstpnn[pos.x - u / 2] +
                           dstpnn[pos.x - ((u + 1) / 2)] + 2) /
                          4;
                    min = diff;
                    dir = u;
                }
            }
        }
    }

    if (min != d.nt8) {
        out_line[pos.x] = val;
        mout_line[pos.x] = neutral + (dir << shift2);
    } else {
        const int dt = 4 >> d.subSampling;
        const int uStart2 = mmax(-int(pos.x) + 1, -dt);
        const int uStop2 = mmin(int(width) - 2 - int(pos.x), dt);
        const uint minm = mmin(dstp[pos.x], dstpnn[pos.x]);
        const uint maxm = mmax(dstp[pos.x], dstpnn[pos.x]);
        min = d.nt7;

        for (int u = uStart2; u <= uStop2; u++) {
            const int p1 = dstp[pos.x + u / 2] + dstp[pos.x + ((u + 1) / 2)];
            const int p2 =
                dstpnn[pos.x - u / 2] + dstpnn[pos.x - ((u + 1) / 2)];
            const uint diff = abs(dstp[pos.x - 1] - dstpnn[pos.x - u - 1]) +
                              abs(dstp[pos.x] - dstpnn[pos.x - u]) +
                              abs(dstp[pos.x + 1] - dstpnn[pos.x - u + 1]) +
                              abs(dstpnn[pos.x - 1] - dstp[pos.x + u - 1]) +
                              abs(dstpnn[pos.x] - dstp[pos.x + u]) +
                              abs(dstpnn[pos.x + 1] - dstp[pos.x + u + 1]) +
                              abs(p1 - p2);
            if (diff < min) {
                const uint valt = (p1 + p2 + 2) / 4;
                if (valt >= minm && valt <= maxm) {
                    val = valt;
                    min = diff;
                    dir = u;
                }
            }
        }

        out_line[pos.x] = val;
        mout_line[pos.x] = (min == d.nt7) ? neutral : neutral + (dir << shift2);
    }
}

kernel void KERNEL_NAME(postProcess)(constant EEDI2Param &d [[buffer(0)]],
                                     const device TYPE *nmsk [[buffer(1)]],
                                     const device TYPE *omsk [[buffer(2)]],
                                     device TYPE *dst [[buffer(3)]],
                                     uint2 pos [[thread_position_in_grid]]) {
    const uint width = d.width;
    const uint height = d.height;
    uint y = d.field ? 2 * pos.y + 1 : 2 * pos.y;

    if (pos.x >= width || pos.y >= height)
        return;

    const uint pitch = d.d_pitch;
    const TYPE peak = numeric_limits<TYPE>::max();
    const TYPE neutral = peak / 2;
    const uint shift = d.shift;
    const uint shift2 = shift + 2;

    if (y < 1 || y >= height * 2 - 1)
        return;

    device const TYPE *nmskp =
        (device const TYPE *)((device const char *)nmsk + y * pitch);
    device const TYPE *omskp =
        (device const TYPE *)((device const char *)omsk + y * pitch);
    device TYPE *dstp = (device TYPE *)((device char *)dst + y * pitch);
    device const TYPE *dstpp =
        (device const TYPE *)((device const char *)dst + (y - 1) * pitch);
    device const TYPE *dstpn =
        (device const TYPE *)((device const char *)dst + (y + 1) * pitch);

    const int lim = limlut[abs(nmskp[pos.x] - int(neutral)) >> shift2] << shift;
    if (abs(nmskp[pos.x] - omskp[pos.x]) > lim && omskp[pos.x] != peak &&
        omskp[pos.x] != neutral)
        dstp[pos.x] = (dstpp[pos.x] + dstpn[pos.x] + 1) / 2;
}

#undef GET_LINE
#undef GET_LINE_W
#undef GET_POINT
#undef TYPE
#undef SUFFIX