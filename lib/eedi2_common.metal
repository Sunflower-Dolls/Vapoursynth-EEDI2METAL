/*
 * EEDI2METAL: EEDI2 filter using METAL
 *
 * Copyright (C) 2005-2006 Kevin Stone
 * Copyright (C) 2014-2019 HolyWu
 * Copyright (C) 2021 Misaki Kasumi
 * Copyright (C) 2025 Sunflower Dolls
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include <metal_stdlib>
using namespace metal;

struct EEDI2Param {
    uint width, height, d_pitch, field;
    uint nt4, nt7, nt8, nt13, nt19;
    uint mthresh, lthresh, vthresh;
    uint estr, dstr, maxd;
    uint subSampling;
    uint shift;
};

constant int limlut[33] = {6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10,
                           11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                           12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};

#define PASTE(a, b) a##b
#define KERNEL_NAME_INTERNAL(name, suffix) PASTE(name, suffix)
#define KERNEL_NAME(name) KERNEL_NAME_INTERNAL(name, SUFFIX)

template <typename T> METAL_FUNC T mmax(T last) { return last; }

template <typename T, typename... Args>
METAL_FUNC T mmax(T first, Args... remaining) {
    auto candidate = mmax(remaining...);
    return first < candidate ? candidate : first;
}

template <typename T> METAL_FUNC T mmin(T last) { return last; }

template <typename T, typename... Args>
METAL_FUNC T mmin(T first, Args... remaining) {
    auto candidate = mmin(remaining...);
    return first < candidate ? first : candidate;
}

METAL_FUNC int round_div(int a, int b) { return (a + (b / 2)) / b; }

// TODO: Use optimal sorting networks for small sizes
namespace bose {
template <typename T, size_t I, size_t J> METAL_FUNC void P(thread T *arr) {
    thread T &a = arr[I - 1];
    thread T &b = arr[J - 1];
    const T c = a;
    a = a < b ? a : b;
    b = c < b ? b : c;
}

template <typename T, size_t I, size_t X, size_t J, size_t Y>
METAL_FUNC void Pbracket(thread T *arr) {
    constexpr size_t A = X / 2, B = (X & 1) ? (Y / 2) : ((Y + 1) / 2);

    if constexpr (X == 1 && Y == 1)
        P<T, I, J>(arr);
    else if constexpr (X == 1 && Y == 2) {
        P<T, I, J + 1>(arr);
        P<T, I, J>(arr);
    } else if constexpr (X == 2 && Y == 1) {
        P<T, I, J>(arr);
        P<T, I + 1, J>(arr);
    } else {
        Pbracket<T, I, A, J, B>(arr);
        Pbracket<T, I + A, X - A, J + B, Y - B>(arr);
        Pbracket<T, I + A, X - A, J, B>(arr);
    }
}

template <typename T, size_t I, size_t M> METAL_FUNC void Pstar(thread T *arr) {
    constexpr size_t A = M / 2;

    if constexpr (M > 1) {
        Pstar<T, I, A>(arr);
        Pstar<T, I + A, M - A>(arr);
        Pbracket<T, I, A, I + A, M - A>(arr);
    }
}
} // namespace bose

template <typename T, size_t N>
METAL_FUNC void boseSortArray(thread T (&arr)[N]) {
    bose::Pstar<T, 1, N>(arr);
}