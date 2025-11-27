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