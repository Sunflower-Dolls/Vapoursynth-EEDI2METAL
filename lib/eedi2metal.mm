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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <VSHelper.h>
#include <VapourSynth.h>

struct EEDI2Param {
    uint width, height, d_pitch, field;
    uint nt4, nt7, nt8, nt13, nt19;
    uint mthresh, lthresh, vthresh;
    uint estr, dstr, maxd;
    uint subSampling;
    uint shift;
};

using namespace std::string_literals;

struct EEDI2MetalData {
    VSNodeRef* node;
    const VSVideoInfo* vi;

    // Filter parameters
    int field;
    int mthresh, lthresh, vthresh;
    int estr, dstr, maxd;
    int map, nt, pp;
    std::vector<int> planes;
    int device_id;
    int bits_per_sample;

    // Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSDictionary<NSString*, id<MTLComputePipelineState>>* psos;

    // Buffers
    id<MTLBuffer> d_src; // Holds the source plane
    id<MTLBuffer> d_msk; // The edge mask
    id<MTLBuffer> d_tmp; // A temporary buffer, same size as msk
    id<MTLBuffer> d_dst; // The direction map / final non-2x output

    // 2x Buffers
    id<MTLBuffer> d_dst2;
    id<MTLBuffer> d_msk2;
    id<MTLBuffer> d_tmp2;
    id<MTLBuffer> d_tmp2_2;
    id<MTLBuffer> d_tmp2_3;
    id<MTLBuffer> d_dst2M;

    id<MTLBuffer> params_buffer;
};

static id<MTLComputePipelineState> get_pso(EEDI2MetalData* d,
                                           const std::string& name) {
    std::string suffix = (d->bits_per_sample == 8) ? "_u8" : "_u16";
    id<MTLComputePipelineState> pso =
        d->psos[[NSString stringWithUTF8String:(name + suffix).c_str()]];
    if (pso == nullptr) {
        throw std::runtime_error("Error: PSO for " + name + suffix +
                                 " not found!");
    }
    return pso;
}

static inline void bitblt(void* dstp, int dst_stride, const void* srcp,
                          int src_stride, size_t row_size, size_t height) {
    if (src_stride == dst_stride && src_stride == (int)row_size) {
        memcpy(dstp, srcp, row_size * height);
    } else {
        const auto* srcp8 = static_cast<const uint8_t*>(srcp);
        auto* dstp8 = static_cast<uint8_t*>(dstp);
        for (size_t i = 0; i < height; i++) {
            memcpy(dstp8, srcp8, row_size);
            srcp8 += src_stride;
            dstp8 += dst_stride;
        }
    }
}

static void VS_CC eedi2Init(VSMap* /*unused*/, VSMap* /*unused*/,
                            void** instanceData, VSNode* node,
                            VSCore* /*unused*/, const VSAPI* vsapi) {
    auto* d = static_cast<EEDI2MetalData*>(*instanceData);
    VSVideoInfo vi = *d->vi;
    if (d->map != 1 && d->map != 2) {
        vi.height *= 2;
    }
    vsapi->setVideoInfo(&vi, 1, node);
}

static const VSFrameRef* VS_CC eedi2GetFrame(int n, int activationReason,
                                             void** instanceData,
                                             void** /*unused*/,
                                             VSFrameContext* frameCtx,
                                             VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<EEDI2MetalData*>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrameRef* src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);

    VSVideoInfo vi = *d->vi;
    if (d->map != 1 && d->map != 2) {
        vi.height *= 2;
    }

    VSFrameRef* dst_frame =
        vsapi->newVideoFrame(vi.format, vi.width, vi.height, src_frame, core);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [d->queue commandBuffer];

        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
            if (std::ranges::find(d->planes, plane) == d->planes.end()) {
                bitblt(vsapi->getWritePtr(dst_frame, plane),
                       vsapi->getStride(dst_frame, plane),
                       vsapi->getReadPtr(src_frame, plane),
                       vsapi->getStride(src_frame, plane),
                       (size_t)vsapi->getFrameWidth(src_frame, plane) *
                           d->vi->format->bytesPerSample,
                       vsapi->getFrameHeight(src_frame, plane));
                continue;
            }

            const int width = vsapi->getFrameWidth(src_frame, plane);
            const int height = vsapi->getFrameHeight(src_frame, plane);
            const int src_stride = vsapi->getStride(src_frame, plane);
            const int dst_stride = vsapi->getStride(dst_frame, plane);

            const size_t metal_stride =
                ((size_t)d->vi->width * d->vi->format->bytesPerSample + 63) &
                ~63;

            // Upload
            bitblt([d->d_src contents], (int)metal_stride,
                   vsapi->getReadPtr(src_frame, plane), src_stride,
                   (size_t)width * d->vi->format->bytesPerSample, height);
            [d->d_src didModifyRange:NSMakeRange(0, d->d_src.length)];

            auto* params =
                static_cast<EEDI2Param*>([d->params_buffer contents]);
            params->width = width;
            params->height = height;
            params->d_pitch = (uint)metal_stride;
            params->subSampling = (plane > 0) ? d->vi->format->subSamplingW : 0;
            params->shift = (d->bits_per_sample == 16) ? 8 : 0;
            if (d->field > 1) {
                if ((n & 1) != 0) {
                    params->field = (d->field == 2) ? 1 : 0;
                } else {
                    params->field = (d->field == 2) ? 0 : 1;
                }
            } else {
                params->field = d->field;
            }
            params->mthresh = d->mthresh * d->mthresh;
            params->lthresh = d->lthresh;
            params->vthresh = d->vthresh * 81;
            unsigned nt_shifted = d->nt << params->shift;
            params->nt4 = nt_shifted * 4;
            params->nt7 = nt_shifted * 7;
            params->nt8 = nt_shifted * 8;
            params->nt13 = nt_shifted * 13;
            params->nt19 = nt_shifted * 19;
            params->estr = d->estr;
            params->dstr = d->dstr;
            params->maxd = d->maxd;

            id<MTLComputeCommandEncoder> encoder =
                [cmd_buf computeCommandEncoder];
            MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
            MTLSize numThreadgroups = MTLSizeMake(
                (width + threadsPerGroup.width - 1) / threadsPerGroup.width,
                (height + threadsPerGroup.height - 1) / threadsPerGroup.height,
                1);

#ifdef DUMP_METAL_BUFFERS
            auto dump_buffer = [&](id<MTLBuffer> buffer,
                                   const std::string& name, bool is_2x) {
                [encoder endEncoding];
                id<MTLBlitCommandEncoder> blit = [cmd_buf blitCommandEncoder];
                [blit synchronizeResource:buffer];
                [blit endEncoding];
                [cmd_buf commit];
                [cmd_buf waitUntilCompleted];

                std::string filename_str = "dump_metal_" + name + ".bin";
                std::ofstream ofs(filename_str, std::ios::binary);
                if (ofs) {
                    const char* ptr = (const char*)[buffer contents];
                    int h = is_2x ? height * 2 : height;
                    size_t row_size =
                        (size_t)width * d->vi->format->bytesPerSample;
                    for (int y = 0; y < h; y++) {
                        ofs.write(ptr + y * metal_stride, row_size);
                    }
                }

                cmd_buf = [d->queue commandBuffer];
                encoder = [cmd_buf computeCommandEncoder];
            };
#else
            auto dump_buffer = [&](id<MTLBuffer> /*buffer*/,
                                   const std::string& /*name*/,
                                   bool /*is_2x*/) { /* Do nothing */ };
#endif

            auto run_kernel = [&](const std::string& name,
                                  const std::vector<id<MTLBuffer>>& buffers) {
                id<MTLComputePipelineState> pso = get_pso(d, name);
                if (!pso) {
                    return;
                }
                [encoder setComputePipelineState:pso];
                [encoder setBuffer:d->params_buffer offset:0 atIndex:0];
                for (int i = 0; i < static_cast<int>(buffers.size()); ++i) {
                    [encoder setBuffer:buffers[i] offset:0 atIndex:i + 1];
                }
                [encoder dispatchThreadgroups:numThreadgroups
                        threadsPerThreadgroup:threadsPerGroup];
            };

            run_kernel("buildEdgeMask", {d->d_src, d->d_msk});
            run_kernel("erode", {d->d_msk, d->d_tmp});
            run_kernel("dilate", {d->d_tmp, d->d_msk});
            run_kernel("erode", {d->d_msk, d->d_tmp});
            run_kernel("removeSmallHorzGaps", {d->d_tmp, d->d_msk});

            id<MTLBuffer> final_buffer;

            if (d->map == 1) { // view edge mask
                final_buffer = d->d_msk;
            } else { // full pipeline
                MTLSize tg_size_calc = MTLSizeMake(64, 1, 1);
                MTLSize tg_num_calc = MTLSizeMake(
                    (width + tg_size_calc.width - 1) / tg_size_calc.width,
                    (height + tg_size_calc.height - 1) / tg_size_calc.height,
                    1);
                id<MTLComputePipelineState> pso_calc =
                    get_pso(d, "calcDirections");
                if (pso_calc != nullptr) {
                    if (plane == 0) {
                        dump_buffer(d->d_msk, "msk_before_calc", false);
                    }
                    [encoder setComputePipelineState:pso_calc];
                    [encoder setBuffer:d->params_buffer offset:0 atIndex:0];
                    [encoder setBuffer:d->d_src offset:0 atIndex:1];
                    [encoder setBuffer:d->d_msk offset:0 atIndex:2];
                    [encoder setBuffer:d->d_tmp offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:tg_num_calc
                            threadsPerThreadgroup:tg_size_calc];
                    if (plane == 0) {
                        dump_buffer(d->d_tmp, "dmsk_calc", false);
                    }
                }

                run_kernel("filterDirMap", {d->d_msk, d->d_tmp, d->d_dst});
                if (plane == 0) {
                    dump_buffer(d->d_dst, "dmsk_filtered1", false);
                }
                run_kernel("expandDirMap", {d->d_msk, d->d_dst, d->d_tmp});
                if (plane == 0) {
                    dump_buffer(d->d_tmp, "dmsk_expanded", false);
                }
                run_kernel("filterMap", {d->d_msk, d->d_tmp, d->d_dst});
                if (plane == 0) {
                    dump_buffer(d->d_dst, "dmsk_filtered2", false);
                }

                if (d->map == 2) {
                    final_buffer = d->d_dst;
                } else {
                    // 2x Upscaling Pipeline

                    // Clear buffers
                    [encoder endEncoding];

                    id<MTLBlitCommandEncoder> blit =
                        [cmd_buf blitCommandEncoder];
                    [blit fillBuffer:d->d_dst2
                               range:NSMakeRange(0, d->d_dst2.length)
                               value:0];
                    // tmp2 needs to be filled with 255 (peak for 8bit) or 65535 (peak for 16bit)
                    // If 16-bit, we need to fill with 0xFF.
                    [blit fillBuffer:d->d_tmp2
                               range:NSMakeRange(0, d->d_tmp2.length)
                               value:0xFF];
                    [blit endEncoding];

                    encoder = [cmd_buf computeCommandEncoder];

                    run_kernel("enlarge2", {d->d_src, d->d_dst2});
                    run_kernel("enlarge2", {d->d_dst, d->d_tmp2_2});
                    run_kernel("enlarge2", {d->d_msk, d->d_msk2});

                    // 2x Kernels
                    // markDirections2X
                    run_kernel("markDirections2X",
                               {d->d_msk2, d->d_tmp2_2, d->d_tmp2});
                    if (plane == 0) {
                        dump_buffer(d->d_tmp2, "dmsk_marked2x", true);
                    }

                    // filterDirMap2X
                    run_kernel("filterDirMap2X",
                               {d->d_msk2, d->d_tmp2, d->d_dst2M});
                    if (plane == 0) {
                        dump_buffer(d->d_dst2M, "dmsk_filtered2x_1", true);
                    }

                    // expandDirMap2X
                    run_kernel("expandDirMap2X",
                               {d->d_msk2, d->d_dst2M, d->d_tmp2});
                    if (plane == 0) {
                        dump_buffer(d->d_tmp2, "dmsk_expanded2x", true);
                    }

                    // fillGaps2X & fillGaps2XStep2 (3 passes)
                    run_kernel("fillGaps2X",
                               {d->d_msk2, d->d_tmp2, d->d_tmp2_3});
                    run_kernel("fillGaps2XStep2",
                               {d->d_msk2, d->d_tmp2, d->d_tmp2_3, d->d_dst2M});
                    if (plane == 0) {
                        dump_buffer(d->d_dst2M, "dmsk_gaps_filled1", true);
                    }

                    run_kernel("fillGaps2X",
                               {d->d_msk2, d->d_dst2M, d->d_tmp2_3});
                    run_kernel("fillGaps2XStep2",
                               {d->d_msk2, d->d_dst2M, d->d_tmp2_3, d->d_tmp2});
                    if (plane == 0) {
                        dump_buffer(d->d_tmp2, "dmsk_gaps_filled2", true);
                    }

                    if (d->map == 3) {
                        final_buffer = d->d_tmp2;
                    } else {
                        [encoder endEncoding];
                        id<MTLBlitCommandEncoder> blit =
                            [cmd_buf blitCommandEncoder];

                        const int current_field =
                            static_cast<int>(params->field);
                        const size_t width_bytes =
                            (size_t)width * d->vi->format->bytesPerSample;
                        const size_t pitch = params->d_pitch;

                        if (current_field !=
                            0) { // field == 1, copy line height*2-2 to height*2-1
                            [blit copyFromBuffer:d->d_dst2
                                     sourceOffset:pitch * (height * 2 - 2)
                                         toBuffer:d->d_dst2
                                destinationOffset:pitch * (height * 2 - 1)
                                             size:width_bytes];
                        } else { // field == 0, copy line 1 to 0
                            [blit copyFromBuffer:d->d_dst2
                                     sourceOffset:pitch
                                         toBuffer:d->d_dst2
                                destinationOffset:0
                                             size:width_bytes];
                        }

                        // Copy tmp2 -> tmp2_3
                        [blit copyFromBuffer:d->d_tmp2
                                 sourceOffset:0
                                     toBuffer:d->d_tmp2_3
                            destinationOffset:0
                                         size:d->d_tmp2.length];
                        [blit endEncoding];
                        encoder = [cmd_buf computeCommandEncoder];

                        // interpolateLattice
                        // Args: d, omsk, dmsk, dst, dmsk_2
                        // CUDA: d, tmp2_2, tmp2, dst2, tmp2_3
                        run_kernel(
                            "interpolateLattice",
                            {d->d_tmp2_2, d->d_tmp2, d->d_dst2, d->d_tmp2_3});
                        if (plane == 0) {
                            dump_buffer(d->d_dst2, "output_interp", true);
                            dump_buffer(d->d_tmp2, "dmsk_interp", true);
                        }

                        if (d->pp == 1) {
                            run_kernel("filterDirMap2X",
                                       {d->d_msk2, d->d_tmp2_3, d->d_dst2M});
                            run_kernel("expandDirMap2X",
                                       {d->d_msk2, d->d_dst2M, d->d_tmp2});
                            if (plane == 0) {
                                dump_buffer(d->d_tmp2, "dmsk_filtered", true);
                            }
                            // postProcess: d, nmsk, omsk, dst
                            // CUDA: d, tmp2, tmp2_3, dst2
                            run_kernel("postProcess",
                                       {d->d_tmp2, d->d_tmp2_3, d->d_dst2});
                            if (plane == 0) {
                                dump_buffer(d->d_dst2, "output_pp", true);
                            }
                        }

                        final_buffer = d->d_dst2;
                    }
                }
            }

            [encoder endEncoding];

            id<MTLBlitCommandEncoder> blitEncoder =
                [cmd_buf blitCommandEncoder];
            [blitEncoder synchronizeResource:final_buffer];
            [blitEncoder endEncoding];

            // TODO: download would happen after the whole command buffer is committed
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            // This is a temporary copy for prototyping
            bitblt(vsapi->getWritePtr(dst_frame, plane), dst_stride,
                   [final_buffer contents], (int)metal_stride,
                   (size_t)width * d->vi->format->bytesPerSample,
                   (d->map == 1 || d->map == 2) ? height : height * 2);

            // Re-create command buffer for next plane if any
            if (plane < d->vi->format->numPlanes - 1) {
                cmd_buf = [d->queue commandBuffer];
            }
        }
    }

    vsapi->freeFrame(src_frame);
    return dst_frame;
}

static void VS_CC eedi2Free(void* instanceData, VSCore* /*unused*/,
                            const VSAPI* vsapi) {
    auto d = std::unique_ptr<EEDI2MetalData>(
        static_cast<EEDI2MetalData*>(instanceData));
    vsapi->freeNode(d->node);
}

static void VS_CC eedi2Create(const VSMap* in, VSMap* out, void* /*unused*/,
                              VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<EEDI2MetalData>();
    int err = 0;

    auto set_error = [&](const std::string& errorMessage) {
        vsapi->setError(out, ("EEDI2Metal: "s + errorMessage).c_str());
    };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    if ((isConstantFormat(d->vi) == 0) ||
        d->vi->format->sampleType != stInteger ||
        (d->vi->format->bytesPerSample != 1 &&
         d->vi->format->bytesPerSample != 2)) {
        set_error("only constant format 8 or 16 bit integer input supported");
        vsapi->freeNode(d->node);
        return;
    }
    d->bits_per_sample = d->vi->format->bitsPerSample;

    d->field = static_cast<int>(vsapi->propGetInt(in, "field", 0, &err));
    if (err != 0) {
        set_error("field is a required argument");
        vsapi->freeNode(d->node);
        return;
    }

    d->mthresh = static_cast<int>(vsapi->propGetInt(in, "mthresh", 0, &err));
    if (err != 0) {
        d->mthresh = 10;
    }
    d->lthresh = static_cast<int>(vsapi->propGetInt(in, "lthresh", 0, &err));
    if (err != 0) {
        d->lthresh = 20;
    }
    d->vthresh = static_cast<int>(vsapi->propGetInt(in, "vthresh", 0, &err));
    if (err != 0) {
        d->vthresh = 20;
    }
    d->estr = static_cast<int>(vsapi->propGetInt(in, "estr", 0, &err));
    if (err != 0) {
        d->estr = 2;
    }
    d->dstr = static_cast<int>(vsapi->propGetInt(in, "dstr", 0, &err));
    if (err != 0) {
        d->dstr = 4;
    }
    d->maxd = static_cast<int>(vsapi->propGetInt(in, "maxd", 0, &err));
    if (err != 0) {
        d->maxd = 24;
    }
    d->map = static_cast<int>(vsapi->propGetInt(in, "map", 0, &err));
    if (err != 0) {
        d->map = 0;
    }
    d->nt = static_cast<int>(vsapi->propGetInt(in, "nt", 0, &err));
    if (err != 0) {
        d->nt = 50;
    }
    d->pp = static_cast<int>(vsapi->propGetInt(in, "pp", 0, &err));
    if (err != 0) {
        d->pp = 1;
    }

    int num_planes_prop = vsapi->propNumElements(in, "planes");
    if (num_planes_prop > 0) {
        for (int i = 0; i < num_planes_prop; i++) {
            d->planes.push_back(
                static_cast<int>(vsapi->propGetInt(in, "planes", i, nullptr)));
        }
    } else {
        for (int i = 0; i < d->vi->format->numPlanes; i++) {
            d->planes.push_back(i);
        }
    }

    @autoreleasepool {
        d->device_id =
            static_cast<int>(vsapi->propGetInt(in, "device_id", 0, &err));
        if (err != 0) {
            d->device_id = 0;
        }
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (d->device_id >= static_cast<int>(devices.count)) {
            set_error("invalid device_id");
            return;
        }
        d->device = devices[d->device_id];

//NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
        constexpr char utils_metal[] = {
#embed "utils.metal"
        };
        constexpr char eedi2_common_metal[] = {
#embed "eedi2_common.metal"
        };
        constexpr char eedi2_kernels_metal[] = {
#embed "eedi2_kernels.metal"
        };
#pragma clang diagnostic pop
        //NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)

        // Construct complete Metal source
        std::string metal_source_str;
        metal_source_str += "#include <metal_stdlib>\n";
        metal_source_str += "using namespace metal;\n\n";

        metal_source_str +=
            "\n\n// ===== eedi2_common.metal.h (shared helpers) =====\n";
        metal_source_str +=
            std::string(static_cast<const char*>(eedi2_common_metal),
                        sizeof(eedi2_common_metal));

        metal_source_str += "\n\n// ===== 8-bit kernels =====\n";
        metal_source_str += "#define TYPE uchar\n";
        metal_source_str += "#define SUFFIX _u8\n";
        metal_source_str += "// utils.metal (8-bit)\n";
        metal_source_str += std::string(static_cast<const char*>(utils_metal),
                                        sizeof(utils_metal));
        metal_source_str += "\n// eedi2_kernels.metal (8-bit)\n";
        metal_source_str +=
            std::string(static_cast<const char*>(eedi2_kernels_metal),
                        sizeof(eedi2_kernels_metal));
        metal_source_str += "#undef TYPE\n";
        metal_source_str += "#undef SUFFIX\n";

        metal_source_str += "\n\n// ===== 16-bit kernels =====\n";
        metal_source_str += "#define TYPE ushort\n";
        metal_source_str += "#define SUFFIX _u16\n";
        metal_source_str += "// utils.metal (16-bit)\n";
        metal_source_str += std::string(static_cast<const char*>(utils_metal),
                                        sizeof(utils_metal));
        metal_source_str += "\n// eedi2_kernels.metal (16-bit)\n";
        metal_source_str +=
            std::string(static_cast<const char*>(eedi2_kernels_metal),
                        sizeof(eedi2_kernels_metal));
        metal_source_str += "#undef TYPE\n";
        metal_source_str += "#undef SUFFIX\n";

        NSString* lib_src =
            [NSString stringWithUTF8String:metal_source_str.c_str()];
        NSError* error = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        d->library = [d->device newLibraryWithSource:lib_src
                                             options:opts
                                               error:&error];
        if (d->library == nullptr) {
            set_error("Failed to compile metal library: "s +
                      [error.localizedDescription UTF8String]);
            return;
        }

        d->queue = [d->device newCommandQueue];

        NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* psos =
            [NSMutableDictionary dictionary];
        std::vector<std::string> kernel_names = {
            "buildEdgeMask",    "erode",
            "dilate",           "removeSmallHorzGaps",
            "calcDirections",   "filterDirMap",
            "expandDirMap",     "filterMap",
            "markDirections2X", "filterDirMap2X",
            "expandDirMap2X",   "fillGaps2X",
            "fillGaps2XStep2",  "interpolateLattice",
            "postProcess",      "enlarge2"};
        std::string suffix = (d->bits_per_sample == 8) ? "_u8" : "_u16";

        for (const auto& name : kernel_names) {
            std::string funcNameStr = name;
            NSString* funcName =
                [NSString stringWithUTF8String:(name + suffix).c_str()];

            id<MTLFunction> func = [d->library newFunctionWithName:funcName];
            if (func == nullptr) {
                set_error("Failed to create function: "s +
                          [funcName UTF8String]);
                return;
            }
            psos[[NSString stringWithUTF8String:(name + suffix).c_str()]] =
                [d->device newComputePipelineStateWithFunction:func
                                                         error:&error];
            if (psos[[NSString stringWithUTF8String:(name + suffix).c_str()]] ==
                nullptr) {
                set_error("Failed to create PSO: "s + name);
                return;
            }
        }
        d->psos = psos;

        // Calculate plane size based on max possible stride
        // For plane 0: width * bytesPerSample, aligned to some reasonable value
        size_t stride =
            ((size_t)d->vi->width * d->vi->format->bytesPerSample + 63) &
            ~63; // 64-byte alignment
        size_t plane_size = (size_t)d->vi->height * stride;
        size_t plane_size_2x = plane_size * 2;

        MTLResourceOptions options = MTLResourceStorageModeManaged;
        d->d_src = [d->device newBufferWithLength:plane_size options:options];
        d->d_msk = [d->device newBufferWithLength:plane_size options:options];
        d->d_tmp = [d->device newBufferWithLength:plane_size options:options];
        d->d_dst = [d->device newBufferWithLength:plane_size options:options];

        if (d->map == 0 || d->map == 3) {
            d->d_dst2 = [d->device newBufferWithLength:plane_size_2x
                                               options:options];
            d->d_msk2 = [d->device newBufferWithLength:plane_size_2x
                                               options:options];
            d->d_tmp2 = [d->device newBufferWithLength:plane_size_2x
                                               options:options];
            d->d_tmp2_2 = [d->device newBufferWithLength:plane_size_2x
                                                 options:options];
            d->d_tmp2_3 = [d->device newBufferWithLength:plane_size_2x
                                                 options:options];
            d->d_dst2M = [d->device newBufferWithLength:plane_size_2x
                                                options:options];
        }

        d->params_buffer =
            [d->device newBufferWithLength:sizeof(EEDI2Param)
                                   options:MTLResourceStorageModeShared];
    }

    vsapi->createFilter(in, out, "EEDI2", eedi2Init, eedi2GetFrame, eedi2Free,
                        fmParallelRequests, 0, d.release(), core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc,
                      VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.Sunflower-dolls.eedi2metal", "eedi2metal",
               "EEDI2 filter using Metal", VAPOURSYNTH_API_VERSION, 1, plugin);

    const char* eedi2_args =
        "clip:clip;field:int;mthresh:int:opt;lthresh:int:opt;vthresh:int:"
        "opt;estr:int:opt;dstr:int:opt;maxd:int:opt;map:int:opt;nt:int:opt;pp:"
        "int:opt;planes:int[]:opt;device_id:int:opt;";

    registerFunc("EEDI2", eedi2_args, eedi2Create, nullptr, plugin);
}