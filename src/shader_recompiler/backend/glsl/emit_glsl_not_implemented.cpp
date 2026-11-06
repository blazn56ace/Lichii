// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <cstdlib>

#include <string>

#include "common/logging.h"
#include "shader_recompiler/backend/glsl/emit_glsl_instructions.h"
#include "shader_recompiler/backend/glsl/glsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLSL {

void EmitGetRegister(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetRegister(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetPred(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetPred(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetGotoVariable(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetGotoVariable(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetIndirectBranchVariable(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetIndirectBranchVariable(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetZFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetSFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetCFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetOFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetZFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetSFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetCFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitSetOFlag(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetZeroFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetSignFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetCarryFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetOverflowFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetSparseFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitGetInBoundsFromOp(EmitContext& ctx) {
    NotImplemented();
}

void EmitIsbeRead(EmitContext& ctx) {
    NotImplemented();
}

void EmitIsbeWriteAttribute(EmitContext& ctx) {
    NotImplemented();
}

void EmitIsbeWriteMapU8(EmitContext& ctx) {
    NotImplemented();
}

void EmitIsbeWriteMapU32(EmitContext& ctx) {
    NotImplemented();
}

void EmitTensorMulAdd(EmitContext& ctx, IR::Inst& inst, const IR::Value& a,
                      const IR::Value& b, const IR::Value& c) {
    const u32 depth = inst.Flags<u32>();
    if (depth != 8 && depth != 16) {
        throw NotImplementedException("Unsupported GLSL HMMA depth {}", depth);
    }

    // GLSL software HMMA mirrors the SPIR-V lane mapping and sequential FMA order.
    // readInvocationARB requires GL_ARB_shader_ballot.
    if (ctx.header.find("#extension GL_ARB_shader_ballot") == std::string::npos) {
        ctx.header.insert(0, "#extension GL_ARB_shader_ballot : require\n");
    }
    if (ctx.header.find("LichiiSoftwareHmma(") == std::string::npos) {
        // Diagnostic only: 0=explicit rounding (baseline), 1=pack/unpack, 2=no intermediate rounding.
        const char* mode_env = std::getenv("EDEN_NX2_GLSL_HMMA_ROUND_MODE");
        const std::string_view mode = mode_env ? std::string_view(mode_env) : std::string_view{};
        const unsigned round_mode = mode == "1" ? 1u : mode == "2" ? 2u : 0u;
        static bool logged_hmma_mode = false;
        if (!logged_hmma_mode) {
            LOG_INFO(Render_OpenGL, "GLSL software HMMA active: rounding mode={} depth={} (0=explicit, 1=pack/unpack, 2=none)", round_mode, depth);
            logged_hmma_mode = true;
        }
        ctx.header += "#define LICHII_HMMA_ROUND_MODE " + std::to_string(round_mode) + "\n";
        ctx.header += R"GLSL(
// Match the SPIR-V software HMMA's explicit FP16 quantization, including
// ties-to-even and subnormals. Avoid pack/unpack round trips that a GLSL
// optimizer may fold away when the intermediate FP32 value is reused.
float LichiiRoundToF16(float value) {
    uint bits = floatBitsToUint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint tie = (magnitude >> 13u) & 1u;
    uint rounded = (magnitude + 0xfffu + tie) & ~0x1fffu;
    uint normal = rounded >= 0x47800000u ? 0x7f800000u : rounded;
    float scaled = uintBitsToFloat(magnitude) * 16777216.0;
    uint subnormal = floatBitsToUint(roundEven(scaled) * (1.0 / 16777216.0));
    uint finite = magnitude >= 0x38800000u ? normal : subnormal;
    uint result = magnitude >= 0x7f800000u ? magnitude : finite;
    return uintBitsToFloat(result | (bits & 0x80000000u));
}

uvec2 LichiiSoftwareHmma(uvec4 matrixA, uvec2 matrixB, uvec2 matrixC, uint depth) {
    vec2 c0 = unpackHalf2x16(matrixC.x);
    vec2 c1 = unpackHalf2x16(matrixC.y);
    vec4 accum = vec4(c0, c1);
    uint lane = uint(gl_SubGroupInvocationARB);
    // readInvocationARB uses subgroup-relative lane indices, not absolute IDs.
    uint rowBase = lane & 28u;
    uint colBase = (lane & 3u) << 3u;
    for (uint pair = 0u; pair < depth / 2u; ++pair) {
        uint component = pair & 3u;
        uint aLane = rowBase + component;
        uint bLane = colBase + component;
        uint upper = pair >= 4u ? 2u : 0u;
        uint bPacked = matrixB[upper >> 1u];
        vec2 b0 = unpackHalf2x16(readInvocationARB(bPacked, bLane));
        vec2 b1 = unpackHalf2x16(readInvocationARB(bPacked, bLane + 4u));
        vec2 a0 = unpackHalf2x16(readInvocationARB(matrixA[upper], aLane));
        vec2 a1 = unpackHalf2x16(readInvocationARB(matrixA[upper + 1u], aLane));
        accum.x = fma(a0.x, b0.x, accum.x);
        accum.x = fma(a0.y, b0.y, accum.x);
        accum.y = fma(a0.x, b1.x, accum.y);
        accum.y = fma(a0.y, b1.y, accum.y);
        accum.z = fma(a1.x, b0.x, accum.z);
        accum.z = fma(a1.y, b0.y, accum.z);
        accum.w = fma(a1.x, b1.x, accum.w);
        accum.w = fma(a1.y, b1.y, accum.w);
        if (component == 3u) {
            // HMMA.F16 rounds at each K=8 boundary.
            #if LICHII_HMMA_ROUND_MODE == 0
            accum.x = LichiiRoundToF16(accum.x);
            accum.y = LichiiRoundToF16(accum.y);
            accum.z = LichiiRoundToF16(accum.z);
            accum.w = LichiiRoundToF16(accum.w);
            #elif LICHII_HMMA_ROUND_MODE == 1
            accum = vec4(unpackHalf2x16(packHalf2x16(accum.xy)),
                         unpackHalf2x16(packHalf2x16(accum.zw)));
            #endif
        }
    }
    return uvec2(packHalf2x16(accum.xy), packHalf2x16(accum.zw));
}
)GLSL";
    }
    ctx.AddU32x2("{}=LichiiSoftwareHmma({}, {}, {}, {}u);", inst,
                 ctx.var_alloc.Consume(a), ctx.var_alloc.Consume(b),
                 ctx.var_alloc.Consume(c), depth);
}

} // namespace Shader::Backend::GLSL
