#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel, 
            float gate_scale = 1.0f, float up_scale = 1.0f, float output_scale = 1.0f);
}