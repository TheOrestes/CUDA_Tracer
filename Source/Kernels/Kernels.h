#pragma once
#include <cuda_runtime.h>

//---------------------------------------------------------------------------------------------------------------------
extern "C" void RunRayTracingKernel(cudaGraphicsResource_t res, int cuWidth, int cuHeight, float t);