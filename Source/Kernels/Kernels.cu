
#include "Kernels.h"

#include <corecrt_math.h>
#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------
__global__ void RayTracer(cudaSurfaceObject_t surface, int width, int height, float time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) 
	{
		float u = x / (float)width;
		float v = y / (float)height;

		float r = 0.5f + 0.5f * sinf(time + u * 6.28f);
		float g = 0.5f + 0.5f * cosf(time + v * 6.28f);
		float b = 0.5f + 0.5f * sinf(time + (u + v) * 6.28f);

		// x * sizeof(float4) is CRITICAL for surf2Dwrite
		surf2Dwrite(make_float4(r, g, b, 1.0f), surface, x * sizeof(float4), y);
	}
}

//---------------------------------------------------------------------------------------------------------------------
// This function
// 1. Lets CUDA take control of OpenGL texture
// 2. Runs CUDA kernel
// 3. Writes data to the texture
// 4. Gives back control of the updated texture to OpenGL
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource, int cuWidth, int cuHeight, float time)
{
	// Map the OpenGL resource, post this CUDA controls the texture...
	cudaGraphicsMapResources(1, &cuda_graphics_resource, 0);

	// Get CUDA specific handle to the texture's memory!
	// cudaArray_t is a special, hardware optimized data type for textures allowing efficient 2D/3D access patterns.
	cudaArray_t cuArray;
	cudaGraphicsSubResourceGetMappedArray(&cuArray, cuda_graphics_resource, 0, 0);

	// Creare Surface object!
	// Surface object allows read/write access to the texture memory from the CUDA kernel! 
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaSurfaceObject_t surface;
	cudaCreateSurfaceObject(&surface, &resDesc);

	// Configure the kernel launch grid
	dim3 blockSize(16, 16);
	dim3 gridSize((cuWidth + blockSize.x - 1) / blockSize.x, (cuHeight + blockSize.y - 1) / blockSize.y);

	// Launch the CUDA Kernel!
	RayTracer << <gridSize, blockSize >> > (surface, cuWidth, cuHeight, time);

	// Cleanup!
	cudaDestroySurfaceObject(surface);

	// OpenGL now regains the control of updated texture!
	cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
}