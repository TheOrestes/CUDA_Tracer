
#include "GLRenderer/Common.h"

#include <corecrt_math.h>
#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------
__device__ float3 GetMissColor(const RT::Ray& r)
{
	float3 unit_direction = unit_vector(r.Direction);
	float t = 0.5f * (unit_direction.y + 1.0f);

	// Lerp: White (1,1,1) to Blue (0.5, 0.7, 1.0)
	float3 white = make_float3(1.0f, 1.0f, 1.0f);
	float3 blue = make_float3(0.5f, 0.7f, 1.0f);

	return white * (1.0f - t) + blue * t;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void RayTracer(cudaSurfaceObject_t surface, int width, int height, RT::Camera cam)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)  return;

	// Normalized coordinates
	float u = float(x) / float(width - 1);
	float v = float(y) / float(height - 1);

	// 1. Generate Ray from Camera
	RT::Ray r = cam.GetRay(u, v);

	// 2. Calculate Color
	float3 pixelColor = GetMissColor(r);

	// 4. Write final color to the framebuffer!
	// x * sizeof(float4) is CRITICAL for surf2Dwrite
	float4 finalColor = make_float4(pixelColor.x, pixelColor.y, pixelColor.z, 1.0f);
	surf2Dwrite(finalColor, surface, x * sizeof(float4), y);
}

//---------------------------------------------------------------------------------------------------------------------
// This function
// 1. Lets CUDA take control of OpenGL texture
// 2. Runs CUDA kernel
// 3. Writes data to the texture
// 4. Gives back control of the updated texture to OpenGL
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource, int cuWidth, int cuHeight, RT::Camera camera)
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
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cuWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (cuHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Launch the CUDA Kernel!
	RayTracer << <blocksPerGrid, threadsPerBlock>> > (surface, cuWidth, cuHeight, camera);

	// Cleanup!
	cudaDestroySurfaceObject(surface);

	// OpenGL now regains the control of updated texture!
	cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
}