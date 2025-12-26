
#include "GLRenderer/Common.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------
// --- Helper: Random float in [0, 1] ---
__device__ inline float rand_float(curandState* state)
{
	return curand_uniform(state);
}

//---------------------------------------------------------------------------------------------------------------------
__device__ float3 GetMissColor(const RT::Ray& r, curandState state)
{
	const float3 unit_direction = unit_vector(r.Direction);
	float t = 0.5f * (unit_direction.y + 1.0f);

	//--- Enable this to visualize noise & accumulation!
	//const float3 white = make_float3(rand_float(&state), rand_float(&state), rand_float(&state));
	//const float3 blue = make_float3(rand_float(&state), rand_float(&state), rand_float(&state));

	// Lerp: White (1,1,1) to Blue (0.5, 0.7, 1.0)
	const float3 white = make_float3(1, 1, 1);
	const float3 blue = make_float3(0.5f, 0.7f, 1.0f);

	return white * (1.0f - t) + blue * t;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ bool HitWorld(const RT::Ray& r, float t_min, float t_max, RT::HitRecord& rec, float3& outColor)
{
	RT::HitRecord temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	// Define simple scene
	const RT::Sphere spheres[] = 
	{
		{ make_float3(0.0f, 0.0f, -1.0f),    0.5f,   make_float3(1.0f, 0.0f, 0.0f) }, // Red Sphere
		{ make_float3(0.0f, -100.5f, -1.0f), 100.0f, make_float3(0.0f, 1.0f, 0.0f) }  // Green Floor
	};

	constexpr int num_spheres = 2;

	for (int i = 0; i < num_spheres; i++) 
	{
		if (spheres[i].hit(r, t_min, closest_so_far, temp_rec)) 
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;

			// --- COLORING STRATEGY ---
			// Option A: Use the sphere's flat color
			outColor = spheres[i].color; 

			// Option B: Normal Visualization (Rainbow colors based on curve)
			// Maps normal [-1, 1] to color [0, 1]
			//outColor = 0.5f * (rec.Normal + make_float3(1.0f, 1.0f, 1.0f));
		}
	}

	return hit_anything;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void RayTracer(cudaSurfaceObject_t surface, int width, int height, RT::Camera cam, int frameCount, float4* accumBuffer)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)  return;

	// Initialize RNG per pixel
	const int pixelIndex = y * width + x;
	curandState localState;
	curand_init(1984 + pixelIndex, frameCount, 0, &localState);

	// Normalized coordinates
	float u = (float(x) + rand_float(&localState)) / float(width - 1);
	float v = (float(y) + rand_float(&localState)) / float(height - 1);

	// 1. Generate Ray from Camera
	RT::Ray r = cam.GetRay(u, v);

	RT::HitRecord rec;
	float3 hitColor;
	float3 currentColor;

	// 2. Check Intersection & Calculate Color
	if(HitWorld(r, 0.001f, 1000.0f, rec, hitColor))
	{
		currentColor = hitColor;
	}
	else
	{
		currentColor = GetMissColor(r, localState);
	}

	const float4 currentColorRGBA = make_float4(currentColor.x, currentColor.y, currentColor.z, 1.0f);

	// Accumulation logic!
	float4 finalColor;

	if(frameCount == 1)
	{
		// First frame, just save the current color!
		finalColor = currentColorRGBA;
		accumBuffer[pixelIndex] = finalColor;
	}
	else
	{
		// subsequent frames = Blend!
		const float4 oldColor = accumBuffer[pixelIndex];

		float n = float(frameCount);

		finalColor.x = oldColor.x + (currentColorRGBA.x - oldColor.x) / n;
		finalColor.y = oldColor.y + (currentColorRGBA.y - oldColor.y) / n;
		finalColor.z = oldColor.z + (currentColorRGBA.z - oldColor.z) / n;
		finalColor.w = 1.0f;

		// store back to buffer
		accumBuffer[pixelIndex] = finalColor;
	}

	// can do gamma correction here if required!

	// 4. Write final color to the framebuffer!
	// x * sizeof(float4) is CRITICAL for surf2Dwrite
	surf2Dwrite(finalColor, surface, x * sizeof(float4), y);
}

//---------------------------------------------------------------------------------------------------------------------
// This function
// 1. Lets CUDA take control of OpenGL texture
// 2. Runs CUDA kernel
// 3. Writes data to the texture
// 4. Gives back control of the updated texture to OpenGL
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource, int cuWidth, int cuHeight, RT::Camera camera, float4* accumBuffer, int frameCount)
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
	RayTracer << <blocksPerGrid, threadsPerBlock>> > (surface, cuWidth, cuHeight, camera, frameCount, accumBuffer);

	// Cleanup!
	cudaDestroySurfaceObject(surface);

	// OpenGL now regains the control of updated texture!
	cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
}