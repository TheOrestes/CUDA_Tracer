
#include "RT_Common.cuh"
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------
// --- Helper: Random float in [0, 1] ---
__device__ float RandomFloat(curandState* state)
{
	return curand_uniform(state);
}

//---------------------------------------------------------------------------------------------------------------------
__device__ float3 Reflect(const float3& V, const float3& N)
{
	return V - 2.0f * dot(V, N) * N;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ bool Refract(const float3& V, const float3& N, float ni_over_nt, float3& refracted)
{
	const float3 uv = unit_vector(V);
	const float dt = dot(uv, N);
	const float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);

	if (discriminant > 0.0f)
	{
		refracted = ni_over_nt * (uv - N * dt) - N * sqrtf(discriminant);
		return true;
	}

	return false; // Total internal reflection
}

// -------------------------------------------------------------------------------------------------------------------- 
// Schlick's approximation for Fresnel reflectance
__device__ float Schlick(float cosine, float ref_idx)
{
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

//---------------------------------------------------------------------------------------------------------------------
// Rejection method to find a random point in unit sphere
__device__ float3 RandomVectorInUnitSphere(curandState* state)
{
	float3 p;

	do
	{
		p = 2.0f * make_float3(RandomFloat(state), RandomFloat(state), RandomFloat(state)) - make_float3(1, 1, 1);
	} while (dot(p, p) >= 1.0f);

	return p;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ float3 GetHeatmapColor(float t)
{
	// t is 0.0 (Cheap) to 1.0 (Expensive)
	// Simple Blue -> Green -> Red gradient
	if (t < 0.5f) 
	{
		// Blue to Green
		return (1.0f - 2.0f * t) * make_float3(0, 0, 1) + (2.0f * t) * make_float3(0, 1, 0);
	}
	else 
	{
		// Green to Red
		return (1.0f - 2.0f * (t - 0.5f)) * make_float3(0, 1, 0) + (2.0f * (t - 0.5f)) * make_float3(1, 0, 0);
	}
}

//---------------------------------------------------------------------------------------------------------------------
__device__ float3 GetMissColor(const RT::Ray& r, curandState state)
{
	const float3 unit_direction = unit_vector(r.Direction);
	const float t = 0.5f * (unit_direction.y + 1.0f);

	//--- Enable this to visualize noise & accumulation!
	//const float3 white = make_float3(RandomFloat(&state), RandomFloat(&state), RandomFloat(&state));
	//const float3 blue = make_float3(RandomFloat(&state), RandomFloat(&state), RandomFloat(&state));

	// Lerp: White (1,1,1) to Blue (0.5, 0.7, 1.0)
	const float3 white = make_float3(1, 1, 1);
	const float3 blue = make_float3(0.5f, 0.7f, 1.0f);

	return white * (1.0f - t) + blue * t;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ bool HitSphere(const RT::SphereData& s, const RT::Ray& r, float t_min, float t_max, RT::HitRecord& rec) 
{
	const float3 oc = r.Origin - s.center;
	const float a = dot(r.Direction, r.Direction);
	const float b = 2.0f * dot(oc, r.Direction);
	const float c = dot(oc, oc) - s.radius * s.radius;
	const float discriminant = b * b - 4 * a * c;

	if (discriminant > 0)
	{
		float temp = (-b - sqrtf(discriminant)) / (2.0f * a);
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.P = r.GetAt(rec.t);
			rec.Normal = (rec.P - s.center) / s.radius;

			const float phi = atan2(rec.Normal.z, rec.Normal.x);
			const float theta = asin(rec.Normal.y);

			rec.UV.x = 1.0f - (phi + RT_PI) / (2.0f * RT_PI);
			rec.UV.y = (theta + RT_PI / 2.0f) / RT_PI;

			return true;
		}

		temp = (-b + sqrtf(discriminant)) / (2.0f * a);
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.P = r.GetAt(rec.t);
			rec.Normal = (rec.P - s.center) / s.radius;

			const float phi = atan2(rec.Normal.z, rec.Normal.x);
			const float theta = asin(rec.Normal.y);

			rec.UV.x = 1.0f - (phi + RT_PI) / (2.0f * RT_PI);
			rec.UV.y = (theta + RT_PI / 2.0f) / RT_PI;

			return true;
		}
	}

	return false;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ bool HitObject(const RT::SceneObject& obj, RT::Ray& r, float t_min, float t_max, RT::HitRecord& rec)
{
	bool hit = false;

	switch (obj.type)
	{
	case RT::SPHERE:
		hit = HitSphere(obj.sphere, r, t_min, t_max, rec);
		break;
	}

	if (hit)
		rec.MaterialID = obj.MaterialID;

	return hit;
}

//---------------------------------------------------------------------------------------------------------------------
__device__ bool HitWorld(RT::Ray& r, float t_min, float t_max, RT::HitRecord& rec, RT::SceneObject* objects, int numObjects)
{
	RT::HitRecord temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	// Iterate through all the scene objects!
	for (int i = 0; i < numObjects; i++) 
	{
		if (HitObject(objects[i], r, t_min, closest_so_far, temp_rec)) 
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void RayTracer(cudaSurfaceObject_t surface, int width, int height, RT::Camera cam, float4* pAccumBuffer, int currentSPP, RT::SceneObject* pObjects, int numObject, RT::Material* pMaterials, bool showHeatmap)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)  return;

	// Initialize RNG per pixel
	const int pixelIndex = y * width + x;
	curandState localState;
	curand_init(1984 + pixelIndex, currentSPP, 0, &localState);

	// Normalized coordinates
	float u = (float(x) + RandomFloat(&localState)) / float(width - 1);
	float v = (float(y) + RandomFloat(&localState)) / float(height - 1);

	// 1. Generate Ray from Camera
	RT::Ray r = cam.GetRay(u, v);
	float3 currentColor = make_float3(1, 1, 1);
	float3 pixelColor = make_float3(0, 0, 0);

	// Heatmap - Track actual bounces taken!
	int actualBounces = 0;

	// Iterative Bounce Loop (Recursion Depth = 50)
	for (int depth = 0 ; depth < 50 ; ++depth)
	{
		RT::HitRecord rec;

		// 2. Check Intersection & Calculate Color
		if (HitWorld(r, 0.001f, 1000.0f, rec, pObjects, numObject))
		{
			// Heatmap - Increment the bounce count!
			++actualBounces;

			// Fetch material data from the gloabl memory
			RT::Material material = pMaterials[rec.MaterialID];

			switch (material.type)
			{
				case RT::LAMBERTIAN:
				{
					float3 target = rec.P + rec.Normal + RandomVectorInUnitSphere(&localState);
					r = RT::Ray(rec.P, target - rec.P);
					currentColor = currentColor * material.Albedo;

					break;
				}

				case RT::METAL:
				{
					float3 reflected = Reflect(unit_vector(r.Direction), rec.Normal);

					// Add Fuzziness (roughness) to the reflection
					float3 scattered = reflected + material.Fuzz * RandomVectorInUnitSphere(&localState);

					// only scatter if reflection isn't absorbed
					if(dot(scattered, rec.Normal) > 0.0f)
					{
						r = RT::Ray(rec.P, scattered);
						currentColor = currentColor * material.Albedo;
					}

					break;
				}

				case RT::PHONG:
				{
					break;
				}

				case RT::TRANSPARENT:
				{
					float3 outward_normal;
					float3 reflected = Reflect(r.Direction, rec.Normal);
					float ni_over_nt;
					float3 refracted;
					float reflect_prob;
					float cosine;

					// Determine if ray is entering or exiting the material
					if (dot(r.Direction, rec.Normal) > 0.0f)
					{
						// Exiting the material
						outward_normal = -rec.Normal;
						ni_over_nt = material.IoR;
						cosine = material.IoR * dot(r.Direction, rec.Normal) / sqrtf(dot(r.Direction, r.Direction));
					}
					else
					{
						// Entering the material
						outward_normal = rec.Normal;
						ni_over_nt = 1.0f / material.IoR;
						cosine = -dot(r.Direction, rec.Normal) / sqrtf(dot(r.Direction, r.Direction));
					}

					// Try to refract
					if (Refract(r.Direction, outward_normal, ni_over_nt, refracted))
					{
						reflect_prob = Schlick(cosine, material.IoR);
					}
					else
					{
						// Total internal reflection
						reflect_prob = 1.0f;
					}

					// Randomly choose between reflection and refraction based on Fresnel
					if (RandomFloat(&localState) < reflect_prob)
					{
						r = RT::Ray(rec.P, reflected);
					}
					else
					{
						r = RT::Ray(rec.P, refracted);
					}

					// Transparent materials don't attenuate color (or use material.Albedo for tinted glass)
					currentColor = currentColor * make_float3(1.0f, 1.0f, 1.0f);

					break;
				}
			}
		}
		else
		{
			// NO: We hit the sky (Light Source)
			pixelColor = currentColor * GetMissColor(r, localState);
			break;
		}
	}

	// Heatmap Visualization!
	float4 pixelColorRGBA;
	if(showHeatmap)
	{
		// Normalize bounce count to [0,1] range
		float t = static_cast<float>(actualBounces) / 50;
		float3 heatmapColor = GetHeatmapColor(t);
		pixelColorRGBA = make_float4(heatmapColor.x, heatmapColor.y, heatmapColor.z, 1.0f);
	}
	else
	{
		pixelColorRGBA = make_float4(pixelColor.x, pixelColor.y, pixelColor.z, 1.0f);
	}

	// Accumulation logic!
	float4 finalColor;

	if(currentSPP == 1)
	{
		// First frame, just save the current color!
		finalColor = pixelColorRGBA;
		pAccumBuffer[pixelIndex] = finalColor;
	}
	else
	{
		// subsequent frames = Blend!
		const float4 oldColor = pAccumBuffer[pixelIndex];

		float n = float(currentSPP);

		finalColor.x = oldColor.x + (pixelColorRGBA.x - oldColor.x) / n;
		finalColor.y = oldColor.y + (pixelColorRGBA.y - oldColor.y) / n;
		finalColor.z = oldColor.z + (pixelColorRGBA.z - oldColor.z) / n;
		finalColor.w = 1.0f;

		// store back to buffer
		pAccumBuffer[pixelIndex] = finalColor;
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
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource, int cuWidth, int cuHeight, RT::Camera camera, float4* pAccumBuffer, int currentSPP, RT::SceneObject* pObjects, int numObjects, RT::Material* pMaterials, bool showHeatmap)
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
	RayTracer <<<blocksPerGrid, threadsPerBlock>>>(surface, cuWidth, cuHeight, camera, pAccumBuffer, currentSPP, pObjects, numObjects, pMaterials, showHeatmap);

	// Cleanup!
	cudaDestroySurfaceObject(surface);

	// OpenGL now regains the control of updated texture!
	cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
}