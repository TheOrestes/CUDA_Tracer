#pragma once

#include <cuda_runtime.h>
#include <cmath>

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator/(const float3& a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 unit_vector(const float3& v)
{
    float len = sqrtf(dot(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

//-----------------------------------------------------------------------------------------------------------------
namespace RT
{
    //-----------------------------------------------------------------------------------------------------------------
    struct Ray
    {
        float3 Origin;
        float3 Direction;

        __host__ __device__ Ray() {}

        __host__ __device__ Ray(const float3& o, const float3& d)
	        : Origin(o), Direction(d) { }

        __host__ __device__ float3 GetAt(float t) const { return Origin + Direction * t; }
    };

    //-----------------------------------------------------------------------------------------------------------------
    struct Camera
    {
        float3 Origin;
        float3 Lower_Left_Corner;
        float3 Horizontal;
        float3 Vertical;

        // Default constructor!
        __host__ __device__ Camera() {}

        // Initialize Camera parameters on the Host (or Device if needed)
        __host__ __device__ Camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect_ratio)
        {
            float theta = vfov * 3.14159265358979323846f / 180.0f;
            float h = tanf(theta / 2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            float3 w = unit_vector(lookfrom - lookat);
            float3 u = unit_vector(cross(vup, w));
            float3 v = cross(w, u);

            Origin = lookfrom;
            Horizontal = u * viewport_width;
            Vertical = v * viewport_height;
            Lower_Left_Corner = Origin - Horizontal / 2.0f - Vertical / 2.0f - w;
        }

        // get a Ray for UV coords
        __host__ __device__ Ray GetRay(float u, float v)
        {
            return Ray(Origin, Lower_Left_Corner + Horizontal * u + Vertical * v - Origin);
        }
    };
}

//---------------------------------------------------------------------------------------------------------------------
extern "C" void RunRayTracingKernel(cudaGraphicsResource_t res, int cuWidth, int cuHeight, RT::Camera cam);
