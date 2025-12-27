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
    //---
    struct Ray
    {
        float3 Origin;
        float3 Direction;

        __host__ __device__ Ray() {}

        __host__ __device__ Ray(const float3& o, const float3& d)
	        : Origin(o), Direction(d) { }

        __host__ __device__ float3 GetAt(float t) const { return Origin + Direction * t; }
    };

    //---
    struct HitRecord
    {
        float t;
        float3 P;
        float3 Normal;
        float3 Albedo;
    };

    //---
    struct Sphere
    {
        float3 center;
        float radius;
        float3 color;

        __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
        {
            const float3 oc = r.Origin - center;
            const float a = dot(r.Direction, r.Direction);
            const float b = 2.0f * dot(oc, r.Direction);
            const float c = dot(oc, oc) - radius * radius;
            const float discriminant = b * b - 4 * a * c;

            if(discriminant > 0)
            {
                float temp = (-b - sqrtf(discriminant)) / (2.0f * a);
                if (temp < t_max && temp > t_min)
                {
                    rec.t = temp;
                    rec.P = r.GetAt(rec.t);
                    rec.Normal = (rec.P - center) / radius;
                    rec.Albedo = color;
                    return true;
                }

                temp = (-b + sqrtf(discriminant)) / (2.0f *a);
                if (temp < t_max && temp > t_min) 
                {
                    rec.t = temp;
                    rec.P = r.GetAt(rec.t);
                    rec.Normal = (rec.P - center) / radius;
                    rec.Albedo = color;
                    return true;
                }
            }

            return false;
        }
    };

    //---
    struct Camera
    {
        float3 Origin;
        float3 Lower_Left_Corner;
        float3 Horizontal;
        float3 Vertical;

        //--- Store basis vectors to allow easy movement updates
        float3 u, v, w;
        float vFov, Aspect_ratio;

        //--- Default constructor!
        __host__ __device__ Camera() {}

        //--- Parameterized constructor!
        __host__ __device__ Camera(float3 lookfrom, float3 lookat, float3 vup, float _vfov, float _aspect_ratio)
        {
            Init(lookfrom, lookat, vup, _vfov, _aspect_ratio);
        }

        //--- Initialize Camera parameters on the Host (or Device if needed) 
        __host__ __device__ void Init(float3 lookfrom, float3 lookat, float3 vup, float _vfov, float _aspect_ratio)
    	{
            Origin = lookfrom;
            vFov = _vfov;
            Aspect_ratio = _aspect_ratio;

            const float theta = vFov * 3.14159265358979323846f / 180.0f;
            const float h = tanf(theta / 2.0f);
            const float viewport_height = 2.0f * h;
            const float viewport_width = Aspect_ratio * viewport_height;

            // Calculate orthonormal basis
            w = unit_vector(lookfrom - lookat); // Backward vector
            u = unit_vector(cross(vup, w));     // Right vector
            v = cross(w, u);                    // Up vector

            Horizontal = u * viewport_width;
            Vertical = v * viewport_height;
            Lower_Left_Corner = Origin - Horizontal / 2.0f - Vertical / 2.0f - w;
        }

        //--- Helper to move the camera
		// 'w' is backward, so -w is forward
        __host__ void move(float3 offset)
    	{
            Origin = Origin + offset;
            // Re-calculate corners based on new origin (orientation stays same)
            Lower_Left_Corner = Origin - Horizontal / 2.0f - Vertical / 2.0f - w;
        }


        //--- get a Ray for UV coords
        __host__ __device__ Ray GetRay(float u, float v)
        {
            return Ray(Origin, Lower_Left_Corner + Horizontal * u + Vertical * v - Origin);
        }
    };
}

//---------------------------------------------------------------------------------------------------------------------
extern "C" void RunRayTracingKernel(cudaGraphicsResource_t res, int cuWidth, int cuHeight, RT::Camera cam, float4* accumBuffer, int frameCount);
