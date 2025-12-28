#pragma once

#include <corecrt_math.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define RT_PI 3.14159265358979323846f

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
	const float len = sqrtf(dot(v, v));
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
        __host__ __device__ Ray(const float3& o, const float3& d) : Origin(o), Direction(d) {}
        __host__ __device__ float3 GetAt(float t) const { return Origin + Direction * t; }
    };

    //---
    struct HitRecord
    {
        float   t;
        float3  P;
        float3  Normal;
        float2  UV;
        int     MaterialID;
    };

    enum ObjectType
    {
        SPHERE,
        MESH
    };

    struct SphereData
    {
        float3 center;
        float radius;
    };

    struct TriangleData
    {
        float3 V0, V1, V2;
        float2 UV0, UV1, UV2;
        float3 Normal;
    };

    struct SceneObject
    {
        ObjectType type;
        int MaterialID;

        union
        {
            SphereData sphere;
            TriangleData triangle;
        };

        // Default Hit function!
        __device__ bool Hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const;
    };

    enum MaterialType
    {
        LAMBERTIAN,
        METAL,
        PHONG,
        TRANSPARENT
    };

    struct Material
    {
        MaterialType type;
        float3 Albedo;
        float Fuzz;
        float IoR;
    };



    //---
    struct Sphere
    {
        float3 center;
        float radius;
        float3 color;
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
        __host__ Camera() {}

        //--- Initialize Camera parameters on the Host (or Device if needed) 
        __host__ void Init(float3 lookfrom, float3 lookat, float3 vup, float _vfov, float _aspect_ratio)
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
        __device__ Ray GetRay(float u, float v)
        {
            return Ray(Origin, Lower_Left_Corner + Horizontal * u + Vertical * v - Origin);
        }
    };
}

//---------------------------------------------------------------------------------------------------------------------
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource, int cuWidth, int cuHeight, RT::Camera camera, float4* pAccumBuffer, int frameCount, RT::SceneObject* pObjects, int numObjects, RT::Material* pMaterials);
