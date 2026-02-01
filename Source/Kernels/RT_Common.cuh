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
__host__ __device__ inline float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
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
__host__ __device__ inline float length(const float3& v)
{
    const float len = sqrtf(dot(v, v));
    return len;
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 unit_vector(const float3& v)
{
	const float len = sqrtf(dot(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

//-----------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}



//-----------------------------------------------------------------------------------------------------------------
namespace RT
{
    //-----------
    struct Ray
    {
        float3 Origin;
        float3 Direction;

        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const float3& o, const float3& d) : Origin(o), Direction(d) {}
        __host__ __device__ float3 GetAt(float t) const { return Origin + Direction * t; }
    };

    //-----------
    struct HitRecord
    {
        float   t;
        float3  P;
        float3  Normal;
        float2  UV;
        int     MaterialID;
    };

    //-----------
    struct AABB
    {
	    float3 min;
	    float3 max;
    };

    //-----------
    struct BVHNode
    {
        AABB bounds;
        int left_or_leaf;       // left child index OR leaf primitive index
        int right_or_count;     // right child index OR split axis (-1 for leaf)
        int is_leaf;            // 1 = leaf, 0 = internal
        int depth;
    };

    //-----------
    enum ObjectType
    {
        SPHERE,
        MESH
    };

    //-----------
    struct SphereData
    {
        float3 center;
        float radius;
    };

    //-----------
    struct TriangleData
    {
        float3 V0, V1, V2;
        float2 UV0, UV1, UV2;
        float3 Normal;
    };

    //-----------
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

    //-----------
    enum MaterialType
    {
        LAMBERTIAN,
        METAL,
        PHONG,
        TRANSPARENT
    };

    //-----------
    struct Material
    {
        MaterialType type;
        float3 Albedo;
        float Fuzz;
        float IoR;
    };

    //-----------
    struct Sphere
    {
        float3 center;
        float radius;
        float3 color;
    };

    //-----------
    struct Camera
    {
        float3 Origin;
        float3 Lower_Left_Corner;
        float3 Horizontal;
        float3 Vertical;

        //--- Store basis vectors to allow easy movement updates
        float3 u, v, w;
        float vFov, Aspect_ratio;

        // Rotation angles in radians!
        float yaw, pitch;

        //--- Default constructor!
        __host__ Camera(): yaw(0.0f), pitch(0.0f) {}

        //--- Initialize Camera parameters on the Host (or Device if needed) 
        __host__ void Init(float3 lookfrom, float3 lookat, float3 vup, float _vfov, float _aspect_ratio)
        {
            Origin = lookfrom;
            vFov = _vfov;
            Aspect_ratio = _aspect_ratio;

            // Calculate initial yaw and pitch from lookfrom -> lookat direction
            const float3 direction = unit_vector(lookat - lookfrom);
            yaw = atan2f(direction.x, -direction.z);
            pitch = asinf(direction.y);

            UpdateVectors();
        }

        //--- Update camera vectors based on yaw and pitch
        __host__ void UpdateVectors()
        {
            float3 forward;
            forward.x = sinf(yaw) * cosf(pitch);
            forward.y = sinf(pitch);
            forward.z = -cosf(yaw) * cosf(pitch);
            forward = unit_vector(forward);

            // calculate basis vectors
            w = -forward;                                       // backward vector
            u = unit_vector(cross(make_float3(0, 1, 0), w));    // right vector
            v = cross(w, u);

            // Update viewport based on new orientation
            const float theta = vFov * 3.14159265358979323846f / 180.0f;
            const float h = tanf(theta / 2.0f);
            const float viewport_height = 2.0f * h;
            const float viewport_width = Aspect_ratio * viewport_height;

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

        //--- Rotate camera by delta angles (in radians)
        __host__ void Rotate(float delta_yaw, float delta_pitch)
        {
            yaw += delta_yaw;
            pitch += delta_pitch;

            // Clamp pitch to avoid gimbal lock
            constexpr float max_pitch = 1.553343f; // ~89 degrees
            if (pitch > max_pitch) pitch = max_pitch;
            if (pitch < -max_pitch) pitch = -max_pitch;

            UpdateVectors();
        }

        //--- get a Ray for UV coords
        __device__ Ray GetRay(float u, float v)
        {
            return Ray(Origin, Lower_Left_Corner + Horizontal * u + Vertical * v - Origin);
        }
    };
}

//---------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float surface_area(const RT::AABB& aabb)
{
    float dx = aabb.max.x - aabb.min.x;
    float dy = aabb.max.y - aabb.min.y;
    float dz = aabb.max.z - aabb.min.z;
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

//---------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline RT::AABB make_aabb(float minx, float miny, float minz,
                                       float maxx, float maxy, float maxz)
{
	RT::AABB box;
    box.min.x = minx; box.min.y = miny; box.min.z = minz;
    box.max.x = maxx; box.max.y = maxy; box.max.z = maxz;
    return box;
}

//---------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline RT::AABB sphere_to_aabb(const RT::SphereData& s)
{
    RT::AABB box;
    box.min = s.center - make_float3(s.radius, s.radius, s.radius);
    box.max = s.center + make_float3(s.radius, s.radius, s.radius);
    return box;
}

//---------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline RT::AABB combine_aabb(const RT::AABB& a, const RT::AABB& b)
{
    return make_aabb(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z),
        fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z));
}

//---------------------------------------------------------------------------------------------------------------------
void RunRayTracingKernel(cudaGraphicsResource_t cuda_graphics_resource,
                        int cuWidth, int cuHeight,
                        RT::Camera camera,
                        float4* pAccumBuffer,
                        int frameCount,
                        RT::SceneObject* pObjects,
                        int numObjects,
                        RT::Material* pMaterials,
                        RT::BVHNode* pNodes,
						int bvhNodeCount,
                        bool useBVH,
						bool showHeatmap);
