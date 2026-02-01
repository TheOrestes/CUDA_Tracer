#pragma once

#include "GL/glew.h"
#include "Kernels/RT_Common.cuh"
#include <vector>

class BVHDebugRenderer
{
public:
    BVHDebugRenderer();
    ~BVHDebugRenderer();

    // Initialize with BVH data
    void Initialize(RT::BVHNode* nodes, int nodeCount);

    // Render the wireframe
    void Render(const RT::Camera& camera);

    // Cleanup
    void Cleanup();

    // Settings
    void SetShowInternalNodes(bool show) { m_showInternal = show; }
    void SetShowLeafNodes(bool show) { m_showLeaves = show; }
    void EnableDepthColorMode(bool byDepth) { m_colorByDepth = byDepth; }
    void SetMaxDepth(int depth) { m_maxDepth = depth; }
    void SetBaseLineWidth(float width) { m_lineWidth = width; }

    // Helper to get color by depth
    void GetDepthColor(int depth, float color[3]);

private:
    struct LineVertex
    {
        float position[3];
        float color[3];
    };

    struct Matrix4x4
    {
        float m[16];
    };

    // Store lines organized by depth
	std::vector<std::vector<LineVertex>> m_linesByDepth;

    // Generate line geometry from BVH
    std::vector<LineVertex> GenerateLinesForDepth(RT::BVHNode* nodes, int nodeCount, int depth);

    // Shader helpers
    GLuint LoadShaders();
    GLuint CompileShader(GLenum type, const char* source);

    // Matrix helpers
    Matrix4x4 GetViewMatrix(const RT::Camera& camera);
    Matrix4x4 GetProjectionMatrix(float fovY, float aspect, float nearPlane, float farPlane);
    Matrix4x4 MultiplyMatrix(const Matrix4x4& a, const Matrix4x4& b);

    // OpenGL objects
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_shaderProgram;
    int m_lineCount;

    // Settings
    float m_lineWidth;
    bool m_showInternal;
    bool m_showLeaves;
    bool m_initialized;
    bool m_colorByDepth;
    int m_maxDepth;
    int m_treeMaxDepth;         // actual max depth in the tree!
};


