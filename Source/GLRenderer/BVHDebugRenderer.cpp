#include "BVHDebugRenderer.h"
#include <iostream>
#include <cstring>
#include <cmath>

//---------------------------------------------------------------------------------------------------------------------
BVHDebugRenderer::BVHDebugRenderer()
    : m_vao(0)
    , m_vbo(0)
    , m_shaderProgram(0)
    , m_lineCount(0)
    , m_lineWidth(2.0f)
    , m_showInternal(true)
    , m_showLeaves(true)
    , m_initialized(false)
	, m_treeMaxDepth(0)
{
}

//---------------------------------------------------------------------------------------------------------------------
BVHDebugRenderer::~BVHDebugRenderer()
{
    Cleanup();
}

//---------------------------------------------------------------------------------------------------------------------
void BVHDebugRenderer::Initialize(RT::BVHNode* nodes, int nodeCount)
{
    if (m_initialized)
    {
        Cleanup();
    }

    std::cout << "Initializing BVH Debug Renderer...\n";

    // Find max depth
    m_treeMaxDepth = 0;
    for (int i = 0; i < nodeCount; i++)
    {
        if (nodes[i].depth > m_treeMaxDepth)
            m_treeMaxDepth = nodes[i].depth;
    }
    std::cout << "BVH Max Depth: " << m_treeMaxDepth << '\n';

    // Generate lines organized by depth
    m_linesByDepth.clear();
    m_linesByDepth.resize(m_treeMaxDepth + 1);

    int totalVertices = 0;
    for (int depth = 0; depth <= m_treeMaxDepth; depth++)
    {
        m_linesByDepth[depth] = GenerateLinesForDepth(nodes, nodeCount, depth);
        totalVertices += m_linesByDepth[depth].size();
    }

    m_lineCount = totalVertices;
    std::cout << "Generated " << m_lineCount << " vertices for BVH wireframe\n";

    // Create VAO and VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // Don't buffer data yet - will be done per-depth in Render()

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex), (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Load shaders
    m_shaderProgram = LoadShaders();

    m_initialized = true;
    std::cout << "BVH Debug Renderer initialized successfully\n";
}

//---------------------------------------------------------------------------------------------------------------------
void BVHDebugRenderer::Render(const RT::Camera& camera)
{
    if (!m_initialized || m_vao == 0 || m_lineCount == 0)
        return;

    // Build view-projection matrix
    const Matrix4x4 view = GetViewMatrix(camera);
    const Matrix4x4 projection = GetProjectionMatrix(camera.vFov, camera.Aspect_ratio, 0.1f, 100.0f);
    const Matrix4x4 viewProj = MultiplyMatrix(projection, view);

    // Use shader
    glUseProgram(m_shaderProgram);

    // Set view-projection uniform
    const GLint vpLoc = glGetUniformLocation(m_shaderProgram, "viewProjection");
    glUniformMatrix4fv(vpLoc, 1, GL_FALSE, viewProj.m);

    // Enable rendering settings
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glBindVertexArray(m_vao);

    // NEW: Render each depth with different line width
    for (int depth = 0; depth <= m_treeMaxDepth; depth++)
    {
        if (m_linesByDepth[depth].empty())
            continue;

        // Calculate line width: thick for root, thin for leaves
        const float depthRatio = static_cast<float>(depth) / static_cast<float>(m_treeMaxDepth);
        const float lineWidth = m_lineWidth * (1.0f - depthRatio * 0.8f);
        glLineWidth(std::max(1.0f, lineWidth));

        // Upload and draw this depth's lines
        glBufferData(GL_ARRAY_BUFFER,
            m_linesByDepth[depth].size() * sizeof(LineVertex),
            m_linesByDepth[depth].data(),
            GL_DYNAMIC_DRAW);

        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_linesByDepth[depth].size()));
    }

    glBindVertexArray(0);

    // Restore defaults
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
}

//---------------------------------------------------------------------------------------------------------------------
void BVHDebugRenderer::Cleanup()
{
    if (m_vao != 0)
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_vbo != 0)
    {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    if (m_shaderProgram != 0)
    {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }

    m_lineCount = 0;
    m_initialized = false;
}

//---------------------------------------------------------------------------------------------------------------------
std::vector<BVHDebugRenderer::LineVertex> BVHDebugRenderer::GenerateLinesForDepth(RT::BVHNode* nodes, int nodeCount, int targetDepth)
{
    std::vector<LineVertex> lines;

    for (int i = 0; i < nodeCount; i++)
    {
        // Only process nodes at this specific depth
        if (nodes[i].depth != targetDepth)
            continue;

        // Skip based on settings
        if (nodes[i].is_leaf && !m_showLeaves)
            continue;
        if (!nodes[i].is_leaf && !m_showInternal)
            continue;

        const RT::AABB box = nodes[i].bounds;

        // Color based on depth
        float color[3];
        GetDepthColor(targetDepth, color);

        // 8 corners of AABB
        const float corners[8][3] = {
            {box.min.x, box.min.y, box.min.z},
            {box.max.x, box.min.y, box.min.z},
            {box.max.x, box.max.y, box.min.z},
            {box.min.x, box.max.y, box.min.z},
            {box.min.x, box.min.y, box.max.z},
            {box.max.x, box.min.y, box.max.z},
            {box.max.x, box.max.y, box.max.z},
            {box.min.x, box.max.y, box.max.z}
        };

        // 12 edges of the box
        int edges[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},
            {4,5}, {5,6}, {6,7}, {7,4},
            {0,4}, {1,5}, {2,6}, {3,7}
        };

        for (auto& edge : edges)
        {
            LineVertex v1, v2;
            memcpy(v1.position, corners[edge[0]], sizeof(float) * 3);
            memcpy(v1.color, color, sizeof(float) * 3);
            memcpy(v2.position, corners[edge[1]], sizeof(float) * 3);
            memcpy(v2.color, color, sizeof(float) * 3);

            lines.push_back(v1);
            lines.push_back(v2);
        }
    }

    return lines;
}

//---------------------------------------------------------------------------------------------------------------------
GLuint BVHDebugRenderer::LoadShaders()
{
    // Vertex shader source (embedded)
    const char* vertexShaderSource = R"(
        #version 460 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        
        uniform mat4 viewProjection;
        
        out vec3 fragColor;
        
        void main()
        {
            gl_Position = viewProjection * vec4(position, 1.0);
            fragColor = color;
        }
    )";

    // Fragment shader source (embedded)
    const char* fragmentShaderSource = R"(
        #version 460 core
        in vec3 fragColor;
        out vec4 outColor;
        
        void main()
        {
            outColor = vec4(fragColor, 1.0);
        }
    )";

    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Link program
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check linking
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
    }

    // Cleanup
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

//---------------------------------------------------------------------------------------------------------------------
GLuint BVHDebugRenderer::CompileShader(GLenum type, const char* source)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check compilation
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }

    return shader;
}

//---------------------------------------------------------------------------------------------------------------------
void BVHDebugRenderer::GetDepthColor(int depth, float color[3])
{
    const float colors[][3] = {
        {1.0f, 0.0f, 0.0f},  // Red
        {1.0f, 0.5f, 0.0f},  // Orange
        {1.0f, 1.0f, 0.0f},  // Yellow
        {0.0f, 1.0f, 0.0f},  // Green
        {0.0f, 1.0f, 1.0f},  // Cyan
        {0.0f, 0.5f, 1.0f},  // Light Blue
        {0.0f, 0.0f, 1.0f},  // Blue
        {0.5f, 0.0f, 1.0f},  // Purple
        {1.0f, 0.0f, 1.0f},  // Magenta
        {1.0f, 0.0f, 0.5f},  // Pink
    };

    const int numColors = sizeof(colors) / sizeof(colors[0]);
    const int colorIndex = depth % numColors;

    color[0] = colors[colorIndex][0];
    color[1] = colors[colorIndex][1];
    color[2] = colors[colorIndex][2];
}

//---------------------------------------------------------------------------------------------------------------------
BVHDebugRenderer::Matrix4x4 BVHDebugRenderer::GetViewMatrix(const RT::Camera& camera)
{
    Matrix4x4 view;

    // Camera basis vectors - w points BACKWARD (away from view direction)
    const float3 right = camera.u;
    const float3 up = camera.v;
    const float3 back = camera.w;  
    const float3 pos = camera.Origin;

    // Build view matrix (world-to-camera transform)
    view.m[0] = right.x;  view.m[4] = right.y;  view.m[8] = right.z;  view.m[12] = -dot(right, pos);
    view.m[1] = up.x;     view.m[5] = up.y;     view.m[9] = up.z;     view.m[13] = -dot(up, pos);
    view.m[2] = back.x;   view.m[6] = back.y;   view.m[10] = back.z;   view.m[14] = -dot(back, pos);
    view.m[3] = 0;        view.m[7] = 0;        view.m[11] = 0;        view.m[15] = 1;

    return view;
}

//---------------------------------------------------------------------------------------------------------------------
BVHDebugRenderer::Matrix4x4 BVHDebugRenderer::GetProjectionMatrix(float fovY, float aspect, float nearPlane, float farPlane)
{
    Matrix4x4 proj;

    const float tanHalfFovy = tanf(fovY * 0.5f * 3.14159265f / 180.0f);

    proj.m[0] = 1.0f / (aspect * tanHalfFovy);
    proj.m[1] = 0;
    proj.m[2] = 0;
    proj.m[3] = 0;

    proj.m[4] = 0;
    proj.m[5] = 1.0f / tanHalfFovy;
    proj.m[6] = 0;
    proj.m[7] = 0;

    proj.m[8] = 0;
    proj.m[9] = 0;
    proj.m[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    proj.m[11] = -1;

    proj.m[12] = 0;
    proj.m[13] = 0;
    proj.m[14] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    proj.m[15] = 0;

    return proj;
}

//---------------------------------------------------------------------------------------------------------------------
BVHDebugRenderer::Matrix4x4 BVHDebugRenderer::MultiplyMatrix(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 result;

    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            result.m[row + col * 4] = 0;
            for (int k = 0; k < 4; k++)
            {
                result.m[row + col * 4] += a.m[row + k * 4] * b.m[k + col * 4];
            }
        }
    }

    return result;
}
