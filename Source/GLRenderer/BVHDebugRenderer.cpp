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

    // Generate line geometry
    const std::vector<LineVertex> lines = GenerateLines(nodes, nodeCount);
    m_lineCount = static_cast<int>(lines.size());

    std::cout << "Generated " << m_lineCount << " vertices for BVH wireframe\n";

    // Create VAO and VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(LineVertex),
        lines.data(), GL_STATIC_DRAW);

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
    glLineWidth(m_lineWidth);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Draw lines
    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINES, 0, m_lineCount);
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
std::vector<BVHDebugRenderer::LineVertex> BVHDebugRenderer::GenerateLines(RT::BVHNode* nodes, int nodeCount)
{
    std::vector<LineVertex> lines;

    for (int i = 0; i < nodeCount; i++)
    {
        // Skip based on settings
        if (nodes[i].is_leaf && !m_showLeaves)
            continue;
        if (!nodes[i].is_leaf && !m_showInternal)
            continue;

        const RT::AABB box = nodes[i].bounds;

        // Color based on leaf/internal
        float color[3];
        if (nodes[i].is_leaf)
        {
            const int leaf_id = nodes[i].left_or_leaf;
            if (leaf_id % 6 == 0) { color[0] = 1; color[1] = 0; color[2] = 0; }  // Red
            else if (leaf_id % 6 == 1) { color[0] = 0; color[1] = 1; color[2] = 0; }  // Green
            else if (leaf_id % 6 == 2) { color[0] = 0; color[1] = 0; color[2] = 1; }  // Blue
            else if (leaf_id % 6 == 3) { color[0] = 1; color[1] = 1; color[2] = 0; }  // Yellow
            else if (leaf_id % 6 == 4) { color[0] = 1; color[1] = 0; color[2] = 1; }  // Magenta
            else { color[0] = 0; color[1] = 1; color[2] = 1; }  // Cyan
        }
        else
        {
            color[0] = 0.7f; color[1] = 0.7f; color[2] = 0.7f;  // Gray for internal
        }

        // 8 corners of AABB
        const float corners[8][3] = {
            {box.min.x, box.min.y, box.min.z},  // 0
            {box.max.x, box.min.y, box.min.z},  // 1
            {box.max.x, box.max.y, box.min.z},  // 2
            {box.min.x, box.max.y, box.min.z},  // 3
            {box.min.x, box.min.y, box.max.z},  // 4
            {box.max.x, box.min.y, box.max.z},  // 5
            {box.max.x, box.max.y, box.max.z},  // 6
            {box.min.x, box.max.y, box.max.z}   // 7
        };

        // 12 edges of the box
        int edges[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},  // Bottom face
            {4,5}, {5,6}, {6,7}, {7,4},  // Top face
            {0,4}, {1,5}, {2,6}, {3,7}   // Vertical edges
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
