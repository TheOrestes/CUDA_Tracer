// CUDA_Tracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define WIN32_LEAN_AND_MEAN

#define GLEW_STATIC
#include <algorithm>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <iomanip>

#include <vector>
#include <iostream>

#include "Kernels/RT_Common.cuh"
#include "GLRenderer/BVHDebugRenderer.h"

//---------------------------------------------------------------------------------------------------------------------
GLFWwindow* window = nullptr;
float4* gAccumulationBuffer = nullptr;

// Window Settings globals
constexpr int width = 600;
constexpr int height = 450;

// Scene globals
RT::Camera gCamera;
RT::SceneObject* dSceneObject = nullptr;
RT::BVHNode* d_nodes = nullptr;
BVHDebugRenderer* gBVHRenderer = nullptr;
RT::Material* dMaterial = nullptr;
int gNumObjects = 0;
int gBVHNodeCount = 0;

// Render settings globals
bool accumulationComplete = false;
int currentSPP = 0;
constexpr int targetSPP = 50;
float accumulationStartTime = 0.0f;  
float totalRenderTime = 0.0f;
bool showHeatmap = false;
bool showBVH = false;
int bvhDebugDepth = 3;

// Input State globals
bool keys[1024] = { false };
bool rightMousePressed = false;
bool cameraRotated = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;
bool firstMouse = true;
constexpr float mouseSensitivity = 0.003f; // Adjust for faster/slower rotation

//---------------------------------------------------------------------------------------------------------------------
float GetRandom01()
{
	return (static_cast<float>(rand()) / (RAND_MAX + 1));
}

//---------------------------------------------------------------------------------------------------------------------
// Helper to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) 
	{
		std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}

//---------------------------------------------------------------------------------------------------------------------
// 1. Initialize GLFW & Create Window - Opengl context
void InitGLFW()
{
	// Initialize & Setup basic 
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a window!
	window = glfwCreateWindow(width, height, "CUDA RT", nullptr, nullptr);

	if (!window)
	{
		std::cout << "Create Window FAILED!!!\n";
		glfwTerminate();
		return;
	}

	// Window is created, now create context for the same window...
	glfwMakeContextCurrent(window);
}

//---------------------------------------------------------------------------------------------------------------------
// 2. Initialize GLEW
void InitGLEW()
{
	// Ensure glew uses all the modern techniques...
	glewExperimental = GL_TRUE;

	// Initialize GLEW
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Initialize GLEW FAILED!!!\n";
		return;
	}
}

//---------------------------------------------------------------------------------------------------------------------
// 3. Inputs
void KeyHandler(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
			keys[key] = true;
		else if (action == GLFW_RELEASE)
			keys[key] = false;
	}

	// Heatmap toggle!
	if (key == GLFW_KEY_H && action == GLFW_PRESS)
	{
		showHeatmap = !showHeatmap;
		std::cout << "Heatmap mode: " << (showHeatmap ? "ON" : "OFF") << '\n';

		// Reset accumulation when toggling to see immediate effect
		currentSPP = 0;
		accumulationComplete = false;

		constexpr size_t bufferSize = width * height * sizeof(float4);
		cudaMemset(gAccumulationBuffer, 0, bufferSize);
	}

	// BVH wireframe toggle!
	// Toggle with keyboard
	if (key == GLFW_KEY_B && action == GLFW_PRESS)
	{
		showBVH = !showBVH;
		std::cout << "BVH Wireframe: " << (showBVH? "ON" : "OFF") << '\n';
	}

	// Increase - Decrease the BVH visualization depth!
	if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)  // + key
	{
		bvhDebugDepth++;
	}

	if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)  // - key
	{
		bvhDebugDepth = std::max(0, bvhDebugDepth - 1);
	}
}


//---------------------------------------------------------------------------------------------------------------------
// 3. Mouse button callback
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if (action == GLFW_PRESS)
		{
			rightMousePressed = true;
			firstMouse = true; // Reset to avoid jump on first move
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Hide cursor
		}
		else if (action == GLFW_RELEASE)
		{
			rightMousePressed = false;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // Show cursor
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------
// 3. Mouse movement callback
void MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (!rightMousePressed) return;

	if (firstMouse)
	{
		lastMouseX = xpos;
		lastMouseY = ypos;
		firstMouse = false;
		return;
	}

	const double xoffset = xpos - lastMouseX;
	const double yoffset = lastMouseY - ypos; // Reversed: y increases downward

	lastMouseX = xpos;
	lastMouseY = ypos;

	gCamera.Rotate(static_cast<float>(xoffset) * mouseSensitivity, static_cast<float>(yoffset) * mouseSensitivity);

	cameraRotated = true;
}

//---------------------------------------------------------------------------------------------------------------------
// 4. Create GL texture & register with CUDA!
void CreateTextureCUDA(GLuint* textureID, cudaGraphicsResource_t* cudaResource)
{
	glGenTextures(1, textureID);
	glBindTexture(GL_TEXTURE_2D, *textureID);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Allocate texture memory (RGBA float format)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	// Unbind to ensure we aren't modifying it accidentally
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register the texture.
	// Flags:
	// - cudaGraphicsRegisterFlagsNone: Default
	// - cudaGraphicsRegisterFlagsReadOnly: CUDA will not write to this texture
	// - cudaGraphicsRegisterFlagsSurfaceLoadStore: Enable writing via CUDA surfaces
	const cudaError_t err = cudaGraphicsGLRegisterImage(cudaResource, *textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	checkCudaError(err, "Registering Image");

	std::cout << "Successfully registered OpenGL texture " << *textureID << " with CUDA." << '\n';
}

// -------------------------------------------------------------------------------------------------------------------- -
// 5. Process Input
bool ProcessInput(float dt)
{
	bool moved = false;
	const float cameraSpeed = 2.5f * dt; // Adjust speed as needed

	// Camera-relative movement vectors
	const float3 forward = -gCamera.w;
	const float3 right = gCamera.u;
	const float3 up = gCamera.v; // Use camera's up for true free-fly

	float3 movement = { 0.0f, 0.0f, 0.0f };

	// Movement
	if (keys[GLFW_KEY_W]) movement = movement + forward * cameraSpeed;
	if (keys[GLFW_KEY_S]) movement = movement - forward * cameraSpeed;
	if (keys[GLFW_KEY_A]) movement = movement - right * cameraSpeed;
	if (keys[GLFW_KEY_D]) movement = movement + right * cameraSpeed;

	// Vertical movement
	if (keys[GLFW_KEY_E]) movement = movement + make_float3(0, 1, 0) * cameraSpeed;
	if (keys[GLFW_KEY_Q]) movement = movement - make_float3(0, 1, 0) * cameraSpeed;

	// Apply movement
	if (dot(movement, movement) > 0.0f) 
	{
		gCamera.move(movement);
		moved = true;
	}

	// check if camera rotated
	if(cameraRotated)
	{
		moved = true;
		cameraRotated = false;
	}

	return moved;
}

//---------------------------------------------------------------------------------------------------------------------
RT::AABB ComputeOverallBounds(const std::vector<RT::SceneObject>& objects)
{
	if (objects.empty())
		return { make_float3(0,0,0), make_float3(0,0,0) };

	// Initialize with the FIRST object's bounds
	RT::AABB bounds = sphere_to_aabb(objects[0].sphere);

	for (size_t i = 1; i < objects.size(); i++)
	{
		if (objects[i].type == RT::SPHERE)
		{
			bounds = combine_aabb(bounds, sphere_to_aabb(objects[i].sphere));
		}
	}

	return bounds;
}

//---------------------------------------------------------------------------------------------------------------------
int buildBVH_simple(RT::BVHNode* nodes, std::vector<RT::SceneObject>& objects, int start, int end, int& node_count)
{
	const int node_idx = node_count++;

	// Leaf node
	if (end - start <= 1)
	{
		const RT::AABB bounds = sphere_to_aabb(objects[start].sphere);

		nodes[node_idx].bounds = bounds;
		nodes[node_idx].left_or_leaf = start;
		nodes[node_idx].right_or_count = end - start;
		nodes[node_idx].is_leaf = 1;
		return node_idx;
	}

	// Simple middle split
	const int mid = (start + end) / 2;

	// Sort by X coordinate (works for any object type)
	std::sort(objects.begin() + start, objects.begin() + end,
		[](const RT::SceneObject& a, const RT::SceneObject& b)
		{
			float3 center_a, center_b;

			if (a.type == RT::SPHERE)
				center_a = a.sphere.center;
			else // MESH/Triangle
				center_a = (a.triangle.V0 + a.triangle.V1 + a.triangle.V2) / 3.0f;

			if (b.type == RT::SPHERE)
				center_b = b.sphere.center;
			else // MESH/Triangle
				center_b = (b.triangle.V0 + b.triangle.V1 + b.triangle.V2) / 3.0f;

			return center_a.x < center_b.x;
		});

	const int left = buildBVH_simple(nodes, objects, start, mid, node_count);
	const int right = buildBVH_simple(nodes, objects, mid, end, node_count);

	// Internal node
	nodes[node_idx].bounds = combine_aabb(nodes[left].bounds, nodes[right].bounds);
	nodes[node_idx].left_or_leaf = left;
	nodes[node_idx].right_or_count = right;
	nodes[node_idx].is_leaf = 0;

	return node_idx;
}

//---------------------------------------------------------------------------------------------------------------------
void InitSimpleScene(std::vector<RT::SceneObject>& objects, std::vector<RT::Material>& mats)
{
	// Initialize Scene!
	gCamera.Init(make_float3(0, 0, 2), make_float3(0, 0, -1), make_float3(0, 1, 0), 45.0f, static_cast<float>(width) / static_cast<float>(height));

	// Center Transparent sphere
	RT::SceneObject centerSphere;
	centerSphere.type = RT::SPHERE;
	centerSphere.MaterialID = 0;
	centerSphere.sphere.center = make_float3(0, 0, 0.35f);
	centerSphere.sphere.radius = 0.5f;

	objects.push_back(centerSphere);

	// Ground Sphere
	RT::SceneObject groundSphere;
	groundSphere.type = RT::SPHERE;
	groundSphere.MaterialID = 1;
	groundSphere.sphere.center = make_float3(0.0f, -100.5f, -1.0f);
	groundSphere.sphere.radius = 100.0f;

	objects.push_back(groundSphere);

	// Left Sphere
	RT::SceneObject leftSphere;
	leftSphere.type = RT::SPHERE;
	leftSphere.MaterialID = 2;
	leftSphere.sphere.center = make_float3(-1.2f, 0, -1.0f);
	leftSphere.sphere.radius = 0.5f;

	objects.push_back(leftSphere);

	// Right Sphere
	RT::SceneObject rightSphere;
	rightSphere.type = RT::SPHERE;
	rightSphere.MaterialID = 3;
	rightSphere.sphere.center = make_float3(1.2f, 0, -1.0f);
	rightSphere.sphere.radius = 0.5f;

	objects.push_back(rightSphere);

	gNumObjects = static_cast<int>(objects.size());

	// Materials
	mats.push_back({ RT::TRANSPARENT, {0.8f, 0.8f, 0.0f}, 0.0f, 1.5f });	// small sphere
	mats.push_back({ RT::LAMBERTIAN,{0.8f, 0.8f, 0.8f}, 0.0f, 0.0f });	// ground sphere
	mats.push_back({ RT::METAL,{0.2f, 0.2f, 0.7f}, 0.0f, 0.0f });		// left shiny sphere
	mats.push_back({ RT::METAL,{0.7f, 0.2f, 0.2f}, 0.3f, 0.0f });		// right fuzzy sphere
}

//---------------------------------------------------------------------------------------------------------------------
void InitRandomScene(std::vector<RT::SceneObject>& objects, std::vector<RT::Material>& mats)
{
	// Initialize Scene!
	gCamera.Init(make_float3(5.0f, 2.5f, 5.0f), make_float3(0, 0, 0), make_float3(0, 1, 0), 45.0f, static_cast<float>(width) / static_cast<float>(height));

	// Ground sphere
	RT::SceneObject Sphere0;
	Sphere0.type = RT::SPHERE;
	Sphere0.MaterialID = 0;
	Sphere0.sphere.center = make_float3(0, -1000.0f, 0.0f);
	Sphere0.sphere.radius = 1000.0f;

	objects.push_back(Sphere0);
	mats.push_back({ RT::LAMBERTIAN,{0.5f, 0.5f, 0.5f}, 0.0f, 0.0f });	// ground sphere

	int i = 1; int objDims = 3;
	for (int a = -objDims; a < objDims; a++)
	{
		for(int b = -objDims; b < objDims; b++)
		{
			float randomMat = GetRandom01();

			float3 center = make_float3(a + 0.9f * GetRandom01(), 0.2f, b + 0.9f * GetRandom01());
			if(length(center - make_float3(4.0f, 0.2f, 0.0f)) > 0.9f)
			{
				if(randomMat < 0.8f)
				{
					// Lambertian
					RT::SceneObject temp;
					temp.type = RT::SPHERE;
					temp.MaterialID = i;
					temp.sphere.center = center;
					temp.sphere.radius = 0.2f;

					objects.push_back(temp);

					float3 lambertColor = make_float3(GetRandom01() * GetRandom01(), GetRandom01() * GetRandom01(), GetRandom01() * GetRandom01());
					mats.push_back({ RT::LAMBERTIAN, lambertColor, 0.0f, 0.0f });
				}
				else if(randomMat < 0.95f)
				{
					// Metal
					RT::SceneObject temp;
					temp.type = RT::SPHERE;
					temp.MaterialID = i;
					temp.sphere.center = center;
					temp.sphere.radius = 0.2f;

					objects.push_back(temp);

					float3 metalColor = make_float3(0.5f * (1 + GetRandom01()), 0.5f * (1 + GetRandom01()), 0.5f * (1 + GetRandom01()));
					mats.push_back({ RT::METAL, metalColor, GetRandom01(), 0.0f });
				}
				else
				{
					// Transparent
					RT::SceneObject temp;
					temp.type = RT::SPHERE;
					temp.MaterialID = i;
					temp.sphere.center = center;
					temp.sphere.radius = 0.2f;

					objects.push_back(temp);

					float3 glassColor = make_float3(GetRandom01(), GetRandom01(), GetRandom01());
					mats.push_back({ RT::TRANSPARENT, glassColor, 0.0f, 1.5f });
				}
			}

			i++;
		}
	}

	RT::SceneObject Sphere1;
	Sphere1.type = RT::SPHERE;
	Sphere1.MaterialID = i++;
	Sphere1.sphere.center = make_float3(0, 1.0f, 0.0f);
	Sphere1.sphere.radius = 1.0f;

	objects.push_back(Sphere1);

	float3 sphere1Color = make_float3(GetRandom01(), GetRandom01(), GetRandom01());
	mats.push_back({ RT::TRANSPARENT, sphere1Color, 0.0f, 1.5f });

	RT::SceneObject Sphere2;
	Sphere2.type = RT::SPHERE;
	Sphere2.MaterialID = i++;
	Sphere2.sphere.center = make_float3(-4.0f, 1.0f, 0.0f);
	Sphere2.sphere.radius = 1.0f;

	objects.push_back(Sphere2);

	float3 sphere2Color = make_float3(0.4f, 0.2f, 0.1f);
	mats.push_back({ RT::LAMBERTIAN, sphere2Color, 0.1f, 0.0f });

	RT::SceneObject Sphere3;
	Sphere3.type = RT::SPHERE;
	Sphere3.MaterialID = i++;
	Sphere3.sphere.center = make_float3(4.0f, 1.0f, 0.0f);
	Sphere3.sphere.radius = 1.0f;

	objects.push_back(Sphere3);

	float3 sphere3Color = make_float3(0.7f, 0.6f, 0.5f);
	mats.push_back({ RT::METAL, sphere3Color, 0.0f, 0.0f});

	// !!! IMPORTANT !!!
	gNumObjects = static_cast<int>(objects.size());
}

//---------------------------------------------------------------------------------------------------------------------
void SetupScene()
{
	std::vector<RT::SceneObject> sceneObjects;
	std::vector<RT::Material> materials;

	//InitSimpleScene(sceneObjects, materials);
	InitRandomScene(sceneObjects, materials);

	// Optional: Check scene bounds
	const RT::AABB sceneBounds = ComputeOverallBounds(sceneObjects);
	printf("Scene bounds: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)\n",
		sceneBounds.min.x, sceneBounds.min.y, sceneBounds.min.z,
		sceneBounds.max.x, sceneBounds.max.y, sceneBounds.max.z);

	// Build BVH on host
	const int MAX_NODES = gNumObjects * 2;
	RT::BVHNode* h_nodes = new RT::BVHNode[MAX_NODES];

	int root = buildBVH_simple(h_nodes, sceneObjects, 0, gNumObjects, gBVHNodeCount);
	printf("BVH built: %d nodes for %d objects\n", gBVHNodeCount, gNumObjects);

	// Allocate device memory
	cudaMalloc(&dSceneObject, sceneObjects.size() * sizeof(RT::SceneObject));
	cudaMalloc(&d_nodes, gBVHNodeCount * sizeof(RT::BVHNode));
	cudaMalloc(&dMaterial, materials.size() * sizeof(RT::Material));

	cudaMemcpy(dSceneObject, sceneObjects.data(), sceneObjects.size() * sizeof(RT::SceneObject), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, h_nodes, gBVHNodeCount * sizeof(RT::BVHNode), cudaMemcpyHostToDevice);
	cudaMemcpy(dMaterial, materials.data(), materials.size() * sizeof(RT::Material), cudaMemcpyHostToDevice);

	// Initialize BVH Debug renderer
	gBVHRenderer = new BVHDebugRenderer();
	gBVHRenderer->Initialize(h_nodes, gBVHNodeCount);
	gBVHRenderer->SetLineWidth(2.5f);

	delete[] h_nodes;
}


//---------------------------------------------------------------------------------------------------------------------
int main()
{
	InitGLFW();
	InitGLEW();

	glfwSetKeyCallback(window, KeyHandler);							// Keyboard callback
	glfwSetMouseButtonCallback(window, MouseButtonCallback);		// Mouse button callback
	glfwSetCursorPosCallback(window, MouseCallback);				// Mouse movement callback

	GLuint fbTexture;
	cudaGraphicsResource_t fbCudaResource;

	// Call the function
	CreateTextureCUDA(&fbTexture, &fbCudaResource);

	// Create Framebuffer object & attach CUDA-written texture to it!
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);

	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTexture, 0);

	// Allocate Accumulation buffer on GPU!
	constexpr size_t bufferSize = width * height * sizeof(float4);
	cudaMalloc(&gAccumulationBuffer, bufferSize);
	cudaMemset(gAccumulationBuffer, 0, bufferSize);

	// Create scene and compute AABB!
	SetupScene();

	float deltaTime = 0.0f;
	float lastFrameTime = 0.0f;

	// Message Loop!
	while (!glfwWindowShouldClose(window))
	{
		const float currentTime = static_cast<float>(glfwGetTime());
		deltaTime = currentTime - lastFrameTime;
		lastFrameTime = currentTime;

		glfwPollEvents();
		const bool cameraMoved = ProcessInput(deltaTime);

		// If camera has moved, then RESET the accumulation! 
		if (cameraMoved)
		{
			currentSPP = 0;
			accumulationComplete = false;						
			accumulationStartTime = currentTime;				// reset timer!
			cudaMemset(gAccumulationBuffer, 0, bufferSize);
		}

		if(!accumulationComplete)
		{
			// Start timer on first sample
			if(currentSPP == 0)
			{
				accumulationStartTime = currentTime;
			}

			++currentSPP;

			// Run CUDA kernel!
			RunRayTracingKernel(fbCudaResource, width, height, gCamera, gAccumulationBuffer, currentSPP, dSceneObject, gNumObjects, dMaterial, d_nodes, gBVHNodeCount, true, showHeatmap);

			// Print progress every 1/5th step...
			if (currentSPP % (targetSPP / 5) == 0)
			{
				totalRenderTime = currentTime - accumulationStartTime;

				const float progress = static_cast<float>(currentSPP) / targetSPP * 100.0f;
				const float sppPerSecond = static_cast<float>(currentSPP) / totalRenderTime;

				std::cout << "Progress: " << currentSPP << "/" << targetSPP
					<< " SPP (" << std::fixed << std::setprecision(1)
					<< progress << "%) - "
					<< totalRenderTime << "s - "
					<< std::setprecision(2) << sppPerSecond << " SPP/s"
					<< '\n';
			}

			// Check if complete!
			if (currentSPP >= targetSPP)
			{
				accumulationComplete = true;

				// Format time nicely
				const int minutes = static_cast<int>(totalRenderTime / 60.0f);
				const float seconds = totalRenderTime - (minutes * 60.0f);

				std::cout << "\n=== Accumulation Complete ===" << '\n';
				std::cout << "Total SPP: " << currentSPP << '\n';
				std::cout << "Total Time: ";

				if (minutes > 0)
					std::cout << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
				else
					std::cout << std::fixed << std::setprecision(2) << totalRenderTime << "s";

				std::cout << '\n';
				std::cout << "Average: " << std::fixed << std::setprecision(2) << (currentSPP / totalRenderTime) << " SPP/s" << '\n';
				std::cout << "============================\n" << '\n';
			}
		}
		

		// Bind the FBO as the "Read" source
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		// Bind the default screen (0) as the "Draw" destination
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

		// Render BVH wireframe!
		if(showBVH && gBVHRenderer)
		{
			gBVHRenderer->Render(gCamera);
		}

		glfwSwapBuffers(window);
	}

	// 4. Cleanup
	if(gBVHRenderer)
	{
		delete gBVHRenderer;
		gBVHRenderer = nullptr;
	}

	// Always unregister before destroying the GL texture
	cudaGraphicsUnregisterResource(fbCudaResource);
	cudaFree(gAccumulationBuffer);

	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &fbTexture);
	glfwTerminate();
    
	return 0;
}
