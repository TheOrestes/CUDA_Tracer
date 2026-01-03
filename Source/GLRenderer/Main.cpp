// CUDA_Tracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define WIN32_LEAN_AND_MEAN

#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <iomanip>

#include <vector>
#include <iostream>

#include "Kernels/RT_Common.cuh"

//---------------------------------------------------------------------------------------------------------------------
GLFWwindow* window = nullptr;
float4* gAccumulationBuffer = nullptr;

// Window Settings globals
constexpr int width = 600;
constexpr int height = 450;

// Scene globals
RT::Camera gCamera;
RT::SceneObject* dSceneObject = nullptr;
RT::Material* dMaterial = nullptr;
int gNumObjects = 0;

// Render settings globals
bool accumulationComplete = false;
int currentSPP = 0;
constexpr int targetSPP = 100;

// Input State globals
bool keys[1024] = { false };
bool rightMousePressed = false;
bool cameraRotated = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;
bool firstMouse = true;
constexpr float mouseSensitivity = 0.003f; // Adjust for faster/slower rotation

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

	std::cout << "Successfully registered OpenGL texture " << *textureID << " with CUDA." << std::endl;
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
void SetupScene()
{
	std::vector<RT::SceneObject> sceneObjects;

	// Center Transparent sphere
	RT::SceneObject centerSphere;
	centerSphere.type = RT::SPHERE;
	centerSphere.MaterialID = 0;
	centerSphere.sphere.center = make_float3(0, 0, 0.35f);
	centerSphere.sphere.radius = 0.5f;

	sceneObjects.push_back(centerSphere);

	// Ground Sphere
	RT::SceneObject groundSphere;
	groundSphere.type = RT::SPHERE;
	groundSphere.MaterialID = 1;
	groundSphere.sphere.center = make_float3(0.0f, -100.5f, -1.0f);
	groundSphere.sphere.radius = 100.0f;

	sceneObjects.push_back(groundSphere);

	// Left Sphere
	RT::SceneObject leftSphere;
	leftSphere.type = RT::SPHERE;
	leftSphere.MaterialID = 2;
	leftSphere.sphere.center = make_float3(-1.2f, 0, -1.0f);
	leftSphere.sphere.radius = 0.5f;

	sceneObjects.push_back(leftSphere);

	// Right Sphere
	RT::SceneObject rightSphere;
	rightSphere.type = RT::SPHERE;
	rightSphere.MaterialID = 3;
	rightSphere.sphere.center = make_float3(1.2f, 0, -1.0f);
	rightSphere.sphere.radius = 0.5f;

	sceneObjects.push_back(rightSphere);

	gNumObjects = static_cast<int>(sceneObjects.size());

	// Materials
	std::vector<RT::Material> mats;
	mats.push_back({RT::TRANSPARENT, {0.8f, 0.8f, 0.0f}, 0.0f, 1.5f });	// small sphere
	mats.push_back({RT::LAMBERTIAN,{0.8f, 0.8f, 0.8f}, 0.0f, 0.0f });	// ground sphere
	mats.push_back({ RT::METAL,{0.2f, 0.2f, 0.7f}, 0.0f, 0.0f });		// left shiny sphere
	mats.push_back({ RT::METAL,{0.7f, 0.2f, 0.2f}, 0.3f, 0.0f });		// right fuzzy sphere

	cudaMalloc(&dSceneObject, sceneObjects.size() * sizeof(RT::SceneObject));
	cudaMemcpy(dSceneObject, sceneObjects.data(), sceneObjects.size() * sizeof(RT::SceneObject), cudaMemcpyHostToDevice);

	cudaMalloc(&dMaterial, mats.size() * sizeof(RT::Material));
	cudaMemcpy(dMaterial, mats.data(), mats.size() * sizeof(RT::Material), cudaMemcpyHostToDevice);
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

	// Initialize Scene!
	gCamera.Init(make_float3(0, 0, 2), make_float3(0, 0, -1), make_float3(0, 1, 0), 45.0f, static_cast<float>(width) / static_cast<float>(height));

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
			cudaMemset(gAccumulationBuffer, 0, bufferSize);
		}

		if(!accumulationComplete)
		{
			++currentSPP;

			// Run CUDA kernel!
			RunRayTracingKernel(fbCudaResource, width, height, gCamera, gAccumulationBuffer, currentSPP, dSceneObject, gNumObjects, dMaterial);

			// Print progress every 1/10th step...
			if (currentSPP % (targetSPP / 10) == 0)
			{
				const float progress = static_cast<float>(currentSPP) / targetSPP * 100.0f;
				std::cout << "Progress: " << currentSPP << "/" << targetSPP << " SPP (" << std::fixed << std::setprecision(1) << progress << "%)\r";
			}

			if (currentSPP >= targetSPP)
			{
				accumulationComplete = true;
				std::cout << "Accumulation complete! (" << currentSPP << " SPP)\n";
			}
		}


		// Bind the FBO as the "Read" source
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		// Bind the default screen (0) as the "Draw" destination
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

		glBlitFramebuffer(
			0, 0, width, height,
			0, 0, width, height,
			GL_COLOR_BUFFER_BIT,
			GL_NEAREST
		);

		glfwSwapBuffers(window);
	}

	// 4. Cleanup
	// Always unregister before destroying the GL texture
	cudaGraphicsUnregisterResource(fbCudaResource);
	cudaFree(gAccumulationBuffer);

	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &fbTexture);
	glfwTerminate();
    
	return 0;
}
