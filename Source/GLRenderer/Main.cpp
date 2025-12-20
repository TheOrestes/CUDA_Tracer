// CUDA_Tracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Kernels/Kernels.h"

//---------------------------------------------------------------------------------------------------------------------
GLFWwindow* window = nullptr;

constexpr int width = 960;
constexpr int height = 540;

//---------------------------------------------------------------------------------------------------------------------
// Helper to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
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
	//if (key == GLFW_KEY_R && action == GLFW_PRESS)
	//{
	//}
	//
	//if (key == GLFW_KEY_G && action == GLFW_PRESS)
	//{
	//}
	//
	//if (key == GLFW_KEY_B && action == GLFW_PRESS)
	//{
	//}
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

//---------------------------------------------------------------------------------------------------------------------
int main()
{
	InitGLFW();
	InitGLEW();

	glfwSetKeyCallback(window, KeyHandler);

	GLuint fbTexture;
	cudaGraphicsResource_t fbCudaResource;

	// Call the function
	CreateTextureCUDA(&fbTexture, &fbCudaResource);

	// Create Framebuffer object & attach CUDA-written texture to it!
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);

	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTexture, 0);

	// Message Loop!
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		const float time = static_cast<float>(glfwGetTime());

		// Run CUDA kernel!
		RunRayTracingKernel(fbCudaResource, width, height, time);

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

	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &fbTexture);
	glfwTerminate();
    
	return 0;
}
