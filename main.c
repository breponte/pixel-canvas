#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

void OpenCLPixelCanvas(Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_out;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    # if __APPLE__
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    #else
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    #endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    char buildLog[4096]; 
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL); 
    printf("Build Log:\n%s\n", buildLog);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    CHECK_ERR(err, "clCreateKernel");

    // Allocate GPU memory here
    device = clCreateBuffer(context,
                            CL_MEM_WRITE_ONLY,
                            result->shape[0] * result->shape[1] * sizeof(float),
                            NULL,
                            &err);
    CHECK_ERR(err, "clCreateBuffer result");

    //@@ define local and global work sizes
    size_t image_dimensions[2] = {result->shape[0], result->shape[1]};
    size_t global_work_size[2] = {image_dimensions[1], image_dimensions[0]};
    size_t local_work_size[2] = {image_dimensions[1] / 4, image_dimensions[0] / 4};

    // Set the arguments to our compute kernel
    // __global int *C,
    // const unsigned int numCRows, const unsigned int numCColumns
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &result->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &result->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 2");

    // Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, 
                                    kernel, 
                                    2,
                                    NULL, 
                                    global_work_size, 
                                    local_work_size,
                                    0, 
                                    NULL, 
                                    NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    // Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue,
                                device,
                                CL_TRUE,
                                0,
                                result->shape[0] * result->shape[1] * sizeof(float),
                                result->data,
                                0,
                                NULL,
                                NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer result");

    // Free the GPU memory here
    err = clReleaseMemObject(device);
    CHECK_ERR(err, "clReleaseMemObject result");

    // Free OpenCL Objects
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file = argv[1];

    // Host input and output vectors and sizes
    Matrix host;
    
    cl_int err;

    // Allocate the memory for the target.
    host.shape[0] = 16;
    host.shape[1] = 16;
    host.data = (int *)malloc(sizeof(int) * host.shape[0] * host.shape[1]);

    // Call your matrix multiply.
    OpenCLPixelCanvas(&host);

    // // Call to print the matrix
    // PrintMatrix(&host);

    // Save the matrix
    SaveMatrix(input_file, &host);

    // Release host memory
    free(host.data);

    return 0;
}
