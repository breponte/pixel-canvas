#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "opencl_utils.h"

#define IMAGE_WIDTH 16
#define IMAGE_HEIGHT 16

int main()
{
    cl_int err;
    OpenCL ocl;

    opencl_setup(&ocl, "kernel", "kernel.cl", NULL);

    size_t bytes = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);

    float *hostData = (float *)malloc(bytes);

    cl_mem deviceBuffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    checkErr(err, "clCreateBuffer");

    err = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &deviceBuffer);
    checkErr(err, "clSetKernelArg(0)");

    size_t globalSize[2] = { (size_t)IMAGE_WIDTH, (size_t)IMAGE_HEIGHT };
    size_t localSize[2]  = { 4, 4 };

    err = clEnqueueNDRangeKernel(ocl.queue,
                                 ocl.kernel,
                                 2,         // work_dim (2D)
                                 NULL,      // global_work_offset
                                 globalSize,// global_work_size
                                 localSize, // local_work_size (optional)
                                 0,
                                 NULL,
                                 NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    clFinish(ocl.queue);

    err = clEnqueueReadBuffer(ocl.queue, deviceBuffer, CL_TRUE, 0, bytes, hostData, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    printf("Result:\n");
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            printf("%.0f ", hostData[j * image_width + i]);
        }
        printf("\n");
    }

    free(hostData);
    clReleaseMemObject(deviceBuffer);
    opencl_cleanup(&ocl);

    return 0;
}
