 // OpenCLKernel.h - Code to create and produce a kernel

#ifndef OPENCLKERNEL_H
#define OPENCLKERNEL_H

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <cstdlib>

typedef float vec2[2];

class OpenCLKernel
{
    
public:
    OpenCLKernel();
    void checkStatus(std::string where, cl_int status, bool abortOnError);
    void reportVersion(cl_platform_id platform);
    void showProgramBuildLog(cl_program pgm, cl_device_id dev);
    int typicalOpenCLProlog(cl_device_type desiredDeviceType);
    void doTheEdgeKernelLaunch(cl_device_id dev, int* h_edges, float* vertexValues, float level, size_t rows, size_t cols);
    void doTheEdgePointsKernelLaunch(cl_device_id dev, int* h_edges, float* vertexValues, float level, size_t rows, size_t cols, vec2* pointsArray, size_t arraySize);
    
    cl_uint numPlatforms;
    cl_platform_id* platforms;
    cl_platform_id curPlatform;
    cl_uint numDevices;
    cl_device_id* devices;
    bool debug = false;
    
};

#endif
