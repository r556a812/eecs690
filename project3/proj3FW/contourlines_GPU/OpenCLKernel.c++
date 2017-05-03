#include "OpenCLKernel.h"

const char* readSource(const char* filename);

OpenCLKernel::OpenCLKernel()
{
    // 1) Platforms
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = nullptr;

    // 2) Devices
    cl_uint numDevices = 0;
    cl_device_id* devices = nullptr;
}

void OpenCLKernel::checkStatus(std::string where, cl_int status, bool abortOnError)
{
        if (debug || (status != 0))
                std::cout << "Step " << where << ", status = " << status << '\n';
        if ((status != 0) && abortOnError)
                exit(1);
}

void OpenCLKernel::reportVersion(cl_platform_id platform)
{
        // Get the version of OpenCL supported on this platform
        size_t strLength;
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &strLength);
        char* version = new char[strLength+1];
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, strLength+1, version, &strLength);
        std::cout << version << '\n';
        delete [] version;
}

void OpenCLKernel::showProgramBuildLog(cl_program pgm, cl_device_id dev)
{
        size_t size;
        clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        char* log = new char[size+1];
        clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, size+1, log, nullptr);
        std::cout << "LOG:\n" << log << "\n\n";
        delete [] log;
}


// Return value is device index to use; -1 ==> no available devices
int OpenCLKernel::typicalOpenCLProlog(cl_device_type desiredDeviceType)
{
        // ----------------------------------------------------
        // Discover and initialize the platforms
        // ----------------------------------------------------

        cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
        checkStatus("clGetPlatformIDs-0", status, true);
        if (numPlatforms <= 0)
        {
                std::cout << "No platforms!\n";
                return -1;
        }

        platforms = new cl_platform_id[numPlatforms];

        status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
        checkStatus("clGetPlatformIDs-1", status, true);
        curPlatform = platforms[1];
        if (numPlatforms > 1)
        {
                size_t platformNameSize = 0;
                clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize);
                char* name = new char[platformNameSize+1];
                clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, platformNameSize+1, name, nullptr);
                std::cout << "Found " << numPlatforms << " platforms. Arbitrarily using: " << name << '\n';
                delete [] name;
        }

        reportVersion(curPlatform);

        // ---------------------vertexValues---------------------------------------------
        // Discover and initialize the devices on a specific platform
        // ------------------------------------------------------------------

        status = clGetDeviceIDs(curPlatform, desiredDeviceType, 0, nullptr, &numDevices);
        checkStatus("clGetDeviceIDs-0", status, true);
        if (numDevices <= 0)
        {
                std::cout << "No devices on platform!\n";
                return -1;
        }

        devices = new cl_device_id[numDevices];

        status = clGetDeviceIDs(curPlatform, desiredDeviceType, numDevices, devices, nullptr);
        checkStatus("clGetDeviceIDs-1", status, true);
        int devIndex = 0;
        if (numDevices > 1)
        {
                size_t nameLength;
                for (int idx=0 ; idx<numDevices ; idx++)
                {
                        clGetDeviceInfo(devices[idx], CL_DEVICE_NAME, 0, nullptr, &nameLength);
                        char* name = new char[nameLength+1];
                        clGetDeviceInfo(devices[idx], CL_DEVICE_NAME, nameLength+1, name, nullptr);
                        // You can also query lots of other things about the device capability,
                        // for example, CL_DEVICE_EXTENSIONS to see if "cl_khr_fp64". (See also
                        // the first line of daxpy.cl.)
                        std::cout << "Device " << idx << ": " << name << '\n';
                }
                devIndex = -1;
                while ((devIndex < 0) || (devIndex >= numDevices))
                {
                        std::cout << "Which device do you want to use? ";
                        std::cin >> devIndex;
                }
        }
        else if (numDevices <= 0)
                std::cout << "No devices found\n";
        else
                std::cout << "Only one device detected\n";
        return devIndex;
}

void OpenCLKernel::doTheEdgeKernelLaunch(cl_device_id dev, int* h_edges, float* vertexValues, float level, size_t rows, size_t cols)
{
        // --------------------------------------------------
        // Create a context for the one chosen device
        // --------------------------------------------------

        cl_int status;
        cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
        checkStatus("clCreateContext", status, true);

        // ------------------------------------------------------------
        // Create a command queue for one device in the context
        // (There is one queue per device per context.)
        // ------------------------------------------------------------

        cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
        checkStatus("clCreateCommandQueue", status, true);

        // ---------------------------------------------------------
        // Create device bufferSpot associated with the context
        // ---------------------------------------------------------

        size_t datasizeF = rows * cols * sizeof(float);
        size_t datasizeI = sizeof(int);

        cl_mem bufferVertexValues = clCreateBuffer( // Input array on the device
                context, CL_MEM_READ_ONLY, datasizeF, nullptr, &status);
        checkStatus("clCreateBuffer-X", status, true);

        cl_mem bufferEdges = clCreateBuffer( // Output array on the device
                context, CL_MEM_WRITE_ONLY, datasizeI, nullptr, &status);
        checkStatus("clCreateBuffer-Z", status, true);

        // ------------------------------------------------------
        // Use the command queue to encode requests to write host
        // data to the device bufferSpot
        // ------------------------------------------------------

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferEdges, CL_FALSE, 0, datasizeI,
                h_edges, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-Z", status, true);

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferVertexValues, CL_TRUE, 0, datasizeF,
                vertexValues, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-X", status, true);

        // ----------------------------------------------------
        // Create, compile, and link the program
        // ----------------------------------------------------

        const char* programSource[] = { readSource("OpenCLKernel.cl") };

        cl_program program = clCreateProgramWithSource(context,
                1, programSource, nullptr, &status);
        checkStatus("clCreateProgramWithSource", status, true);

        status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
        checkStatus("clBuildProgram", status, false);
        if (status != 0)
                showProgramBuildLog(program, dev);

        // ---------------------------------------------------------------------
        // Create a kernel using a "__kernel" function in the ".cl" file
        // ---------------------------------------------------------------------

        cl_kernel kernel = clCreateKernel(program, "EdgeExpected", &status);

        // ----------------------------------------------------
        // Set the kernel arguments
        // ----------------------------------------------------

        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferVertexValues);
        checkStatus("clSetKernelArg-0", status, true);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferEdges);
        checkStatus("clSetKernelArg-1", status, true);
        status = clSetKernelArg(kernel, 2, sizeof(float), &level);
        checkStatus("clSetKernelArg-2", status, true);
        status = clSetKernelArg(kernel, 3, sizeof(int), &rows);
        checkStatus("clSetKernelArg-3", status, true);
        status = clSetKernelArg(kernel, 4, sizeof(int), &cols);
        checkStatus("clSetKernelArg-4", status, true);

        // ----------------------------------------------------
        // Configure the work-item structure
        // ----------------------------------------------------

        size_t globalWorkSize[] = { (rows) * (cols) };

        // ---------------------------------------------------
        // Enqueue the kernel for execution
        // ----------------------------------------------------

        status = clEnqueueNDRangeKernel(cmdQueue,
                kernel, 1, nullptr, globalWorkSize,
                nullptr, 0, nullptr, nullptr);
        checkStatus("clEnqueueNDRangeKernel", status, true);

        // ----------------------------------------------------
        // Read the output buffer back to the host
        // ----------------------------------------------------

        clEnqueueReadBuffer(cmdQueue,
                bufferEdges, CL_TRUE, 0, datasizeI,
                h_edges, 0, nullptr, nullptr);

        // ----------------------------------------------------
        // Release OpenCL resources
        // ----------------------------------------------------

        // Free OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(cmdQueue);
        clReleaseMemObject(bufferVertexValues);
        clReleaseMemObject(bufferEdges);
        clReleaseContext(context);

        // Free host resources
        delete [] platforms;
        delete [] devices;
}

void OpenCLKernel::doTheEdgePointsKernelLaunch(cl_device_id dev, int* h_edges, float* vertexValues, float level, size_t rows, size_t cols, vec2* h_pointsArray, size_t arraySize)
{
        // --------------------------------------------------
        // Create a context for the one chosen device
        // --------------------------------------------------

        cl_int status;
        cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
        checkStatus("clCreateContext", status, true);

        // ------------------------------------------------------------
        // Create a command queue for one device in the context
        // (There is one queue per device per context.)
        // ------------------------------------------------------------

        cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
        checkStatus("clCreateCommandQueue", status, true);

        // ---------------------------------------------------------
        // Create device bufferSpot associated with the context
        // ---------------------------------------------------------

        size_t datasizeF = rows * cols * sizeof(float);
        size_t datasizeI = sizeof(int);
        size_t datasizeVec = arraySize * sizeof(vec2);
        int* spot = new int[1];
        spot[0] = 0;

        cl_mem bufferVertexValues = clCreateBuffer( // Input array on the device
                context, CL_MEM_READ_ONLY, datasizeF, nullptr, &status);
        checkStatus("clCreateBuffer-X", status, true);

        cl_mem bufferPointsArray = clCreateBuffer( // Output array on the device
                context, CL_MEM_WRITE_ONLY, datasizeVec, nullptr, &status);
        checkStatus("clCreateBuffer-y", status, true);

        cl_mem bufferEdges = clCreateBuffer( // Output array on the device
                context, CL_MEM_WRITE_ONLY, datasizeI, nullptr, &status);
        checkStatus("clCreateBuffer-Z", status, true);

        cl_mem bufferSpot = clCreateBuffer( // Output array on the device
                context, CL_MEM_READ_WRITE, datasizeI, nullptr, &status);
        checkStatus("clCreateBuffer-S", status, true);

        // ------------------------------------------------------
        // Use the command queue to encode requests to write host
        // data to the device buffers
        // ------------------------------------------------------

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferVertexValues, CL_TRUE, 0, datasizeF,
                vertexValues, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-X", status, true);

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferPointsArray, CL_TRUE, 0, datasizeVec,
                h_pointsArray, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-Y", status, true);

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferEdges, CL_TRUE, 0, datasizeI,
                h_edges, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-Z", status, true);

        status = clEnqueueWriteBuffer(cmdQueue,
                bufferSpot, CL_TRUE, 0, datasizeI,
                spot, 0, nullptr, nullptr);
        checkStatus("clEnqueueWriteBuffer-S", status, true);

        // ----------------------------------------------------
        // Create, compile, and link the program
        // ----------------------------------------------------

        const char* programSource[] = { readSource("OpenCLKernel.cl") };

        cl_program program = clCreateProgramWithSource(context,
                1, programSource, nullptr, &status);
        checkStatus("clCreateProgramWithSource", status, true);

        status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
        checkStatus("clBuildProgram", status, false);
        if (status != 0)
                showProgramBuildLog(program, dev);

        // ---------------------------------------------------------------------
        // Create a kernel using a "__kernel" function in the ".cl" file
        // ---------------------------------------------------------------------

        cl_kernel kernel = clCreateKernel(program, "ActualEdges", &status);

        // ----------------------------------------------------
        // Set the kernel arguments
        // ----------------------------------------------------

        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferVertexValues);
        checkStatus("clSetKernelArg-0", status, true);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferEdges);
        checkStatus("clSetKernelArg-1", status, true);
        status = clSetKernelArg(kernel, 2, sizeof(float), &level);
        checkStatus("clSetKernelArg-2", status, true);
        status = clSetKernelArg(kernel, 3, sizeof(int), &rows);
        checkStatus("clSetKernelArg-3", status, true);
        status = clSetKernelArg(kernel, 4, sizeof(int), &cols);
        checkStatus("clSetKernelArg-4", status, true);
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferPointsArray);
        checkStatus("clSetKernelArg-5", status, true);
        status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &bufferSpot);
        checkStatus("clSetKernelArg-6", status, true);

        // ----------------------------------------------------
        // Configure the work-item structure
        // ----------------------------------------------------

        size_t globalWorkSize[] = { (rows) * (cols) };

        // ---------------------------------------------------
        // Enqueue the kernel for execution
        // ----------------------------------------------------

        status = clEnqueueNDRangeKernel(cmdQueue,
                kernel, 1, nullptr, globalWorkSize,
                nullptr, 0, nullptr, nullptr);
        checkStatus("clEnqueueNDRangeKernel", status, true);

        // ----------------------------------------------------
        // Read the output buffer back to the host
        // ----------------------------------------------------

        clEnqueueReadBuffer(cmdQueue,
                bufferEdges, CL_TRUE, 0, datasizeI,
                h_edges, 0, nullptr, nullptr);

        clEnqueueReadBuffer(cmdQueue,
                bufferPointsArray, CL_TRUE, 0, datasizeVec,
                h_pointsArray, 0, nullptr, nullptr);

        // ----------------------------------------------------
        // Release OpenCL resources
        // ----------------------------------------------------

        // Free OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(cmdQueue);
        clReleaseMemObject(bufferVertexValues);
        clReleaseMemObject(bufferEdges);
        clReleaseMemObject(bufferPointsArray);
        clReleaseMemObject(bufferSpot);
        clReleaseContext(context);

        // Free host resources
        delete [] platforms;
        delete [] devices;
        delete [] spot;
}

