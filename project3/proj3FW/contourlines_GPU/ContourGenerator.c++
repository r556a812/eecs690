// ContourGenerator.c++ - Code to read a scalar data field and produce
// contours at requested levels.

#include "ContourGenerator.h"
#include "OpenCLKernel.h"

ContourGenerator::ContourGenerator(std::istream& inp) :
	vertexValues(nullptr)
{
	inp >> nRowsOfVertices >> nColsOfVertices;
	std::string scalarDataFieldFileName;
	inp >> scalarDataFieldFileName;
	std::ifstream scalarFieldFile(scalarDataFieldFileName.c_str());
	if (scalarFieldFile.good())
	{
		readData(scalarFieldFile);
		scalarFieldFile.close();
	}
	else
	{
		std::cerr << "Could not open " << scalarDataFieldFileName
		          << " for reading.\n";
		nRowsOfVertices = nColsOfVertices = 0;
	}
}

ContourGenerator::~ContourGenerator()
{
	if (vertexValues != nullptr)
		delete [] vertexValues;
	// Delete any GPU structures (buffers) associated with this model as well!
}

int ContourGenerator::computeContourEdgesFor(float level, vec2*& lines)
{
        //Declare a kernel and its variables to use
        OpenCLKernel edgeKernel;
        int* edgeArray = new int[1]; //Store the approximated amount of edges in edgeArray[0]
        edgeArray[0] = 0;  //Initialize edgeArray[0] to 0

        //Discover and initialize platforms and devices for the kernel
        cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
        int devIndex = edgeKernel.typicalOpenCLProlog(devType);

        // Fire a kernel to determine expected number of edges at the given "level'
        if (devIndex >= 0)
        {
            edgeKernel.doTheEdgeKernelLaunch(edgeKernel.devices[devIndex], edgeArray, vertexValues, level, nRowsOfVertices, nColsOfVertices);
        }

        //Set the estimated amount of edges in edgeArray[0] to numExpectedEdges
        int numExpectedEdges = edgeArray[0];

	// Create space for the line end points on the device
	int numExpectedPoints = 2 * numExpectedEdges; // each edge is: (x,y), (x,y)
        vec2* pointsArray = new vec2[numExpectedPoints];

        //Declare a kernel and its variables to use
        OpenCLKernel actualEdgeKernel;
        edgeArray[0] = 0;   //Initialize edgeArray back to 0 to use again to find exact amount of edges

        //Discover and initialize platforms and devices for the kernel
        devType = CL_DEVICE_TYPE_DEFAULT;
        devIndex = actualEdgeKernel.typicalOpenCLProlog(devType);

        // Fire a kernel to compute the edge end points (determimes "numActualEdges") and fill pointsArray
        if (devIndex >= 0)
        {
            actualEdgeKernel.doTheEdgePointsKernelLaunch(actualEdgeKernel.devices[devIndex], edgeArray, vertexValues, level, nRowsOfVertices, nColsOfVertices, pointsArray, numExpectedPoints);
        }

        //Set numActualEdges found from kernel and set numActualPoints
        int numActualEdges = edgeArray[0];
        int numActualPoints = 2 * numActualEdges; // each edge is: (x,y), (x,y)

	// Get the point coords back, storing them into "lines"
        lines = new vec2[numActualPoints];
        lines = pointsArray;

        //Delete edgeArray now that done
        delete[] edgeArray;

	// return number of coordinate pairs in "lines":
	return numActualPoints;
}

void ContourGenerator::readData(std::ifstream& scalarFieldFile)
{
	vertexValues = new float[nRowsOfVertices * nColsOfVertices];
	int numRead = 0, numMissing = 0;
	float val;
	float minVal = 1.0, maxVal = -1.0;
	scalarFieldFile.read(reinterpret_cast<char*>(&val),sizeof(float));
	while (!scalarFieldFile.eof())
	{
		vertexValues[numRead++] = val;
		if (val == -9999)
			numMissing++;
		else if (minVal > maxVal)
			minVal = maxVal = val;
		else
		{
			if (val < minVal)
				minVal = val;
			else if (val > maxVal)
				maxVal = val;
		}
		scalarFieldFile.read(reinterpret_cast<char*>(&val),sizeof(float));
	}
	std::cout << "read " << numRead << " values; numMissing: " << numMissing
	          << "; range of values: " << minVal << " <= val <= " << maxVal << '\n';
}
