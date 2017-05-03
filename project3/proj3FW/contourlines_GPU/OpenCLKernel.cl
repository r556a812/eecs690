#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef float vec2[2];

//Function to compute the edge point from the given vertices
float computeEdges(float vertex1, float vertex2, float level, int rowOrColumn)
{
    float f = 0;
    float c = 0;
   
    f = (level - vertex1)/(vertex2 - vertex1);
    c = ((1-f)*rowOrColumn) + (f*(rowOrColumn+1));
    if (get_global_id(0) == 0)
    {
    printf("point1:  %f  point2: %f  F: %f  C:  %f  level:  %f  rc:  %i\n", vertex1, vertex2, f, c, level, rowOrColumn);
    }
    
    return c;
    
}

//Function to calculate the amount for over/under and the over/underArrays
void calculateOverUnder(float A, float B, float C, float D, float level, int* overunderArray, float* overArray, float* underArray)
{
    int over = 0;
    int under = 0;
    
    if (A > level)
    {
        overArray[over] = A;
        over++;
    }
    else
    {
        underArray[under] = A;
        under++;
    }

    if (B > level)
    {
        overArray[over] = B;
        over++;
    }
    else
    {
        underArray[under] = B;
        under++;
    }

    if (C > level)
    {
        overArray[over] = C;
        over++;
    }
    else
    {
        underArray[under] = C;
        under++;
    }

    if (D > level)
    {
        overArray[over] = D;
        over++;
    }
    else
    {
        underArray[under] = D;
        under++;
    }
    
    overunderArray[0] = over;
    overunderArray[1] = under;
}


//This kernel is responsible for getting an estimate of the number of edges.
//Keeps track of how many vertices in its box is over/under and determines if there 
//can be a possible edge(s) within that box.  Uses the function atom_inc to make sure 
//that the incrementing of the value in edgeArray is done correctly.
__kernel
void EdgeExpected(__global float* vertexValues, __global int* edgeArray, float level, int rows, int cols)
{
    //Get the thread id and its corresponding row and column
    int place = get_global_id(0);
    int under = 0;
    int over = 0;
    
    //Only enters if the thread is dealing in a "box" of vertices.
    if ((place%cols) != (cols-1) && place < (cols*rows)-cols)
    {
        //Vertex values that the thread will be working with
        float A = vertexValues[place];
        float B = vertexValues[place+1];
        float C = vertexValues[place+cols];
        float D = vertexValues[place+1+cols];
        
        //Array to determine count for over/under
        int overunderArray[2];
        int* ptr1 = overunderArray;
        
        //Dont use these arrays, just used to fill in for the function call 
        float overArray[4];
        float underArray[4];
        float* ptr2 = overArray;
        float* ptr3 = underArray;
        
        //Call function to get over/under and fill arrays (Dont actually use the arrays over/underArray)
        calculateOverUnder(A, B, C, D, level, ptr1, ptr2, ptr3);
        over = overunderArray[0];
        under = overunderArray[1];
        
        //If-else statements used for estimate on edge count
        if ((over == 3 || over == 2) && under == 1)
        {
            atom_inc(&edgeArray[0]);
        }
        else if ((under == 3 || under == 2) && over == 1)
        {
            atom_inc(&edgeArray[0]);
        }
        else if (over == 2 && under == 2)
        {
            atom_inc(&edgeArray[0]);
            atom_inc(&edgeArray[0]);
        }
    }
}


//This kernel calculates the exact number of edges and its corresponding endpoints.
//Keeps track of how many vertices in its box is over/under and determines if there 
//is/are an edge(s) within that box.  Uses the function atom_inc to make sure 
//that the incrementing of the value in edgeArray is done correctly.  Based on which
//vertex is over/under, determines the endpoints and exact amount of edges.
//Uses atomic_add on variable spot to put the found endpoints into the array pointsArray
__kernel
void ActualEdges(__global float* vertexValues, __global int* edgeArray, float level, int rows, int cols, __global vec2* pointsArray,__global int* spot)
{
    //Get the thread id and its corresponding row and column
    int place = get_global_id(0);
    int myRow = (place/cols);
    int myCol = place%cols;
        
    //Only enters if the thread is dealing in a "box" of vertices.
    if ((place%cols) != (cols-1) && place < (cols*rows)-cols)
    {
        //Vertex values that the thread will be working with
        float A = vertexValues[place];
        float B = vertexValues[place+1];
        float C = vertexValues[place+cols];
        float D = vertexValues[place+1+cols];
        
        //Variables over/under keep track of the count that is over/under level
        //Arrays over/underArray keep track of which vertices are the over/under
        //myArray keeps the amount of under/over calculated from calculateOverUnder
        int over = 0;
        int under = 0;
        float overArray[4];
        float underArray[4];
        int myArray[2];
        
        //Pointers to my local variables to pass to function calculateOverUnder
        int* ptr1 = myArray;
        float* ptr2 = overArray;
        float* ptr3 = underArray;
        
        //The endpoints that will be calculated. If value is -1 then no endpoints for the thread
        float endpoints[4][2] = { {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1} };
        int counter = 0;
        int mySpot = 0;

        //Call function to get over/under and fill arrays
        calculateOverUnder(A, B, C, D, level, ptr1, ptr2, ptr3);
        over = myArray[0];
        under = myArray[1];

        //If-else statements to determine the endpoints 
        //based on the over/under and arrays
        if (over == 3 && under == 1)
        {
            atom_inc(&edgeArray[0]);
            counter++;
            
            if (underArray[0] == A)
            {
                endpoints[0][0] = computeEdges(A, B, level, myCol);
                endpoints[0][1] = myRow;
                endpoints[1][1] = computeEdges(A, C, level, myRow);
                endpoints[1][0] = myCol;
            }
            else if (underArray[0] == B)
            {
                endpoints[0][0] = computeEdges(A, B, level, myCol);
                endpoints[0][1] = myRow;
                endpoints[1][1] = computeEdges(B, D, level, myRow);
                endpoints[1][0] = myCol+1;
            }
            else if (underArray[0] == C)
            {
                endpoints[0][0] = computeEdges(C, D, level, myCol);
                endpoints[0][1] = myRow+1;
                endpoints[1][1] = computeEdges(A, C, level, myRow);
                endpoints[1][0] = myCol;
            }
            else 
            {
                endpoints[0][0] = computeEdges(C, D, level, myCol);
                endpoints[0][1] = myRow+1;
                endpoints[1][1] = computeEdges(B, D, level, myRow);
                endpoints[1][0] = myCol+1;
            }
        }
        else if (over == 2 && under == 1)
        {
            if (underArray[0] == A)
            {
                if (overArray[0] == B && overArray[1] == C)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(A, C, level, myRow);
                    endpoints[1][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else if (underArray[0] == B)
            {
                if (overArray[0] == A && overArray[1] == D)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else if (underArray[0] == C)
            {
                if (overArray[0] == A && overArray[1] == D)
                {
                    endpoints[0][0] = computeEdges(C, D, level, myCol);
                    endpoints[0][1] = myRow+1;
                    endpoints[1][1] = computeEdges(A, C, level, myRow);
                    endpoints[1][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else 
            {
                if (overArray[0] == B && overArray[1] == C)
                {
                    endpoints[0][0] = computeEdges(C, D, level, myCol);
                    endpoints[0][1] = myRow+1;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
        }
        else if (under == 3 && over == 1)
        {
            atom_inc(&edgeArray[0]);
            counter++;
            
            if (overArray[0] == A)
            {
                endpoints[0][0] = computeEdges(A, B, level, myCol);
                endpoints[0][1] = myRow;
                endpoints[1][1] = computeEdges(A, C, level, myRow);
                endpoints[1][0] = myCol;
            }
            else if (overArray[0] == B)
            {
                endpoints[0][0] = computeEdges(A, B, level, myCol);
                endpoints[0][1] = myRow;
                endpoints[1][1] = computeEdges(B, D, level, myRow);
                endpoints[1][0] = myCol+1;
            }
            else if (overArray[0] == C)
            {
                endpoints[0][0] = computeEdges(C, D, level, myCol);
                endpoints[0][1] = myRow+1;
                endpoints[1][1] = computeEdges(A, C, level, myRow);
                endpoints[1][0] = myCol;
            }
            else 
            {
                endpoints[0][0] = computeEdges(C, D, level, myCol);
                endpoints[0][1] = myRow+1;
                endpoints[1][1] = computeEdges(B, D, level, myRow);
                endpoints[1][0] = myCol+1;
            }
        }
        else if (under == 2 && over == 1)
        {
            if (overArray[0] == A)
            {
                if (underArray[0] == B && underArray[1] == C)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(A, C, level, myRow);
                    endpoints[1][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else if (overArray[0] == B)
            {
                if (underArray[0] == A && underArray[1] == D)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else if (overArray[0] == C)
            {
                if (underArray[0] == A && underArray[1] == D)
                {
                    endpoints[0][0] = computeEdges(C, D, level, myCol);
                    endpoints[0][1] = myRow+1;
                    endpoints[1][1] = computeEdges(A, C, level, myRow);
                    endpoints[1][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else 
            {
                if (overArray[0] == B && underArray[1] == C)
                {
                    endpoints[0][0] = computeEdges(C, D, level, myCol);
                    endpoints[0][1] = myRow+1;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
        }
        else if (over == 2 && under == 2)
        {  
            if (overArray[0] == A)
            {
                if (overArray[1] == D)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(A, C, level, myRow);
                    endpoints[1][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                    
                    endpoints[2][0] = computeEdges(C, D, level, myCol);
                    endpoints[2][1] = myRow+1;
                    endpoints[3][1] = computeEdges(B, D, level, myRow);
                    endpoints[3][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
                else if (overArray[1] == B)
                {
                    endpoints[0][1] = computeEdges(A, C, level, myRow);
                    endpoints[0][0] = myCol;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
                else 
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][0] = computeEdges(C, D, level, myCol);
                    endpoints[1][1] = myRow+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
            }
            else if (overArray[0] == B)
            {
                if (overArray[1] == C)
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][1] = computeEdges(B, D, level, myRow);
                    endpoints[1][0] = myCol+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                    
                    endpoints[2][0] = computeEdges(C, D, level, myCol);
                    endpoints[2][1] = myRow+1;
                    endpoints[3][1] = computeEdges(A, C, level, myRow);
                    endpoints[3][0] = myCol;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
                else
                {
                    endpoints[0][0] = computeEdges(A, B, level, myCol);
                    endpoints[0][1] = myRow;
                    endpoints[1][0] = computeEdges(C, D, level, myCol);
                    endpoints[1][1] = myRow+1;
                    atom_inc(&edgeArray[0]);
                    counter++;
                }
                
            }
            else
            {
                endpoints[0][1] = computeEdges(A, C, level, myRow);
                endpoints[0][0] = myCol;
                endpoints[1][1] = computeEdges(B, D, level, myRow);
                endpoints[1][0] = myCol+1;
                atom_inc(&edgeArray[0]);
                counter++;
            }
        }
        
        //Increment the spot counter to the amount of points in my thread
        counter = counter *2;
        mySpot = atom_add(&spot[0], counter);
        
        //Put the points into the pointsArray based on mySpot
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (endpoints[i][j] != -1)
                {
                    pointsArray[mySpot][j] = endpoints[i][j];
                    
                    if(place == 4)
                    {
                        printf("%f ", endpoints[i][j]);
                    }
                }
            }
            mySpot++;
        }
    }
}
