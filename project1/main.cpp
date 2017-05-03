/**
*	@file : main.cpp
*	@author :  Richard Aviles
*	@date : 2017.02.20
*	Purpose: The main that interacts with the user.
*/

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "Barrier.h"

using namespace std;

int** tracks;
int* trainArray;
int numExpected;
std::mutex myMutex;
Barrier startB;
Barrier firstB;
Barrier secondB;

//Method used to convert thread number to a letter
char getName(int i)
{
    return static_cast<char>('A' + i);
}

//Method to star the routes
void startRoutes(std::vector<int> myRoutes, int num)
{
    //Start barrier to make sure all threads have arrived
    startB.barrier(numExpected);

    //Declare variables
    int length = myRoutes.size() - 1;
    int spot = 0;
    int time = 0;
    int station = myRoutes[0];
    int trainsRunning = 0;
    char n = getName(num);
    bool change = false;

    //While loop that runs until the entire route is completed
    while (spot < length)
    {
        //Check for the amount of trains still running
        trainsRunning = 0;
        for(int i = 0; i < numExpected; i++)
        {
            if (trainArray[i] == -1)
            {
                trainsRunning++;
            }
        }

        //Mutex lock to see if track is available
        myMutex.lock();

        //If track is available, progress. If not, then stay
        if (tracks[myRoutes[spot]][myRoutes[spot + 1]] == 1)
        {
            std::cout << "At time step: " << time
                      << " train " << n
                      << " is going from station " << myRoutes[spot]
                      << " to station " << myRoutes[spot + 1] << "\n";

            tracks[myRoutes[spot]][myRoutes[spot + 1]] = -1;
            tracks[myRoutes[spot + 1]][myRoutes[spot]] = -1;
            spot++;
            change = true;
            station = myRoutes[spot];
        }
        else
        {
            std::cout << "At time step: " << time
                      << " train " << n
                      << " must stay at station " << myRoutes[spot] << "\n";
        }

        myMutex.unlock();

        //First barrier after locks to wait for all threads to arrive
        firstB.barrier(trainsRunning);

        //Reset the track array to allow for to be available again
        if(change)
        {
        tracks[myRoutes[spot-1]][myRoutes[spot]] = 1;
        tracks[myRoutes[spot]][myRoutes[spot-1]] = 1;
        }

        change = false;

        //Update the trainArray
        if (!(spot < length))
        {
            trainArray[num] = time;
        }

        //Increment time
        time++;

        //Second barrier to allow all threads to reset track array before starting again
        secondB.barrier(trainsRunning);
    }
}

int main(int argc, char *argv[])
{
    //Make sure to get a file name to use
    if (argc < 2)
    {
        std::cout << "Error: Need to give an input file name";
    }

    //Declare variables
    int numT = 0;
    int numS = 0;
    int numR = 0;
    int x = 0;

    //Read the file given from argv1
    ifstream infile;
    infile.open(argv[1]);

    //Get the number of trains and stations
    infile >> numT;
    infile >> numS;
    numExpected = numT;

    //Populate the trainArray
    trainArray = new int[numT];
    for (int i = 0; i < numT; i++)
    {
        trainArray[i] = -1;
    }

    //Create the thread and route arrays
    std::thread** t = new std::thread*[numT];
    std::vector<int>* myvector = new std::vector<int>[numT];

    //Initialize tracks
    tracks = new int*[numS];
    for(int i = 0; i < numS; i++)
    {
        tracks[i] = new int[numS];

        for(int j = 0; j < numS; j++)
        {
            tracks[i][j] = -1;
        }
    }

    //Populate the routes and tracks array
    for (int i = 0; i < numT; i++)
    {
        infile >> numR;
        std::vector<int> myVec(numR);

        for (int j = 0; j < numR; j++)
        {
            infile >> x;
            myVec[j] = x;
        }

        for (int k = 0; k < numR - 1; k++)
        {
            if (tracks[myVec[k]][myVec[k+1]] == -1)
            {
                tracks[myVec[k]][myVec[k+1]] = 1;
                tracks[myVec[k+1]][myVec[k]] = 1;
            }
        }

        myvector[i] = myVec;
    }

    std::cout << "Starting simulation...\n";

    //Create the threads and give them their routes
    for(int i = 0; i < numT; i++)
    {
        t[i] = new std::thread(startRoutes, myvector[i], i);
    }

    //Wait until the threads are done
    for(int i = 0; i < numT; i++)
    {
        t[i]->join();
    }

    std::cout << "Simulation complete.\n";


}
