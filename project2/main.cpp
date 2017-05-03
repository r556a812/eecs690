/**
*	@file : main.cpp
*	@author :  Richard Aviles
*	@date : 2017.03.29
*	Purpose: The main that interacts with the user.
*/

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>
#include <mpi.h>
#include <sstream>


using namespace std;

int numElements = 500;
std::string filename;

static std::string get_name(int place)
{
    int count = 1;
    std::string s;
    std::string a;
    std::string state;
    std::string city;
    std::string name;

    ifstream infile;
    infile.open(filename);

    while(std::getline(infile, s))
    {
        std::istringstream iss{s};

        if (count == place)
        {
            std::getline(iss, a, ',');
            state = a;
            std::getline(iss, a, ',');
            city = a;
            break;
        }
        count++;
    }

    name = city + ", " + state;
    return name;
}

static void do_rank_0_srWork(std::string compareChoice, int place, int compare, std::string gtLt, int communicatorSize)
{
    //Declare local variables
    int size = numElements/communicatorSize;
    double arr[numElements];
    double scattArray[size];
    double results[1];
    double answer[1];
    MPI::Op op;

    //Declare variables for file reading
    int count = 1;
    int spot = 0;
    std::string s;
    std::string a;
    std::string header;

    ifstream infile;
    infile.open(filename);

    //Get the column header
    std::getline(infile, s);
    std::istringstream iss{s};
    while (std::getline(iss, a, ','))
    {
        if(count == place)
        {
            header = a;
            count = 1;
            break;
        }
        count++;
    }

    //Get the column data
    while(std::getline(infile, s))
    {
        std::istringstream iss{s};

        while (std::getline(iss, a, ','))
        {
            if (a[0] == '"')
            {
                count--;
            }
            if(count == place)
            {
                double temp = atof(a.c_str());
                arr[spot] = temp;
                spot++;
                break;
            }
            count++;
        }

        count = 1;
    }

    //Scatter the data to the other ranks
    MPI_Scatter(arr, size, MPI_DOUBLE, scattArray, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Do my work
    double returnNum = scattArray[0];

    if (compareChoice == "avg")
    {
        for(int i = 1; i < size; i++)
        {
            returnNum += scattArray[i];
        }

        op = MPI_SUM;
    }
    else if (compareChoice == "min")
    {
        for(int i = 1; i < size; i++)
        {
            if (returnNum > scattArray[i])
            {
                returnNum = scattArray[i];
            }
        }

        op = MPI_MIN;
    }
    else if (compareChoice == "max")
    {
        for(int i = 1; i < size; i++)
        {
            if (returnNum < scattArray[i])
            {
                returnNum = scattArray[i];
            }
        }

        op = MPI_MAX;
    }
    else if (compareChoice == "number")
    {
        returnNum = 0;
        if (gtLt == "gt")
        {
            for(int i = 0; i < size; i++)
            {
                if (scattArray[i] > compare)
                {
                    returnNum++;
                }
            }
        }
        else
        {
            for(int i = 0; i < size; i++)
            {
                if (scattArray[i] < compare)
                {
                    returnNum++;
                }
            }
        }

        op = MPI_SUM;
    }
    results[0] = returnNum;

    //Receive back the data from other ranks, place into my answer array
    MPI_Reduce(results, answer, 1, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);

    //Output results to the screen
    if(compareChoice == "avg")
    {
        double ans = (answer[0])/numElements;
        std::cout << "Average " << header << " = " << ans << "\n";
    }
    else if (compareChoice == "number")
    {
        std::cout << "Number of cities with " << header << " " << gtLt << " " << compare << " = " << answer[0] << "\n";
    }
    else
    {
        std::cout << header << " = " << answer[0] << "\n";
    }
}
static void do_rank_i_srWork(std::string compareChoice,int compare, std::string gtLt, int communicatorSize)
{
    //Declare local variables
    int size = numElements/communicatorSize;
    double scattArray[size];
    double results[1];
    MPI::Op op;

    //Receive my work
    MPI_Scatter(nullptr, 0, MPI_DOUBLE, scattArray, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double returnNum = scattArray[0];

    //Do work for the given comparison
    if (compareChoice == "avg")
    {
        for(int i = 1; i < size; i++)
        {
            returnNum += scattArray[i];
        }

        op = MPI_SUM;
    }
    else if (compareChoice == "min")
    {
        for(int i = 1; i < size; i++)
        {
            if (returnNum > scattArray[i])
            {
                returnNum = scattArray[i];
            }
        }

        op = MPI_MIN;
    }
    else if (compareChoice == "max")
    {
        for(int i = 1; i < size; i++)
        {
            if (returnNum < scattArray[i])
            {
                returnNum = scattArray[i];
            }
        }

        op = MPI_MAX;
    }
    else if (compareChoice == "number")
    {
        returnNum = 0;
        if (gtLt == "gt")
        {
            for(int i = 0; i < size; i++)
            {
                if (scattArray[i] > compare)
                {
                    returnNum++;
                }
            }
        }
        else
        {
            for(int i = 0; i < size; i++)
            {
                if (scattArray[i] < compare)
                {
                    returnNum++;
                }
            }
        }

        op = MPI_SUM;
    }
    results[0] = returnNum;

    //Return the results to the rank 0 process
    MPI_Reduce(results, nullptr, 1, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
}


static void do_rank_0_bgWork(std::string compareChoice, std::vector<int> placeVec , int numPlaces, int communicatorSize)
{
    //Declare local variables
    double* data = new double[numElements * numPlaces];
    int cities[numPlaces];
    double answer[numPlaces];
    std::string headerArray[numPlaces];

    //Declare variables for file reading
    std::string s;
    std::string a;

    //Get all column data
    for (int i = 0; i < numPlaces; i++)
    {
        int count = 1;
        int spot = 0;

        ifstream infile;
        infile.open(filename);

        //Get the column header
        std::getline(infile, s);
        std::istringstream iss{s};

        while (std::getline(iss, a, ','))
        {
            if(count == placeVec[i])
            {
                headerArray[i] = a;
                count = 1;
                break;

            }

            count++;
        }

        while(std::getline(infile, s))
        {
            std::istringstream iss{s};

            while (std::getline(iss, a, ','))
            {
                if (a[0] == '"')
                {
                    std::getline(iss, a, ',');
                    //count--;
                }

                if(count == placeVec[i])
                {
                    double temp = atof(a.c_str());
                    data[(i*numElements) + spot] = temp;
                    spot++;
                    //break;
                }

                count++;
            }

            count = 1;
        }
    }

    //If statement to force wait for fill of array
    if(data[numElements*numPlaces - 1] != 0)
    {
        //Broadcast array to all processes
        MPI_Bcast(data, (numElements * numPlaces), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Declare local variables for my work
        int start = 0;
        int end = numElements;
        double numReturn = data[start];
        int city = 0;

        //Do my work based on comparison given
        if (compareChoice == "max")
        {
            for (int i = start+1; i < end; i++)
            {
                if (numReturn < data[i])
                {
                    numReturn = data[i];
                    city = i;
                }
            }
        }
        else if (compareChoice == "min")
        {
            for (int i = start+1; i < end; i++)
            {
                if (numReturn > data[i])
                {
                    numReturn = data[i];
                    city = i;
                }
            }
        }
        else if (compareChoice == "avg")
        {
            for (int i = start+1; i < end; i++)
            {
                numReturn = numReturn + data[i];
            }

            numReturn = numReturn/numElements;
        }

        answer[0] = numReturn;
        cities[0] = city + 2;

        //Gather back my answers and the associated city numbers
        MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, answer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, 0, MPI_INT, cities, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //Print results to the screen
        for (int a = 0; a < numPlaces; a++)
        {
            std::string name = get_name(cities[a]);
            if (compareChoice == "avg")
            {
                std::cout << compareChoice << " " << headerArray[a] << " = " << answer[a] << "\n";
            }
            else
            {
                std::cout << compareChoice << " " << headerArray[a] << " = " << answer[a] << "; " << name <<  "\n";
            }
        }
    }

    delete[]data;
}

static void do_rank_i_bgWork(int rank, std::string compareChoice, int numPlaces, int communicatorSize)
{
    //Declare local variables
    double* myArray = new double[numElements * numPlaces];
    double answer[1];
    int cities[1];

    //Get data that is being broadcasted from rank 0
    MPI_Bcast(myArray, (numElements * numPlaces), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Local variables needed to do my work
    int start = rank * numElements;
    int end = (rank+1) * numElements;
    double numReturn = myArray[start];
    int city = start;

    //Do my work based on the comparison given
    if (compareChoice == "max")
    {
        for (int i = start+1; i < end; i++)
        {
            if (numReturn < myArray[i])
            {
                numReturn = myArray[i];
                city = i;
            }
        }
    }
    else if (compareChoice == "min")
    {
        for (int i = start+1; i < end; i++)
        {
            if (numReturn > myArray[i])
            {
                numReturn = myArray[i];
                city = i;
            }
        }
    }
    else if (compareChoice == "avg")
    {
        for (int i = start+1; i < end; i++)
        {
            numReturn = numReturn + myArray[i];
        }

        numReturn = numReturn/numElements;
    }

    answer[0] = numReturn;
    cities[0] = city - (rank*numElements) + 2;

    //Send data back to the rank 0 process
    MPI_Gather(answer, 1, MPI_DOUBLE, nullptr, numPlaces, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(cities, 1, MPI_INT, nullptr, numPlaces, MPI_INT, 0, MPI_COMM_WORLD);

    delete[] myArray;
}

int main(int argc, char *argv[])
{
    //Create MPI world
    MPI_Init(&argc, &argv);

    int rank;
    int communicatorSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

    //Make sure given enough args
    if (argc < 4)
    {
        if (rank == 0)
        {
            std::cout << "\nError: Need to give more conditions\n";
        }
    }
    else
    {
        //Get information from command line
        std::string communicationChoice = argv[1];
        std::string compareChoice = argv[2];
        std::string column = argv[3];

        //Use only if compareChoice is number and need it
        int compare = 0;
        std::string gtLt = "";

        if (compareChoice == "number")
        {
            gtLt = argv[4];
            compare = atoi(argv[5]);
        }

        if (rank == 0)
        {
            std::cout << "Enter the name of the file to open: ";
            std::cin >> filename;
        }

        if (communicationChoice == "sr")
        {
            if (rank == 0)
            {
                //Figure out the distance for the given column
                int place = 0;
                if (column.length() == 2)
                {
                    place = (26 * (column[0] - 64)) + (column[1] - 64);
                }
                else
                {
                    place = column[0] - 64;
                }

                //Start rank 0 work
                do_rank_0_srWork(compareChoice, place, compare, gtLt, communicatorSize);
            }
            else
            {
                //Start other rank work
                do_rank_i_srWork(compareChoice, compare, gtLt, communicatorSize);
            }
        }
        else if (communicationChoice == "bg")
        {
            int numColumns = argc - 3;

            //Give error message if columns not equal to processes
            if(communicatorSize != numColumns)
            {
                if (rank == 0)
                {
                    std::cout << "Error: Number of processes does not equal the number of columns";
                }
            }
            else
            {
                if (rank == 0)
                {
                    std::vector<int> placeVec(numColumns);
                    for (int i = 3; i < argc; i++)
                    {
                        //Figure out the distance for the given column
                        column = argv[i];
                        int place = 0;
                        if (column.length() == 2)
                        {
                            place = (26 * (column[0] - 64)) + (column[1] - 64);
                        }
                        else
                        {
                            place = column[0] - 64;
                        }
                        placeVec[i-3] = place;

                        //Start rank 0 work
                        do_rank_0_bgWork(compareChoice, placeVec, numColumns, communicatorSize);
                    }
                }
                else
                {
                    //Start other rank work
                    do_rank_i_bgWork(rank, compareChoice, numColumns, communicatorSize);
                }
            }
        }
        else
        {
            std::cout << "\nError: Invalid communication choice. Type sr for Scatter Reduce or bg for Broadcast Gather.\n";
        }

    }

    MPI_Finalize();
}
