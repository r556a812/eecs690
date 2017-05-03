project2: main.o
	mpicxx -std=c++11 -g -Wall main.o -o project2 -pthread -Wl,-rpath -Wl,/usr/lib64/openmpi/lib -Wl,--enable-new-dtags -L/usr/lib64/openmpi/lib -lmpi_cxx -lmpi
main.o: main.cpp
	mpicxx -std=c++11 -g -Wall -c main.cpp
clean: 
	rm *.o project2
	echo clean done
	
