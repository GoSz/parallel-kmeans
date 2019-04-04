CC=g++

objects = kmeans_tool.o parallel_kmeans.o

kmeans_tool : $(objects)
	$(CC) -g -O2 -std=c++11 -fopenmp -o kmeans_tool $(objects)

kmeans_tool.o : kmeans_tool.cpp parallel_kmeans.h
	$(CC) -g -O2 -std=c++11 -c kmeans_tool.cpp

parallel_kmeans.o : parallel_kmeans.cpp parallel_kmeans.h
	$(CC) -g -O2 -std=c++11 -mavx -msse4 -fopenmp -c parallel_kmeans.cpp

.PHONY : clean
clean :
	-rm -rf kmeans_tool $(objects)
