all: seq seqg mpi mpig pthread pthreadg

seq: sequential.cpp
	g++ sequential.cpp -o seq -O2 -std=c++11

seqg: sequential.cpp
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm sequential.cpp -o seqg -DGUI -O2 -std=c++11

mpi: mpi.cpp
	mpic++ mpi.cpp -o mpi -std=c++11

mpig: mpi.cpp
	mpic++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm mpi.cpp -o mpig -DGUI -std=c++11

pthread: pthread.cpp
	g++ pthread.cpp -lpthread -o pthread -O2 -std=c++11

pthreadg: pthread.cpp
	g++ -I/usr/include -L/usr/local/lib -L/usr/lib -lglut -lGLU -lGL -lm -lpthread pthread.cpp -o pthreadg -DGUI -O2 -std=c++11

run_seq: seq
	./seq $(X_RESN) $(Y_RESN) $(max_iteration)

run_seqg: seqg	
	./seqg $(X_RESN) $(Y_RESN) $(max_iteration)

run_mpi: mpi
	mpirun -np $(n_proc) ./mpi $(X_RESN) $(Y_RESN) $(max_iteration)

run_mpig: mpig	
	mpirun -np $(n_proc) ./mpig $(X_RESN) $(Y_RESN) $(max_iteration)

run_pthread: pthread
	./pthread $(X_RESN) $(Y_RESN) $(max_iteration) $(n_thd)

run_pthreadg: pthreadg
	./pthreadg $(X_RESN) $(Y_RESN) $(max_iteration) $(n_thd)



.PHONY: clean run_seq run_seqg run_mpi run_mpig run_pthread run_pthreadg
clean:
	rm -f seq seqg mpi mpig pthread pthreadg