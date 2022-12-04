CXX:=g++
MPICXX:=mpic++
CXXFLAGS:=-std=c++14

all: gen ssort psort check

gen: test_data_generator.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

ssort: odd_even_sequential_sort.cpp
	$(CXX) $(CXXFLAGS) -O2 -o $@ $^

psort: odd_even_parallel_sort.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^

check: check_sorted.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_ssort: ssort
	./ssort $(num) $(input)

.PHONY: clean run_ssort
clean:
	-rm gen ssort psort check