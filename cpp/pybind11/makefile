$(CXX) = g++
options += -O3 -Wall -std=c++11 
options += -shared  -fPIC 
options += `python3 -m pybind11 --includes` 
src = example.cpp 
out = -o example`python3-config --extension-suffix`
run:
	$(CXX) $(options) $(src) $(out)
