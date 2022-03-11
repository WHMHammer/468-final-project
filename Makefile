build:
	g++ -O3 -o regress_cpp regress.cpp
	nvcc -O3 -lineinfo -o regress_cuda regress.cu

dependencies:
	python3 -m venv venv
	venv/bin/pip3 install matplotlib numpy

run:
	venv/bin/python3 run.py

clean:
	rm -f regress_cpp regress_cuda in.txt out.txt

clobber: clean
	rm -rf training.png testing.png venv
