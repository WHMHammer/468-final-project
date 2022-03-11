build: cuda

dependencies:
	python3 -m venv venv
	venv/bin/pip3 install matplotlib numpy

cuda:
	nvcc -O3 -lineinfo -o regress regress.cu

cpp:
	g++ -O3 -o regress regress.cpp

run:
	venv/bin/python3 run.py

clean:
	rm -f regress in.txt out.txt

clobber: clean
	rm -rf training.png testing.png venv
