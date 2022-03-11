# Robust Mini-Batch Gradient Descent

This repository contains the source codes for the final project of COMP_SCI 468: Programming Massively Parallel Processors with CUDA at Northwestern University.

In this project, we implemented the best model (ε-then-Z-score-trimmed Huber Loss without preprocessing) found in [Robust Mini-Batch Gradient Descent](https://github.com/WHMHammer/robust-mini-batch-gradient-descent) with C++ (baseline) and CUDA.

**Note**: the CUDA codes are optimized for RTX 2060 Super graphics cards in the Wilkinson lab.

For more details on the model itself, please visit the [Robust Mini-Batch Gradient Descent](https://github.com/WHMHammer/robust-mini-batch-gradient-descent) repo.

## Run Tests

```
make dependencies    # only before the first run
make
make run
# make clean         # to clean up the binaries and input/output files generated
# make clobber       # to also clean up the graphs and python virtual environment
```

Two graphs, `testing.png` and `training.png` will be generated, showing the regression lines under the training set and the testing set. These graphs are generated to check the correctness of the implementation. For more test cases, please visit the [Robust Mini-Batch Gradient Descent](https://github.com/WHMHammer/robust-mini-batch-gradient-descent) repo.

## Contribution

[Hanming Wang](https://github.com/WHMHammer)

- Implemented the baseline implementation in C++

## Acknowledgement

We appreciate prof. [Nikos Hardavellas](https://users.cs.northwestern.edu/~hardav/) for the wonderful lectures throughout the quarter. Hanming especially thanks Nikos for the time spent during and beyond the office hours, answering the numerous questions (some being stupid) about CUDA.
