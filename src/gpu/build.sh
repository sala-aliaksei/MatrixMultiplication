#!/bin/bash

nvcc -ccbin=/usr/bin/gcc-13 -o matmul -Xcompiler -lstdc++ matrixMul.cu
