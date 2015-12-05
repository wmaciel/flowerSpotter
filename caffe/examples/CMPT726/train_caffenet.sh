#!/usr/bin/env sh

./build/tools/caffe train -gpu 1,2 \
    --solver=models/CMPT726/solver.prototxt
