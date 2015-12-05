#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/CMPT726
DATA=data/CMPT726
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/CMPT726_train_lmdb \
  $DATA/CMPT726_mean.binaryproto

echo "Done."
