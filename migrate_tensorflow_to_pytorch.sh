#!/bin/bash


name="tutorial_example"
datashape="(32,32,3)"


archit_in="subclassing"
archit_out="subclassing"


python tf2torch/tf2torch.py "output/${name}/tf_nn_${archit_in}.py" ${archit_in} ${archit_out} --datashape ${datashape}

