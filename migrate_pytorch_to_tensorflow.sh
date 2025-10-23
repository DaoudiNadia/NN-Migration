#!/bin/bash


name="alexnet"
datashape="(224,224,3)"


archit_in="subclassing"
archit_out="subclassing"


python torch2tf/torch2tf.py "output/${name}/pytorch_nn_${archit_in}.py" ${archit_in} ${archit_out}


