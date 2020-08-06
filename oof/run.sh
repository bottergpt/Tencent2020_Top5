#!/bin/bash
tm_now=`date +%F_%T`
echo "run model_result_to npy..."
/root/.conda/envs/tf1/bin/python -u model_result_to_npy.py
echo "run odel_result_to npy, Done!"