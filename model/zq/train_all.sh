#!/bin/bash
tm_now=`date +%F_%T`
echo "run_bear..."
/root/.conda/envs/tf1/bin/python -u run_bear.py --fold -1 --tm_now ${tm_now} --seed 736127 --cvseed 736127
echo "run_bear, Done!"
tm_now=`date +%F_%T`
echo "run_cat..."
/root/.conda/envs/tf1/bin/python -u run_cat.py --fold -1 --tm_now ${tm_now} --seed 712 --cvseed 712
echo "run_cat, Done!"
tm_now=`date +%F_%T`
echo "run_dog..."
/root/.conda/envs/tf1/bin/python -u run_dog.py --fold -1 --tm_now ${tm_now} --seed 666 --cvseed 666
echo "run_dog, Done!"
tm_now=`date +%F_%T`
echo "run_lion..."
/root/.conda/envs/tf1/bin/python -u run_lion.py --fold -1 --tm_now ${tm_now} --seed 2020 --cvseed 2020
echo "run_lion, Done!"
# tm_now=`date +%F_%T`
# echo "run_tiger..."
# /root/.conda/envs/tf1/bin/python -u run_tiger.py --fold -1 --tm_now ${tm_now} --seed 911 --cvseed 911
# echo "run_tiger, Done!"