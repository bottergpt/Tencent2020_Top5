
#!/bin/bash
tm_now=`date +%F_%T`
echo "run_model1..."
/root/.conda/envs/tf2/bin/python -u model1.py --fold -1 --tm_now ${tm_now} --seed 736127 --cvseed 736127
echo "run_model1, Done!"
tm_now=`date +%F_%T`
echo "run_model2..."
/root/.conda/envs/tf2/bin/python -u model2.py --fold -1 --tm_now ${tm_now} --seed 712 --cvseed 712
echo "run_model2, Done!"
tm_now=`date +%F_%T`
echo "run_model3..."
/root/.conda/envs/tf2/bin/python -u model3.py --fold -1 --tm_now ${tm_now} --seed 666 --cvseed 666
echo "run_model3, Done!"
tm_now=`date +%F_%T`
echo "run_model4..."
/root/.conda/envs/tf2/bin/python -u model4.py --fold -1 --tm_now ${tm_now} --seed 2020 --cvseed 2020
echo "run_model4, Done!"