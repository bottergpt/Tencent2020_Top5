#!/bin/bash
/root/.conda/envs/torch/bin/python -u Multi_Head_ResNet.py
/root/.conda/envs/torch/bin/python -u Multi_Head_ResNext.py
/root/.conda/envs/torch/bin/python -u Multi_Head_ResNext_4seeds_5folds.py
/root/.conda/envs/torch/bin/python -u Multi_Head_ResNext_seed_34_aug.py
/root/.conda/envs/torch/bin/python -u Multi_Head_ResNext_seed1111_aug.py




