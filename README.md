# 2020腾讯广告算法大赛Top5 Solution

具体方案可以参考答辩PPT以及答辩视频。

#### 1. 队伍名称及成员：BANJITINO

[zhangqibot](https://github.com/zhangqibot)、[阿郑zzz](https://github.com/zhenglinghan)、[贝壳er](https://github.com/lixiangwang)

#### 2. 运行环境

```bash
# tf1.12
conda create -y -n tf1 python=3.6 tensorflow-gpu=1.12 keras=2.2.4 &&
conda activate tf1 &&
pip install pandas tqdm sklearn
# pip install glove_python gensim 

# tf2.2
conda create -y -n tf2 python=3.8 tensorflow-gpu=2.2 &&
conda activate tf2 &&
pip install pandas tqdm sklearn

# torch1.4
conda create -y -n torch python=3.8 pytorch=1.4.0 &&
conda activate torch &&
pip install gensim pandas tqdm sklearn
```

环境详见requirements_tf1.txt，requirements_tf2.txt 和 requirements_torch.txt

#### 3. 运行

```bash
bash run.sh
```

运行顺序是：

```bash
#!/bin/bash
bash preprocess/run.sh &&  # 数据预处理
bash get_emb/run.sh &&     # embedding
bash model/run.sh &&       # 模型的训练，均采用20分类进行预测
bash oof/run.sh &&         # 对模型训练的输出结果进行归并，处理成model.npy，里面是[400w,20]的矩阵
bash ensemble/run.sh       # 对模型的oof做stacking，用的是Ridge
echo "all done!"
```

#### 4. Reference

- [2020 腾讯广告算法大赛](https://algo.qq.com/)
- [易观性别年龄预测第一名解决方案](https://github.com/chizhu/yiguan_sex_age_predict_1st_solution)
- [HUAWEI-DIGIX-AgeGroup](https://github.com/luoda888/HUAWEI-DIGIX-AgeGroup)

#### 5. Email

hi@zhangqibot.com

