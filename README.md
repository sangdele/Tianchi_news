# 天池大赛 零基础入门推荐系统

比赛地址: https://tianchi.aliyun.com/competition/entrance/531842/introduction

在baseline基础上做了两处优化：
```
方案一：在召回层增加了 source-aware 的归一化和显式权重控制，使多路召回的融合过程可配置、可对比，而非依赖隐式比例假设。
方案二：将召回阶段的来源标记与相似度分数显式作为特征输入排序模型，让模型学习不同召回源在不同用户下的可靠性
```

### 复现步骤
```
操作系统 ubuntu 
python版本 3.8
数据集路径 tcdata/
```
### 离线
```
pip install -r requirements.txt
cd code
bash test.sh
```

### 在线
```
pip install requirements.txt
cd code
bash test_online.sh
```

### 使用baseline/方案一/方案二
```
recall.py、recall_feature.py文件内设置：
USE_SCHEME_1 = True / False #是否使用source-aware召回权重
USE_SCHEME_2 = True / False #是否使用召回信息显式作为排序特征
```
