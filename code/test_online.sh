time=$(date "+%Y-%m-%d-%H:%M:%S")
处理数据
python data.py --mode online --logfile "${time}.log"

# itemcf 召回
python recall_itemcf.py --mode online --logfile "${time}.log"

# binetwork 召回
python recall_binetwork.py --mode onlined --logfile "${time}.log"

# w2v 召回
python recall_w2v.py --mode online --logfile "${time}.log"

# 召回合并
python recall.py --mode online --logfile "${time}.log"

# 排序特征
python rank_feature.py --mode online --logfile "${time}.log"

# 使用lgb模型排序
python rank_lgb.py --mode online --logfile "${time}.log"
