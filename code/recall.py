import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

# ===== 策略开关 =====
USE_SCHEME_1 = True   # 是否使用source-aware召回权重
USE_SCHEME_2 = True   # 是否使用召回信息显式作为排序特征


warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):
    df = df.copy()

    grp = df.groupby(['recall_source', 'user_id'])['sim_score']

    min_s = grp.transform('min')
    max_s = grp.transform('max')

    # 防止除 0
    denom = (max_s - min_s).replace(0, 1e-9)

    df['sim_score_norm'] = (df['sim_score'] - min_s) / denom

    return df


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = ['itemcf', 'w2v', 'binetwork']

    weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
    recall_list = []
    recall_dict = {}
   
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        
        # 标记召回来源
        recall_result['recall_source'] = recall_method
        if USE_SCHEME_2:
            # 保留单路原始分数
            recall_result[f'{recall_method}_raw_score'] = recall_result['sim_score']

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度
    import pdb
    pdb.set_trace()
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)


    if USE_SCHEME_1:
        # 归一化
        recall_final = mms(recall_final)

        # 显式权重融合
        recall_final['weight'] = recall_final['recall_source'].map(weights)
        recall_final['sim_score_final'] = (
            recall_final['sim_score_norm'] * recall_final['weight']
        )
    else:
        # 直接用原始 sim_score
        recall_final['sim_score_final'] = recall_final['sim_score']
    
    if USE_SCHEME_2:
        # 召回来源 one-hot
        for src in recall_methods:
            recall_final[f'is_{src}'] = (
                recall_final['recall_source'] == src
            ).astype(int)

        # 每路召回的归一化分数
        for src in recall_methods:
            recall_final[f'{src}_score'] = np.where(
                recall_final['recall_source'] == src,
                recall_final['sim_score_norm'] if USE_SCHEME_1 else recall_final['sim_score'],
                0.0
            )


    agg_dict = {'sim_score_final': 'sum'}

    if USE_SCHEME_2:
        for src in recall_methods:
            agg_dict[f'is_{src}'] = 'max'
            agg_dict[f'{src}_score'] = 'max'


    recall_score = recall_final.groupby(
        ['user_id', 'article_id']
    ).agg(agg_dict).reset_index()

    
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')


    recall_final.sort_values(['user_id', 'sim_score_final'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)

    if USE_SCHEME_2:
        log.debug(
            recall_final[['sim_score_final',
                        'is_itemcf', 'itemcf_score',
                        'is_w2v', 'w2v_score',
                        'is_binetwork', 'binetwork_score']].head(20)
        )
    else:
        log.debug(
        recall_final[['sim_score_final']].head(20)
    )

    log.debug(f'df_useful_recall: {df_useful_recall.head()}')



    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score_final'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 限制每个用户的召回数量
    df_useful_recall = (
        df_useful_recall
        .groupby('user_id', group_keys=False)
        .head(200)
    )


    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')
