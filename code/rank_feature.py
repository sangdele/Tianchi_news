import argparse
import os
import pandas as pd
import warnings

from utils import Logger

warnings.filterwarnings('ignore')

# ===== 策略开关=====
USE_SCHEME_1 = True   # 是否使用source-aware召回权重
USE_SCHEME_2 = True   # 是否使用召回信息显式作为排序特征


# 命令行参数
parser = argparse.ArgumentParser(description='rank feature (light)')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='rank_feature.log')
args = parser.parse_args()

mode = args.mode
logfile = args.logfile

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'rank feature (light), mode={mode}')

if __name__ == '__main__':
    if mode == 'valid':
        df = pd.read_pickle('../user_data/data/offline/recall.pkl')
    else:
        df = pd.read_pickle('../user_data/data/online/recall.pkl')

    log.debug(f'load recall, shape={df.shape}')
    log.debug(f'columns={df.columns.tolist()}')

    #文章静态特征
    df_article = pd.read_csv(
        '../../data/articles.csv'
    )
    df_article = df_article[
        ['article_id', 'category_id', 'words_count', 'created_at_ts']
    ]

    df_article['created_at_ts'] = (df_article['created_at_ts'] / 1000).astype(int)

    df = df.merge(df_article, how='left', on='article_id')


    # 每个 user 的召回候选数
    user_cnt = df['user_id'].value_counts()
    df['user_recall_cnt'] = df['user_id'].map(user_cnt)

    # 每个 article 被召回的次数（隐式热度）
    article_cnt = df['article_id'].value_counts()
    df['article_recall_cnt'] = df['article_id'].map(article_cnt)

    # 缺失值处理
    fill_zero_cols = [
        'sim_score_final',
        'itemcf_score', 'w2v_score', 'binetwork_score',
        'is_itemcf', 'is_w2v', 'is_binetwork',
        'user_recall_cnt',
        'article_recall_cnt'
    ]

    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # 控制使用哪种相似度分数
    if USE_SCHEME_1:
        # 使用 recall 阶段已经处理好的分数
        df['rank_score'] = df['sim_score_final']
        log.info('Rank uses sim_score_final (scheme 1 ON)')
    else:
        # 回退到原始相似度
        if 'sim_score' in df.columns:
            df['rank_score'] = df['sim_score']
        else:
            df['rank_score'] = df['sim_score_final']
        log.info('Rank uses raw sim_score (scheme 1 OFF)')

    # 控制是否使用多路召回结构特征
    struct_cols = []

    if USE_SCHEME_2:
        struct_cols = [
            'is_itemcf', 'itemcf_score',
            'is_w2v', 'w2v_score',
            'is_binetwork', 'binetwork_score',
        ]
        log.info('Rank uses recall source features (scheme 2 ON)')
    else:
        log.info('Rank ignores recall source features (scheme 2 OFF)')



    # 确定最终特征
    final_cols = [
        'user_id',
        'article_id',
        'label',

        'rank_score',

        'user_recall_cnt',
        'article_recall_cnt',

        'category_id',
        'words_count',
        'created_at_ts',
    ] + struct_cols

    df = df[final_cols]

    log.debug(f'final feature shape={df.shape}')
    log.debug(f'final columns={df.columns.tolist()}')
    log.debug(f'final feature columns: {df.columns.tolist()}')


    if mode == 'valid':
        df.to_pickle('../user_data/data/offline/feature.pkl')
    else:
        df.to_pickle('../user_data/data/online/feature.pkl')

    log.info('rank feature done')
