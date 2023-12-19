from rouge import Rouge 
import numpy as np
from nltk.translate import meteor_score

def compute_rouge_score(references_batch, candidates_batch):
    # 初始化 ROUGE
    rouge = Rouge(metrics=['rouge-l'])
    scores = []
    for references, candidates in zip(references_batch, candidates_batch):
        for ref, cand in zip(references, candidates):
            # 将词列表转换为字符串
            ref_str = ' '.join(ref)
            cand_str = ' '.join(cand)
            # 计算 ROUGE-L 分数
            rouge_l_score = rouge.get_scores(cand_str, ref_str)
            f_measure = rouge_l_score[0]['rouge-l']['f']
            scores.append(f_measure)
    return np.mean(scores)

def compute_meteor_score(references_batch, candidates_batch):
    scores = []
    for references, candidates in zip(references_batch, candidates_batch):
        for ref, cand in zip(references, candidates):
            # 将词列表转换为字符串
            ref_str = ' '.join(ref)
            cand_str = ' '.join(cand)
            # 计算 METEOR 分数
            score = meteor_score.single_meteor_score(ref_str, cand_str)
            scores.append(score)
    return np.mean(scores)