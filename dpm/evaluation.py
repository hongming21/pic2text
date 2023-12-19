from rouge import Rouge 
import numpy as np
from nltk.translate import meteor_score

def compute_rouge_score(references, candidates):
    # 计算 ROUGE-L 分数
    rouge = Rouge(metrics=['rouge-l'])
    scores = []
    for cand, ref in zip(candidates, references):
        # 保证 cand 和 ref 都得是字符串才行
        rouge_L = [rouge.get_scores(' '.join(cand), ' '.join(ref))]
        # 取每个 ROUGE-L 的平均值？
        f = np.mean([rl[0]['rouge-l']['f'] for rl in rouge_L])
        scores.append(f)
    return np.mean(scores)

def compute_meteor_score(references, candidates):
    # 计算 METEOR 分数
    scores = [
        meteor_score.meteor_score([ref], cand)
        for ref, cand in zip(references, candidates)
    ]
    return np.mean(scores)