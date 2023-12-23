from rouge import Rouge 
import numpy as np
from nltk.translate import meteor_score
from nltk import word_tokenize

def compute_rouge_score(references_batch, candidates_batch):
    # 初始化 ROUGE
    rouge = Rouge(metrics=['rouge-l'])
    scores = []
    assert len(references_batch)==len(candidates_batch)
    for i in range(len(references_batch)):
        
            cand_str=candidates_batch[i]
            ref_str=references_batch[i]
            # 计算 ROUGE-L 分数
            try:
                rouge_l_score = rouge.get_scores(cand_str, ref_str)
                f_measure = rouge_l_score[0]['rouge-l']['f']
                scores.append(f_measure)
            except:
                scores.append(0.0)
    return np.mean(scores)

def compute_meteor_score(references, candidates):
    # 计算 METEOR 分数
    scores = [
        meteor_score.meteor_score([ref], cand)
        for ref, cand in zip(references, candidates)
    ]
    return np.mean(scores)