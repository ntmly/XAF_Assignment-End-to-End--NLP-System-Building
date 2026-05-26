import argparse
import re
from typing import List, Tuple
from sklearn.metrics import f1_score

def normalize_answer(s: str) -> str:
    """Chuẩn hóa câu trả lời để so sánh"""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def exact_match(pred: str, gold: str) -> int:
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0

def f1_score_per_pair(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate(pred_file: str, gold_file: str) -> Tuple[float, float]:
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f if line.strip()]
    with open(gold_file, 'r', encoding='utf-8') as f:
        golds = [line.strip() for line in f if line.strip()]
    
    assert len(preds) == len(golds), "Số lượng câu trả lời không khớp"
    
    em_sum = 0
    f1_sum = 0
    for pred, gold in zip(preds, golds):
        em_sum += exact_match(pred, gold)
        f1_sum += f1_score_per_pair(pred, gold)
    
    em = em_sum / len(preds)
    f1 = f1_sum / len(preds)
    
    return em, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    args = parser.parse_args()
    
    em, f1 = evaluate(args.pred, args.gold)
    print(f"Exact Match (EM): {em:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()