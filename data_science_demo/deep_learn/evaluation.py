from argparse import ArgumentParser
import numpy as np
import pandas as pd


##################################################
# Evaluation for regression
##################################################
def rmse(y_trues, y_preds, k=None):
    """
        Args:
            y_trues (list): label scores in each class
                            [[1, 0, 1], [1, 1, 0], ...]
            y_preds (list): predict scores in each class
                            [[1, 3, 2], [2, 1, 1], ...]
            k       (int):  at top K
    """
    num = k or len(y_preds[0])
    rmse = []
    for y_true, y_pred in zip(y_trues, y_preds):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        index = y_true == 1

        # Positive
        postive_scores = y_pred[index]
        pos_mse = np.sum((1 - postive_scores) ** 2)

        # Negative
        negative_scores = y_pred[~index]
        if k:
            negative_scores = np.sort(negative_scores)
            negative_scores = negative_scores[-k+sum(index):]
        neg_mse = np.sum((0 - negative_scores) ** 2)

        rmse.append(pos_mse + neg_mse)

    rmse = np.sqrt(np.asarray(rmse) / num)
    mean_rmse = np.mean(rmse)
    return mean_rmse


##################################################
# Evaluation for ranking
##################################################
# F1-score @ K
def f1_score_k(y_trues, y_preds, k=None):
    p = precision_k(y_trues, y_preds, k)
    r = recall_k(y_trues, y_preds, k)
    f1 = (2 * p * r) / (p + r + 1e-15)
    return f1


# Precision @ K
def precision_k(y_trues, y_preds, k=None):
    """
        Precision @ K =
        Num(relevant items in y_pred at top-k) / Num(items in y_pred at top-k)
        Args:
            y_trues (list): each sample's label indexes
                            [[1, 3, 2], [2, 1], ...]
            y_preds (list): each sample's predict indexes
            k       (int):  Precision@K
        Returns:
            avg_precision_k (float): average precision as top-K
    """

    precision = []
    for y_true, y_pred in zip(y_trues, y_preds):
        y_pred = y_pred[:k] if k else y_pred
        p = len(np.intersect1d(y_true, y_pred)) / (len(y_pred) + 1e-14)
        precision.append(p)
    avg_precision_k = np.mean(precision).item()
    return avg_precision_k


# Recall @ K
def recall_k(y_trues, y_preds, k=None):
    """
        Recall @ K =
        Num(relevant items in y_pred at top-k) / Num(total relevant items)
        Args:
            y_trues (list): each sample's label indexes
                            [[1, 3, 2], [2, 1], ...]
            y_preds (list): each sample's predict indexes
            k       (int):  Precision@K
        Returns:
            avg_recall_k (float): average recall as top-K
    """

    recall = []
    for y_true, y_pred in zip(y_trues, y_preds):
        y_pred = y_pred[:k] if k else y_pred
        p = len(np.intersect1d(y_true, y_pred)) / (len(y_true) + 1e-14)
        recall.append(p)
    avg_recall_k = np.mean(recall).item()
    return avg_recall_k


##################################################
# Evaluation for multi-label classification
##################################################
# Accuracy @ K
def accuracy_k(y_trues, y_preds, k=None):
    """
        Num(correct) / Num(y)
        correct: y_pred is correct when items in y_pred at top-k
                 are include one of item in y_pred
    """
    accuracy = 0
    for y_true, y_pred in zip(y_trues, y_preds):
        y_pred = y_pred[:k] if k else y_pred
        correct = len(np.intersect1d(y_true, y_pred)) > 0
        accuracy = accuracy + 1 if correct else accuracy

    return accuracy / len(y_trues)


# GAP @ K
def gap(y_trues, y_preds, top_k=20):
    """ Global average precision in top K
        Args:
        y_preds (np.array): confidence (probabilities).
                            dimension is (batch, num_classes)
        y_trues (np.array): one-hot format
                            dimension is (batch, num_classes)
    """
    # Ranking by confidence in each videos
    index = np.argsort(-y_preds, axis=1)
    for i, idx in enumerate(index):
        y_preds[i] = y_preds[i, idx]
        y_trues[i] = y_trues[i, idx]
    # Pick top k
    y_preds = y_preds[:, :top_k]
    y_trues = y_trues[:, :top_k]

    # Ranking by confidence (global)
    correct = y_trues.reshape(-1)
    y_preds = y_preds.reshape(-1)
    idx = np.argsort(-y_preds)
    y_preds = y_preds[idx]
    correct = correct[idx]

    # Precision
    precision = correct.cumsum() / (np.arange(len(correct)) + 1)
    # Delta recall
    delta_recall = correct / correct.sum()
    # GAP = SUM(Precision * Delta recall)
    gap = precision * delta_recall
    return gap.sum()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ground_truth_path', '-gt', type=str,
                        help="file path of ground truth reslut (txt)")
    parser.add_argument('--predict_path', '-p', type=str,
                        help="file path of predicted reslut (txt)")
    parser.add_argument('--eval', '-e', type=str,
                        help="evaluation function")
    parser.add_argument('--K', '-k', default=None, type=int,
                        help="@K")
    parser.add_argument('--file_type', '-type', default='txt', type=str)
    args = parser.parse_args()
    gt_path = args.ground_truth_path
    predict_path = args.predict_path
    eval_function = args.eval
    k = args.K
    file_type = args.file_type

    print(">>> Evaluation")
    ##################################################
    # Load ground truth and prediction results
    ##################################################
    print(f"Load {gt_path}")
    print(f"Load {predict_path}")
    if file_type == 'txt':
        with open(gt_path, 'r') as f:
            y_trues = [d.split()[-1].split(',') for d in f.read().split('\n')]
        with open(predict_path, 'r') as f:
            y_preds = [d.split()[-1].split(',') for d in f.read().split('\n')]
    elif file_type == 'csv':
        y_preds = pd.read_csv(predict_path)['LabelConfidencePairs'].tolist()
        y_preds = np.asarray([e.split()[1::2] for e in y_preds], dtype=float)

        indexes = pd.read_csv(gt_path)['Labels'].tolist()
        indexes = [list(map(int, e.split())) for e in indexes]
        y_trues = np.zeros_like(y_preds)
        for y_true, idx in zip(y_trues, indexes):
            y_true[idx] = 1

    ##################################################
    # Evaluation
    ##################################################
    if eval_function == 'f1_score_k':
        result = f1_score_k(y_trues, y_preds, k=k)
    elif eval_function == 'precision_k':
        result = precision_k(y_trues, y_preds, k=k)
    elif eval_function == 'recall_k':
        result = recall_k(y_trues, y_preds, k=k)
    elif eval_function == 'gap':
        result = gap(y_trues, y_preds, top_k=k)

    # Show results
    print(f"{eval_function}: K={k}")
    print(f"{result:.6f}\n")
