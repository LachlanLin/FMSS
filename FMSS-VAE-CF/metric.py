import numpy as np
import bottleneck as bn


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=5):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.sum(axis=1).astype(np.int32)])
    return DCG / IDCG


def Recall_Precision_F1_OneCall_at_k_batch(X_pred, heldout_batch, k=5):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / X_true_binary.sum(axis=1)
    precision = tmp / k
    f1 = 2 * recall * precision / (recall + precision)
    oneCall = (tmp > 0).astype(np.float32)
    return recall, precision, f1, oneCall


def AUC_at_k_batch(X_train, X_pred, heldout_batch):
    train_set_num = X_train.sum(axis=1)
    test_set_num = heldout_batch.sum(axis=1)
    sorted_id = np.argsort(X_pred, axis=1)
    rank = np.argsort(sorted_id) + 1
    molecular = (heldout_batch * rank).sum(axis=1) - test_set_num * (
                test_set_num + 1) / 2 - test_set_num * train_set_num
    denominator = (X_pred.shape[1] - train_set_num - test_set_num) * test_set_num
    aucs = molecular / denominator
    return aucs
