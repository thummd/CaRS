import numpy as onp
from sklearn import metrics as sklearn_metrics
import networkx as nx



def define_big_mat(N,regime, L,g_list,tr_lab):
            if tr_lab ==False:
                        all_estimated_g = onp.zeros((N*regime,N*regime))
                        all_estimated_g_lag = onp.zeros((N*regime,N*regime))
                        for i in range(regime):
                                    all_estimated_g[i*N:(i+1)*N,i*N:(i+1)*N] = L[i][0]



                                    all_estimated_g_lag[i*N:(i+1)*N,i*N:(i+1)*N] = L[i][1]
                        all_estimated_g[all_estimated_g<0.4] = 0
                        all_estimated_g[all_estimated_g>0.4] = 1

                        all_estimated_g_lag[all_estimated_g_lag<0.4] = 0
                        all_estimated_g_lag[all_estimated_g_lag>0.4] = 1
                        return all_estimated_g,all_estimated_g_lag
            else:
                        g_3_regime = onp.zeros((N*regime,N*regime))
                        g_3_regime_lag = onp.zeros((N*regime,N*regime))
                        for i in range(regime):
                                    g_3_regime[(i)*N:(i+1)*N,(i)*N:(i+1)*N] = nx.to_numpy_array(g_list[i])[N:2*N,N:2*N]
                                    g_3_regime_lag[(i)*N:(i+1)*N,(i)*N:(i+1)*N] = nx.to_numpy_array(g_list[i])[:N,N:2*N]
                                    
                        return g_3_regime,g_3_regime_lag

def shd(g, h):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    Args:
        g:  [..., d, d]
        h:  [..., d, d]
    """
    assert g.ndim == h.ndim
    abs_diff =  onp.abs(g - h)
    mistakes = abs_diff + onp.swapaxes(abs_diff, -2, -1)  # mat + mat.T (transpose of last two dims)

    # ignore double edges
    mistakes_adj = onp.where(mistakes > 1, 1, mistakes)

    return onp.triu(mistakes_adj).sum((-1, -2))


def n_edges(g):
    """
    Args:
        g:  [..., d, d]
    """
    return g.sum((-1, -2))


def is_acyclic(g):
    """
       Args:
           g:  [d, d]
       """
    n_vars = g.shape[-1]
    mat = onp.eye(n_vars) + g / n_vars
    mat_pow = onp.linalg.matrix_power(mat, n_vars)
    acyclic_constr = onp.trace(mat_pow) - n_vars
    return onp.isclose(acyclic_constr, 0.0)


def is_cyclic(g):
    """
    Args:
        g:  [d, d]
    """
    return not is_acyclic(g)


def classification_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            true_flat, pred_flat, average="binary")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    else:
        # no true positives, but we predicted some positives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def threshold_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        fpr, tpr, _ = sklearn_metrics.roc_curve(true_flat, pred_flat)
        precision, recall, _ = sklearn_metrics.precision_recall_curve(true_flat, pred_flat)
        ave_prec = sklearn_metrics.average_precision_score(true_flat, pred_flat)
        roc_auc = sklearn_metrics.auc(fpr, tpr)
        prc_auc = sklearn_metrics.auc(recall, precision)

        return {
            "auroc": roc_auc,
            "auprc": prc_auc,
            "ap": ave_prec,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "auroc": 1.0,
            "auprc": 1.0,
            "ap": 1.0,
        }

    else:
        # no true positives, but we predicted some positives
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "ap": 0.0,
        }
        
def normalized_hamming_distance(prediction, target):
  '''
  prediction and target are edge lists
  calculate the normalized hamming distance

  For a graph with m nodes, the distance is given by ∑m i,j=1 1 m2 1Gij 6=G′ ij , 
  the number of edges that are present in one graph but not the other, 
  divided by the total number of all possible edges.
  '''
  prediction = set(prediction)
  target = set(target)
  total_nodes = set()
  for i,j in target:
    total_nodes.add(i)
    total_nodes.add(j)
  no_overlap = len(prediction.union(target)) - len(prediction.intersection(target))
  nhd = no_overlap / (len(total_nodes) ** 2)
  reference_nhd = (len(prediction) + len(target))/ (len(total_nodes) ** 2)
  return nhd, reference_nhd, nhd / reference_nhd

def adj_mat_to_edge_list(adj_mat):
    n = adj_mat.shape[0]
    edge_list = []
    for i in range(n):
        for j in range(n):
            if adj_mat[i][j] == 1:
                edge_list.append((i, j))
    return edge_list