from sklearn import metrics
def AUC(g_o):
    y_true,y_score=[],[]
    for u,v,d in g_o.edges(data=True):
        y_true.append(d['label'])
        y_score.append(d['score'])

    auc=metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    return auc

