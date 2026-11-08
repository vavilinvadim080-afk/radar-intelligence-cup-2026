def pair_metrics(gold,pred):
    if len(gold)!=len(pred):raise ValueError('Lengths differ')
    tp=fp=fn=0
    for i in range(len(gold)):
        for j in range(i):
            expected=gold[i] is not None and gold[i]==gold[j]
            got=pred[i] is not None and pred[i]==pred[j]
            tp+=bool(expected and got);fp+=bool(got and not expected);fn+=bool(expected and not got)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                coverage=sum(p is not None for p in pred)/len(pred) if pred else None)
