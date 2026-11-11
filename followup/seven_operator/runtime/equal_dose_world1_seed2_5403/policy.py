import numpy as np

CLASSES='ABCDEFG'


class Policy:
    def __init__(self,path):
        self.weights=np.load(path)
        assert self.weights.shape==(17,9)

    def predict(self,words,histories):
        positions=np.arange(9)[None,:];lengths=np.array([len(w) for w in words]);counts=np.array([len(h) for h in histories]);assert all(lengths<=8) and all(counts<=16)
        logits=self.weights[counts].copy();logits[positions>lengths[:,None]]=-np.inf;attention=np.exp(logits-np.max(logits,axis=1,keepdims=True));attention/=attention.sum(axis=1,keepdims=True);values=np.zeros((len(words),9,8))
        for i,w in enumerate(words):
            for j,g in enumerate(w):values[i,j,CLASSES.index(g)]=1.
            values[i,len(w),7]=1.
        probabilities=np.einsum('bi,bij->bj',attention,values);categories=probabilities.argmax(axis=1)
        return [{'text':CLASSES[c] if c<7 else 'STOP','category_id':int(c),'category_probabilities':p.tolist(),'position_attention':v.tolist()} for c,p,v in zip(categories,probabilities,attention)]

    def certificate(self):
        rows=[]
        for n in range(1,9):
            for k in range(n+1):
                logits=self.weights[k,:n+1];a=np.exp(logits-logits.max());a/=a.sum();rows.append({'length':n,'completed_steps':k,'correct_position_mass':float(a[k])})
        return {'all_assignments_certificate_pass':all(r['correct_position_mass']>.5 for r in rows),'minimum_correct_position_mass':min(r['correct_position_mass'] for r in rows),'checks':rows}
