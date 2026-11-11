"""Exact cell closure and held-out four-operator development outcomes."""
import argparse,itertools as it,json
from collections import Counter,defaultdict
from pathlib import Path
from rules import OPS,FACES,active_faces
HERE=Path(__file__).resolve().parent


def rank(columns):
    pivots={}
    for v in columns:
        while v:
            j=v.bit_length()-1
            if j in pivots:v^=pivots[j]
            else:pivots[j]=v;break
    return len(pivots)


def betti(scores,theta=.8):
    vertices=[v for v in OPS if scores[v]>=theta];edges=[''.join(e) for e in it.combinations(vertices,2) if scores[''.join(e)]>=theta];faces=[f for f in FACES if all(v in vertices for v in f) and scores[f]>=theta and all(''.join(e) in edges for e in it.combinations(f,2))]
    d1=[sum(1<<vertices.index(v) for v in e) for e in edges];d2=[sum(1<<edges.index(''.join(e)) for e in it.combinations(f,2)) for f in faces];r1=rank(d1);r2=rank(d2)
    return [len(vertices)-r1,len(edges)-r1-r2,len(faces)-r2]
