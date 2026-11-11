"""Independent three-symbol context rules on seven primitive operators."""
import itertools as it,random
OPS='ABCDEFG'
FACES=tuple(map(''.join,it.combinations(OPS,3)))


def primitive(x,g,world=1):
    a=list(x)
    if world==1:
        if g=='A':return tuple(a[::-1])
        if g=='B':a[0],a[3]=a[3],a[0]
        elif g=='C':a[0]=(a[0]+3)%8
        elif g=='D':
            if a[0]>a[1]:a[2],a[3]=a[3],a[2]
        elif g=='E':a[:4]=a[1:4]+a[:1]
        elif g=='F':a[:3]=a[1:3]+a[:1]
        elif g=='G':a[5],a[7]=a[7],a[5]
    else:
        if g=='A':return tuple(a[1:]+a[:1])
        if g=='B':a[:3]=a[1:3]+a[:1]
        elif g=='C':a[4]=(a[4]+3)%8
        elif g=='D':
            if a[0]>a[1]:a[5],a[7]=a[7],a[5]
        elif g=='E':a[:4]=a[:4][::-1]
        elif g=='F':a=[a[j] for j in (2,1,4,3,6,5,0,7)]
        elif g=='G':a[1],a[6]=a[6],a[1]
    return tuple(a)


def coefficients(world):
    rng=random.Random(511020261110+world)
    return {f:tuple(rng.randrange(8) for _ in range(8)) for f in FACES}


def execute(root,word,coef,world=1):
    x=tuple(root);out=[]
    for i,g in enumerate(word):
        x=primitive(x,g,world)
        if i>=2 and len(set(word[i-2:i+1]))==3:
            v=coef[''.join(sorted(word[i-2:i+1]))]
            x=tuple(a^b for a,b in zip(x,v))
        out.append(x)
    return out


PERMUTATIONS={
 1:{'A':(7,6,5,4,3,2,1,0),'B':(3,1,2,0,4,5,6,7),'E':(1,2,3,0,4,5,6,7),'F':(1,2,0,3,4,5,6,7),'G':(0,1,2,3,4,7,6,5)},
 2:{'A':(1,2,3,4,5,6,7,0),'B':(1,2,0,3,4,5,6,7),'E':(3,2,1,0,4,5,6,7),'F':(2,1,4,3,6,5,0,7),'G':(0,6,2,3,4,5,1,7)}}


def reference(root,word,coef,world=1):
    """Independent table-based primitive and set-based context interpreter."""
    x=tuple(root);history=[];out=[]
    for g in word:
        if g in PERMUTATIONS[world]:x=tuple(x[i] for i in PERMUTATIONS[world][g])
        elif g=='C':
            j=0 if world==1 else 4;x=tuple(((v+3)&7) if i==j else v for i,v in enumerate(x))
        elif g=='D':
            i,j=(2,3) if world==1 else (5,7)
            if x[0]>x[1]:x=tuple(x[j] if k==i else x[i] if k==j else x[k] for k in range(8))
        else:raise ValueError(g)
        recent=history[-2:]+[g]
        if len(recent)==3 and len(frozenset(recent))==3:
            v=coef[''.join(sorted(frozenset(recent)))];x=tuple(x[j]^v[j] for j in range(8))
        history.append(g);out.append(x)
    return out


def active_faces(word):
    return {''.join(sorted(word[i:i+3])) for i in range(len(word)-2) if len(set(word[i:i+3]))==3}


def prompt(root,word):
    return 'Input: '+' '.join(map(str,root))+'\nProgram: '+' '.join(word)+'\nPrint the state after each instruction. Separate states by |.'


def answer(trace):return ' | '.join(' '.join(map(str,x)) for x in trace)
