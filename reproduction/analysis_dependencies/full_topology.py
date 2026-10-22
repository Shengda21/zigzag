"""Exact F2 homology maps and original-node zigzag interval decomposition.
No simplex micro-update ordering is introduced. Endpoint counts also have an
exact rank formula, independently checked against interval decompositions.
"""
from functools import lru_cache
from itertools import product
import json,random
import numpy as np
from local_topology import CELLS
DIMS=[0]*4+[1]*5+[2]*2
BOUND=[0,0,0,0,(1<<0)|(1<<1),(1<<1)|(1<<2),(1<<0)|(1<<2),(1<<0)|(1<<3),(1<<3)|(1<<2),(1<<4)|(1<<5)|(1<<6),(1<<6)|(1<<7)|(1<<8)]
CLOSURE=[]
for i,b in enumerate(BOUND):
 c=(1<<i)|b
 for j in range(i):
  if b>>j&1:c|=CLOSURE[j]
 CLOSURE.append(c)

def rank(vectors):
 piv={}
 for x in vectors:
  x=int(x)
  while x:
   j=x.bit_length()-1
   if j in piv:x^=piv[j]
   else:piv[j]=x;break
 return len(piv)

def basis(vectors):
 selected=[]
 for x in vectors:
  if rank(selected+[x])>len(selected):selected.append(x)
 return selected

def span(b):
 result=[0]
 for x in b:result += [y^x for y in result]
 return result

def boundary(chain):
 ans=0
 for i,b in enumerate(BOUND):
  if chain>>i&1:ans^=b
 return ans

def is_closed(mask):return all(not(mask>>i&1) or mask&c==c for i,c in enumerate(CLOSURE))

def close_scores(scores,threshold):
 raw=sum((1<<i) for i,x in enumerate(scores) if x>=threshold)
 return sum((1<<i) for i,c in enumerate(CLOSURE) if raw&c==c)

@lru_cache(None)
def homology(mask,k):
 mask=int(mask);assert is_closed(mask)
 chains=[1<<i for i,d in enumerate(DIMS) if d==k and mask>>i&1]
 bd=basis([BOUND[i] for i,d in enumerate(DIMS) if d==k+1 and mask>>i&1])
 cycles=[x for x in span(chains) if boundary(x)==0];reps=[];combined=bd[:]
 for x in cycles:
  if rank(combined+[x])>len(combined):reps.append(x);combined.append(x)
 coord={}
 for h,z in enumerate(span(reps)):
  for b in span(bd):coord[z^b]=h
 assert len(cycles)==len(coord)
 return tuple(reps),coord

@lru_cache(None)
def induced(source,target,k):
 source=int(source);target=int(target);assert source&target==source
 reps,_=homology(source,k);_,coord=homology(target,k)
 return tuple(coord[r] for r in reps)

def augmented(masks,bridge='union'):
 out=[]
 for i,m in enumerate(masks):
  if i:out.append(int(masks[i-1])|int(m) if bridge=='union' else int(masks[i-1])&int(m))
  out.append(int(m))
 return out

def diagram(masks,k=1,bridge='union'):
 nodes=augmented(masks,bridge);dims=[len(homology(m,k)[0]) for m in nodes];arrows=[]
 for i in range(len(nodes)-1):
  forward=(i%2==0)==(bridge=='union');s,t=(i,i+1) if forward else (i+1,i)
  arrows.append((s,t,induced(nodes[s],nodes[t],k)))
 return dims,arrows

def nullspace(equations,n):
 rows=[int(x) for x in equations if x];pivots=[];pos=0
 for j in range(n):
  chosen=next((i for i in range(pos,len(rows)) if rows[i]>>j&1),None)
  if chosen is None:continue
  rows[pos],rows[chosen]=rows[chosen],rows[pos]
  for i in range(len(rows)):
   if i!=pos and rows[i]>>j&1:rows[i]^=rows[pos]
  pivots.append(j);pos+=1
 free=[j for j in range(n) if j not in pivots];out=[]
 for j in free:
  x=1<<j
  for row,p in zip(rows,pivots):
   if row>>j&1:x|=1<<p
  out.append(x)
 return out

def generalized_rank(dims,arrows,a,b):
 offsets={};n=0
 for i in range(a,b+1):offsets[i]=n;n+=dims[i]
 if n==0:return 0
 equations=[];relations=[]
 for s,t,cols in arrows:
  if not (a<=s<=b and a<=t<=b):continue
  for j,col in enumerate(cols):relations.append((1<<(offsets[s]+j))^(int(col)<<offsets[t]))
  for j in range(dims[t]):
   row=1<<(offsets[t]+j)
   for l,col in enumerate(cols):
    if col>>j&1:row^=1<<(offsets[s]+l)
   equations.append(row)
 kernel=nullspace(equations,n)
 images=[x&((1<<dims[a])-1) for x in kernel]
 return rank(relations+images)-rank(relations)

def decompose(dims,arrows):
 n=len(dims);r={}
 for a in range(n):
  for b in range(a,n):r[a,b]=generalized_rank(dims,arrows,a,b)
 bars=[]
 for a in range(n):
  for b in range(a,n):
   count=r[a,b]-r.get((a-1,b),0)-r.get((a,b+1),0)+r.get((a-1,b+1),0)
   assert count>=0,(a,b,count)
   bars.extend([(a,b)]*count)
 assert all(sum(a<=i<=b for a,b in bars)==dims[i] for i in range(n))
 return bars

@lru_cache(maxsize=50000)
def barcode(masks,k=1,bridge='union'):
 return tuple(decompose(*diagram(masks,k,bridge)))

def endpoints(masks,k=1,bridge='union',exclude_boundary=False):
 dims,arrows=diagram(masks,k,bridge);ends=np.array(dims,dtype=int)*2
 for s,t,cols in arrows:
  r=rank(cols);ends[s]-=r;ends[t]-=r
 if exclude_boundary:ends[0]=0;ends[-1]=0
 return np.array([sum(ends[2*i:2*i+3]) for i in range(len(masks)-1)])

def enriched_bars(masks,steps,k,bridge):
 other='intersection' if bridge=='union' else 'union';bars=barcode(tuple(masks),k,bridge);otherbars=barcode(tuple(masks),k,other)
 # Endpoint/support signature comparison is descriptive, not cross-bridge class identity.
 out=[]
 for a,b in bars:
  real=[i//2 for i in range(a,b+1) if i%2==0]
  out.append(dict(start_node=a,end_node=b,start_kind='checkpoint' if a%2==0 else 'bridge',end_kind='checkpoint' if b%2==0 else 'bridge',real_checkpoint_steps=[steps[i] for i in real],transition_only=not real,left_censored=a==0,right_censored=b==2*len(masks)-2,bridge_signature_sensitive=bars.count((a,b))!=otherbars.count((a,b))))
 return out

def tests():
 # Validate boundary squared and all closed subcomplexes.
 assert all(boundary(b)==0 for b in BOUND)
 complexes=[m for m in range(1<<11) if is_closed(m)]
 for m in complexes:
  betti=[len(homology(m,k)[0]) for k in (0,1,2)]
  assert sum((-1)**DIMS[i] for i in range(11) if m>>i&1)==betti[0]-betti[1]+betti[2]
 A=CLOSURE[9]^(1<<9);B=CLOSURE[10]^(1<<10)
 assert barcode((A,A))==((0,2),)
 assert barcode((A,B))==((0,1),(1,2))
 edge1=A^(1<<4);edge2=A^(1<<5)
 assert barcode((edge1,edge2))==((1,1),)
 assert barcode((A,A|(1<<9)))==((0,0),)
 assert barcode((A,A^(1<<4)))==((0,1),)
 outer=15|sum(1<<i for i in [4,5,7,8]);assert len(homology(outer,1)[0])==1
 assert [len(homology(2047,k)[0]) for k in (0,1,2)]==[1,0,0]
 # Independent known-interval quivers test generalized-rank decomposition.
 rng=random.Random(1729)
 for _ in range(100):
  n=rng.randrange(2,10);truth=[]
  for j in range(rng.randrange(1,12)):
   a=rng.randrange(n);b=rng.randrange(a,n);truth.append((a,b))
  active=[[j for j,(a,b) in enumerate(truth) if a<=i<=b] for i in range(n)];dims=list(map(len,active));arrows=[]
  for i in range(n-1):
   s,t=(i,i+1) if rng.randrange(2) else (i+1,i)
   arrows.append((s,t,tuple((1<<active[t].index(j)) if j in active[t] else 0 for j in active[s])))
  assert sorted(decompose(dims,arrows))==sorted(truth)
 for _ in range(100):
  masks=tuple(rng.choice(complexes) for _ in range(4))
  for k,bridge in product((0,1),('union','intersection')):
   bars=barcode(masks,k,bridge)
   for cens in (False,True):
    expected=[]
    for i in range(3):expected.append(sum(2*i<=x<=2*i+2 and (not cens or x not in (0,6)) for ab in bars for x in ab))
    assert np.array_equal(endpoints(masks,k,bridge,cens),expected)
 return dict(closed_complexes=len(complexes),known_interval_quivers=100,random_diagrams=400,boundary_squared_zero=True,outer_ring=True,endpoint_identity=True,toy_checks='passed')

if __name__=='__main__':print(json.dumps(tests(),indent=2))

