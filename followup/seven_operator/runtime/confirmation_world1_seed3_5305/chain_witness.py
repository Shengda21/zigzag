"""Exact labelled chain witnesses for the pre-specified ACEF bridge class."""
import itertools as it,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;V='ABCDEFG';E=tuple(map(''.join,it.combinations(V,2)));F=tuple(map(''.join,it.combinations(V,3)))


def insert(basis,v):
    while v:
        j=v.bit_length()-1
        if j in basis:v^=basis[j]
        else:basis[j]=v;return True
    return False


def member(basis,v):
    while v:
        j=v.bit_length()-1
        if j not in basis:return False
        v^=basis[j]
    return True


def boundary2(f):return sum(1<<E.index(''.join(e)) for e in it.combinations(f,2))


def cycles2(faces):
    pivots={};cycles=[]
    for f in sorted(faces):
        chain=1<<F.index(f);v=boundary2(f)
        while v:
            j=v.bit_length()-1
            if j in pivots:v^=pivots[j][0];chain^=pivots[j][1]
            else:pivots[j]=(v,chain);break
        if not v:cycles.append(chain)
    return cycles


def witness(A,B):
    I=A&B;U=A|B;span={}
    for v in [*cycles2(A),*cycles2(B)]:insert(span,v)
    target=sum(1<<F.index(''.join(f)) for f in it.combinations('ACEF',3));faces={F[j] for j in range(len(F)) if target>>j&1};delta=boundary2('ACF');bounds={}
    for f in I:insert(bounds,boundary2(f))
    left=boundary2('AEF')^boundary2('ACE')^boundary2('CEF');right=boundary2('ACF')
    out={'A_H2':len(cycles2(A)),'B_H2':len(cycles2(B)),'intersection_H2':len(cycles2(I)),'union_H2':len(cycles2(U)),'endpoint_cycle_image_span':len(span),'connecting_rank':len(cycles2(U))-len(span),'ACEF_faces':sorted(faces),'ACEF_in_union':faces<=U,'ACEF_in_A':faces<=A,'ACEF_in_B':faces<=B,'ACEF_in_endpoint_span':member(span,target),'connecting_cycle_edges':['AC','AF','CF'],'connecting_cycle_nonboundary_in_intersection':not member(bounds,delta),'A_filling':['AEF','ACE','CEF'],'B_filling':['ACF'],'fillings_checked':left==right==delta and {'AEF','ACE','CEF'}<=A and 'ACF' in B}
    out['all_witness_checks_pass']=out['connecting_rank']==1 and out['ACEF_in_union'] and not out['ACEF_in_A'] and not out['ACEF_in_B'] and not out['ACEF_in_endpoint_span'] and out['connecting_cycle_nonboundary_in_intersection'] and out['fillings_checked']
    return out
