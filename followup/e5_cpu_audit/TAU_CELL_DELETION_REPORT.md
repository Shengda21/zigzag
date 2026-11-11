# Tau and one-sided kappa: exploratory 2-cell deletion audit

Let Z+=Z_k(w_high), W=Z_k(A_low)+Z_k(B_low), and D=B_k(w_low), all embedded in C_k(w_low). Every (k+1)-cell in w_low belongs to A_low or B_low, so its boundary belongs to the corresponding cycle space, giving D subset W. Under q:Z_k(w_low)->H_k(w_low), I=q(Z+) and J=q(W)=W/D. Therefore dim(I+J)-dim J=dim(Z+ + W)-dim W. Removing (k+1)-cells changes neither Z+ nor W, proving tau invariance. The argument applies to any degree over the fixed field, with subcomplex inclusions and cellular chains in one ambient complex.

All six trajectories and 288 original bridges at each frozen scale delta=0/.05: 0/576 tau mismatches.

Observed difference counts by scale:
{
  "0": {
    "tau": 0,
    "left": 7,
    "right": 4,
    "birth": 7,
    "death": 4
  },
  "0.05": {
    "tau": 0,
    "left": 2,
    "right": 2,
    "birth": 2,
    "death": 2
  }
}

For a one-sided source A_low, D need not be contained in Z_k(A_low): boundaries contributed by B_low can be absent from A_low. Instead kappa_left=dim(Z+ + Z_k(A_low)+D)-dim(Z_k(A_low)+D). Thus one-sided kappa can depend on (k+1)-cells. Since tau is invariant, b=kappa_left-tau and d=kappa_right-tau can also depend on them. A valid counterexample is A a path with two triangle edges and B the filled triangle: original H1(A union B)=0, while deleting B face gives H1=1 and H1(A)=0. Left kappa and birth change 0 to 1, tau stays 0 because B after deletion contains the boundary cycle. Swapping A/B gives the right/death example. Assign all included cells score 1 and absent cells score 0: this works at theta=.8 for both delta=0 and .05.

Validated triangle counterexample:
{
  "original": {
    "tau": 0,
    "left": 0,
    "right": 0,
    "birth": 0,
    "death": 0
  },
  "deleted": {
    "tau": 0,
    "left": 1,
    "right": 0,
    "birth": 1,
    "death": 0
  }
}

All bridge-scale values and every difference are in tau_cell_deletion.json. No empirical prediction, coverage, mechanism or novelty claim follows. No GPU/cloud or manuscript edits. Run python tau_cell_deletion.py.