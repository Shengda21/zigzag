# Two-cell deletion: exploratory alpha1 audit

All six full-complex trajectories at theta=.8: 0/294 alpha mismatches. Three independent worlds; checkpoints/seeds are repeated observations.

W=sum Z_k(L_s) contains B_k(U), since the boundary of each (k+1)-cell lies in a checkpoint containing that cell. Hence S=W/B_k(U), H_k(U)/S is isomorphic to Z_k(U)/W, and alpha=dim Z_k(U)-dim W. Alpha_k depends only on checkpoint k-skeleta.

| Run | Final alpha original/deleted | Final beta original/deleted | Checkpoints with beta difference |
|---|---:|---:|---:|
| world635_S1 | 3/3 | 6/7 | 35 |
| world635_S2 | 3/3 | 6/7 | 35 |
| world636_S1 | 3/3 | 6/7 | 34 |
| world636_S2 | 3/3 | 6/7 | 36 |
| world637_S1 | 3/3 | 6/7 | 34 |
| world637_S2 | 3/3 | 6/7 | 35 |

At world636-S1 step240, original killed/becomes-supported = 1/0; after deletion = 0/1. These values are computed from the equality-condition subspace ranks. The original killing description is technically correct for those maps; it is not evidence that alpha detects fillings or that filling is necessary for its change.

{
  "original": {
    "loss_of_unsupported_directions": 1,
    "previous_artifact_becomes_supported": 0
  },
  "deleted": {
    "loss_of_unsupported_directions": 0,
    "previous_artifact_becomes_supported": 1
  }
}

All 294 alpha/beta pairs and historical connecting ranks are in two_cell_deletion.json. No training, GPU, outcome fitting, manuscript edits or confirmation claims. The original audit script/results are unchanged.

Run: python two_cell_deletion.py. Requires the existing repository routines and NumPy. Outputs contain no absolute paths, timestamps or random computations, allowing exact numeric comparison between both package locations.