from timeit import timeit
import random

import alifedata_phyloinformatics_convert as apc
import dendropy as dp
from dendropy.calculate.treemeasure import treeness as dp_treeness
import numba as nb

from pylib import shim


if __name__ == "__main__":

    # Generate a random tree
    dp_tree = dp.model.coalescent.pure_kingman_tree_shape(
        num_leaves=2000,
        pop_size=2000,
        rng=random.Random(1),
    )
    print(f"{len(dp_tree.nodes())=}")
    for node in dp_tree:
        node.edge.length = 1
    dp_result = dp_treeness(dp_tree)

    alifestd_df = apc.RosettaTree(dp_tree).as_alife
    print(f"{len(alifestd_df)=}")
    shim_tree = shim.Tree(alifestd_df)

    jit_treeness = nb.njit(dp_treeness)  # <--- no internal modifications
    shim_result = jit_treeness(shim_tree)

    print(f"{dp_result=} {shim_result=}")

    shim_time = timeit(lambda: jit_treeness(shim_tree), number=10**3)
    dp_time = timeit(lambda: dp_treeness(dp_tree), number=10**3)
    print(f"{dp_time=} {shim_time=}")
