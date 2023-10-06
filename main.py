from timeit import timeit
import random

from pylib.treeness import treeness as shim_treeness
from pylib import shim

import alifedata_phyloinformatics_convert as apc
import dendropy as dp
from dendropy.calculate.treemeasure import treeness as dp_treeness


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
    shim_result = shim_treeness(shim_tree)

    print(f"{dp_result=} {shim_result=}")

    shim_time = timeit(lambda: shim_treeness(shim_tree), number=10**3)
    dp_time = timeit(lambda: dp_treeness(dp_tree), number=10**3)
    print(f"{dp_time=} {shim_time=}")
