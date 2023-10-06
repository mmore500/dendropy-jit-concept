from timeit import timeit
import random

import alifedata_phyloinformatics_convert as apc
import dendropy as dp
from dendropy.calculate.treemeasure import treeness as dp_treeness
import numba as nb

from pylib import shim


if __name__ == "__main__":

    # PART I: DO DEMO SETUP
    #######################
    # generate a random tree
    dp_tree = dp.model.coalescent.pure_kingman_tree_shape(
        num_leaves=2000,
        pop_size=2000,
        rng=random.Random(1),
    )
    for node in dp_tree:
        node.edge.length = 1  # simplification, for proof of concept
    print(f"{len(dp_tree.nodes())=}")  # check how many nodes in tree

    # prepare numpy-based, jit-compatible Tree shim
    # populated with tree structure from dp_tree
    # (note: shim impl is in other source file)
    alifestd_df = apc.RosettaTree(dp_tree).as_alife
    # ^ first, use existing tool to convert tree to DataFrame format
    print(f"{len(alifestd_df)=}")  # check num nodes in converted tree
    shim_tree = shim.Tree(alifestd_df)

    # create just-in-time compiled (jit) version of library func
    # note that this requires zero internal library modifications
    jit_treeness = nb.njit(dp_treeness)

    # PART II: DO EXAMPLE CALCULATIONS
    ##################################
    # do treeness calculation with pure-py dendropy tree & func
    dp_result = dp_treeness(dp_tree)

    # do treeness calculation with numpy-based tree & jitted func
    # (warms up the jit compiler cache for benchmark below)
    shim_result = jit_treeness(shim_tree)

    # compare pure-py and jitted treeness result, should be identical
    print(f"{dp_result=} {shim_result=}")

    # compare pure-py and jitted treeness performance
    shim_time = timeit(lambda: jit_treeness(shim_tree), number=10**3)
    dp_time = timeit(lambda: dp_treeness(dp_tree), number=10**3)
    print(f"{dp_time=} {shim_time=}")
