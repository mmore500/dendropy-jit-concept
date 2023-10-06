from timeit import timeit

from pylib.treeness import treeness as shim_treeness
from pylib import shim

import alifedata_phyloinformatics_convert as apc
import dendropy as dp
from dendropy.calculate.treemeasure import treeness as dp_treeness


if __name__ == "__main__":

    # Number of taxa (tips) you want in your tree
    num_taxa = 500

    # Generate a random tree with 5 taxa
    # taxa = dendropy.TaxonNamespace(['t{0}'.format(i) for i in range(num_taxa)])
    dp_tree = dp.simulate.treesim.birth_death_tree(
        birth_rate=1.0,
        death_rate=0.5,
        max_time=6.0,
    )
    for node in dp_tree:
        node.edge.length = 1
    dp_result = dp_treeness(dp_tree)

    shim_tree = shim.Tree(
        apc.RosettaTree(dp_tree).as_alife,
    )
    shim_result = shim_treeness(shim_tree)

    print(f"{num_taxa=} {dp_result=} {shim_result=}")
    print(f"{timeit(lambda: shim_treeness(shim_tree))=}")
    print(f"{timeit(lambda: dp_treeness(dp_tree))=}")
