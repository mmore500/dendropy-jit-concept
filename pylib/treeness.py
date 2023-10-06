from numba import njit

from .shim import Tree


@njit
def treeness(tree: Tree):
    """
    Returns the proportion of total tree length that is taken up by
    internal branches.
    """
    internal = 0.0
    external = 0.0
    for nd in tree.postorder_node_iter():
        if not nd._parent_node:
            continue
        if nd.is_leaf():
            external += nd.edge.length
        else:
            internal += nd.edge.length
    return internal / (external + internal)
