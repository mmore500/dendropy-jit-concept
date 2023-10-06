import typing

from hstrat import _auxiliary_lib as hstrat_aux
from numba import int32 as nb_int32
from numba.experimental import jitclass
import numpy as np
import pandas as pd


@jitclass({})
class Edge:
    def __init__(self: "Edge") -> None:
        pass

    @property
    def length(self: "Edge") -> int:
        return 1


@jitclass(
    {
        "_ancestor_ids": nb_int32[:],
        "_pos": nb_int32,
    },
)
class Node:

    _ancestor_ids: np.array
    _pos: int

    def __init__(
        self: "Node",
        ancestor_ids: np.array,
        pos: int,
    ) -> None:
        self._ancestor_ids = ancestor_ids
        self._pos = pos

    @property
    def _parent_node(self: "Node") -> typing.Optional["Node"]:
        pos = self._pos
        ancestor_id = self._ancestor_ids[pos]
        if ancestor_id == pos:
            return None
        else:
            return Node(
                self._ancestor_ids,
                ancestor_id,
            )

    @property
    def edge(self: "Node") -> Edge:
        return Edge()

    def is_leaf(self: "Node") -> bool:
        pos = self._pos
        return not pos in self._ancestor_ids


@jitclass(
    {
        "_ancestor_ids": nb_int32[:],
        "_pos": nb_int32,
    },
)
class _Tree:

    _ancestor_ids: np.array

    def __init__(
        self: "_Tree",
        ancestor_ids: pd.DataFrame,
    ) -> None:
        self._ancestor_ids = ancestor_ids

    def postorder_node_iter(self: "_Tree") -> typing.Iterator[Node]:
        size = len(self._ancestor_ids)
        for pos in range(size, 0, -1):
            yield Node(self._ancestor_ids, pos)


def Tree(data: pd.DataFrame) -> _Tree:
    ancestor_ids = hstrat_aux.alifestd_to_working_format(
        data,
    )["ancestor_id"].astype("int32").values
    return _Tree(ancestor_ids)
