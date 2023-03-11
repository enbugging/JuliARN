from sage.all import *
import numpy as np
import time


class PolytopeAlgebra:
    def __init__(self, polytope=None, dimension=2, base_ring=ZZ):
        self.dimension = dimension
        if polytope is None:
            self.polytope = Polyhedron(
                ambient_dim=dimension, base_ring=base_ring)
        else:
            self.polytope = Polyhedron(vertices=polytope, base_ring=base_ring)

    def __mul__(self, other):
        if self.polytope.dim() == -1:
            return other
        elif other.polytope.dim() == -1:
            return self
        else:
            result = PolytopeAlgebra(dimension=self.dimension)
            result.polytope = self.polytope + other.polytope
            return result

    def __add__(self, other):
        result = PolytopeAlgebra(dimension=self.dimension)
        result.polytope = self.polytope.convex_hull(other.polytope)
        return result
