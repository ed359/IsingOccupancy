# How to run this code
# Install sagemath (10.2 or later)
# run "load('canaug.pyx')" in a cell to compile the code

import cython

from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy

from cysignals.signals cimport sig_check
from cysignals.memory cimport sig_free, sig_malloc

from sage.data_structures.bitset_base cimport *

from sage.graphs.base.c_graph cimport CGraph
from sage.graphs.base.dense_graph cimport DenseGraph
from sage.graphs.base.dense_graph import DenseGraphBackend
from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs

from sage.groups.perm_gps.partn_ref.data_structures cimport *

from sage.plot.plot import graphics_array

from sage.rings.integer import Integer

from local_view import LocalView

def gen_local_views(d, spin_depth=1, spins=None, spin_orbits=None, forbidden_subgraphs=None, verbose=False, tqdm=None):
    if spins is None:
        spins = [1,2,3,4]
    if type(spins) is Integer or spins is int:
        spins = [i for i in range(1,spins+1)]
    if spin_orbits is None:
        spin_orbits = [[s for s in spins]]
    if forbidden_subgraphs is None:
        forbidden_subgraphs = []
    if tqdm is None:
        tqdm = lambda x, *args, **kwargs: x

    G = Graph(1, data_structure='dense', loops=False, multiedges=False)
    data = [(d, G, [[0]], [0])]

    def add_new_layer_helper(gb):
        d, G, last_partition, last_layer = gb
        yield from add_new_layer_gen(d, G, last_partition, last_layer, forbidden_subgraphs)

    def fill_layer_helper(gb):
        d, G, last_partition, last_layer = gb
        yield from fill_layer_gen(d, G, last_partition, last_layer, forbidden_subgraphs)

    def add_new_layer_disjoint_helper(gb):
        d, G, last_partition, last_layer = gb
        yield from add_new_layer_disjoint_gen(d, G, last_partition, last_layer)

    for i in tqdm(range(spin_depth-1)):
        if verbose:
            print(f'Adding layer {i+1} starting with {len(data)} graphs')
        data = [y for x in data for y in add_new_layer_helper(x)]

        if verbose:
            print(f'Filling layer {i+1} starting with {len(data)} graphs')
        data = [y for x in data for y in fill_layer_helper(x)]

    if verbose:
        print(f'Adding layer {spin_depth} disjointly to {len(data)} graphs')
    data = [y for x in data for y in add_new_layer_disjoint_helper(x)]

    if verbose:
        print(f'Assigning spins to layer {spin_depth} starting with {len(data)} graphs')
    extend = lambda x: (*x, spins, spin_orbits)
    yield from (y for x in map(extend, data) for y in assign_spins_gen(x))

def gen_local_view_1(d, spins, spin_orbits):
    yield from gen_local_views(d, 1, spins, spin_orbits)

def gen_local_view_2(d, spins, spin_orbits):
    yield from gen_local_views(d, 2, spins, spin_orbits)

# def gen_local_views_par(d, spin_depth=1, spins=None, spin_orbits=None, verbose=False, tqdm=None):
#     if spins is None:
#         spins = [0,1]
#     if spin_orbits is None:
#         spin_orbits = [[s] for s in spins]
#     if tqdm is None:
#         tqdm = lambda x, *args, **kwargs: x

#     G = Graph(1, data_structure='dense', loops=False, multiedges=False)
#     data = [(d, G, [[0]], [0])]
#     with ProcessPoolExecutor() as executor:
#         for i in tqdm(range(spin_depth-1)):
#             if verbose:
#                 print(f'Adding layer {i+1} starting with {len(data)} graphs')
#             data = [y for ys in executor.map(add_new_layer, data) for y in ys]

#             if verbose:
#                 print(f'Filling layer {i+1} starting with {len(data)} graphs')
#             data = [y for ys in executor.map(fill_layer, data) for y in ys]

#         if verbose:
#             print(f'Adding layer {spin_depth} disjointly to {len(data)} graphs')
#         data = [y for ys in executor.map(add_new_layer_disjoint, data) for y in ys]

#         if verbose:
#             print(f'Assigning spins to layer {spin_depth} starting with {len(data)} graphs')
#         extend = lambda x: (*x, spins, spin_orbits)
#         yield from (y for ys in executor.map(assign_spins, map(extend, data)) for y in ys)

############
# LAYER CODE
# def add_new_layer(gb):
#     d, G, last_partition, last_layer = gb
#     if not isinstance(G._backend, DenseGraphBackend): # Workaround for https://github.com/sagemath/sage/issues/38900
#         G = G.copy(data_structure="dense")
#     r = []
#     next_layer = []
#     for v in last_layer:
#         for w in range(d-G.degree(v)):
#             next_layer.append(G.add_vertex())
    
#     partition = last_partition + [next_layer]
#     aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
#     for X in canaug_new_layer(d, G, partition, last_layer, next_layer, aut_gens):
#         clean_X = copy(X)
#         clean_partition = deepcopy(partition)
#         clean_next_layer = copy(next_layer)
#         for v in next_layer:
#             if clean_X.degree(v) == 0:
#                 clean_X.delete_vertex(v)
#                 clean_partition[-1].remove(v)
#                 clean_next_layer.remove(v)
#         if not clean_partition[-1]:
#             clean_partition.pop()
#         r.append((d, clean_X, clean_partition, clean_next_layer))
#     return r

def add_new_layer_gen(d, G, last_partition, last_layer, forbidden_subgraphs):
    next_layer = []
    for v in last_layer:
        for w in range(d-G.degree(v)):
            next_layer.append(G.add_vertex())
    
    partition = last_partition + [next_layer]
    aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
    for X in canaug_new_layer(d, G, partition, last_layer, next_layer, aut_gens, forbidden_subgraphs):
        clean_X = copy(X)
        clean_partition = deepcopy(partition)
        clean_next_layer = copy(next_layer)
        for v in next_layer:
            if clean_X.degree(v) == 0:
                clean_X.delete_vertex(v)
                clean_partition[-1].remove(v)
                clean_next_layer.remove(v)
        if not clean_partition[-1]:
            clean_partition.pop()
        yield (d, clean_X, clean_partition, clean_next_layer)

# def add_new_layer_disjoint(gb):
#     d, G, last_partition, last_layer = gb
#     if not isinstance(G._backend, DenseGraphBackend): # Workaround for https://github.com/sagemath/sage/issues/38900
#         G = G.copy(data_structure="dense")
#     next_layer = []
#     for v in last_layer:
#         Nv = []
#         for w in range(d-G.degree(v)):
#             Nv.append(G.add_vertex())
#         G.add_edges((v, w) for w in Nv)
#         next_layer.extend(Nv)
#     partition = deepcopy(last_partition + [next_layer])
#     return [(d, G, partition, next_layer)]

def add_new_layer_disjoint_gen(d, G, last_partition, last_layer):
    next_layer = []
    for v in last_layer:
        Nv = []
        for w in range(d-G.degree(v)):
            Nv.append(G.add_vertex())
        G.add_edges((v, w) for w in Nv)
        next_layer.extend(Nv)
    partition = deepcopy(last_partition + [next_layer])
    yield (d, G, partition, next_layer)

# def fill_layer(gb):
#     d, G, partition, layer = gb
#     if not isinstance(G._backend, DenseGraphBackend): # Workaround for https://github.com/sagemath/sage/issues/38900
#         G = G.copy(data_structure="dense")
#     r = []
#     aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
#     for X in canaug_fill_layer(d, G, partition, layer, aut_gens):
#         r.append((d, X, partition, layer))
#     return r

def fill_layer_gen(d, G, partition, layer, forbidden_subgraphs):
    aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
    for X in canaug_fill_layer(d, G, partition, layer, aut_gens, forbidden_subgraphs):
        yield (d, X, partition, layer)

##############
# ASSIGN SPINS
def assign_spins(pv):
    d, G, partition, layer, spins, spin_orbits = pv
    if not isinstance(G._backend, DenseGraphBackend): # Workaround for https://github.com/sagemath/sage/issues/38900
        G = G.copy(data_structure="dense")
    r = []
    spin_indices = {s: G.add_vertex() for s in spins}
    spin_vertices = list(spin_indices.values())

    partition += [[spin_indices[s] for s in orbit] for orbit in spin_orbits]
    aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
    for X in canaug_assign_spins(G, partition, layer, spin_vertices, aut_gens):
        r.append(LocalView(X, copy(spins), copy(spin_vertices), deepcopy(partition)))
    return r

def assign_spins_gen(pv):
    d, G, partition, layer, spins, spin_orbits = pv
    spin_indices = {s: G.add_vertex() for s in spins}
    spin_vertices = list(spin_indices.values())

    partition += [[spin_indices[s] for s in orbit] for orbit in spin_orbits]
    aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
    for X in canaug_assign_spins(G, partition, layer, spin_vertices, aut_gens):
        yield LocalView(X, copy(spins), copy(spin_vertices), deepcopy(partition))

#############
# CANAUG CODE
def canaug_new_layer(d, X, partition, last_layer, new_layer, aut_gens, forbidden_subgraphs):
    cY_can:     DenseGraph
    mY:         cython.list
    upper_reps: cython.list
    uppers:     cython.set
    e:          cython.tuple[cython.int, cython.int]
    ystar:      cython.tuple
    ystar_orig: cython.tuple
    
    unfinished_last = [v for v in last_layer if X.degree(v) < d]
    unfinished_new = [v for v in new_layer if X.degree(v) < d]

    # yield the local view if all elements of the last layer now have degree d
    if not unfinished_last:
        yield X
        return

    uppers = {e for e in gen_bipartite_edges(unfinished_last, unfinished_new) if not X.has_edge(e) and 
                                                                                 no_forbidden_subgraphs(X, e, forbidden_subgraphs)}

    # use the automorphism group to do some isomorphism checking
    upper_reps = find_upper_reps(aut_gens, uppers)
    for e in upper_reps:
        # Y is X augmented with e; cY is the underlying C graph
        Y = copy(X)
        Y.add_edge(e)

        # compute the automorphism group of Y and the canonical labeling
        Y_aut_gens, cY_can, cr = search_tree(Y._backend.c_graph()[0], partition, True, True)
        icr = {v: k for k, v in cr.items()} # inverse of relableling
        last_layer_can = sorted(cr[v] for v in last_layer)
        new_layer_can = sorted(cr[v] for v in new_layer)
        ystar = max(f for f in gen_bipartite_edges(last_layer_can, new_layer_can) if cY_can.has_arc(f[0], f[1]))
        ystar_orig = tuple(sorted([icr[ystar[0]], icr[ystar[1]]]))

        mY = find_orbit(ystar_orig, Y_aut_gens)
        if e in mY:
            yield from canaug_new_layer(d, Y, partition, last_layer, new_layer, Y_aut_gens, forbidden_subgraphs)


def canaug_fill_layer(d, X, partition, layer, aut_gens, forbidden_subgraphs):
    cY_can:     DenseGraph
    mY:         cython.list
    upper_reps: cython.list
    uppers:     cython.set
    e:          cython.tuple
    ystar:      cython.tuple
    ystar_orig: cython.tuple
    
    yield X

    unfinished = [v for v in layer if X.degree(v) < d]
    if not unfinished:
        return

    uppers = {e for e in gen_all_edges(unfinished) if not X.has_edge(e) and 
                                                      no_forbidden_subgraphs(X, e, forbidden_subgraphs)}

    # use the automorphism group to do some isomorphism checking
    upper_reps = find_upper_reps(aut_gens, uppers)
    for e in upper_reps:
        # Y is X augmented with e; cY is the underlying C graph
        Y = copy(X)
        Y.add_edge(e)

        # compute the automorphism group of Y and the canonical labeling
        Y_aut_gens, cY_can, cr = search_tree(Y._backend.c_graph()[0], partition, True, True)
        icr = {v: k for k, v in cr.items()} # inverse of relableling
        layer_can = sorted(cr[v] for v in layer)
        ystar = max(f for f in gen_all_edges(layer_can) if cY_can.has_arc(f[0], f[1]))
        ystar_orig = tuple(sorted([icr[ystar[0]], icr[ystar[1]]]))

        mY = find_orbit(ystar_orig, Y_aut_gens)
        if e in mY:
            yield from canaug_fill_layer(d, Y, partition, layer, Y_aut_gens, forbidden_subgraphs)


def canaug_assign_spins(X, partition, domain, spins, aut_gens):
    cY_can:     DenseGraph
    mY:         cython.list
    upper_reps: cython.list
    uppers:     cython.set
    e:          cython.tuple
    ystar:      cython.tuple
    ystar_orig: cython.tuple

    # yield the local view if all neighbors of u have been assigned spins, and terminate
    if all_assigned_spins(X, domain, spins):
        yield X
        return
    
    # uppers are the possible spin assignments to proceed with
    unassigned = [v for v in domain if not any(X.has_edge(v,s) for s in spins)]
    uppers = {f for f in gen_bipartite_edges(unassigned, spins) if not X.has_edge(f)}

    # use the automorphism group to do some isomorphism checking
    upper_reps = find_upper_reps(aut_gens, uppers)
    for e in upper_reps:
        # Y is X augmented with e; cY is the underlying C graph
        Y = copy(X)
        Y.add_edge(e)

        # compute the automorphism group of Y and the canonical labeling
        Y_aut_gens, cY_can, cr = search_tree(Y._backend.c_graph()[0], partition, True, True)
        icr = {v: k for k, v in cr.items()} # inverse of relableling
        domain_can = sorted(cr[v] for v in domain)
        spins_can = sorted(cr[v] for v in spins)
        ystar = max(f for f in gen_bipartite_edges(domain_can, spins_can) if cY_can.has_arc(f[0], f[1]))
        ystar_orig = tuple(sorted([icr[ystar[0]], icr[ystar[1]]]))

        mY = find_orbit(ystar_orig, Y_aut_gens)
        if e in mY:
            yield from canaug_assign_spins(Y, partition, domain, spins, Y_aut_gens)


################
# CANAUG HELPERS
@cython.cfunc
@cython.inline
def gen_bipartite_edges(A: cython.list, B: cython.list):
    return ((a,b) for a in A for b in B)

@cython.cfunc
@cython.inline
def gen_all_edges(A: cython.list):
    return ((A[i],A[j]) for i in range(len(A)) for j in range(i+1,len(A)))

@cython.cfunc
@cython.inline
def all_assigned_spins(X, domain, spins):
    return all(any(X.has_edge(v,s) for s in spins) for v in domain)
@cython.cfunc
@cython.inline
def permute_edge(e: cython.tuple, gen: cython.list) -> cython.tuple:
    f: cython.tuple
    f = (gen[e[0]], gen[e[1]])
    return f if f[0] < f[1] else (f[1], f[0])

@cython.cfunc
@cython.inline
def reverse(t: cython.tuple) -> cython.tuple:
    return (t[1], t[0])

@cython.ccall
def find_upper_reps(aut_gens: cython.list, uppers: cython.set) -> cython.list:
    """
    Compute a list of representatives of each isomorphism class of uppers under
    the action of the permutation group given by aut_gens

    INPUT:
    -  ``aut_gens`` - a list of generators for the group in permutation list.
       format, i.e. the generator g maps i to g[i].
    -  ``uppers`` - a set of edges (tuples of length 2) to permute
    -  ``end_sort`` - a flag indicating whether to sort the reperesentatives 
       by reverse lexicographic order.
    
    OUTPUT:
    -  A list of representatives
    """
    orbit: cython.list
    orbits: cython.list
    sorbit: cython.set
    
    orbits = []
    # WARNING the set that is iterated over gets modified in the loop
    while uppers:
        orbit = find_orbit(next(iter(uppers)), aut_gens)
        orbits.append(orbit)
        for e in orbit:
            uppers.discard(e)

    return sorted([orbit[0] for orbit in orbits])

@cython.ccall
def find_orbit(e: cython.tuple, aut_gens: cython.list) -> cython.list:
    """
    Compute the orbit of an edge under the action of a permutation group

    INPUT:
    -  ``e`` - an edge (tuple of length 2).
    -  ``aut_gens`` - a list of generators for the group in permutation list 
       format, i.e. the generator g maps i to g[i].
    
    OUTPUT:
    -  The orbit of e as a list
    """
    lorbit: cython.list
    sorbit: cython.set
    i: cython.int

    lorbit = [e]
    sorbit = {e}
    i = 0
    while i < len(lorbit):
        f = lorbit[i]
        for gen in aut_gens:
            g = permute_edge(f, gen)
            if g not in sorbit:
                sorbit.add(g)
                lorbit.append(g)
        i += 1
    lorbit.sort()
    return lorbit

def no_forbidden_subgraphs(X, e, forbidden_subgraphs):
    X.add_edge(e)
    r = all(not H.is_subgraph(X, up_to_isomorphism=True) for H in forbidden_subgraphs)
    X.delete_edge(e)
    return r

##############
# LIBRARY CODE

# Import cython implementations of partition refinement algorithms from the
# sage standard library (v8.0 2017-07-21)
from sage.groups.perm_gps.partn_ref.automorphism_group_canonical_label cimport (
    aut_gp_and_can_lab, deallocate_agcl_output, get_aut_gp_and_can_lab)
from sage.groups.perm_gps.partn_ref.data_structures cimport (
    PartitionStack, PS_dealloc, PS_from_list, PS_num_cells)
from sage.groups.perm_gps.partn_ref.refinement_graphs cimport (
    GraphStruct)

cdef bint all_children_are_equivalent(PartitionStack *PS, void *S) noexcept:
    """
    Return True if every refinement of the current partition results in the
    same structure.

    WARNING:

    Converse does not hold in general!  See Lemma 2.25 of [1] for details.

    INPUT:

    - ``PS`` -- the partition stack to be checked
    - ``S`` -- a graph struct object
    """
    cdef GraphStruct GS = <GraphStruct> S
    if GS.directed or GS.loops:
        return 0
    cdef int i, n = PS.degree
    cdef bint in_cell = 0
    cdef int nontrivial_cells = 0
    cdef int total_cells = PS_num_cells(PS)
    if n <= total_cells + 4:
        return 1
    for i from 0 <= i < n-1:
        if PS.levels[i] <= PS.depth:
            if in_cell:
                nontrivial_cells += 1
            in_cell = 0
        else:
            in_cell = 1
    if in_cell:
        nontrivial_cells += 1
    if n == total_cells + nontrivial_cells:
        return 1
    if n == total_cells + nontrivial_cells + 1:
        return 1
    return 0

cdef int compare_graphs(int *gamma_1, int *gamma_2, void *S1, void *S2, int degree) noexcept:
    r"""
    Compare gamma_1(S1) and gamma_2(S2).

    Return -1 if gamma_1(S1) < gamma_2(S2), 0 if gamma_1(S1) ==
    gamma_2(S2), 1 if gamma_1(S1) > gamma_2(S2).

    INPUT:

    - ``gamma_1``, ``gamma_2`` -- list permutations (inverse)
    - ``S1``, ``S2`` -- graph struct objects
    """
    cdef size_t i, j
    cdef GraphStruct GS1 = <GraphStruct> S1
    cdef GraphStruct GS2 = <GraphStruct> S2
    cdef CGraph G1 = GS1.G
    cdef CGraph G2 = GS2.G
    if G1.active_vertices.size != G2.active_vertices.size or \
       not bitset_cmp(G1.active_vertices, G2.active_vertices):
        for i from 0 <= i < <size_t>degree:
            if G1.has_vertex(gamma_1[i]) != G2.has_vertex(gamma_2[i]):
                return G1.has_vertex(gamma_1[i]) - G2.has_vertex(gamma_2[i])
    for i from 0 <= i < G1.num_verts:
        for j from 0 <= j < G1.num_verts:
            if G1.has_arc_unsafe(gamma_1[i], gamma_1[j]):
                if not G2.has_arc_unsafe(gamma_2[i], gamma_2[j]):
                    return 1
            elif G2.has_arc_unsafe(gamma_2[i], gamma_2[j]):
                return -1
    return 0

cdef inline int degree(PartitionStack *PS, CGraph G, int entry, int cell_index, bint reverse) noexcept:
    """
    Return the number of edges from the vertex corresponding to entry to
    vertices in the cell corresponding to cell_index.

    INPUT:

    - ``PS`` -- the partition stack to be checked
    - ``S`` -- a graph struct object
    - ``entry`` -- the position of the vertex in question in the entries of PS
    - ``cell_index`` -- the starting position of the cell in question in the entries
      of PS
    - ``reverse`` -- whether to check for arcs in the other direction
    """
    cdef int num_arcs = 0
    entry = PS.entries[entry]
    if not reverse:
        while True:
            if G.has_arc_unsafe(PS.entries[cell_index], entry):
                num_arcs += 1
            if PS.levels[cell_index] > PS.depth:
                cell_index += 1
            else:
                break
    else:
        while True:
            if G.has_arc_unsafe(entry, PS.entries[cell_index]):
                num_arcs += 1
            if PS.levels[cell_index] > PS.depth:
                cell_index += 1
            else:
                break
    return num_arcs

cdef int refine_by_degree(PartitionStack *PS, void *S, int *cells_to_refine_by, int ctrb_len) noexcept:
    r"""
    Refine the input partition by checking degrees of vertices to the given
    cells.

    INPUT:

    - ``PS`` -- a partition stack, whose finest partition is the partition to be
      refined
    - ``S`` -- a graph struct object, which contains scratch space, the graph in
      question, and some flags
    - ``cells_to_refine_by`` -- a list of pointers to cells to check degrees against
      in refining the other cells (updated in place). Must be allocated to
      length at least the degree of PS, since the array may grow
    - ``ctrb_len`` -- how many cells in cells_to_refine_by

    OUTPUT:

    An integer invariant under the orbits of $S_n$.  That is, if $\gamma$ is a
    permutation of the vertices, then
    $$ I(G, PS, cells_to_refine_by) = I( \gamma(G), \gamma(PS), \gamma(cells_to_refine_by) ) .$$
    """
    cdef GraphStruct GS = <GraphStruct> S
    cdef CGraph G = GS.G
    cdef int current_cell_against = 0
    cdef int current_cell, i, r
    cdef int first_largest_subcell
    cdef int invariant = 1
    cdef int max_degree
    cdef int *degrees = GS.scratch # length 3n+1
    cdef bint necessary_to_split_cell
    cdef int against_index
    if <int>G.num_verts != PS.degree and PS.depth == 0:
        # should be less verts, then, so place the "nonverts" in separate cell at the end
        current_cell = 0
        while current_cell < PS.degree:
            i = current_cell
            r = 0
            while True:
                if G.has_vertex(PS.entries[i]):
                    degrees[i-current_cell] = 0
                else:
                    r = 1
                    degrees[i-current_cell] = 1
                i += 1
                if PS.levels[i-1] <= PS.depth:
                    break
            if r != 0:
                sort_by_function(PS, current_cell, degrees)
            current_cell = i
    while not PS_is_discrete(PS) and current_cell_against < ctrb_len:
        invariant += 1
        current_cell = 0
        while current_cell < PS.degree:
            invariant += 50
            i = current_cell
            necessary_to_split_cell = 0
            max_degree = 0
            while True:
                degrees[i-current_cell] = degree(PS, G, i, cells_to_refine_by[current_cell_against], 0)
                if degrees[i-current_cell] != degrees[0]:
                    necessary_to_split_cell = 1
                if degrees[i-current_cell] > max_degree:
                    max_degree = degrees[i-current_cell]
                i += 1
                if PS.levels[i-1] <= PS.depth:
                    break
            # now, i points to the next cell (before refinement)
            if necessary_to_split_cell:
                invariant += 10
                first_largest_subcell = sort_by_function(PS, current_cell, degrees)
                invariant += first_largest_subcell + max_degree
                against_index = current_cell_against
                while against_index < ctrb_len:
                    if cells_to_refine_by[against_index] == current_cell:
                        cells_to_refine_by[against_index] = first_largest_subcell
                        break
                    against_index += 1
                r = current_cell
                while True:
                    if r == current_cell or PS.levels[r-1] == PS.depth:
                        if r != first_largest_subcell:
                            cells_to_refine_by[ctrb_len] = r
                            ctrb_len += 1
                    r += 1
                    if r >= i:
                        break
                invariant += (i - current_cell)
            current_cell = i
        if GS.directed:
            # if we are looking at a digraph, also compute
            # the reverse degrees and sort by them
            current_cell = 0
            while current_cell < PS.degree: # current_cell is still a valid cell
                invariant += 20
                i = current_cell
                necessary_to_split_cell = 0
                max_degree = 0
                while True:
                    degrees[i-current_cell] = degree(PS, G, i, cells_to_refine_by[current_cell_against], 1)
                    if degrees[i-current_cell] != degrees[0]:
                        necessary_to_split_cell = 1
                    if degrees[i-current_cell] > max_degree:
                        max_degree = degrees[i-current_cell]
                    i += 1
                    if PS.levels[i-1] <= PS.depth:
                        break
                # now, i points to the next cell (before refinement)
                if necessary_to_split_cell:
                    invariant += 7
                    first_largest_subcell = sort_by_function(PS, current_cell, degrees)
                    invariant += first_largest_subcell + max_degree
                    against_index = current_cell_against
                    while against_index < ctrb_len:
                        if cells_to_refine_by[against_index] == current_cell:
                            cells_to_refine_by[against_index] = first_largest_subcell
                            break
                        against_index += 1
                    against_index = ctrb_len
                    r = current_cell
                    while True:
                        if r == current_cell or PS.levels[r-1] == PS.depth:
                            if r != first_largest_subcell:
                                cells_to_refine_by[against_index] = r
                                against_index += 1
                                ctrb_len += 1
                        r += 1
                        if r >= i:
                            break
                    invariant += (i - current_cell)
                current_cell = i
        current_cell_against += 1
    if GS.use_indicator:
        return invariant
    else:
        return 0
    
cpdef search_tree(DenseGraph G, list partition, bint lab, bint certificate):
    """
    Compute automorphism groups and canonical labels of graphs. This function 
    is copied from sage.groups.perm_groups.partn_ref.refinement_graphs (v8.0 
    2017-07-21) but ancilliary code has been removed.

    INPUT:
    -  ``G`` - a DenseGraph object.
    -  ``partitions`` - a list of lists representing a partition of V(G). The
       regurned group fixes parts of this partition.
    -  ``lab`` - a flab indicating whether the canonically relabeled G should
       be returned as a DenseGraph.
    -  ``certificate`` - a flag indicating whether a dictionary of the 
       canonical relabeling should be returned.

    OUTPUT (as a tuple if multiple items returned):
    -  A list of generators for aut(G) (repsecting the partition)
    -  A DenseGraph which is the canoncial labeling of G (if lab)
    -  A dictionary cr such that cr[v] is the canonical label of v (if 
       certificate)
    """
    cdef int i, j, n
    cdef aut_gp_and_can_lab *output
    cdef PartitionStack *part

    n = G.num_verts

    cdef GraphStruct GS = GraphStruct()
    GS.G = <CGraph>G
    GS.directed = 0
    GS.loops = 0
    GS.use_indicator = 1

    if n == 0:
        return_tuple = [[]] # no aut gens
        if lab:
            G_C = DenseGraph(n)
            return_tuple.append(G_C)
        if certificate:
            return_tuple.append({})
        if len(return_tuple) == 1:
            return return_tuple[0]
        else:
            return tuple(return_tuple)

    GS.scratch = <int *> sig_malloc( (3*G.num_verts + 1) * sizeof(int) )
    part = PS_from_list(partition)
    if GS.scratch is NULL or part is NULL:
        PS_dealloc(part)
        sig_free(GS.scratch)
        raise MemoryError

    output = get_aut_gp_and_can_lab(<void *>GS, part, G.num_verts, 
                                    all_children_are_equivalent, 
                                    refine_by_degree, compare_graphs, 
                                    lab, NULL, NULL, NULL)
    sig_free(GS.scratch)

    # prepare output
    list_of_gens = []
    for i in xrange(output.num_gens):
        list_of_gens.append([output.generators[j+i*G.num_verts] 
                            for j in xrange(G.num_verts)])
    return_tuple = [list_of_gens]
    if lab:
        G_C = DenseGraph(n)
        for i in xrange(n):
            for j in G.out_neighbors(i):
                G_C.add_arc(output.relabeling[i],output.relabeling[j])
        return_tuple.append(G_C)
    if certificate:
        cr = {}
        for i in xrange(G.num_verts):
            cr[i] = output.relabeling[i]
        return_tuple.append(cr)
    PS_dealloc(part)
    deallocate_agcl_output(output)
    if len(return_tuple) == 1:
        return return_tuple[0]
    else:
        return tuple(return_tuple)

