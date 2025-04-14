import os.path
import subprocess
import sys

from collections import namedtuple, Counter

from sage.all import diff, graphs, ln, load, save, sqrt, var, matrix, vector
from sage.numerical.mip import MixedIntegerLinearProgram

load("potts_local_view.py")
load("potts_canaug.pyx")

def mono(G, sigma):
    """Count the number of monochromatic edges of a graph G under a spin assignment sigma."""
    return sum(1 for e in G.edges(labels=False) if sigma[e[0]] == sigma[e[1]])

def Z(G, b, q):
    """Compute the partition function of the Potts model on a graph G with edge activity b."""
    return sum(
        exp(-b * m)
        for m, sigma in PottsLocalView(G, q=q, spin_vertices=[]).gen_all_spin_assignments()
    )

def F(G, b, q):
    ZG = Z(G, b, q)
    return ln(ZG)/G.order()

def occ(G, b, q):
    varb = var("varb")
    FG = F(G, varb, q)
    return -diff(FG, varb).subs(varb=b)

def compute_probabilities(L, depth, b=None, tqdm=None):
    """Compute the probabilities for a local view L."""
    if b is None:
        b = var("b")
    u = L.u
    d = L.G.degree(u)
    q = L._q

    q_parts = [p for p in Partitions(d) if len(p) <= q]

    # p is the probability that a uniform random edge incident to u is monochromatic
    p = 0
    # gs[i][j] is the probability that the simple random walk of length i from u terminates in a vertex which sees the partition indexed by j
    gs = [[0 for _ in q_parts] for _ in range(depth)]

    # Z is the partition function of the local view L
    Z = 0

    for m, sigma in L.gen_all_spin_assignments(tqdm):
        weight = exp(-b * m)
        Z += weight

        p += weight * sum(1 for v in L.G.neighbors(u) if sigma[u] == sigma[v]) / d

        for i in range(depth):
            Apow = L.G.adjacency_matrix()**i

            for v in L.G:
                if Apow[u, v] != 0:
                    part = sorted(Counter(sigma[w] for w in L.G.neighbors(v)).values(), reverse=True)
                    j = q_parts.index(part)
                    gs[i][j] += weight * Apow[u, v] / d**i

    # Normalize by Z at the end
    p /= Z
    for i in range(depth):
        gs[i] = [g / Z for g in gs[i]]

    L.data = {
        "p": p,
        "gs": gs,
        "Z": Z,
    }


# Local view data generation/loading
LData = namedtuple("LData", ["Ls", "version"])
DATA_VERSION = 1
def default_filename(d, spin_depth, forbidden):
    if len(forbidden) == 1 and forbidden[0].is_isomorphic(graphs.CompleteGraph(3)):
        return f"data/potts_d{d}_depth{spin_depth}_triangle_free.sobj"
        
    return f"data/potts_d{d}_depth{spin_depth}.sobj"

def gen_data(d, spin_depth=2, forbidden=None, filename=None, verbose=False, tqdm=None):
    if filename is None:
        filename = default_filename(d, spin_depth, forbidden)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Ls = []
    for L in gen_local_views(d, spin_depth, q_max=None, forbidden_subgraphs=forbidden, verbose=verbose, tqdm=tqdm):
        compute_probabilities(L, spin_depth, tqdm=tqdm)
        Ls.append(L)
    
    data = LData(Ls, DATA_VERSION)
    save(data, filename)
    return data

def gen_data_par(d, spin_depth=2, filename=None, verbose=False, tqdm=None):
    if filename is None:
        filename = default_filename(d, spin_depth)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Ls = []
    for L in gen_local_views_par(d, spin_depth, q_max=None, spins=ising_spins, verbose=verbose, tqdm=tqdm):
        compute_probabilities(L, spin_depth, tqdm=tqdm)
        Ls.append(L)
    
    data = LData(Ls, DATA_VERSION)
    save(data, filename)
    return data

def load_data(d, spin_depth=2, forbidden=None, filename=None):
    if filename is None:
        filename = default_filename(d, spin_depth, forbidden)

    data = load(filename)
    return data

def get_data(d, spin_depth=2, forbidden=None, filename=None):
    if filename is None:
        filename = default_filename(d, spin_depth, forbidden)

    if os.path.isfile(filename):
        data = load_data(d, spin_depth, forbidden, filename)
        if data.version == DATA_VERSION:
            return data

    return gen_data(d, spin_depth, forbidden, filename)


# WARNING: this destructively modifies the input Ls
def sub_Ls(Ls, bval, qval, precision=None):
    for L in Ls:
        data = L.data
        data["p"] = data["p"].subs(b=bval, q=qval)
        data["gs"] = [[g.subs(b=bval, q=qval) for g in gs] for gs in data["gs"]]
        data["Z"] = data["Z"].subs(b=bval, q=qval)

        if precision != None:
            data["p"] = data["p"].n(precision)
            data["gs"] = [[g.n(precision) for g in gs] for gs in data["gs"]]
            data["Z"] = data["Z"].n(precision)

def triangle_count(G, u):
    return sum(1 for s in Subsets(G.neighbors(u), 2) if G.has_edge(s[0], s[1]))

# linear programming
def gen_lp(d, spin_depth, bval, qval, probability_precision, Ls=None, solver="PPL", constraints="eq", maximization=False):
    if Ls is None:
        Ls = get_data(d, spin_depth)
    sub_Ls(Ls, bval, qval, precision=probability_precision)

    p = MixedIntegerLinearProgram(maximization=maximization, solver=solver)
    x = p.new_variable(nonnegative=True)

    if constraints == "eq":
        p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) == 1)
        
        num_parts = len([p for p in Partitions(d) if len(p) <= qval])
        for j in range(num_parts):
            for k in range(1, spin_depth):
                p.add_constraint(
                    p.sum((L.data["gs"][0][j] - L.data["gs"][k][j]) * x[i] for i, L in enumerate(Ls)) == 0
                )

    # elif constraints == "ge":
    #     p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) >= 1)
    #     num_parts = len(Ls[0].data["gs"][0]) #this is ugly but it's the number of relevant partitions
    #     for j in range(num_parts):
    #         p.add_constraint(
    #             p.sum((L.data["gs"][0][j] - L.data["gs"][k][j]) * x[i] for i, L in enumerate(Ls)) >= 0
    #         )
    # else: #constraints == "le":
    #     p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) <= 1)
    #     num_parts = len(Ls[0].data["gs"][0]) #this is ugly but it's the number of relevant partitions
    #     for j in range(num_parts):
    #         p.add_constraint(
    #             p.sum((L.data["gs"][0][j] - L.data["gs"][k][j]) * x[i] for i, L in enumerate(Ls)) <= 0
    #         )

    p.set_objective(p.sum(L.data["p"] * x[i] for i, L in enumerate(Ls)))
    return p, x

def gen_lp_via_poly(d, bval, qval, Ls=None):
    if Ls is None:
        Ls = get_data(d).Ls
    
    Ls = sub_Ls(Ls, Bval, lval)

    eqns = []
    ieqs = []

    # probability constraint
    eqns.append([-1] + [1 for L in Ls])

    num_parts = len([p for p in Partitions(d) if len(p) <= qval])
    for j in range(num_parts):
        for k in range(1, len(L.layers)):
            eqns.append([0] + [L.data["gs"][0][j] - L.data["gs"][k][j] for L in Ls])

    # slow
    print("Generating polyhedron")
    pol = Polyhedron(ieqs=ieqs, eqns=eqns, base_ring=AA)

    print("Generating LP")
    p, x = pol.to_linear_program(solver='InteractiveLP', return_variable=True)
    p.set_objective(p.sum(AA(-L.data["p"]) * x[i] for i, L in enumerate(Ls)))
    p.set_min(x, 0)
    return p, x