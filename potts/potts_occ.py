import os.path
import subprocess
import sys

from collections import namedtuple

from sage.all import diff, graphs, ln, load, save, sqrt, var, matrix, vector
from sage.numerical.mip import MixedIntegerLinearProgram

load("local_view.py")
load("canaug.pyx")

# Ising model computations
q = 4
potts_spins = list(range(1,1+q))
potts_orbits = [list(range(1,1+q))]

def mono(G, sigma):
    """Count the number of monochromatic edges of a graph G under a spin assignment sigma."""
    return sum(1 for e in G.edges(labels=False) if sigma[e[0]] == sigma[e[1]])

def Z(G, b):
    """Compute the partition function of the Ising model on a graph G with edge activity B and external field l."""
    return sum(
        exp(-b * mono(G, sigma))
        for sigma in LocalView(G, ising_spins, []).gen_all_spin_assignments()
    )

def F(G, b):
    ZG = Z(G, b)
    return ln(ZG)/G.order()

def occ(G, b):
    varb = var("varb")
    FG = F(G, varb)
    return -diff(FG, varb).subs(varb=b)

def compute_probabilities(L, depth, b=None, tqdm=None):
    """Compute the probabilities for a local view L."""
    if b is None:
        B = var("b")

    u = L.u
    d = L.G.degree(u)

    # Below has to change for Potts
    # want ps[0] (or just p) to be the probability that a uniform random edge incident to u is monochromatic
    # want for each partition d = x_1 + x_2 + ... + x_k we want the probability that 
    # u sees those totals of inidvidual colors
    # and same for a uniform random neighbor v of u,

    # ps[i] is the probability that the simple random walk of length i from u terminates in a vertex which gets +
    ps = [0 for _ in range(depth)]
    # gs[i][j] is the probability that the simple random walk of length i from u terminates in a vertex which has j neighbors that get +
    gs = [[0 for _ in range(d + 1)] for _ in range(depth)]
    # Z is the partition function of the local view L
    Z = 0

    for sigma in L.gen_all_spin_assignments(tqdm):
        weight = B ** mono(L.G, sigma) * l ** nplus(L.G, sigma)
        Z += weight

        for i in range(depth):
            Apow = L.G.adjacency_matrix()**i
            for v in L.G:
                ps[i] += weight * Apow[u, v] * int(sigma[v] == "+") / d**i
                gs[i][sum(1 for w in L.G.neighbors(v) if sigma[w] == "+")] += weight * Apow[u, v] / d**i

    # Normalize by Z at the end
    for i in range(depth):
        ps[i] /= Z
        gs[i] = [g / Z for g in gs[i]]

    L.data = {
        "ps": ps,
        "gs": gs,
        "Z": Z,
    }


# Local view data generation/loading
LData = namedtuple("LData", ["Ls", "version"])
DATA_VERSION = 1
def default_filename(d, spin_depth):
    return f"data/potts_d{d}_depth{spin_depth}.sobj"

def gen_data(d, spin_depth=2, filename=None, verbose=False, tqdm=None):
    if filename is None:
        filename = default_filename(d, spin_depth)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Ls = []
    for L in gen_local_views(d, spin_depth, spins=potts_spins, spin_orbits=potts_orbits, verbose=verbose, tqdm=tqdm):
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
    for L in gen_local_views_par(d, spin_depth, spins=ising_spins, verbose=verbose, tqdm=tqdm):
        compute_probabilities(L, spin_depth, tqdm=tqdm)
        Ls.append(L)
    
    data = LData(Ls, DATA_VERSION)
    save(data, filename)
    return data

def load_data(d, spin_depth=2, filename=None):
    if filename is None:
        filename = default_filename(d, spin_depth)

    data = load(filename)
    return data

def get_data(d, spin_depth=2, filename=None):
    if filename is None:
        filename = default_filename(d, spin_depth)

    if os.path.isfile(filename):
        data = load_data(d, spin_depth, filename)
        if data.version == DATA_VERSION:
            return data

    return gen_data(d, spin_depth, filename)


# WARNING: this destructively modifies the input Ls
def sub_Ls(Ls, Bval, lval):
    for L in Ls:
        data = L.data
        data["ps"] = [p.subs(B=Bval, l=lval) for p in data["ps"]]
        data["gs"] = [[g.subs(B=Bval, l=lval) for g in gs] for gs in data["gs"]]
        data["Z"] = data["Z"].subs(B=Bval, l=lval)

def triangle_count(G, u):
    return sum(1 for s in Subsets(G.neighbors(u), 2) if G.has_edge(s[0], s[1]))

# linear programming
# mflips is a list of indices for Ls we want to add constraints for by flipping a - to a +
def gen_lp(d, spin_depth, Bval, lval, Ls=None, solver="PPL", gams=None, constraints="eq", mflips=[], maximization=False):
    if Ls is None:
        Ls = get_data(d, spin_depth)
    if gams is None:
        gams = range(d+1)
    sub_Ls(Ls, Bval, lval)

    p = MixedIntegerLinearProgram(maximization=maximization, solver=solver)
    x = p.new_variable(nonnegative=True)

    if constraints == "eq":
        p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) == 1)
        # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) == 0)
        for j in gams:
            for k in range(1, spin_depth):
                p.add_constraint(
                    p.sum((L.data["gs"][0][j] - L.data["gs"][k][j]) * x[i] for i, L in enumerate(Ls)) == 0
                )

        # p.add_constraint(
        #     p.sum((triangle_count(L.G, L.u)-sum(triangle_count(L.G,v) for v in L.Nu)/d)*x[i] for i, L in enumerate(Ls)) == 0
        # )
    # elif constraints == "ge":
    #     p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) >= 1)
    #     # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) >= 0)
    #     for j in gams:
    #         p.add_constraint(
    #             p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) >= 0
    #         )
    # else: #constraints == "le":
    #     p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) <= 1)
    #     # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) >= 0)
    #     for j in gams:
    #         p.add_constraint(
    #             p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) <= 0
    #         )

    # for i in mflips:
    #     Lm = Ls[i]["L"]
    #     ms = set(w for w in Lm.N2u if Lm.spin_assignment[w] == "-")
    #     # print(f"mflip index {i}")
    #     # Lm.show()

    #     orbitms = []
    #     while ms:
    #         w = next(iter(ms))
    #         orbit = set(Lm.orbit(w))
    #         ms -= orbit
    #         orbitms.append(orbit)
    #     print(f"mflip index {i}: orbitms = {orbitms}")

    #     for orbitm in orbitms:
    #         w = next(iter(orbitm))
    #         # print(f"mflip index {i}: orbitm = {orbitm}, w={w}")
    #         Lp = Lm.change_spin(w)
    #         # Lp.show()
    #         orbitp = set(Lp.orbit(w))
    #         # print(f"mflip index {i}: orbitp = {orbitp}")

    #         Lpcan = Lp.fullG_can_fixed_spins
    #         j = 0
    #         while (Ls[j]['L'].fullG_can_fixed_spins != Lpcan):
    #             j += 1

    #         p.add_constraint(len(orbitm)*x[i] >= Bval**d/lval * len(orbitp) * x[j])
    #         print(f"mflip index {i}: constraint {len(orbitm)} * x[{i}] >= B^{d}/lam * {len(orbitp)} * x[{j}]")
    

    p.set_objective(p.sum(L.data["ps"][0] * x[i] for i, L in enumerate(Ls)))
    return p, x

def gen_lp_via_poly(d, Bval, lval, Ls=None, gams=None, mflips=[]):
    if Ls is None:
        Ls = get_data(d).Ls
    if gams is None:
        gams = range(d+1)
        
    Ls = sub_Ls(Ls, Bval, lval)

    eqns = []
    ieqs = []

    # probability constraint
    eqns.append([-1] + [1 for L in Ls])
    for j in gams:
        for k in range(1, len(L.layers)):
            eqns.append([0] + [L.data["gs"][0][j] - L.data["gs"][k][j] for L in Ls])

    # for i in mflips:
    #     Lm = Ls[i]["L"]
    #     ms = set(w for w in Lm.N2u if Lm.spin_assignment[w] == "-")
    #     # print(f"mflip index {i}")
    #     # Lm.show()

    #     orbitms = []
    #     while ms:
    #         w = next(iter(ms))
    #         orbit = set(Lm.orbit(w))
    #         ms -= orbit
    #         orbitms.append(orbit)
    #     # print(f"mflip index {i}: orbitms = {orbitms}")

    #     for orbitm in orbitms:
    #         w = next(iter(orbitm))
    #         # print(f"mflip index {i}: orbitm = {orbitm}, w={w}")
    #         Lp = Lm.change_spin(w)
    #         # Lp.show()
    #         orbitp = set(Lp.orbit(w))
    #         # print(f"mflip index {i}: orbitp = {orbitp}")

    #         Lpcan = Lp.fullG_can_fixed_spins
    #         j = 0
    #         while (Ls[j]['L'].fullG_can_fixed_spins != Lpcan):
    #             j += 1
    #         ieq = [0] * (len(Ls) + 1)
    #         ieq[i+1] = len(orbitm)
    #         ieq[j+1] = -Bval**d/lval * len(orbitp)
    #         ieqs.append(ieq)
    #         print(f"mflip index {i}: constraint {len(orbitm)} * x[{i}] >= B^{d}/lam * {len(orbitp)} * x[{j}]")
    
    # slow
    print("Generating polyhedron")
    pol = Polyhedron(ieqs=ieqs, eqns=eqns, base_ring=AA)

    print("Generating LP")
    p, x = pol.to_linear_program(solver='InteractiveLP', return_variable=True)
    p.set_objective(p.sum(AA(-L.data["ps"][0]) * x[i] for i, L in enumerate(Ls)))
    p.set_min(x, 0)
    return p, x