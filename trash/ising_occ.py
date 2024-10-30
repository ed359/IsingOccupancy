import os.path
import subprocess
import sys

from collections import namedtuple

from sage.all import diff, graphs, ln, load, save, sqrt, var, matrix, vector
from sage.numerical.mip import MixedIntegerLinearProgram

load("local_view.py")
load("canaug.pyx")

# Ising model computations
ising_spins = ["+", "-"]

def mono(G, sigma):
    return sum(1 for e in G.edges(labels=False) if sigma[e[0]] == sigma[e[1]])

def nplus(G, sigma):
    return sum(1 for v in G.vertices() if sigma[v] == "+")

def lc(d, B):
    r = var("r")
    s = var("s")
    bc = var("bc")

    ret = (
        (1 - sqrt(r / s))
        * ((1 + sqrt(r * s)) / (1 - sqrt(r * s))) ** ((1 + bc) / (-1 + bc))
    ) / (1 + sqrt(r / s))

    return ret.subs(r=(bc - B) / (bc + B), s=(1 - B) / (1 + B)).subs(bc=(d - 2) / d)


def compute_probabilities(L, B=None, l=None, tqdm=None):
    if B is None:
        B = var("B")
    if l is None:
        l = var("l")

    d = L.G.degree(0)
    pu = 0
    pNu = 0
    Z = 0
    gu = [0 for _ in range(d + 1)]
    gNu = [0 for _ in range(d + 1)]

    for sigma in L.gen_all_spin_assignments(tqdm):
        weight = B ** mono(L.G, sigma) * l ** nplus(L.G, sigma)
        Z += weight

        if sigma[0] == "+":
            pu += weight
        gu[sum(1 for v in L.G.neighbors(0) if sigma[v] == "+")] += weight

        for v in L.G.neighbors(0):
            if sigma[v] == "+":
                pNu += weight / d
            gNu[sum(1 for w in L.G.neighbors(v) if sigma[w] == "+")] += weight / d

    return {
        "pu": pu / Z,
        "pNu": pNu / Z,
        "gu": [g / Z for g in gu],
        "gNu": [g / Z for g in gNu],
        "Z": Z,
    }


# Local view data generation/loading
LData = namedtuple("LData", ["Ls", "version"])
DATA_VERSION = 0
def default_filename(d):
    return f"data/ising_d{d}.sobj"

def gen_data(d, filename=None):
    if filename is None:
        filename = default_filename(d)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Ls = []
    for L in gen_local_view_2(d, spins=ising_spins):
        ps = compute_probabilities(L, tqdm=tqdm)
        ps["L"] = L
        Ls.append(ps)
    data = LData(Ls, DATA_VERSION)
    save(data, filename)
    return data

def load_data(d, filename=None):
    if filename is None:
        filename = default_filename(d)

    data = load(filename)
    return data

def get_data(d, filename=None):
    if filename is None:
        filename = default_filename(d)

    if os.path.isfile(filename):
        data = load_data(d, filename)
        if data.version == DATA_VERSION:
            return data

    return gen_data(d, filename)


# WARNING: this destructively modifies the input Ls
def sub_Ls(Ls, Bval, lval):
    Ls_copy = []
    for L in Ls:
        L_copy = {}
        L_copy["L"] = L["L"]
        L_copy["pu"] = L["pu"].subs(B=Bval, l=lval)
        L_copy["pNu"] = L["pNu"].subs(B=Bval, l=lval)
        L_copy["gu"] = [g.subs(B=Bval, l=lval) for g in L["gu"]]
        L_copy["gNu"] = [g.subs(B=Bval, l=lval) for g in L["gNu"]]
        L_copy["Z"] = L["Z"].subs(B=Bval, l=lval)
        Ls_copy.append(L_copy)
    return Ls_copy


# linear programming
# mflips is a list of indices for Ls we want to add constraints for by flipping a - to a +
def gen_lp(d, Bval, lval, Ls=None, solver="GLPK", gams=None, constraints="eq", mflips=[]):
    if Ls is None:
        Ls = get_data(d).Ls
    if gams is None:
        gams = range(d+1)
    Ls = sub_Ls(Ls, Bval, lval)

    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    x = p.new_variable(nonnegative=True)

    if constraints == "eq":
        p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) == 1)
        # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) == 0)
        for j in gams:
            p.add_constraint(
                p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) == 0
            )
    elif constraints == "ge":
        p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) >= 1)
        # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) >= 0)
        for j in gams:
            p.add_constraint(
                p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) >= 0
            )
    else: #constraints == "le":
        p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) <= 1)
        # p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) >= 0)
        for j in gams:
            p.add_constraint(
                p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) <= 0
            )

    for i in mflips:
        Lm = Ls[i]["L"]
        ms = set(w for w in Lm.N2u if Lm.spin_assignment[w] == "-")
        # print(f"mflip index {i}")
        # Lm.show()

        orbitms = []
        while ms:
            w = next(iter(ms))
            orbit = set(Lm.orbit(w))
            ms -= orbit
            orbitms.append(orbit)
        print(f"mflip index {i}: orbitms = {orbitms}")

        for orbitm in orbitms:
            w = next(iter(orbitm))
            # print(f"mflip index {i}: orbitm = {orbitm}, w={w}")
            Lp = Lm.change_spin(w)
            # Lp.show()
            orbitp = set(Lp.orbit(w))
            # print(f"mflip index {i}: orbitp = {orbitp}")

            Lpcan = Lp.fullG_can_fixed_spins
            j = 0
            while (Ls[j]['L'].fullG_can_fixed_spins != Lpcan):
                j += 1

            p.add_constraint(len(orbitm)*x[i] >= Bval**d/lval * len(orbitp) * x[j])
            print(f"mflip index {i}: constraint {len(orbitm)} * x[{i}] >= B^{d}/lam * {len(orbitp)} * x[{j}]")
    

    p.set_objective(p.sum(L["pu"] * x[i] for i, L in enumerate(Ls)))
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
        eqns.append([0] + [L["gu"][j] - L["gNu"][j] for L in Ls])

    for i in mflips:
        Lm = Ls[i]["L"]
        ms = set(w for w in Lm.N2u if Lm.spin_assignment[w] == "-")
        # print(f"mflip index {i}")
        # Lm.show()

        orbitms = []
        while ms:
            w = next(iter(ms))
            orbit = set(Lm.orbit(w))
            ms -= orbit
            orbitms.append(orbit)
        # print(f"mflip index {i}: orbitms = {orbitms}")

        for orbitm in orbitms:
            w = next(iter(orbitm))
            # print(f"mflip index {i}: orbitm = {orbitm}, w={w}")
            Lp = Lm.change_spin(w)
            # Lp.show()
            orbitp = set(Lp.orbit(w))
            # print(f"mflip index {i}: orbitp = {orbitp}")

            Lpcan = Lp.fullG_can_fixed_spins
            j = 0
            while (Ls[j]['L'].fullG_can_fixed_spins != Lpcan):
                j += 1
            ieq = [0] * (len(Ls) + 1)
            ieq[i+1] = len(orbitm)
            ieq[j+1] = -Bval**d/lval * len(orbitp)
            ieqs.append(ieq)
            print(f"mflip index {i}: constraint {len(orbitm)} * x[{i}] >= B^{d}/lam * {len(orbitp)} * x[{j}]")
    
    # slow
    print("Generating polyhedron")
    pol = Polyhedron(ieqs=ieqs, eqns=eqns, base_ring=AA)

    print("Generating LP")
    p, x = pol.to_linear_program(solver='InteractiveLP', return_variable=True)
    p.set_objective(p.sum(AA(-L["pu"]) * x[i] for i, L in enumerate(Ls)))
    p.set_min(x, 0)
    return p, x