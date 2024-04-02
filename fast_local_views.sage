# %%
from collections import namedtuple
from itertools import product

from sage.all import diff, graphs, ln, load, Rational, save, srange, sqrt, var
from sage.numerical.mip import MixedIntegerLinearProgram

from tqdm.autonotebook import tqdm

# Generating local views
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
    bc = var("bc", latex_name="B_c")

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

    d = L.degree(0)
    pu = 0
    pNu = 0
    Z = 0
    gu = [0 for _ in range(d + 1)]
    gNu = [0 for _ in range(d + 1)]

    for sigma in L.gen_all_spin_assignments(tqdm):
        weight = B ** mono(L, sigma) * l ** nplus(L, sigma)
        Z += weight

        if sigma[0] == "+":
            pu += weight
        gu[sum(1 for v in L.neighbors(0) if sigma[v] == "+")] += weight

        for v in L.neighbors(0):
            if sigma[v] == "+":
                pNu += weight / d
            gNu[sum(1 for w in L.neighbors(v) if sigma[w] == "+")] += weight / d

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


# def default_filename(d):
#     return f"data/ising_d{d}.obj"


def gen_data(d, filename=None):
    # if filename is None:
    #     filename = default_filename(d)

    Ls = []
    for L in gen_local_view_2(d, spins=ising_spins):
        ps = compute_probabilities(L, tqdm=tqdm)
        ps["L"] = L
        Ls.append(ps)
    data = LData(Ls, DATA_VERSION)
    # save(data, filename)
    return data


def load_data(d, filename=None):
    return gen_data(d, filename)
    # if filename is None:
    #     filename = default_filename(d)

    # data = load(filename)
    # return data


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
def gen_lp(d, Bval, lval, Ls=None, solver="GLPK"):
    if Ls is None:
        Ls = load_data(d).Ls
    Ls = sub_Ls(Ls, Bval, lval)

    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    x = p.new_variable(nonnegative=True)

    p.add_constraint(p.sum(x[i] for i, L in enumerate(Ls)) == 1)
    p.add_constraint(p.sum((L["pu"] - L["pNu"]) * x[i] for i, L in enumerate(Ls)) == 0)
    for j in range(d + 1):
        p.add_constraint(
            p.sum((L["gu"][j] - L["gNu"][j]) * x[i] for i, L in enumerate(Ls)) == 0
        )

    p.set_objective(p.sum(L["pu"] * x[i] for i, L in enumerate(Ls)))
    return p


# %%
# WARNING: d=3,4 are tolerable, d=5 is very slow
d = 3

Ls = gen_data(d).Ls

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1, sigma) * l ** nplus(Kd1, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.order()

Bstart = 1 / 100
Bend = (d - 2) / d
Bstep = (Bend - Bstart) / 5
for Bval in srange(start=Bstart, end=Bend, step=Bstep):

    Bval = Rational(Bval)
    lstart = (lc(d, Bval)/1000).n(digits=16)
    lend = lc(d, Bval)
    lstep = (lend - lstart) / 5
    for lval in srange(start=lstart, end=lend, step=lstep):
        lval = Rational(lval.n(digits=16))

        p = gen_lp(d, Bval, lval, Ls, solver="PPL")

        print(
            f"B = {Bval.n(digits=16)}, Bc = {Bend.n(digits=16)}, l = {lval.n(digits=16)}, lc = {lend.n(digits=16)}"
        )
        print(f"\tprog =\t{p.solve()}")
        print(f"\tK_{d+1} =\t{occKd1.subs(B=Bval, l=lval)}")


# %%
