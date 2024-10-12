# %%
import os.path

from collections import namedtuple
from itertools import product

from sage.all import diff, graphs, ln, load, Rational, save, srange, sqrt, var
from sage.numerical.mip import MixedIntegerLinearProgram

from tqdm.autonotebook import tqdm

# Generating local views
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
def gen_lp_via_poly(d, Bval, lval, Ls=None, gams=None):
    if Ls is None:
        Ls = get_data(d).Ls
    if gams is None:
        gams = range(d+1)
        
    Ls = sub_Ls(Ls, Bval, lval)

    constraints = []
    # probability constraint
    constraints.append([-1] + [1 for L in Ls])
    for j in gams:
        constraints.append([0] + [L["gu"][j] - L["gNu"][j] for L in Ls])

    # slow
    pol = Polyhedron(eqns=constraints, base_ring=AA)

    p, x = pol.to_linear_program(solver='InteractiveLP', return_variable=True)
    p.set_objective(p.sum(AA(-L["pu"]) * x[i] for i, L in enumerate(Ls)))
    p.set_min(x, 0)
    return p, x

# %%
d = 3
Ls = get_data(d).Ls # use all local views, including K_4
print(f'using {len(Ls)} local views')
# Bval = 31295/100000 # program gives K_4
Bval = 32/100 # program goes below K_4
lval = lc(d,Bval).simplify_full()
print(f'B = {n(Bval)}...')
print(f'l = {n(lval)}...')

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()
print("building program...")
gams = [0,1,2,3]
p, x = gen_lp_via_poly(d, Bval, lval, Ls, gams=gams)
# p.show()
print("solving program...")
opt = -p.solve()
print(f"prog: {opt} = {opt.n()}")
print(f"K_{d+1} = {occKd1.subs(B=Bval, l=lval).n()}...")
xvals = p.get_values(x)
# %%
