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
def gen_lp(d, Bval, lval, Ls=None, solver="GLPK", gams=None, constraints="eq"):
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

    p.set_objective(p.sum(L["pu"] * x[i] for i, L in enumerate(Ls)))
    return p, x

def investigate_lp(p, verbose=False):

    # convert the program into interactive form so we can see 
    # the standard form as max x.c such that Ax <= b
    primal = p.get_backend().interactive_lp_problem().standard_form()
    dual = primal.dual().standard_form()

    if verbose:
        print('primal:')
        show(primal)
        print()
        print('dual:')
        show(dual)
    
    print(f"primal val: {primal.optimal_value()}")
    print(f"dual val:   {dual.optimal_value()}") 
    
    
    y = dual.optimal_solution()

    A = dual.A()
    b = dual.b()

    # show(y)
    # show(A[:].solve_right(b[:]))

    tight = []
    for i, row in enumerate(A):
        if row*y == b[i]:
            tight.append(i)
            print(f'dual constraint {i} is tight')


    subA = matrix([A.row(i) for i in tight])
    subb = vector([b[i] for i in tight])


    print(y)
    print(subA.solve_right(subb))

    return primal, dual, tight

# %%
# WARNING: d=3,4 are tolerable, d=5 is slow
d = 3

Ls = get_data(d).Ls

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()

Bstart = 1 / 100
Bend = (d - 2) / d
Bstep = (Bend - Bstart) / 5
for Bval in [k*(d-2)/d/100 for k in [50, 92, 96, 99]]:

    Bval = Rational(Bval)
    lstart = (lc(d, Bval)/1000).n(digits=16)
    lend = lc(d, Bval)
    lstep = (lend - lstart) / 5
    for lval in [99/100 * lc(d, Bval)]:
        lval = Rational(lval.n(digits=16))

        p, x = gen_lp(d, Bval, lval, Ls, solver="GLPK")

        print(
            f"B = {Bval.n(digits=16)}, Bc = {Bend.n(digits=16)}, l = {lval.n(digits=16)}, lc = {lend.n(digits=16)}"
        )
        print(f"\tprog =\t{p.solve()}")
        print(f"\tprog =\t{n(p.solve())}")
        print(f"\tK_{d+1} =\t{occKd1.subs(B=Bval, l=lval)}")
        print(f"\tK_{d+1} =\t{occKd1.subs(B=Bval, l=lval).n()}")


# %%
d = 3
Ls = get_data(d).Ls[:-1]
Bval = 1/4
lval = 1/20 

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()
print(f"K_{d+1}:     {occKd1.subs(B=Bval, l=lval)} = {occKd1.subs(B=Bval, l=lval).n()}")

gams = [0,1,2,3]
constraint_type = 'ge'
p, x = gen_lp(d, Bval, lval, Ls, solver="InteractiveLP", gams=gams, constraints=constraint_type)
print(f"program: {p.solve()} = {p.solve().n()}")
xvals = p.get_values(x)

# print({k: v for k, v in xvals.items() if v > 0})
# print(f"constraint type: {constraint_type}")
# for (i, (lb, (indices, coefficients), ub)) in enumerate(p.constraints()):
#     f = sum(coefficients[j] * x[j] for j in indices)
#     g = sum(coefficients[j] * xvals[j] for j in indices)
#     # print(f"{lb} <= {f} <= {ub}")
#     if lb == ub:
#         # skip equality constraints, of course they hold with equality
#         continue

#     if lb is not None:
#         eq = (lb == g)
#         if eq:
#             print(f"c_{i} lb tight: {lb} == {f}")
#     if ub is not None:
#         eq = (ub == g)
#         if eq:
#             print(f"c_{i} ub tight: {ub} == {f}")

investigate_lp(p)

# %%


# %%
d = 3
Ls = get_data(d).Ls[:-1] # no K_{d+1} please
if d == 3:
    tight = [15, 20, 21]
elif d == 4:
    tight = [198, 206, 216, 228]
else:
    raise ValueError(f"no plan for d={d}")

Atranspose = matrix([
    [1] + [Ls[t]["gu"][j] - Ls[t]["gNu"][j] for j in range(d-1)] for t in tight
])
c = vector(Ls[t]["pu"] for t in tight)
ys = Atranspose.solve_right(c)

# TODO: prove for all Ls, ys[0] + sum(ys[j] * (L["gu"][j] - L["gNu"][j]) for j in range(d-1)) <= L["pu"]
for i, L in enumerate(Ls):
    ineq = (ys[0] + sum(ys[j+1] * (L["gu"][j] - L["gNu"][j]) for j in range(d-1)) <= L["pu"])
    print(ineq)
    if not((ys[0] + sum(ys[j+1] * (L["gu"][j] - L["gNu"][j]) for j in range(d-1)) <= L["pu"]).subs(B=16/100,l=19/1000)):
        print(f"counterexample found at index {i}")