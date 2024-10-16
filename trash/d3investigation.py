# %%
# Dependencies:
#   sagemath, tested with version >= 10.2
#   tqdm, installed below
#   wolfram engine with the wolframscript command in the PATH

%pip install tqdm


# %%

import os.path
import subprocess
import sys

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
    pN2u = 0
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

            for w in L.G.neighbors(v):
                if sigma[w] == "+":
                    pN2u += weight / d ** 2

    return {
        "pu": pu / Z,
        "pNu": pNu / Z,
        "pN2u": pN2u / Z,
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
    
    # print(f"primal val: {primal.optimal_value()}")
    # print(f"dual val:   {dual.optimal_value()}") 
    
    y = dual.optimal_solution()

    for yi in y:
        print(f'y: {yi}')

    A = dual.A()
    b = dual.b()

    tight = []
    for i, row in enumerate(A):
        if row*y == b[i]:
            tight.append(i)
            print(f'dual constraint {i} is tight')

    subA = matrix([A.row(i) for i in tight])
    subb = vector([b[i] for i in tight])

    # print(y)
    # print(subA.solve_right(subb))

    return # primal, dual, tight

# %% Generic Setup
d = 3
Ls = get_data(d).Ls[:-1] # exclude K_4
gams = list(range(d-1)) # [0,...,d-2]

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()


# %%
Bval = 32/100
lval = Rational(lc(d,Bval).n(digits=100))
occKd1val = occKd1.subs(B=Bval, l=lval)

print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")

constraint_type = 'eq'
p, x = gen_lp(d, Bval, lval, Ls, solver="InteractiveLP", gams=gams, constraints=constraint_type)
occlb = p.solve()
print(f"program: {occlb.n()}")
if occlb > occKd1val:
    print("Good, program without K_4 is greater than K_4")
    investigate_lp(p)
else:
    print("Bad, program without K_4 is less than K_4")


# %%

# %%

# %%

# %%

tights =  [[15, 20, 21], [8, 16, 21],      [13, 16, 21]]
Branges = [[0, 3/10],    [30/100, 311/1000], [311/1000, 31294/100000]]

# Atranspose = matrix([
#     [Ls[t]["gu"][j] - Ls[t]["gNu"][j] for j in range(d-1)] for t in tight[:d-1]
# ])
# c = vector(Ls[t]["pu"]-occKd1 for t in tight[:d-1])
# ys = Atranspose.solve_right(c)

filename = "data/d3.wls"
with open(filename, "w") as f:

    f.write(
f"""Print["Running code with Wolfram Engine..."];
d = 3;
Bc[d_] := (d-2)/d;
lc[d_, 0]  := 0;
lc[d_, B_] := Block[{{r,s,Bca}},(((1 + Sqrt[r s])/(1 - Sqrt[r s]))^((Bca + 1)/(Bca - 1)) (1 - Sqrt[r/s])/(1 + Sqrt[r/s]) 
            /. {{r -> (Bca - B)/(Bca + B), 
                s -> (1 - B)/(1 + B)}}
            /.Bca -> Bc[d])];
""")


    for i, (tight, [Bmin, Bmax]) in enumerate(zip(tights, Branges), start=1):

        Atranspose = matrix([
            [1] + [Ls[t]["gu"][j] - Ls[t]["gNu"][j] for j in range(d-1)] for t in tight
        ])
        c = vector(Ls[t]["pu"] for t in tight)
        allys = Atranspose.solve_right(c)
        ys = allys[1:] # drop y_p which we will set to alpha(K_4)



        f.write(f"""
ineqs{i} = Table[True, {len(Ls)}];
""")

        for j, L in enumerate(Ls, start=1):
            ineq = 0 <= L["pu"] - occKd1 - sum(ys[k] * (L["gu"][k] - L["gNu"][k]) for k in range(d-1))
            f.write(f"ineqs{i}[[{j}]] = {ineq};\n")

    # f.write("Table[Simplify[ineq,{params}]//AbsoluteTiming, {ineq, ineqs}]\n")

        f.write(f"""
Bmin{i} = {Bmin}; Bmax{i} = {Bmax};

idxchunks = Partition[Range[Length[ineqs{i}]], UpTo[6]];
ineqchunks = Partition[ineqs{i}, UpTo[6]];

(* t{i} := Table[Plot[{{0}}~Join~Table[ineq[[2]]/.l->lc[d,B], {{ineq, ineqchunks[[n]]}}]//Evaluate,{{B,Bmin{i},Bmax{i}}},
          PlotLabel->"Range {i}", PlotLegends->Placed[LineLegend[97,{{0}}~Join~idxchunks[[n]],LegendLayout->Row,LabelStyle->Directive[Small]],{{Center,Top}}], AspectRatio->Full], {{n,Length[idxchunks]}}];
g{i} := GraphicsColumn[Table[Show[p, ImageSize->{{300, 100}}], {{p,t{i}}}], Spacings->{{0,Scaled[0.1]}}]; *)
params{i} = {{lc[d,Bmin{i}]<=l<=lc[d,B] && Bmin{i} <= B <= Bmax{i}}};
a{i} = Table[Simplify[ineq,{{params{i}}}]//AbsoluteTiming, {{ineq, ineqs{i}}}];
time{i} = Sum[r[[1]],{{r,a{i}}}];
ans{i} = And @@ Table[r[[2]],{{r,a{i}}}];
Print["Inequalities{i}: " <> ToString[ans{i}] <> " in time " <> ToString[time{i}] <> "s"]
""")

process = subprocess.Popen(["wolframscript", "-f", filename, "-print all"], stdout=subprocess.PIPE)
for line in process.stdout:
    print(line.decode(), end='')


# %%

tights =  list(map(list, Subsets(range(len(Ls)), 3)))
Bval = 32/100
lval = lc(d,Bval)

for tight in tqdm(tights):

    Atranspose = matrix([
        [1] + [Ls[t]["gu"][j] - Ls[t]["gNu"][j] for j in range(d-1)] for t in tight
    ])
    c = vector(Ls[t]["pu"] for t in tight)
    allys = Atranspose.solve_right(c)
    ys = allys[1:] # drop y_p which we will set to alpha(K_4)

    ans = all( 0 <= (L["pu"] - occKd1 - sum(ys[k] * (L["gu"][k] - L["gNu"][k]) for k in range(d-1))).subs(B=Bval,l=lval) for L in Ls)
    if ans: 
        print(f"{tight}: {ans}")




# %%
d = 3
Ls = get_data(d).Ls[:-1] # exclude K_4

Bval = 30/100
lval = lc(d,Bval)

A = matrix([
    [1 for _ in range(len(Ls))],
    # [L["pu"] - L["pNu"] for L in Ls],
    # [L["pu"] - L["pN2u"] for L in Ls],
    [L["gu"][0] - L["gNu"][0] for L in Ls],
    [L["gu"][1] - L["gNu"][1] for L in Ls],
    [L["gu"][2] - L["gNu"][2] for L in Ls],
    [L["gu"][3] - L["gNu"][3] for L in Ls],
])

As = A.subs(B=Bval, l=lval)

print(As.rank(), A.rank())