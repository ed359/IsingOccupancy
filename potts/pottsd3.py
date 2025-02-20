# Dependencies:
#   sagemath, tested with version >= 10.2
#   tqdm, installed below
#   wolfram engine with the wolframscript command in the PATH

# %% Install tqdm through pip
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

# %% Imports
# from tqdm.autonotebook import tqdm

from sage.symbolic.expression import Expression
from sage.rings.rational import Rational
from sage.rings.integer import Integer

from dataclasses import dataclass
from typing import List

load("ising_occ.py")

@dataclass
class DualFeasibility:
    tight_constraints: int
    Bmin: str
    Bmax: str
    lmin: str
    lmax: str

# %% Setup
run_wolfram = False
d = 3
spin_depth = 2
Ls = get_data(d, spin_depth).Ls[:-1] # exclude K_4 which is the last local view
gams = list(range(d-1)) # [0,...,d-2]

B, l = var("B, l")
K4 = graphs.CompleteGraph(d + 1)
occK4 = occ(K4, B, l)

tights = [
    [15,20,21],
    [15,16,21],
    [8,16,21],
    [13,16,21],
]

# TODO: check carefully that the union of the dfs is closed under making l smaller.
dfs = [
    DualFeasibility(0, '59/100', '1', '0', '(177*B)/200 - (267*B^2)/1000'), # True
    DualFeasibility(0, '0', '59/100', '0', 'B*9/10'), # True
    DualFeasibility(1, '1/10', '1/2', 'B/2', 'B*99/100'), # True
    DualFeasibility(1, '1/2', '9/10', '3/20', 'Min[B,53/100]'), # True
    DualFeasibility(1, '3/5', '98/100', 'B*11/50 + 35/100', 'B*12/50+375/1000'), # True
    DualFeasibility(2, '1/4', '9/20', 'B*49/50', 'B*7/5-85/1000'), # True
    DualFeasibility(2, '56/125', '3/5', 'B*99/100', '133/200 - B*13/50'), # True
    DualFeasibility(3, '26/100', '36/100', 'B*68/50-75/1000', 'B*72/50-90/1000'), # True
]

filename = "data/d3min.wls"
with open(filename, "w") as f:

    f.write(
f"""Print["Running code with Wolfram Engine...\n"];
d = 3;
Bc[d_] := (d-2)/d;
lc[d_, 0]  := 0;
lc[d_, B_] := Block[{{r,s,Bca}},(((1 + Sqrt[r s])/(1 - Sqrt[r s]))^((Bca + 1)/(Bca - 1)) (1 - Sqrt[r/s])/(1 + Sqrt[r/s]) 
            /. {{r -> (Bca - B)/(Bca + B), 
                s -> (1 - B)/(1 + B)}}
            /.Bca -> Bc[d])];
colors = PadRight[ColorData[97, "ColorList"],100,ColorData[97, "ColorList"]];

""")

    for i, tight_constraints in enumerate(tights, start=1):
        f.write(f'\nPrint["Tight constraints {i}: {tight_constraints}"];\n')

        Atranspose = matrix([
            [1] + [Ls[t].data["gs"][0][j] - Ls[t].data["gs"][1][j] for j in gams] for t in tight_constraints
        ])
        c = vector(Ls[t].data["ps"][0] for t in tight_constraints)
        allys = Atranspose.solve_right(c)
        ys = allys[1:] # drop y_p which we will set to occK4

        f.write(f"ineqs{i} = Table[True, {len(Ls)}];\n")

        for j, L in enumerate(Ls, start=1):
            ineq = 0 <= L.data["ps"][0] - occK4 - sum(ys[k] * (L.data["gs"][0][k] - L.data["gs"][1][k]) for k in gams)
            if j-1 in tight_constraints: # Python indices are 0-based, Wolfram indices are 1-based
                f.write(f"ineqs{i}[[{j}]] = {ineq} // Simplify;\n")
            else:
                f.write(f"ineqs{i}[[{j}]] = {ineq};\n")

        f.write(f"tR{i} = Hold[ImplicitRegion[And @@ ineqs{i}, {{B,l}}]];\n")
        f.write(f"tr{i} = Hold[RegionPlot[And @@ ineqs{i}, {{B,0,1}}, {{l,0,1}}, PlotPoints->40, MaxRecursion->4, BoundaryStyle->None, PlotStyle->{{Directive[colors[[1]],Opacity[1]]}}]];\n")

    f.write("\n")
    f.write(f"tRall := RegionUnion @@ {{ {', '.join(f'tR{i} // ReleaseHold' for i, _ in enumerate(tights, start=1))} }};\n")
    f.write(f"Print[""]\n")

    for i, df in enumerate(dfs, start=1):
        t = df.tight_constraints + 1
        f.write(f'\n\n')
        f.write(f'Print["Dual Feasibility {i}: tight constraints {tights[t-1]}"];\n')
        f.write(f'Print["Dual Feasibility {i}: {df.Bmin} <= B <= {df.Bmax}"];\n')
        f.write(f'Print["Dual Feasibility {i}: {df.lmin} <= l <= {df.lmax}"];\n')
        f.write(f'Print[""]\n')

        f.write(f"""
Bmin{i} = {df.Bmin}; Bmax{i} = {df.Bmax};
lmin{i} = {df.lmin}; lmax{i} = {df.lmax};
params{i} = {{lmin{i} <= l <= lmax{i} && Bmin{i} <= B <= Bmax{i}}};

idxchunks{i} = Partition[Range[Length[ineqs{t}]], UpTo[6]];
ineqchunks{i} = Partition[ineqs{t}, UpTo[6]];

dfR{i} := ImplicitRegion[Bmin{i} <= B <= Bmax{i} && lmin{i} <= l <= lmax{i},{{B,l}}];
dfr{i} := RegionPlot[Bmin{i} <= B <= Bmax{i} && lmin{i} <= l <= lmax{i}, {{B,0,1}}, {{l,0,1}}, PlotPoints->100, MaxRecursion->4, BoundaryStyle->None, PlotStyle->{{Directive[colors[[{4}]],Opacity[0.5]]}}];

test{i} := Block[{{}},
{{time{i}, a{i}}} = Table[TrueQ[Simplify[ineq,{{params{i}}}]], {{ineq, ineqs{t}}}]//AbsoluteTiming;
ans{i} = And @@ a{i};
Print["Dual Feasibility {i}: " <> ToString[ans{i}] <> " in time " <> ToString[time{i}] <> "s"]
];
""")

    f.write("\n\n")
    f.write(f"dfRall := RegionUnion @@ {{ {', '.join(f'dfR{i}' for i, _ in enumerate(dfs, start=1))} }};\n")

    f.write("\n")
    f.write('Print["Performing dual feasibility tests..."];\n')
    for i, df in enumerate(dfs, start=1):
        f.write(f"test{i};\n")

if run_wolfram:
    process = subprocess.Popen(["wolframscript", "-f", filename, "-print all"], stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode(), end='')


# %%
