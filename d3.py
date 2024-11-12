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
    tight_constraints: List[List[int]]
    Bmin: str
    Bmax: str
    lmin: str
    lmax: str

# %% Setup
run_wolfram = False
d = 3
spin_depth = 2
Ls = get_data(d, spin_depth).Ls[:-1] # exclude K_4
gams = list(range(d-1)) # [0,...,d-2]

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()

df1 = DualFeasibility([15, 20, 21], '0', '3/10', '0', 'lc[d, B]')
df2 = DualFeasibility([8, 16, 21], '3/10', '311/1000', 'lc[d, 3/10]', 'lc[d, B]')
df3 = DualFeasibility([13, 16, 21], '311/1000', '31294/100000', 'lc[d, 311/1000]', 'lc[d, B]')
df4 = DualFeasibility([15, 16, 21], '1/10', '1/10', '99/1000', '99/1000')
dfs = [df1, df2, df3, df4]

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
colors = PadRight[ColorData[97, "ColorList"],100,ColorData[97, "ColorList"]];
""")


    for i, df in enumerate(dfs, start=1):
        f.write(f'\n\n')
        f.write(f'Print["Dual Feasibility {i}: tight constraints {df.tight_constraints}"];\n')
        f.write(f'Print["Dual Feasibility {i}: {df.Bmin} <= B <= {df.Bmax}"];\n')
        f.write(f'Print["Dual Feasibility {i}: {df.lmin} <= l <= {df.lmax}"];\n')
        f.write(f'Print[""]\n')

        Atranspose = matrix([
            [1] + [Ls[t].data["gs"][0][j] - Ls[t].data["gs"][1][j] for j in range(d-1)] for t in df.tight_constraints
        ])
        c = vector(Ls[t].data["ps"][0] for t in df.tight_constraints)
        allys = Atranspose.solve_right(c)
        ys = allys[1:] # drop y_p which we will set to alpha(K_4)

        f.write(f"ineqs{i} = Table[True, {len(Ls)}];\n")

        for j, L in enumerate(Ls, start=1):
            ineq = 0 <= L.data["ps"][0] - occKd1 - sum(ys[k] * (L.data["gs"][0][k] - L.data["gs"][1][k]) for k in range(d-1))
            f.write(f"ineqs{i}[[{j}]] = {ineq};\n")

        f.write(f"""
Bmin{i} = {df.Bmin}; Bmax{i} = {df.Bmax};
lmin{i} = {df.lmin}; lmax{i} = {df.lmax};
params{i} = {{lmin{i} <= l <= lmax{i} && Bmin{i} <= B <= Bmax{i}}};

idxchunks{i} = Partition[Range[Length[ineqs{i}]], UpTo[6]];
ineqchunks{i} = Partition[ineqs{i}, UpTo[6]];

t{i} := Table[Plot[{{0}}~Join~Table[ineq[[2]]/.l->lc[d,B], {{ineq, ineqchunks{i}[[n]]}}]//Evaluate,{{B,Bmin{i},Bmax{i}}},
          PlotLabel->"Range {i}", PlotLegends->Placed[LineLegend[97,{{0}}~Join~idxchunks{i}[[n]],LegendLayout->Row,LabelStyle->Directive[Small]],{{Center,Top}}], AspectRatio->Full], {{n,Length[idxchunks{i}]}}];
g{i} := GraphicsColumn[Table[Show[p, ImageSize->{{300, 100}}], {{p,t{i}}}], Spacings->{{0,Scaled[0.1]}}];
r{i} := RegionPlot[{{Bmin{i} <= B <= Bmax{i}, lmin{i} <= l <= lmax{i}}}, {{B, 0, 1}}, {{l, 0, 1}}, PlotPoints->100, MaxRecursion->4, PlotStyle->{{Directive[colors[[{i}]],Opacity[0.2]]}}, BoundaryStyle->None];

test{i} := Block[{{}},
{{time{i}, a{i}}} = Table[TrueQ[Simplify[ineq,{{params{i}}}]], {{ineq, ineqs{i}}}]//AbsoluteTiming;
ans{i} = And @@ a{i};
Print["Dual Feasibility {i}: " <> ToString[ans{i}] <> " in time " <> ToString[time{i}] <> "s"]
];
""")
    
    f.write("\n\n\n")
    f.write('Print["Performing dual feasibility tests..."];\n')
    for i, df in enumerate(dfs, start=1):
        f.write(f"test{i};\n")

if run_wolfram:
    process = subprocess.Popen(["wolframscript", "-f", filename, "-print all"], stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode(), end='')



# %%
