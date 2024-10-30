# Dependencies:
#   sagemath, tested with version >= 10.2
#   tqdm, installed below
#   wolfram engine with the wolframscript command in the PATH

# %% Install tqdm through pip
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

# %% Imports
# from tqdm.autonotebook import tqdm

load("ising_occ.py")

# %% Setup
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

# %% Dual feasibility
tights =  [[15, 20, 21], [8, 16, 21],      [13, 16, 21]]
Branges = [[0, 3/10],    [30/100, 311/1000], [311/1000, 31294/100000]]


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
            [1] + [Ls[t].data["gs"][0][j] - Ls[t].data["gs"][1][j] for j in range(d-1)] for t in tight
        ])
        c = vector(Ls[t].data["ps"][0] for t in tight)
        allys = Atranspose.solve_right(c)
        ys = allys[1:] # drop y_p which we will set to alpha(K_4)



        f.write(f"""
ineqs{i} = Table[True, {len(Ls)}];
""")

        for j, L in enumerate(Ls, start=1):
            ineq = 0 <= L.data["ps"][0] - occKd1 - sum(ys[k] * (L.data["gs"][0][k] - L.data["gs"][1][k]) for k in range(d-1))
            f.write(f"ineqs{i}[[{j}]] = {ineq};\n")

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
