# %%
load("ising_occ.py")

d = 3
gen_data(d) # regenerate data
Ls = get_data(d).Ls # do not exclude K_4
gams = list(range(d+1)) # all the gamma constraints [0,...,d]

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

constraint_type = 'eq'
mflips = range(len(Ls))
p, x = gen_lp(d, Bval, lval, Ls, solver="InteractiveLP", gams=gams, constraints=constraint_type, mflips=mflips)
occlb = p.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"program: {occlb.n()}")
if occlb >= occKd1val:
    print("Good, program is at least K_4")
    # investigate_lp(p)
else:
    print("Bad, program without K_4 is less than K_4")




#########
# Old
#########
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