# %%
load("local_occupancy/ising_occ.py")

# %%
d = 3
spin_depth = 2
Ls = get_data(d, spin_depth).Ls # do not exclude K_4
gams = list(range(d+1)) # all the gamma constraints [0,...,d]
mflips = []

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()

# %% Run primal LP with rational sage solver
Bval = 32/100
lval = Rational(lc(d,Bval).n(digits=100))
occKd1val = occKd1.subs(B=Bval, l=lval)

constraint_type = 'eq'
p1, x1 = gen_lp(d, spin_depth, Bval, lval, Ls, solver="PPL", gams=gams, constraints=constraint_type, mflips=mflips)
occlb1 = p1.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"program: {occlb1.n()}")
if occlb1 >= occKd1val:
    print("Good, program is at least K_4")
else:
    print("Bad, program without K_4 is less than K_4")

# %% Run primal LP with exact solver over algebraic reals
Bval = 32/100
lval = lc(d,Bval)
occKd1val = occKd1.subs(B=Bval, l=lval)

p2, x2 = gen_lp_via_poly(d, Bval, lval, Ls, gams, mflips)
occlb2 = -p2.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"program: {n(occlb2)}")
if occlb2 >= occKd1val:
    print("Good, program is at least K_4")
else:
    print("Bad, program without K_4 is less than K_4")


# %%
d = 3
spin_depth = 3
gams = list(range(d+1)) # all the gamma constraints [0,...,d]
mflips = []

B, l = var("B, l")
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()

# %% Run primal LP with rational sage solver
Ls = get_data(d, spin_depth).Ls # do not exclude K_4
Bval = 1/3
lval = Rational(lc(d,Bval).n(digits=5))
occKd1val = occKd1.subs(B=Bval, l=lval)
print('Loaded data...')
constraint_type = 'eq'
p1, x1 = gen_lp(d, spin_depth, Bval, lval, Ls, solver="PPL", gams=gams, constraints=constraint_type, mflips=mflips)
K4idx = len(Ls)-1
p1.add_constraint(x1[K4idx]==0)
print('solving...')
occlb1 = p1.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"program: {n(occlb1)}")
# if occlb1 >= occKd1val:
#     print("Good, program is at least K_4")
# else:
#     print("Bad, program without K_4 is less than K_4")

vals = p1.get_values(x1)
support = []
for i, v in vals.items():
    if v > 0:
        support.append(i)
        print({i: v})
#%%
for i in support:
    Ls[i].show()

#########
# Old
#########
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