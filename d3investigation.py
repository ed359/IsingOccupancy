# %%
load("ising_occ.py")

from itertools import batched
for batch in batched(gen_local_views(3,2), 4):
    graphics_array([L.plot() for L in batch]).show()

for batch in batched(gen_local_views_par(3,2), 4):
    graphics_array([L.plot() for L in batch]).show()

# %%
d = 3
spin_depth = 2
Ls = get_data(d, spin_depth).Ls # do not exclude K_4
gams = list(range(d+1)) # all the gamma constraints [0,...,d]
mflips = []

B, l = var("B, l")
Pet = LocalView(graphs.PetersenGraph(), ising_spins, [])
ZPet = sum(
    B ** mono(Pet.G, sigma) * l ** nplus(Pet.G, sigma)
    for sigma in Pet.gen_all_spin_assignments()
)
occPet = l * diff(ln(ZPet), l) / Pet.G.order()
Kd1 = LocalView(graphs.CompleteGraph(d + 1), ising_spins, [])
ZKd1 = sum(
    B ** mono(Kd1.G, sigma) * l ** nplus(Kd1.G, sigma)
    for sigma in Kd1.gen_all_spin_assignments()
)
occKd1 = l * diff(ln(ZKd1), l) / Kd1.G.order()

# Run primal LP with rational sage solver
Bval = 1/1000
lval = 1/2 # Rational(lc(d,Bval).n(digits=100))
occKd1val = occKd1.subs(B=Bval, l=lval)
occPetval = occPet.subs(B=Bval, l=lval)

constraint_type = 'eq'
p1, x1 = gen_lp(d, spin_depth, Bval, lval, Ls, solver="PPL", gams=gams, constraints=constraint_type, mflips=mflips)
occlb1 = p1.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"Pet:     {occPetval.n()}")

print(f"program: {occlb1.n()}")
# if occlb1 >= occKd1val:
#     print("Good, program is at least K_4")
# else:
#     print("Bad, program without K_4 is less than K_4")

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

Pet = LocalView(graphs.PetersenGraph(), ising_spins, [])
ZPet = sum(
    B ** mono(Pet.G, sigma) * l ** nplus(Pet.G, sigma)
    for sigma in Pet.gen_all_spin_assignments()
)
occPet = l * diff(ln(ZPet), l) / Pet.G.order()

Ls = get_data(d, spin_depth).Ls # do not exclude K_4
Bval = 1/1000
lval = 500/1000 # Rational(lc(d,Bval).n(digits=50))
occKd1val = occKd1.subs(B=Bval, l=lval)
occPetval = occPet.subs(B=Bval, l=lval)

print('Loaded data...')
constraint_type = 'eq'
p1, x1 = gen_lp(d, spin_depth, Bval, lval, Ls, solver="GLPK", gams=gams, constraints=constraint_type, mflips=mflips)
# K4idx = len(Ls)-1
# p1.add_constraint(x1[K4idx]==0)
print('solving...')
occlb1 = p1.solve()
print(f"B: {Bval.n()}, l: {lval.n()}")
print(f"K_{d+1}:     {occKd1val.n()}")
print(f"Pet:     {occPetval.n()}")
print(f"program: {n(occlb1)}")
# if occlb1 >= occKd1val:
#     print("Good, program is at least K_4")
# else:
#     print("Bad, program without K_4 is less than K_4")

vals = p1.get_values(x1)
support = []
for i, v in vals.items():
    if v > 10**-4:
        support.append(i)
        # print({i: v})

for batch in batched((Ls[i].plot() for i in support), 3):
    graphics_array(batch).show()


# Petersen wins:
# B = 0.001, l = 0.5 but not l=0.25, 0.45, 0.75, 0.499 or 0.501
#%%


#########
# Old
#########
# %%

tights =  list(map(list, Subsets(range(len(Ls)), 3)))
Bval = 325/1000
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

# %%

B, l = var("B, l")
z = var('z')
K4 = LocalView(graphs.CompleteGraph(4), ising_spins, [])
ZK4 = sum(
    B ** mono(K4.G, sigma) * l ** nplus(K4.G, sigma)
    for sigma in K4.gen_all_spin_assignments()
)
FK4 = ln(ZK4)/K4.G.order()
occK4 = l * diff(FK4, l)

Pet = LocalView(graphs.PetersenGraph(), ising_spins, [])
ZPet = sum(
    B ** mono(Pet.G, sigma) * l ** nplus(Pet.G, sigma)
    for sigma in Pet.gen_all_spin_assignments()
)
FPet = ln(ZPet)/Pet.G.order()
occPet = l * diff(FPet, l)

plc = implicit_plot3d(l==lc(3,B), (B,0,1/3), (l,0,1), (z,0,1/2), color='green')

pFK4 = plot3d(FK4, (B,0,1/3), (l,0,1), plot_points=100, color='blue')
pFPet = plot3d(FPet, (B,0,1/3), (l,0,1), plot_points=100, color='red')
show(pFK4 + pFPet + plc)

poccK4 = plot3d(occK4, (B,0,1/3), (l,0,1), plot_points=100, color='blue')
poccPet = plot3d(occPet, (B,0,1/3), (l,0,1), plot_points=100, color='red')
show(poccK4 + poccPet + plc)

# %%
poccPet = plot3d(occPet, (B,0,1/3), (l,0,1), plot_points=100, color='blue')
poccK4 = plot3d(occK4, (B,0,1/3), (l,0,1), plot_points=100, color='red')
colors = ['red', 'blue', 'brown','pink','cyan','magenta','yellow']
def occ(G,B,l):
    L = LocalView(G, ising_spins, [])
    Z = sum(
        B ** mono(G, sigma) * l ** nplus(G, sigma)
        for sigma in L.gen_all_spin_assignments()
    )
    F = ln(Z)/G.order()
    return l * diff(F, l)

from itertools import batched
files = [
    # 'cub04.g6',
    # 'cub06.g6',
    # 'cub08.g6',
    # 'cub10.g6',
    # 'cub12.g6',
    'cub14.g6',
    # 'cub16.g6',
    # 'cub18.g6',
    # 'cub20.g6',
]

for file in files:
    print(file)
    with open(f'data/{file}', 'r') as stream:
        for batch in batched(stream, 7):
            Gs = [Graph(line.strip(), format='graph6') for line in batch]
            occs = [occ(G,B,l) for G in Gs]
            plots = [plot3d(occ, (B,0,1/3), (l,0,1), plot_points=100, color=col) for occ, col, label in zip(occs, colors, batch)]
            combined = combined_min + sum(plots)
            print(list(zip(colors, batch)))
            combined.show()

# %%
minimizers = [
    'C~', # K_4
    'I}GOOSE@W', # 10 vertices... GOOSE?
    'IsP@PGXD_', # Petersen
]
minimizer_colors = ['red', 'blue', 'green']

Gs = [Graph(line.strip(), format='graph6') for line in minimizers]
occs = [occ(G,B,l) for G in Gs]
plots = [plot3d(1/2-occ, (B,0,1/3), (l,0,1), plot_points=100, color=col) for occ, col, label in zip(occs, minimizer_colors, minimizers)]
combined = sum(plots)
print(list(zip(minimizer_colors, minimizers)))
combined.show()
combined_min = combined

# %%
r0 = region_plot(occs[0] <= min(occs[1], occs[2]), (B, 0.2, 0.3), (l, 0.9, 1),  incol='red', plot_points=1000)
r1 = region_plot(occs[1] <= min(occs[0], occs[2]), (B, 0.2, 0.3), (l, 0.9, 1),  incol='blue', plot_points=1000)
r2 = region_plot(occs[2] <= min(occs[0], occs[1]), (B, 0.2, 0.3), (l, 0.9, 1),  incol='green', plot_points=1000)
show(graphics_array([r0,r1,r2]))
show(r0+r1+r2)
# %%
plot([occ.subs(l=2/10) for occ in occs], B,0.14,0.2, color=['red','blue','green'], legend_label=['K4', 'GOOSE', 'Petersen'])
# %%
print(str(occs).replace('[',"{").replace(']', "}"))
# %%
