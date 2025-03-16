# %%
load("potts_occ.py")

# %%
d = 3
q = 4
spin_depth = 2
forbidden = [graphs.CompleteGraph(3)]   #triangle free case

# %%
b = var("b")
Pet = graphs.PetersenGraph()
occPet = occ(Pet, b, q_max=q)

print(f"occ Pet symbolic: {occPet}")

# %%
# Run primal LP with rational sage solver
Ls = get_data(d, spin_depth, forbidden).Ls

bval = 3/10
qval = 4
probability_precision = 50 # need to convert all the probabilities to rationals, how many digits to keep
occPetval = occPet.subs(b=bval, q=qval)

p1, x1 = gen_lp(d, spin_depth, bval, qval, probability_precision, Ls, solver="PPL", constraints='eq', maximization=False)
occbd1 = p1.solve()
print(f"b: {bval}")
print(f"Pet: {occPetval.n()}")
print(f"program: {occbd1.n()}")

#%%
vals = p1.get_values(x1)
support = []
for i, v in vals.items():
    if v > 10**-4:
        support.append(i)

for i in support:
    Ls[i].show()

# %%
