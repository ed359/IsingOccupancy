# %%
load("local_occupancy/ising_occ.py")

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
