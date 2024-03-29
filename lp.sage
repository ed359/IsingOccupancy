#%%
B = var('B')
l = var('l')
load("./local_views.sage")

#%%
def PPL(deg, bval, lval):
    lvs_nospins = local_views(deg)
    lvs = []
    for lv_ns in lvs_nospins:
        lvs += assign_spins(lv_ns, deg, target=2)

    props = [get_probabilities(lv, deg, bval, lval) for lv in lvs]
    props_gamma = [get_gamma(lv, deg, bval, lval) for lv in lvs]

    p = MixedIntegerLinearProgram(maximization=False, solver="GLPK")
    x = p.new_variable(nonnegative=True)
    p.add_constraint(p.sum(x[i] for i, L in enumerate(lvs)) == 1)
    p.add_constraint(p.sum((props[i]['pu'] - props[i]['pNu'])*x[i] for i,L in enumerate(lvs)) == 0)
    p.add_constraint(p.sum((props_gamma[i]['gu'] - props_gamma[i]['gNu'])*x[i] for i,L in enumerate(lvs)) == 0)
    p.set_objective(p.sum(props[i]['pu']*x[i] for i, L in enumerate(lvs)))
    #p.show()
    return p.solve()

# %%
r = var('r')
s = var('s')
bc = var('bc', latex_name='B_c')
d = var('d')

lc = ((1 - sqrt(r/s))*((1 + sqrt(r*s))/(1 - sqrt(r*s)))
                       **((1 + bc)/(-1 + bc)))/(1 + sqrt(r/s))

lc = lc.subs(r=(bc-B)/(bc + B), s=(1-B)/(1+B)).subs(bc=(d-2)/d)

# %%
degree = 3
test_bs = [1/10]
test_ls = [3/10] #QQ(lc.subs(B=1/5, d=3).n(digits=5))]

for test_l in test_ls:
    for test_b in test_bs:
        lp_occ = PPL(degree, test_b, test_l)
        complete_occ = occ(test_b, l, graphs.CompleteGraph(degree+1)).subs(l=test_l)
        print("b =", test_b, "l =", test_l)
        print(lp_occ)
        print(complete_occ.n())
        print(bool(lp_occ == complete_occ))

# %%
