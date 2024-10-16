#%%
load('./ising_occ.sage')

#%%
def local_views(d):
    lvs = []
    neighbors = list(range(1, d + 1))   #u is vertex 0, connected to d others

    Gs = graphs(d)
    for G in Gs:
        G.relabel(neighbors)
        lv = G.to_dictionary()
        lv[0] = neighbors
        vcount = d + 1
                
        for i in range(1, d+1):
            deg = G.degree(i)
            lv[i] += list(range(vcount, vcount + d - deg - 1))
            vcount += d - deg - 1
            
        lvs.append(Graph(lv))
       
    return lvs

# target = 2 to assign spins to only 2nd neighbors,
# 0 to assign spins to all other vertices
def assign_spins(G, d, target):
    vs = G.get_vertices()

    s_index = 0
    e_index = d
    if target == 2:
        s_index = d + 1
        e_index = len(vs) - 1
        
    vs_to_assign = list(range(s_index, e_index + 1))
    num_vs = len(vs_to_assign)

    spin_combs = itertools.product([-1, 1], repeat=num_vs)

    graphs = []
    for spin in spin_combs:
        assignment = {}
        for i in range(len(spin)):
            assignment[s_index + i] = spin[i]
        G.set_vertices(assignment)
        graphs.append(copy(G))

    return graphs

def get_probabilities(lv, d, _B=None, _l=None):
    pu = 0
    pNu = 0
    Z = 0

    # every possible spin assignment on the other vertices (not 2nd neighbors)
    all_assignments = assign_spins(lv, d, target=1)
    neighbors = list(range(1, d+1))

    for G in all_assignments:
        vs = G.get_vertices()
        wt = B**M(G) * l**N(G)
        Z += wt
        if vs[0] == 1:
            pu += wt

        for v in neighbors:
            if vs[v] == 1:
                pNu += wt/d
    
    if _B == None or _l == None:
        return {'pu': (pu/Z), 'pNu': (pNu/Z), 'Z': Z}
    else:
        return {'pu': (pu/Z).subs(B=_B, l=_l), 'pNu': (pNu/Z).subs(B=_B,l=_l), 'Z': Z}


# %%
def get_gamma(lv, d, _B=None, _l=None):
    gu = 0
    gNu = 0
    Z = 0

    # every possible spin assignment on the other vertices (not 2nd neighbors)
    all_assignments = assign_spins(lv, d, target=1)
    neighbors = list(range(1, d+1))
    
    for G in all_assignments:
        vs = G.get_vertices()
        wt = B**M(G) * l**N(G)
        Z += wt

        for v in neighbors:
            if vs[v] == 1:
                gu += wt/d

            for w in G.neighbors(v):   #second neighbors
                if vs[w] == 1:
                    gNu += wt/(d**2)

    
    if _B == None or _l == None:
        return {'gu': (gu/Z), 'gNu': (gNu/Z), 'Z': Z}
    else:
        return {'gu': (gu/Z).subs(B=_B, l=_l), 'gNu': (gNu/Z).subs(B=_B,l=_l), 'Z': Z}

# %%
