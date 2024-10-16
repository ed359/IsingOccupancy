#%%
import itertools

#%%
#N_G(+, sigma) = number of positive spins
def N(G):
    vs = G.get_vertices()
    return sum([1 for v in vs.values() if v == 1])

#M_G(sigma) = number of edges w/ same spin at both ends
def M(G):
    vs = G.get_vertices()
    samespin = [1 for e in G.edges() if vs[e[0]] == vs[e[1]]]
    return sum(samespin)

def Z(b, l, G):
    spin_combs = list(itertools.product([0, 1], repeat=len(G.vertices())))
    poly = 0

    for spin in spin_combs:
        G.set_vertices({x: spin[i] for (i, x) in enumerate(G.vertices())})
        b_term = b**M(G)
        l_term = l**N(G)
        poly = poly + (b_term * l_term)
    
    return poly

def occ(b, l, G):
    V = len(G.vertices())
    Z_G = Z(b, l, G)

    return (l/V)*diff(ln(Z_G), l)

# %%
