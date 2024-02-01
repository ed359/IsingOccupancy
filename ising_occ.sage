import itertools

B = var('B')
x = var('x')

#N_G(+, sigma) = number of positive spins
def N(spins):
    return sum(spins)

#M_G(sigma) = number of edges w/ same spin at both ends
def M(G, spins):
    samespin = [1 for e in G.edges() if spins[e[0]] == spins[e[1]]]
    return sum(samespin)

def Z(b, l, G, V):
    spin_combs = itertools.product([0, 1], repeat=V)
    poly = 0

    for spin in spin_combs:
        b_term = b**M(G,spin)
        l_term = l**N(spin)
        poly = poly + (b_term * l_term)
    
    return poly

def occ(b, l, G):
    V = len(G.vertices())
    Z_G = Z(b, l, G, V)

    return (l/V)*diff(ln(Z_G), l)

K4 = graphs.CompleteGraph(4); K4.name("K4")
K2_2 = graphs.CompleteBipartiteGraph(2, 2); K2_2.name("K2_2")
S = graphs.StarGraph(4)

graphs = [K4, K2_2, S]

full_plot = Graphics()
colors = ['blue', 'orange', 'red', 'purple']
for i in range(len(graphs)):
    poly = occ(1, x, graphs[i])
    print(poly)
    full_plot += plot(poly, (x, 0, 10), legend_label=graphs[i].name(), color=colors[i])

full_plot.axes_labels(['x', 'occ_G(B,x)'])
save(full_plot,'./graphs/gtest.png',axes=True)
