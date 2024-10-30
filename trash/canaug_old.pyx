# number of non-isomorphic graphs on n vertices
num_graphs = [1, 1, 2, 4, 11, 34, 156, 1044, 12346, 274668, 12005168]

### GENERATORS
# Generate local views of depth 1, where u has d neighbors,
# there is graph structure on Nu, and assign spins to Nu.
def gen_local_view_1(d, spins=None, spin_orbits=None):

    if spins is None:
        spins = [0,1]
    if spin_orbits is None:
        spin_orbits = [[s] for s in spins]

    u = 0
    Nu = list(range(1,d+1))

    # enumerate Nu graph structure
    for G_Nu in tqdm(graphs(d), desc="N(u) graph structure", total=num_graphs[d], position=0):

        # Construct G and Nu
        G = Graph(d+1, data_structure="dense")
        for v in Nu:
            G.add_edge(u,v)

        # Add graph structure to Nu
        for (v,w) in G_Nu.edges(labels=False):
            G.add_edge(v+1,w+1)

        spin_indices = {s: G.add_vertex() for s in spins}
        spin_vertices = list(spin_indices.values())

        partition = [[u], Nu] + [[spin_indices[s] for s in orbit] for orbit in spin_orbits]
        
        # Assign spins to Nu
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in assign_spins(G, partition, Nu, spin_vertices, aut_gens):
            yield LocalView(X, spins, spin_vertices)

# Generate local views of depth 2 where u has d neighbors,
# there is graph structure on Nu, add second neighbors
# of u (pairwise disjointly) and assign spins to N2u
def gen_local_view_2(d, spins=None, spin_orbits=None):

    if spins is None:
        spins = [0,1]
    if spin_orbits is None:
        spin_orbits = [[s] for s in spins]

    u = 0
    Nu = list(range(1,d+1))

    # enumerate Nu graph structure
    for G_Nu in tqdm(graphs(d), desc="N(u) graph structure", total=num_graphs[d], position=0):

        # Construct G and Nu
        G = Graph(d+1, data_structure="dense")
        for v in Nu:
            G.add_edge(u,v)

        # Add graph structure to Nu
        for (v,w) in G_Nu.edges(labels=False):
            G.add_edge(v+1,w+1)

        # Add N2u
        for v in Nu:
            dv = G.degree(v)
            G.add_edges((v,G.add_vertex()) for w in range(d-dv))

        N2u = list(range(d+1, G.order()))

        spin_indices = {s: G.add_vertex() for s in spins}
        spin_vertices = list(spin_indices.values())

        partition = [[u], Nu]
        if N2u: 
            partition += [N2u]
        partition += [[spin_indices[s] for s in orbit] for orbit in spin_orbits]

        # Assign spins to Nu
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in assign_spins(G, partition, N2u, spin_vertices, aut_gens):
            yield LocalView(X, spins, spin_vertices)


def gen_local_view_3(d, spins=None, spin_orbits=None):

    if spins is None:
        spins = [0,1]
    if spin_orbits is None:
        spin_orbits = [[s] for s in spins]

    G = Graph(1, data_structure='dense', loops=False, multiedges=False)
    data_current = [(G, [[0]], [0])]
    data_next = []

    # print('Initial')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    for (G, last_partition, last_layer) in data_current:
        next_layer = []
        for v in last_layer:
            for w in range(d-G.degree(v)):
                next_layer.append(G.add_vertex())
        
        partition = last_partition + [next_layer]
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in canaug_new_layer(d, G, partition, last_layer, next_layer, aut_gens):
            clean_X = copy(X)
            clean_partition = deepcopy(partition)
            clean_next_layer = copy(next_layer)
            for v in next_layer:
                if clean_X.degree(v) == 0:
                    clean_X.delete_vertex(v)
                    clean_partition[-1].remove(v)
                    clean_next_layer.remove(v)
            if not clean_partition[-1]:
                clean_partition.pop()
            data_next.append((clean_X, clean_partition, clean_next_layer))
    data_current = data_next
    data_next = []

    # print('Added first layer')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    for (G, partition, layer) in data_current:
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in canaug_fill_layer(d, G, partition, layer, aut_gens):
            data_next.append((X, partition, layer))
    data_current = data_next
    data_next = []

    # print('Filled first layer')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    for (G, last_partition, last_layer) in data_current:
        next_layer = []
        for v in last_layer:
            for w in range(d-G.degree(v)):
                next_layer.append(G.add_vertex())
        
        partition = last_partition + [next_layer]
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in canaug_new_layer(d, G, partition, last_layer, next_layer, aut_gens):
            clean_X = copy(X)
            clean_partition = deepcopy(partition)
            clean_next_layer = copy(next_layer)
            for v in next_layer:
                if clean_X.degree(v) == 0:
                    clean_X.delete_vertex(v)
                    clean_partition[-1].remove(v)
                    clean_next_layer.remove(v)
            if not clean_partition[-1]:
                clean_partition.pop()
            data_next.append((clean_X, clean_partition, clean_next_layer))
    data_current = data_next
    data_next = []

    # print('Added second layer')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    for (G, partition, layer) in data_current:
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in canaug_fill_layer(d, G, partition, layer, aut_gens):
            data_next.append((X, partition, layer))
    data_current = data_next
    data_next = []

    # print('Filled second layer')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    for (G, last_partition, last_layer) in data_current:
        next_layer = []
        for v in last_layer:
            Nv = []
            for w in range(d-G.degree(v)):
                Nv.append(G.add_vertex())
            G.add_edges((v, w) for w in Nv)
            next_layer.extend(Nv)
        partition = last_partition + [next_layer]
        data_next.append((G, partition, next_layer))
    data_current = data_next
    data_next = []

    # print('Added third layer')
    # for i, (_, partition, layer) in enumerate(data_current):
    #     print(i, partition, layer)
    # ga = graphics_array([[X.plot() for X, _, _ in data_current]])
    # ga.show()

    # Assign spins to N3u
    for (G, partition, layer) in data_current:
        spin_indices = {s: G.add_vertex() for s in spins}
        spin_vertices = list(spin_indices.values())

        partition += [[spin_indices[s] for s in orbit] for orbit in spin_orbits]
        aut_gens = search_tree(G._backend.c_graph()[0], partition, False, False)
        for X in assign_spins(G, partition, layer, spin_vertices, aut_gens):
            yield LocalView(X, spins, spin_vertices)