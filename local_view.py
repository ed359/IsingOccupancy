from copy import copy
from itertools import product

# A local view of a graph with (optional) preassigned spins
# This class is a thin wrapper around the sage Graph class
class LocalView(object):
    def  __init__(self, full_graph, spins, spin_vertices):
        self.spins = spins

        self.spin_assignment = dict()
        for (s, x) in zip(spins, spin_vertices):
            for v in full_graph.neighbors(x):
                self.spin_assignment[v] = s

        G = copy(full_graph)
        G.delete_vertices(spin_vertices)
        self.G = G.copy(immutable=True)

    def gen_all_spin_assignments(self, tqdm=None):
        """Generate all possible spin assignments to the vertices of the local view that extend any preassigne spins
        """
        if tqdm is None:
            tqdm = lambda x, *args, **kwargs: x

        unassigned = [v for v in self.G.vertices() if v not in self.spin_assignment]
        for sigma_tuple in tqdm(product(self.spins, repeat=len(unassigned)), 
                                desc="Spin assignments",
                                total=len(self.spins)**len(unassigned),
                                leave=False,
                                position=1):
            sigma = self.spin_assignment.copy()
            for v, s in zip(unassigned, sigma_tuple):
                sigma[v] = s
            yield {k: sigma[k] for k in sorted(sigma)}

    def show(self, colors=None):
        """Show the local view with the spins as colors
        """
        if colors is None:
            colors = ['red','blue','green']

        vertex_colors = dict()
        for c, s in zip(colors, self.spins):
            vertex_colors[c] = [v for v, t in self.spin_assignment.items() if s == t]

        return self.G.show(vertex_colors=vertex_colors)

    def u(self):
        return 0

    def Nu(self):
        return self.G.neighbors(self.u())
    
    def N2u(self):
        closedNu = {self.u} | set(self.Nu())
        return [w for w in self.G.vertices() if v not in closedNu]

    def flip(self, w):
        # return a copy of self with the spin of w flipped

    def marked_orbit(self, w):
        # return the orbit of self with w marked under aut(self) in the form [(L, w) ...]
        pass