from copy import copy, deepcopy
from itertools import product, cycle
from functools import cached_property

def invalidate_cached_properties(obj):
    cls = type(obj)
    cached = {
        attr
        for attr in list(obj.__dict__.keys())
        if (descriptor := getattr(cls, attr, None))
        if isinstance(descriptor, cached_property)
    }
    for attr in cached:
        del obj.__dict__[attr]


# A local view of a graph with (optional) preassigned spins
# This class is a thin wrapper around the sage Graph class
class LocalView(object):
    def  __init__(self, full_graph, spins, spin_vertices, full_partition=None):
        self._fullG = full_graph.copy(immutable=True)
        self._spins = spins
        self._spin_vertices = spin_vertices
        self._partition = full_partition

        self._spin_assignment = dict()
        for (s, x) in zip(spins, spin_vertices):
            for v in full_graph.neighbors(x):
                self.spin_assignment[v] = s

        G = copy(full_graph)
        G.delete_vertices(spin_vertices)
        self._G = G.copy(immutable=True)

        self._data = {}

    # Properties
    @property
    def fullG(self):
        return self._fullG

    @property
    def spins(self):
        return self._spins

    @property
    def spin_vertices(self):
        return self._spin_vertices
        
    @property
    def partition(self):
        return self._partition

    @property
    def spin_assignment(self):
        return self._spin_assignment

    @property
    def G(self):
        return self._G

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def u(self):
        return 0

    @cached_property
    def Nu(self):
        return self.G.neighbors(self.u)
    
    @cached_property
    def N2u(self):
        N2u = set()
        for v in self.Nu:
            N2u.update({w for w in self.G.neighbors(v)})
        return list(N2u - set(self.G.neighbors(self.u, closed=True)))

    # @cached_property
    # def layers (self):
    #     return [p for p in self.partition if all(v not in self.spin_vertices for v in p)]

    @cached_property
    def fullG_aut(self):
        return self.fullG.automorphism_group(partition=self.partition)

    # @cached_property
    # def fullG_partition_fixed_spins(self):
    #     partition = [[self.u], 
    #                  self.Nu,
    #                  [w for w in self.fullG.vertices() if w != self.u and w not in self.Nu and w not in self.spin_vertices],
    #                 ]
    #     partition.extend([s] for s in self.spin_vertices)
    #     return partition

    # @cached_property
    # def fullG_partition_permute_spins(self):
    #     partition = [[self.u], 
    #                  self.Nu,
    #                  [w for w in self.fullG.vertices() if w != self.u and w not in self.Nu and w not in self.spin_vertices],
    #                 ]
    #     partition.append(self.spin_vertices)
    #     return partition

    # @cached_property
    # def fullG_aut_fixed_spins(self):
    #     return self.fullG.automorphism_group(partition=self.fullG_partition_fixed_spins)

    # @cached_property
    # def fullG_aut_permute_spins(self):
    #     return self.fullG.automorphism_group(partition=self.fullG_partition_permute_spins)

    # @cached_property
    # def fullG_can_fixed_spins(self):
    #     return self.fullG.canonical_label(partition=self.fullG_partition_fixed_spins)

    # @cached_property
    # def fullG_can_permute_spins(self):
    #     return self.fullG.canonical_label(partition=self.fullG_partition_permute_spins)


    # Methods
    def copy(self):
        return LocalView(self.fullG, copy(self.spins), copy(self.spin_vertices), deepcopy(self.partition))

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
        """Show the local view with the spins as colors.
        """
        return self.plot(colors).show()

    def plot(self, colors=None):
        """Return a :class:`~sage.plot.graphics.Graphics` object representing the local view.
        """
        if colors is None:
            colors = cycle(['red','blue','green','orange','purple',
                            'brown','pink','cyan','magenta','yellow'])

        vertex_colors = dict()
        for c, s in zip(colors, self.spins):
            vertex_colors[c] = [v for v, t in self.spin_assignment.items() if s == t]

        return self.G.plot(vertex_colors=vertex_colors)

    def orbit(self, w):
        return self.fullG_aut.orbit(w)
    
    def change_spin(self, w, s=None):
        # If there are two spins, s defaults to the other spin
        if s is None:
            s = self.spins[1 - self.spins.index(self.spin_assignment[w])]

        full_graph = self.fullG.copy(immutable=False)
        full_graph.delete_edge(w, self.spin_vertices[self.spins.index(self.spin_assignment[w])])
        full_graph.add_edge(w, self.spin_vertices[self.spins.index(s)])
        return LocalView(full_graph, copy(self.spins), copy(self.spin_vertices), deepcopy(self.partition))
