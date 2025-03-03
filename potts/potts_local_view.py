from copy import copy, deepcopy
from itertools import product, cycle
from functools import cached_property

from sage.calculus.var import var

from sage.functions.other import binomial
from sage.arith.misc import falling_factorial

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
class PottsLocalView(object):
    def  __init__(self, full_graph, q_max, spin_vertices, full_partition=None):
        self._fullG = full_graph.copy(immutable=True)
        self._q_max = q_max
        self._spin_vertices = spin_vertices
        self._partition = full_partition

        self._spin_assignment = dict()
        for (s, x) in enumerate(spin_vertices):
            for v in full_graph.neighbors(x):
                self.spin_assignment[v] = s

        if len(self.spin_assignment) == 0:
            self._q_L = 0
        else:
            self._q_L = max(self.spin_assignment.values()) + 1

        G = copy(full_graph)
        G.delete_vertices(spin_vertices)
        self._G = G.copy(immutable=True)

        self._data = {}

    # Properties
    @property
    def fullG(self):
        return self._fullG

    @property
    def q_max(self):
        return self._q_max

    @property
    def spin_vertices(self):
        return self._spin_vertices
        
    @property
    def partition(self):
        return self._partition

    @property
    def q_L(self):
        return self._q_L

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

    @cached_property
    def fullG_aut(self):
        return self.fullG.automorphism_group(partition=self.partition)

    # Methods
    def copy(self):
        return LocalView(self.fullG, self.q_max, copy(self.spin_vertices), deepcopy(self.partition))

    def valid_local_coloring(self, sigma):
        extra_colors = set(range(self.q_L, max(sigma.values()) + 1))
        return set(sigma.values()) & extra_colors == extra_colors

    def gen_all_spin_assignments(self, q=None, tqdm=None):
        """Generate all possible spin assignments to the vertices of the local view that extend any preassigne spins
        """
        if q is None:
            q = var("q")
    
        if tqdm is None:
            tqdm = lambda x, *args, **kwargs: x
        
        unassigned = [v for v in self.G.vertices() if v not in self.spin_assignment]

        if self.q_max is None:
            q_max = len(unassigned) + self.q_L
        else:
            q_max = self.q_max


        spins = range(q_max)
        for sigma_tuple in tqdm(product(spins, repeat=len(unassigned)), 
                                desc="Spin assignments",
                                total=q_max**len(unassigned),
                                leave=False,
                                position=1):

            sigma = self.spin_assignment.copy()
            for v, s in zip(unassigned, sigma_tuple):
                sigma[v] = s
            if self.valid_local_coloring(sigma):
                color_weight = binomial(q-self.q_L, max(max(sigma.values())+1-self.q_L, 0))
                m_sigma = sum(1 for (u, v) in self.G.edges(labels=False) if sigma[u] == sigma[v])
                yield color_weight, m_sigma, {k: sigma[k] for k in sorted(sigma)}

    def show(self, sigma=None, colors=None):
        """Show the local view with the spins as colors.
        """
        return self.plot(sigma, colors).show()

    def plot(self, sigma=None, colors=None):
        """Return a :class:`~sage.plot.graphics.Graphics` object representing the local view.
        """
        if colors is None:
            colors = cycle(['red','blue','green','orange','purple',
                            'brown','pink','cyan','magenta','yellow'])

        if sigma is None:
            sigma = self.spin_assignment
        else:
            sigma = sigma | self.spin_assignment

        vertex_colors = dict()
        if len(sigma) == 0:
            spins = range(0)
        else:
            spins = range(max(sigma.values())+1)
        for c, s in zip(colors, spins):
            vertex_colors[c] = [v for v, t in sigma.items() if s == t]

        return self.G.plot(vertex_colors=vertex_colors)

    def orbit(self, w):
        return self.fullG_aut.orbit(w)

