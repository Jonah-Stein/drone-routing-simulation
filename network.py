"""Network definitions for the drone-routing congestion game.

Every edge e is a directed flight corridor with an affine travel-time function

    c_e(x) = b_e + a_e * x,

where b_e is the free-flow flight time and a_e is the congestion slope.  For the
city grid we write a_e = b_e / cap_e, so c_e(x) = b_e * (1 + x / cap_e): a
corridor carrying exactly its nominal capacity takes twice its free-flow time.
This is the affine (degree-1) member of the BPR family used in traffic studies.

A *commodity* is one company's demand between one origin-destination pair:
(player, origin, destination, demand).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Network:
    n_nodes: int
    tail: np.ndarray          # (E,) start node of each edge
    head: np.ndarray          # (E,) end node of each edge
    a: np.ndarray             # (E,) congestion slope
    b: np.ndarray             # (E,) free-flow time
    pos: dict = field(default_factory=dict)   # node -> (x, y), for plotting only
    name: str = ""

    @property
    def n_edges(self):
        return len(self.tail)

    def cost(self, x):
        return self.b + self.a * x


@dataclass
class Commodity:
    player: int
    origin: int
    dest: int
    demand: float


def _make(edges, n_nodes, name, pos=None):
    edges = np.asarray(edges, dtype=float)
    return Network(n_nodes=n_nodes,
                   tail=edges[:, 0].astype(int), head=edges[:, 1].astype(int),
                   a=edges[:, 2], b=edges[:, 3], pos=pos or {}, name=name)


# ---------------------------------------------------------------------------
# Textbook validation networks
# ---------------------------------------------------------------------------

def pigou():
    """Pigou's example.  Node 0 = s, 1 = t, 2 = dummy midpoint.

    Upper link s->t has constant cost 1.  Lower link s->2->t has cost x.  (The
    dummy node avoids parallel edges, which the shortest-path routine cannot
    represent.)  With one unit of demand: Wardrop cost 1, optimum 3/4, PoA 4/3.
    """
    edges = [(0, 1, 0.0, 1.0),      # constant-cost corridor
             (0, 2, 1.0, 0.0),      # congestible corridor
             (2, 1, 0.0, 0.0)]
    return _make(edges, 3, "Pigou")


def braess():
    """Braess's paradox network with a zero-cost shortcut v->w.

    Nodes: 0 = s, 1 = v, 2 = w, 3 = t.  One unit of demand: Wardrop cost 2,
    optimum 3/2, PoA 4/3.
    """
    edges = [(0, 1, 1.0, 0.0),      # s->v : x
             (1, 3, 0.0, 1.0),      # v->t : 1
             (0, 2, 0.0, 1.0),      # s->w : 1
             (2, 3, 1.0, 0.0),      # w->t : x
             (1, 2, 0.0, 0.0)]      # v->w : 0  (the shortcut)
    return _make(edges, 4, "Braess")


# ---------------------------------------------------------------------------
# Stylised city
# ---------------------------------------------------------------------------

def city_grid(n=6, block_time=1.0, base_cap=2.0, express_time=0.5,
              express_cap=0.5, arterial_rows=(1, 4), arterial_cols=(1, 4),
              no_fly=((2, 2), (3, 2)), cap_noise=0.2, seed=0):
    """An n x n Manhattan-style grid of flight corridors.

    * Each grid edge becomes two directed corridors (one per direction).
    * Corridors along the designated arterial rows/columns are *express
      corridors* (e.g. a higher-altitude lane above a major road or river):
      faster free-flow time ``express_time`` but a narrower capacity
      ``express_cap``.  Ordinary local corridors take ``block_time`` and have
      capacity ``base_cap``.  This fast-but-congestible vs. slow-but-roomy
      trade-off is the network analogue of Pigou's example; without it (all
      corridors equally fast) every used route has the same length and
      selfish and optimal routing nearly coincide.
    * Nodes listed in ``no_fly`` (e.g. a hospital helipad or stadium) are
      removed together with all incident corridors.
    * Capacities get multiplicative noise U(1 - cap_noise, 1 + cap_noise) so the
      network is not perfectly symmetric.

    Node id of grid cell (row r, column c) is r * n + c.
    """
    rng = np.random.default_rng(seed)
    blocked = {r * n + c for r, c in no_fly}
    edges = []
    pos = {}
    for r in range(n):
        for c in range(n):
            u = r * n + c
            pos[u] = (c, n - 1 - r)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr >= n or cc >= n:
                    continue
                v = rr * n + cc
                if u in blocked or v in blocked:
                    continue
                horizontal = dr == 0
                arterial = (horizontal and r in arterial_rows) or \
                           (not horizontal and c in arterial_cols)
                cap = express_cap if arterial else base_cap
                t0 = express_time if arterial else block_time
                for s, t in ((u, v), (v, u)):
                    cap_e = cap * rng.uniform(1 - cap_noise, 1 + cap_noise)
                    edges.append((s, t, t0 / cap_e, t0))
    net = _make(edges, n * n, "City grid", pos)
    net.blocked = blocked
    net.grid_n = n
    return net


def node(n, r, c):
    return r * n + c


def city_od_pairs(n=6):
    """Fixed set of (origin, destination, weight) for the shared-market scenario.

    Three distribution hubs on the west edge ship to three delivery zones on the
    east / north-east side of the city.  Weights sum to 1; total demand is
    scaled separately.
    """
    hubs = [node(n, 0, 0), node(n, 3, 0), node(n, 5, 0)]
    zones = [node(n, 1, 5), node(n, 3, 5), node(n, 5, 4)]
    w = np.array([[3, 2, 1],
                  [2, 3, 2],
                  [1, 2, 3]], dtype=float)
    w /= w.sum()
    return [(hubs[i], zones[j], w[i, j]) for i in range(3) for j in range(3)]


def shared_market(n_players, total_demand, shares=None, n=6):
    """Scenario A: every company serves a share of every OD pair.

    This isolates the effect of *market structure* (how many companies, how
    concentrated) from geography.  K = 1 is the centralised (monopoly /
    coordinated) case; equal shares is the symmetric oligopoly.
    """
    if shares is None:
        shares = np.full(n_players, 1.0 / n_players)
    shares = np.asarray(shares, dtype=float)
    shares = shares / shares.sum()
    comms = []
    for k, s in enumerate(shares):
        for o, d, w in city_od_pairs(n):
            comms.append(Commodity(k, o, d, total_demand * s * w))
    return comms


# Depot locations around the perimeter for scenario B (row, col).
DEPOTS = [(0, 0), (5, 0), (0, 3), (5, 3), (2, 0), (4, 5)]
CUSTOMER_ZONES = [(1, 5), (3, 5), (5, 4), (1, 2), (4, 2)]


def own_depots(n_players, total_demand, n=6):
    """Scenario B: company k ships from its own depot to all customer zones.

    Companies are now geographically differentiated (asymmetric), which is the
    setting where atomic equilibria can be worse than Wardrop equilibria.
    Total demand is split equally across companies and zones.
    """
    assert n_players <= len(DEPOTS)
    zones = [node(n, r, c) for r, c in CUSTOMER_ZONES]
    comms = []
    for k in range(n_players):
        o = node(n, *DEPOTS[k])
        for d in zones:
            if d != o:
                comms.append(Commodity(k, o, d,
                                       total_demand / (n_players * len(zones))))
    return comms
