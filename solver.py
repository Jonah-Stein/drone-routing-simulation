"""Equilibrium and optimum solvers.

All three flow concepts we need are minimisers of a convex quadratic over the
same feasible set (each commodity routes its demand from origin to destination):

  * Social optimum (SO):      min  sum_e x_e c_e(x_e)
  * Wardrop equilibrium (WE): min  sum_e int_0^{x_e} c_e        (Beckmann potential)
  * Nash equilibrium (NE) of the atomic splittable game, affine costs:
                              min  sum_e [ b_e x_e + a_e/2 (x_e^2 + sum_k (x_e^k)^2) ]

The NE potential follows because d/dx_e^k of it equals c_e(x_e) + a_e x_e^k,
which is exactly company k's marginal cost on edge e, i.e. the first-order
condition of company k's own problem  min_{x^k} sum_e x_e^k c_e(x_e).  So with
affine costs the atomic splittable game is a potential game with a strictly
convex potential in the aggregate-per-player flows, and its equilibrium is
unique (cf. Cominetti, Correa & Stier-Moses 2009, Sec. 4).

Fixed per-edge tolls tau_e add sum_e tau_e x_e to every objective.  Tolls are
transfers, so they are *excluded* from the reported social cost.

We minimise with path-based gradient projection, a standard traffic-assignment
method: every commodity keeps a small set of paths, its current cheapest path
(under the gradient of the objective, i.e. the *perceived* marginal cost) is
added each sweep, and flow is shifted onto it with a Newton-type step.
Convergence is measured by the relative Frank-Wolfe duality gap.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EPS_W = 1e-12   # zero-cost edges would be dropped by the sparse graph


class Solver:
    def __init__(self, net, comms):
        self.net = net
        self.comms = comms
        self.players = sorted({c.player for c in comms})
        self.edge_id = {(int(u), int(v)): i
                        for i, (u, v) in enumerate(zip(net.tail, net.head))}
        assert len(self.edge_id) == net.n_edges, "parallel edges not supported"
        self.demand = np.array([c.demand for c in comms])
        self.player_of = np.array([c.player for c in comms])

    # -- flow bookkeeping ---------------------------------------------------
    def player_flows(self, X):
        """(n_players, E) flow of each company on each edge."""
        P = np.zeros((len(self.players), self.net.n_edges))
        for i, k in enumerate(self.players):
            P[i] = X[self.player_of == k].sum(axis=0)
        return P

    def social_cost(self, X):
        x = X.sum(axis=0)
        return float(np.dot(x, self.net.cost(x)))

    def player_costs(self, X, tolls=None):
        x = X.sum(axis=0)
        c = self.net.cost(x) + (0 if tolls is None else tolls)
        return self.player_flows(X) @ c

    # -- gradients ------------------------------------------------------------
    def edge_gradients(self, X, mode, tolls):
        """Return (n_players, E): gradient of the objective w.r.t. the flow of
        any commodity of that player on each edge."""
        net = self.net
        x = X.sum(axis=0)
        if mode == "so":
            g = net.b + 2 * net.a * x
            G = np.tile(g, (len(self.players), 1))
        elif mode == "wardrop":
            G = np.tile(net.cost(x), (len(self.players), 1))
        elif mode == "nash":
            G = net.cost(x)[None, :] + net.a[None, :] * self.player_flows(X)
        else:
            raise ValueError(mode)
        if tolls is not None:
            G = G + tolls[None, :]
        return G

    # -- all-or-nothing assignment -----------------------------------------------
    def all_or_nothing(self, G, free):
        """Route each free commodity entirely on its cheapest path under the
        player-specific edge weights G.  Returns the target flow matrix Y."""
        net = self.net
        Y = np.zeros((len(self.comms), net.n_edges))
        for i, k in enumerate(self.players):
            idx = [j for j, c in enumerate(self.comms) if c.player == k and free[j]]
            if not idx:
                continue
            W = csr_matrix((G[i] + EPS_W, (net.tail, net.head)),
                           shape=(net.n_nodes, net.n_nodes))
            origins = sorted({self.comms[j].origin for j in idx})
            _, pred = dijkstra(W, directed=True, indices=origins,
                               return_predecessors=True)
            row = {o: r for r, o in enumerate(origins)}
            for j in idx:
                c = self.comms[j]
                v = c.dest
                p = pred[row[c.origin]]
                while v != c.origin:
                    u = p[v]
                    if u < 0:
                        raise RuntimeError("destination unreachable")
                    Y[j, self.edge_id[(u, v)]] += c.demand
                    v = u
        return Y

    # -- path-based gradient projection ----------------------------------------
    def _hess(self, mode):
        """d(gradient on e)/d(own commodity flow on e): the per-edge curvature
        used for the Newton-like step size."""
        a = self.net.a
        return {"so": 2 * a, "wardrop": a, "nash": 2 * a}[mode]

    def solve(self, mode, tolls=None, X0=None, free_players=None,
              tol=1e-7, max_iter=5000):
        """Minimise the objective for `mode` ('so', 'wardrop', 'nash').

        Path-based gradient projection (Jayakrishnan et al. 1994): each
        commodity keeps a set of used paths; every sweep we add its current
        cheapest path (column generation) and shift flow from each costlier
        path onto it with a Newton step  (g_p - g_q) / sum_{e in p xor q} h_e.
        This converges much faster than Frank-Wolfe on these instances.

        free_players: if given, only those companies' flows move (others are
        held fixed at X0) -- used to compute best responses.
        Returns (X, info) with X the (n_comms, E) commodity-edge flow matrix
        and info['rel_gap'] the relative Frank-Wolfe duality gap
        <grad, X - Y> / <grad, X> at the returned point (0 at the optimum).
        """
        n_c, E = len(self.comms), self.net.n_edges
        free = np.ones(n_c, bool) if free_players is None else \
            np.isin(self.player_of, list(free_players))
        pidx = {k: i for i, k in enumerate(self.players)}
        rows = np.array([pidx[c.player] for c in self.comms])
        H = self._hess(mode) + 1e-12

        X = np.zeros((n_c, E)) if X0 is None else X0.copy()
        # path sets: list of (edge-index array, flow) per commodity
        paths = [[] for _ in range(n_c)]
        if X0 is not None:
            for j in np.where(free)[0]:
                paths[j] = self._decompose(j, X0[j])

        gap = np.inf
        for it in range(max_iter):
            G = self.edge_gradients(X, mode, tolls)
            Y = self.all_or_nothing(G, free)
            Y[~free] = X[~free]
            if it > 0 or X0 is not None:
                g0 = float(np.sum(G[rows] * (X - Y)))
                denom = float(np.sum(G[rows] * X * free[:, None]))
                gap = g0 / max(denom, 1e-300)
                if gap < tol:
                    break
            x = X.sum(axis=0)
            P = self.player_flows(X)
            for j in np.where(free)[0]:
                c, i = self.comms[j], rows[j]
                q = np.flatnonzero(Y[j] > 0)
                if not any(len(e) == len(q) and np.array_equal(e, q)
                           for e, _ in paths[j]):
                    paths[j].append([q, 0.0 if paths[j] else c.demand])
                    if len(paths[j]) == 1:
                        X[j, q] += c.demand
                        x[q] += c.demand
                        P[i, q] += c.demand
                # Gauss-Seidel: shift flow one path at a time, re-evaluating
                # the marginal costs after every shift so steps never pile up
                g = self._grad_row(x, P[i], mode, tolls)
                best = int(np.argmin([g[e].sum() for e, _ in paths[j]]))
                eb = paths[j][best][0]
                for m, (e, f) in enumerate(paths[j]):
                    if m == best or f <= 0:
                        continue
                    diff = np.setxor1d(e, eb)
                    step = min(f, (g[e].sum() - g[eb].sum()) / H[diff].sum())
                    if step <= 0:
                        continue
                    paths[j][m][1] -= step
                    paths[j][best][1] += step
                    for arr in (X[j], x, P[i]):
                        arr[e] -= step
                        arr[eb] += step
                    g = self._grad_row(x, P[i], mode, tolls)
                # drop empty paths (keep at least one)
                paths[j] = [p for p in paths[j] if p[1] > 1e-15] or paths[j][:1]
            np.maximum(X, 0, out=X)
        return X, {"iters": it + 1, "rel_gap": gap}

    def _grad_row(self, x, x_player, mode, tolls):
        """Objective gradient on each edge for one company's commodities, given
        total edge flow x and that company's edge flow x_player."""
        net = self.net
        if mode == "so":
            g = net.b + 2 * net.a * x
        elif mode == "wardrop":
            g = net.cost(x)
        else:
            g = net.cost(x) + net.a * x_player
        return g if tolls is None else g + tolls

    def _decompose(self, j, xrow):
        """Split one commodity's edge flow into origin-destination paths."""
        c = self.comms[j]
        x = xrow.copy()
        out_edges = {}
        for e, u in enumerate(self.net.tail):
            out_edges.setdefault(int(u), []).append(e)
        paths, remaining = [], c.demand
        while remaining > 1e-12:
            v, path = c.origin, []
            while v != c.dest:
                e = max(out_edges.get(v, []), key=lambda e: x[e])
                path.append(e)
                v = int(self.net.head[e])
                if len(path) > self.net.n_edges:
                    raise RuntimeError("cycle in flow decomposition")
            f = min(remaining, x[path].min())
            if f <= 1e-15:
                break
            x[path] -= f
            remaining -= f
            paths.append([np.array(sorted(path)), f])
        return paths

    def nash_regret(self, X, tolls=None, tol=1e-8):
        """Largest relative gain any single company could get by unilaterally
        re-routing its whole fleet (0 at an exact Nash equilibrium).  Each
        best response is itself a convex problem solved by Frank-Wolfe."""
        base = self.player_costs(X, tolls)
        worst = 0.0
        for i, k in enumerate(self.players):
            Xbr, _ = self.solve("nash", tolls, X0=X, free_players=[k], tol=tol)
            new = self.player_costs(Xbr, tolls)[i]
            worst = max(worst, (base[i] - new) / base[i])
        return worst


def price_of_anarchy(net, comms, eta=0.0, tol=1e-7, check_regret=False):
    """Compute SO, the atomic Nash equilibrium (with optional marginal-cost
    tolls eta * x_opt * c'(x_opt)), and the Wardrop equilibrium.

    Returns a dict with costs, PoA values and convergence diagnostics.
    """
    s = Solver(net, comms)
    X_so, i_so = s.solve("so", tol=tol)
    x_so = X_so.sum(axis=0)
    tolls = eta * x_so * net.a if eta else None
    X_ne, i_ne = s.solve("nash", tolls=tolls, tol=tol)
    X_we, i_we = s.solve("wardrop", tol=tol)
    C_so, C_ne, C_we = (s.social_cost(X) for X in (X_so, X_ne, X_we))
    out = {
        "C_so": C_so, "C_ne": C_ne, "C_we": C_we,
        "poa_nash": C_ne / C_so, "poa_wardrop": C_we / C_so,
        "gap_so": i_so["rel_gap"], "gap_ne": i_ne["rel_gap"],
        "gap_we": i_we["rel_gap"],
        "total_demand": float(s.demand.sum()),
        # average delay factor at the optimum: actual / free-flow travel time
        "delay_factor_so": C_so / float(np.dot(x_so, net.b) + 1e-300),
        "X_so": X_so, "X_ne": X_ne, "X_we": X_we, "solver": s,
    }
    if check_regret:
        out["nash_regret"] = s.nash_regret(X_ne, tolls)
    return out
