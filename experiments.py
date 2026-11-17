"""Run every experiment in the report, save CSVs to results/ and figures to figs/.

    .venv/bin/python -m sim.experiments
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .network import city_grid, shared_market, own_depots, DEPOTS, CUSTOMER_ZONES, node
from .solver import price_of_anarchy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")

# Palette: categorical slots in fixed order, recessive ink for text / axes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURF = ("#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})

NET = city_grid()
D_BASE = 2.0   # baseline total demand (where Wardrop inefficiency peaks)

KEYS = ["C_so", "C_ne", "C_we", "poa_nash", "poa_wardrop", "gap_so", "gap_ne",
        "gap_we", "delay_factor_so"]


def run(comms, eta=0.0, regret=False, net=None):
    r = price_of_anarchy(NET if net is None else net, comms, eta=eta,
                         check_regret=regret)
    out = {k: r[k] for k in KEYS}
    if regret:
        out["nash_regret"] = r["nash_regret"]
    return out


def save(name, rows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def herfindahl(shares):
    s = np.asarray(shares, float)
    s = s / s.sum()
    return float((s ** 2).sum())


def finish(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------

def exp_network_figure():
    """Figure of the city network with depots / zones (for the report)."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.grid(False)
    for e in range(NET.n_edges):
        u, v = NET.tail[e], NET.head[e]
        if u > v:
            continue
        (x0, y0), (x1, y1) = NET.pos[u], NET.pos[v]
        express = NET.b[e] < 1.0
        ax.plot([x0, x1], [y0, y1], color=SERIES[0] if express else AXIS,
                lw=2.5 if express else 1.2, zorder=1, solid_capstyle="round")
    n = NET.grid_n
    for b in NET.blocked:
        x, y = NET.pos[b]
        ax.scatter(x, y, marker="x", s=60, color=SERIES[4], zorder=3)
    hubs = [node(n, 0, 0), node(n, 3, 0), node(n, 5, 0)]
    zones = [node(n, 1, 5), node(n, 3, 5), node(n, 5, 4)]
    for h in hubs:
        ax.scatter(*NET.pos[h], s=90, marker="s", color=SERIES[1], zorder=4,
                   edgecolor=SURF, linewidth=1.5)
    for z in zones:
        ax.scatter(*NET.pos[z], s=90, marker="o", color=SERIES[2], zorder=4,
                   edgecolor=SURF, linewidth=1.5)
    ax.plot([], [], color=SERIES[0], lw=2.5, label="express corridor (fast, narrow)")
    ax.plot([], [], color=AXIS, lw=1.2, label="local corridor (slow, wide)")
    ax.scatter([], [], s=50, marker="s", color=SERIES[1], label="hub (scenario A)")
    ax.scatter([], [], s=50, marker="o", color=SERIES[2], label="delivery zone (scenario A)")
    ax.scatter([], [], marker="x", s=40, color=SERIES[4], label="no-fly zone")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title("Stylised city: 6 x 6 corridor grid")
    finish(fig, "fig0_network")


def exp1_num_companies():
    """PoA vs number of equal-sized companies (shared market)."""
    Ks = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    rows = []
    for D in (1.0, 2.0, 4.0):
        for K in Ks:
            r = run(shared_market(K, D), regret=K <= 10)
            rows.append({"D": D, "K": K, **r})
            print(f"  E1 D={D} K={K:2d} PoA_NE={r['poa_nash']:.4f} "
                  f"PoA_WE={r['poa_wardrop']:.4f}")
    save("exp1_num_companies", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, D in enumerate((1.0, 2.0, 4.0)):
        sub = [r for r in rows if r["D"] == D]
        ax.plot([r["K"] for r in sub], [r["poa_nash"] for r in sub], "-o",
                color=SERIES[i], label=f"Nash, demand D={D:g}")
        ax.axhline(sub[0]["poa_wardrop"], color=SERIES[i], lw=1, ls="--")
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="Wardrop limit (K → ∞)")
    ax.set_xticks([1, 2, 4, 6, 8, 10, 15, 20])
    ax.set_xlabel("number of equal-sized companies K")
    ax.set_ylabel("price of anarchy")
    ax.set_title("Inefficiency grows with the number of competitors")
    ax.legend(fontsize=7)
    finish(fig, "fig1_num_companies")


def exp2_congestion():
    """PoA vs total demand (congestion level)."""
    Ds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    rows = []
    for D in Ds:
        for K in (2, 5):
            r = run(shared_market(K, D))
            rows.append({"D": D, "K": K, **r})
        print(f"  E2 D={D:5.2f} PoA_WE={r['poa_wardrop']:.4f} "
              f"delay={r['delay_factor_so']:.2f}")
    save("exp2_congestion", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, K in enumerate((2, 5)):
        sub = [r for r in rows if r["K"] == K]
        ax.plot(Ds, [r["poa_nash"] for r in sub], "-o", color=SERIES[i],
                label=f"Nash, K={K} companies")
    sub = [r for r in rows if r["K"] == 2]
    ax.plot(Ds, [r["poa_wardrop"] for r in sub], "-o", color=SERIES[2],
            label="Wardrop (many small operators)")
    ax.set_xscale("log")
    ax.set_xticks(Ds[::2]); ax.set_xticklabels([f"{d:g}" for d in Ds[::2]])
    ax.set_xlabel("total demand D (log scale)")
    ax.set_ylabel("price of anarchy")
    ax.set_title("Inefficiency peaks at moderate congestion")
    ax.legend(fontsize=7)
    finish(fig, "fig2_congestion")


def exp3_concentration():
    """PoA vs market concentration (Herfindahl index) at fixed demand."""
    rows = []
    for K in (2, 3, 5):
        for big in np.linspace(1.0 / K, 0.95, 8):
            shares = [big] + [(1 - big) / (K - 1)] * (K - 1)
            r = run(shared_market(K, D_BASE, shares=shares))
            rows.append({"K": K, "largest_share": big,
                         "H": herfindahl(shares), **r})
        print(f"  E3 K={K} done")
    save("exp3_concentration", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, K in enumerate((2, 3, 5)):
        sub = [r for r in rows if r["K"] == K]
        ax.plot([r["H"] for r in sub], [r["poa_nash"] for r in sub], "-o",
                color=SERIES[i], label=f"K={K} (one large + equal small firms)")
    ax.axhline(rows[0]["poa_wardrop"], color=MUTED, lw=1, ls="--",
               label="Wardrop (H → 0)")
    ax.set_xlabel("Herfindahl index H = Σ (market share)²")
    ax.set_ylabel("price of anarchy")
    ax.set_title(f"Concentrated markets internalise more congestion (D={D_BASE:g})")
    ax.legend(fontsize=7)
    finish(fig, "fig3_concentration")


def exp4_own_depots():
    """Scenario B: geographically differentiated companies."""
    rows = []
    for K in range(1, len(DEPOTS) + 1):
        r = run(own_depots(K, D_BASE), regret=True)
        rows.append({"K": K, **r})
        print(f"  E4 K={K} PoA_NE={r['poa_nash']:.4f} PoA_WE={r['poa_wardrop']:.4f}"
              f" regret={r['nash_regret']:.1e}")
    save("exp4_own_depots", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    Ks = [r["K"] for r in rows]
    ax.plot(Ks, [r["poa_nash"] for r in rows], "-o", color=SERIES[0],
            label="Nash (companies with own depots)")
    ax.plot(Ks, [r["poa_wardrop"] for r in rows], "-o", color=SERIES[2],
            label="Wardrop on the same demand")
    ax.set_xlabel("number of companies K (each adds a depot)")
    ax.set_ylabel("price of anarchy")
    ax.set_title(f"Companies with their own depots (D={D_BASE:g})")
    ax.legend(fontsize=7)
    finish(fig, "fig4_own_depots")


def exp5_tolls():
    """Uniform marginal-cost tolls  tau_e = eta * x_e^opt * c_e'(x_e^opt)."""
    etas = list(np.round(np.linspace(0, 1, 11), 3))
    rows = []
    cases = [("A", 2), ("A", 5), ("A", 10), ("B", 4)]
    for scen, K in cases:
        e_list = sorted(set(etas + [round(1 - 1 / K, 4)]))
        for eta in e_list:
            comms = shared_market(K, D_BASE) if scen == "A" else own_depots(K, D_BASE)
            r = run(comms, eta=eta)
            rows.append({"scenario": scen, "K": K, "eta": eta, **r})
        print(f"  E5 scenario {scen} K={K} done")
    save("exp5_tolls", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (scen, K) in enumerate(cases):
        sub = [r for r in rows if r["scenario"] == scen and r["K"] == K]
        lab = f"shared market, K={K}" if scen == "A" else f"own depots, K={K}"
        ax.plot([r["eta"] for r in sub], [r["poa_nash"] for r in sub], "-o",
                color=SERIES[i], label=lab, markersize=3.5)
        if scen == "A":
            ax.axvline(1 - 1 / K, color=SERIES[i], lw=0.8, ls=":")
    ax.axhline(1.0, color=AXIS, lw=1)
    ax.set_xlabel("toll scale η  (η = 1 is the classic Pigouvian toll)")
    ax.set_ylabel("price of anarchy")
    ax.set_title(f"Congestion pricing (D={D_BASE:g}); dotted: η = 1 − 1/K")
    ax.legend(fontsize=7)
    finish(fig, "fig5_tolls")


def exp6_robustness(n_seeds=20):
    """Repeat the headline comparisons on networks with different random
    capacity draws (seeds), so results are not an artefact of one instance.
    Each row is one (seed, configuration); summary = mean and std. error."""
    configs = [("A", 2, 0.0), ("A", 5, 0.0), ("A", 10, 0.0),
               ("A", 2, 0.5), ("A", 5, 0.8), ("A", 10, 0.9),
               ("A", 5, 1.0), ("B", 4, 0.0), ("B", 4, 0.4), ("B", 4, 1.0)]
    rows = []
    for seed in range(n_seeds):
        net = city_grid(seed=seed)
        for scen, K, eta in configs:
            comms = shared_market(K, D_BASE) if scen == "A" else own_depots(K, D_BASE)
            r = run(comms, eta=eta, net=net)
            rows.append({"seed": seed, "scenario": scen, "K": K, "eta": eta, **r})
    save("exp6_robustness", rows)

    summary = []
    for scen, K, eta in configs:
        sub = [r for r in rows if (r["scenario"], r["K"], r["eta"]) == (scen, K, eta)]
        pn = np.array([r["poa_nash"] for r in sub])
        pw = np.array([r["poa_wardrop"] for r in sub])
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        summary.append({"scenario": scen, "K": K, "eta": eta,
                        "poa_nash_mean": pn.mean(), "poa_nash_se": se(pn),
                        "poa_nash_max": pn.max(),
                        "poa_wardrop_mean": pw.mean(), "poa_wardrop_se": se(pw),
                        "n_seeds": len(sub)})
        print(f"  E6 {scen} K={K:2d} eta={eta:.1f}  PoA_NE {pn.mean():.4f} "
              f"({se(pn):.4f})  PoA_WE {pw.mean():.4f} ({se(pw):.4f})")
    save("exp6_robustness_summary", summary)


def main():
    t = time.time()
    exp_network_figure()
    for f in (exp1_num_companies, exp2_congestion, exp3_concentration,
              exp4_own_depots, exp5_tolls, exp6_robustness):
        print(f.__name__)
        f()
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
