"""Check the solvers against closed-form answers before trusting the city runs.

Run from the project root:  .venv/bin/python -m sim.validate
"""

import numpy as np

from .network import Commodity, braess, pigou, shared_market, city_grid
from .solver import price_of_anarchy


def symmetric(net, o, d, K, demand=1.0):
    return [Commodity(k, o, d, demand / K) for k in range(K)]


def check(label, got, want, tol=1e-3):
    ok = abs(got - want) < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label:<45s} got {got:.5f}  want {want:.5f}")
    return ok


def main():
    all_ok = True
    print("Pigou (one unit of demand)")
    for K in (1, 2, 3, 5, 10, 50):
        r = price_of_anarchy(pigou(), symmetric(pigou(), 0, 1, K), tol=1e-8,
                             check_regret=True)
        # K symmetric players: each puts y on the x-link with x + y = 1 and
        # x = K y  =>  x = K/(K+1);  cost = x^2 + (1 - x)
        x = K / (K + 1)
        all_ok &= check(f"K={K:<3d} PoA(Nash)", r["poa_nash"],
                        (x * x + 1 - x) / 0.75)
        all_ok &= check(f"K={K:<3d} Nash regret", r["nash_regret"], 0.0)
    all_ok &= check("PoA(Wardrop)", r["poa_wardrop"], 4 / 3)
    # Pigouvian toll x c'(x) at the optimum makes Wardrop = SO; for K symmetric
    # players the right uniform toll is (1 - 1/K) x c'(x) (CCS 2009, Sec. 5).
    for K in (2, 5):
        r = price_of_anarchy(pigou(), symmetric(pigou(), 0, 1, K),
                             eta=1 - 1 / K, tol=1e-8)
        all_ok &= check(f"K={K} PoA with toll eta=1-1/K", r["poa_nash"], 1.0)

    print("Braess (one unit of demand)")
    r = price_of_anarchy(braess(), symmetric(braess(), 0, 3, 1), tol=1e-8)
    all_ok &= check("Social optimum cost", r["C_so"], 1.5)
    all_ok &= check("Wardrop cost", r["C_we"], 2.0)
    all_ok &= check("K=1 PoA(Nash)", r["poa_nash"], 1.0)
    prev = 1.0
    for K in (2, 3, 5, 10, 50):
        r = price_of_anarchy(braess(), symmetric(braess(), 0, 3, K), tol=1e-8)
        mono = prev - 1e-6 <= r["poa_nash"] <= 4 / 3 + 1e-6
        print(f"  [{'ok' if mono else 'FAIL'}] K={K:<3d} PoA(Nash) = "
              f"{r['poa_nash']:.5f}  (monotone in K, below Wardrop's 4/3)")
        all_ok &= mono
        prev = r["poa_nash"]

    print("City grid: Nash regret of computed equilibria")
    net = city_grid()
    for K, D in ((2, 2.0), (4, 4.0)):
        r = price_of_anarchy(net, shared_market(K, D), check_regret=True)
        all_ok &= check(f"K={K} D={D} Nash regret", r["nash_regret"], 0.0)
        all_ok &= check(f"K={K} D={D} SO <= NE, PoA <= (3K+1)/(2K+2)",
                        float(r["C_so"] <= r["C_ne"] + 1e-6 and
                              r["poa_nash"] <= (3 * K + 1) / (2 * K + 2)), 1.0)
    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
