import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pylab as plt
import numpy as np

from python import HeisenbergCorrelationSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spins", required=True, type=int)
    parser.add_argument("--Jconst", required=True, type=float)
    parser.add_argument("--prefix", required=False, default="", type=str)

    args = parser.parse_args()
    prefix: Optional[Path] = None

    if args.prefix == "":
        prefix = Path(__file__).parent.joinpath(
            "output_spins_{:d}_j_{:.2E}".format(args.spins, args.Jconst)
        )
    else:
        prefix = Path(__file__).parent.joinpath(args.prefix)

    solver = HeisenbergCorrelationSolver(n_spins=args.spins, j=args.Jconst)

    prefix.mkdir(parents=True, exist_ok=True)
    sys.stdout.write("Report path: {:s}".format(str(prefix)))

    exact_prefix = prefix.joinpath("exact")
    exact_prefix.mkdir(parents=True, exist_ok=True)
    exact_solution = solver.exact()

    exact_prefix.joinpath("ground_state.txt").write_text(str(exact_solution))

    solver.solve()

    solver.get_report().to_csv(
        str(prefix.joinpath("main_report.csv").absolute()),
        index=False
    )

    corr_prefix = prefix.joinpath("correlations")
    corr_prefix.mkdir(parents=True, exist_ok=True)

    real_spins = args.spins*args.spins

    for i, corr_mat in enumerate(solver.get_correlations()):
        np.savetxt(
            fname=str(corr_prefix.joinpath("corr_step_{:d}.txt".format(i)).absolute()),
            fmt="%.4f",
            X=corr_mat
        )

        f: plt.Figure = plt.figure(figsize=(10, 10))
        ax: plt.Axes = f.add_subplot()
        ax.imshow(corr_mat)
        ax.set_xticks(np.arange(real_spins))
        ax.set_yticks(np.arange(real_spins))
        for k in range(real_spins):
            for j in range(real_spins):
                ax.text(
                    j, k, "{:.2f}".format(corr_mat[k, j]),
                    ha="center", va="center", color="w"
                )

        f.savefig(str(corr_prefix.joinpath("corr_plot_step_{:d}.png".format(i)).absolute()), dpi=150)
        plt.close(f)

    sys.stdout.write("Done.")

    sys.stdout.write(str(solver.get_sample()))

    sys.exit(0)