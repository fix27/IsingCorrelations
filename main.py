import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pylab as plt
from mpi4py import MPI
import numpy as np

from python import IsingCorrelationsSolver, HeisenbergCorrelationSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spins", required=True, type=int)
    parser.add_argument("--Hconst", required=True, type=float)
    parser.add_argument("--Jconst", required=True, type=float)
    parser.add_argument("--Ham", required=True, type=str)
    parser.add_argument("--prefix", required=False, default="", type=str)

    args = parser.parse_args()
    prefix: Optional[Path] = None

    if args.prefix == "":
        prefix = Path(__file__).parent.joinpath(
            "output_{:s}_spins_{:d}_h_{:.2E}_j_{:.2E}".format(
                args.Ham, args.spins, args.Hconst, args.Jconst
            )
        )
    else:
        prefix = Path(__file__).parent.joinpath(args.prefix)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.Ham == "ising":
        solver = IsingCorrelationsSolver(
            n_spins=args.spins, h=args.Hconst, j=args.Jconst
        )
    elif args.Ham == "heisenberg":
        solver = HeisenbergCorrelationSolver(n_spins=args.spins, j=args.Jconst)
    else:
        raise ValueError()

    if rank == 0:
        prefix.mkdir(parents=True, exist_ok=True)
        sys.stdout.write("Report path: {:s}\n".format(str(prefix)))
        sys.stdout.flush()

    exact_prefix = prefix.joinpath("exact")
    exact_prefix.mkdir(parents=True, exist_ok=True)
    if (args.spins <= 20 and args.Ham == "ising") or (
        args.spins <= 5 and args.Ham == "heisenberg"
    ):
        exact_solution = solver.exact()
        exact_prefix.joinpath("ground_state.txt").write_text(str(exact_solution))

        exact_corr_mat = solver.exact_corr_mat()
        f: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes = f.add_subplot()
        ax.imshow(exact_corr_mat)
        ax.set_xticks(np.arange(args.spins))
        ax.set_yticks(np.arange(args.spins))
        for k in range(args.spins):
            for j in range(args.spins):
                ax.text(
                    j,
                    k,
                    "{:.2f}".format(exact_corr_mat[k, j]),
                    ha="center",
                    va="center",
                    color="w",
                )

        f.savefig(
            str(exact_prefix.joinpath("exact_corr_mat.png").absolute()),
            dpi=150,
        )
        plt.close(f)

    solver.solve()

    solver.get_report().to_csv(
        str(prefix.joinpath("main_report.csv").absolute()), index=False
    )

    corr_prefix = prefix.joinpath("correlations")
    corr_prefix.mkdir(parents=True, exist_ok=True)

    for i, corr_mat in enumerate(solver.get_correlations()):
        np.savetxt(
            fname=str(corr_prefix.joinpath("corr_step_{:d}.txt".format(i)).absolute()),
            fmt="%.4f",
            X=corr_mat,
        )

        f: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes = f.add_subplot()
        ax.imshow(corr_mat)
        ax.set_xticks(np.arange(args.spins))
        ax.set_yticks(np.arange(args.spins))
        for k in range(args.spins):
            for j in range(args.spins):
                ax.text(
                    j,
                    k,
                    "{:.2f}".format(corr_mat[k, j]),
                    ha="center",
                    va="center",
                    color="w",
                )

        f.savefig(
            str(corr_prefix.joinpath("corr_plot_step_{:d}.png".format(i)).absolute()),
            dpi=150,
        )
        plt.close(f)

    for i, corr_mat in enumerate(solver.get_correlations_variances()):
        np.savetxt(
            fname=str(
                corr_prefix.joinpath("corr_var_step_{:d}.txt".format(i)).absolute()
            ),
            fmt="%.4f",
            X=corr_mat,
        )

        f: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes = f.add_subplot()
        ax.imshow(corr_mat)
        ax.set_xticks(np.arange(args.spins))
        ax.set_yticks(np.arange(args.spins))
        for k in range(args.spins):
            for j in range(args.spins):
                ax.text(
                    j,
                    k,
                    "{:.2f}".format(corr_mat[k, j]),
                    ha="center",
                    va="center",
                    color="w",
                )

        f.savefig(
            str(
                corr_prefix.joinpath("corr_var_plot_step_{:d}.png".format(i)).absolute()
            ),
            dpi=150,
        )
        plt.close(f)

    if rank == 0:
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

    sys.exit(0)
