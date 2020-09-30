import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, Callable

from jax.experimental import stax
import netket as nk
import numpy as np
import pandas as pd
from mpi4py import MPI
from netket.operator.spin import sigmaz
from tqdm import tqdm

warnings.filterwarnings("ignore")


class SpinCorrelationSolver(ABC):
    @abstractmethod
    def _set_graph(self):
        self.graph: Optional[nk.graph.Graph] = None
        raise NotImplementedError()

    @abstractmethod
    def _set_operator(self):
        self.hamiltonian: Optional[nk.operator.GraphOperator] = None
        raise NotImplementedError()

    @property
    def netfun(self) -> "Callable":
        return self._netfun

    @property
    def initfun(self) -> "Callable":
        return self._initfun

    def reset(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self._set_graph()
        if rank == 0:
            sys.stdout.write(
                "Graph one the {:d} vertices.\n".format(self.graph.n_sites)
            )
            sys.stdout.flush()
        self.n_spins = self.graph.n_sites

        self.machine = nk.machine.Jax(
            hilbert=self.hilbert, module=(self.netfun, self.initfun), seed=42
        )
        self.machine = nk.machine.RbmSpin(hilbert=self.hilbert, alpha=2)
        self.machine.init_random_parameters(seed=42, sigma=1.0e-2)
        self.sampler = nk.sampler.MetropolisLocal(self.machine)
        self._set_operator()
        self.optimizer = nk.optimizer.RmsProp()

        use_cholesky = self.machine.n_par < 10000

        self.vmc = nk.Vmc(
            hamiltonian=self.hamiltonian,
            sampler=self.sampler,
            optimizer=self.optimizer,
            n_samples=max([2000, self.n_spins * 50]),
            sr=nk.optimizer.SR(
                lsq_solver="LLT",
                diag_shift=1.0e-2,
                use_iterative=not use_cholesky,
                is_holomorphic=self.sampler.machine.is_holomorphic,
            ),
        )

        if rank == 0:
            sys.stdout.write("RBM with {:d} params.\n".format(self.machine.n_par))
            sys.stdout.write(self.vmc.info())
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.corr_operators = {}

        for i in range(self.n_spins):
            for j in range(self.n_spins):
                self.corr_operators["{:d}-{:d}".format(i, j)] = sigmaz(
                    self.hilbert, i
                ) * sigmaz(self.hilbert, j)
        self.correlations = []

    def exact(self) -> float:
        return nk.exact.lanczos_ed(
            self.hamiltonian, first_n=1, compute_eigenvectors=False, matrix_free=False
        ).eigenvalues[0]

    def exact_corr_mat(self) -> np.ndarray:
        psi = nk.exact.lanczos_ed(
            self.hamiltonian, first_n=1, compute_eigenvectors=True
        ).eigenvectors[0]

        psi /= np.linalg.norm(psi)

        corr_mat = np.zeros((self.n_spins, self.n_spins), dtype=np.float32)
        for i in range(self.n_spins):
            for j in range(self.n_spins):
                corr_mat[i, j] = np.vdot(
                    psi,
                    self.corr_operators["{:d}-{:d}".format(i, j)].to_sparse().dot(psi),
                )

        return corr_mat

    def solve(self, n_iter: int = 2500) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        step = max([1, n_iter // 10])
        early_stopping = 5
        iterator = self.vmc.iter(n_steps=n_iter, step=step)

        steps = deque()
        energies = deque()
        variances = deque()
        acceptances = deque()
        correlations = deque()
        correlations_variance = deque()

        zero_steps = 0

        for i in tqdm(iterator):
            steps.append(i)
            exp = self.vmc.energy

            e = np.real(exp.mean)
            var = np.real(exp.variance)

            corrs = self._compute_correlations()
            correlations.append(corrs[0])
            correlations_variance.append(corrs[1])

            if rank == 0:
                sys.stdout.write(
                    "\tStep: {:d}\tEnergy: {:.4f}\tVariance: {:.4f}\n".format(i, e, var)
                )
                sys.stdout.flush()
            energies.append(e)
            variances.append(var)

            if var < 1e-6:
                zero_steps += 1
            else:
                zero_steps = 0

            acceptances.append(self.sampler.acceptance)

            if zero_steps >= early_stopping:
                if rank == 0:
                    sys.stdout.write(
                        "{:d} rounds with zero variance reached. Stop the process.\n".format(
                            early_stopping
                        )
                    )
                    sys.stdout.flush()
                break

        self.report = pd.DataFrame(
            {
                "steps": list(steps),
                "energies": list(energies),
                "variances": list(variances),
                "acceptances": list(acceptances),
            }
        ).drop_duplicates()

        self.correlations = list(correlations)
        self.correlations_variance = list(correlations_variance)

    def get_report(self) -> pd.DataFrame:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return self.report

    def get_correlations(self) -> List[np.ndarray]:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return self.correlations

    def get_correlations_variances(self) -> List[np.ndarray]:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return self.correlations_variance

    def _compute_correlations(self) -> np.ndarray:
        corrs = self.vmc.estimate(self.corr_operators)
        corr_mat = np.zeros(shape=(self.n_spins, self.n_spins), dtype=np.float64)
        var_mat = np.zeros(shape=(self.n_spins, self.n_spins), dtype=np.float64)
        for i in range(self.n_spins):
            for j in range(self.n_spins):
                corr_mat[i, j] = np.real(corrs["{:d}-{:d}".format(i, j)].mean)
                var_mat[i, j] = np.real(corrs["{:d}-{:d}".format(i, j)].variance)

        return corr_mat, var_mat

    def get_sample(self) -> np.ndarray:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return next(self.sampler)[0, :]
