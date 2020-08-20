import sys
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional

import netket as nk
from netket.operator.spin import sigmaz
import numpy as np
import pandas as pd
from tqdm import tqdm


class SpinCorrelationSolver(ABC):
    @abstractmethod
    def _set_graph(self):
        self.graph: Optional[nk.graph.Graph] = None
        raise NotImplementedError()

    @abstractmethod
    def _set_operator(self):
        self.hamiltonian: Optional[nk.operator.GraphOperator] = None
        raise NotImplementedError()

    def reset(self):
        self._set_graph()
        sys.stdout.write("Graph one the {:d} vertices.\n".format(self.graph.n_sites))
        sys.stdout.flush()
        self.n_spins = self.graph.n_sites
        self.hilbert = nk.hilbert.Spin(graph=self.graph, s=0.5)
        self.machine = nk.machine.RbmSpin(hilbert=self.hilbert, alpha=3)
        self.machine.init_random_parameters(seed=42, sigma=1.0e-2)
        self.sampler = nk.sampler.MetropolisExchange(machine=self.machine)
        self._set_operator()
        self.optimizer = nk.optimizer.RmsProp()

        use_cholesky = self.machine.n_par < 10000

        self.vmc = nk.Vmc(
            hamiltonian=self.hamiltonian,
            sampler=self.sampler,
            optimizer=self.optimizer,
            n_samples=max([1500, self.n_spins * 50]),
            sr=nk.variational._SR(
                lsq_solver="LLT",
                diag_shift=1.0e-2,
                use_iterative=not use_cholesky,
                is_holomorphic=self.sampler.machine.is_holomorphic,
            ),
        )

        sys.stdout.write(self.vmc.info())
        sys.stdout.write("/n")
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

    def solve(self, n_iter: int = 800) -> None:
        step = max([1, n_iter // 5])
        early_stopping = 5
        iterator = self.vmc.iter(n_steps=n_iter, step=step)

        steps = deque()
        energies = deque()
        variances = deque()
        acceptances = deque()
        correlations = deque()

        zero_steps = 0

        for i in tqdm(iterator):
            steps.append(i)
            exp = self.vmc.energy

            e = np.real(exp.mean)
            var = np.real(exp.variance)

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
            correlations.append(self._compute_correlations())

            if zero_steps >= early_stopping:
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

    def get_report(self) -> pd.DataFrame:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return self.report

    def get_correlations(self) -> List[np.ndarray]:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return self.correlations

    def _compute_correlations(self) -> np.ndarray:
        corrs = self.vmc.estimate(self.corr_operators)
        corr_mat = np.zeros(shape=(self.n_spins, self.n_spins), dtype=np.float64)
        for i in range(self.n_spins):
            for j in range(self.n_spins):
                corr_mat[i, j] = np.real(corrs["{:d}-{:d}".format(i, j)].mean)

        return corr_mat

    def get_sample(self) -> np.ndarray:
        if self.report is None:
            raise ValueError("You must solve the problem first!")

        return next(self.sampler)[0, :]
