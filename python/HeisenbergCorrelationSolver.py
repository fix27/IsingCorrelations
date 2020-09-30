from typing import Dict, List, Optional

from jax.experimental import stax
import netket as nk
import numpy as np
import pandas as pd
from netket.operator.spin import sigmaz


from python.JaxUtils import SumLayer, LogCoshLayer
from python.SpinCorrelationSolver import SpinCorrelationSolver


class HeisenbergCorrelationSolver(SpinCorrelationSolver):
    def __init__(self, n_spins: int, j: float) -> None:
        self.n_spins: Optional[int] = n_spins * n_spins
        self.dim = n_spins
        self.j = j

        self.report: Optional[pd.DataFrame] = None
        self.layers = (
            nk.layer.FullyConnected(
                input_size=self.n_spins, output_size=20, use_bias=True
            ),
            nk.layer.FullyConnected(input_size=20, output_size=10, use_bias=True),
            nk.layer.Lncosh(input_size=10),
            nk.layer.SumOutput(input_size=10),
        )
        self.graph: Optional[nk.graph.Graph] = None
        self.hilbert: Optional[nk.hilbert.Hilbert] = None
        self.sampler: Optional[nk.sampler.MetropolisExchange] = None
        self.hamiltonian: Optional[nk.operator.GraphOperator] = None
        self.optimizer: Optional[nk.optimizer.Optimizer] = None
        self.vmc: Optional[nk.Vmc] = None
        self.corr_operators: Optional[Dict[str, nk.operator.LocalOperator]] = None
        self.correlations: List[np.ndarray] = []

        self.reset()
        self.corr_operators = {}
        self.k = self.n_spins // 2 + self.dim // 2

        for i in range(self.n_spins):
            self.corr_operators["{:d}-{:d}".format(self.k, i)] = sigmaz(
                self.hilbert, self.k
            ) * sigmaz(self.hilbert, i)

    def _set_graph(self):
        self.graph = nk.graph.Hypercube(length=self.dim, n_dim=2, pbc=False)

    def _set_operator(self):
        self.hamiltonian = nk.operator.Heisenberg(self.hilbert, self.j, False)

    def _compute_correlations(self):
        corrs = self.vmc.estimate(self.corr_operators)

        corr_mat = np.zeros(shape=self.n_spins, dtype=np.float64)
        var_mat = np.zeros(shape=self.n_spins, dtype=np.float64)
        for i in range(self.n_spins):
            corr_mat[i] = np.real(corrs["{:d}-{:d}".format(self.k, i)].mean)
            var_mat[i] = np.real(corrs["{:d}-{:d}".format(self.k, i)].variance)

        return corr_mat.reshape((self.dim, self.dim)), var_mat.reshape(
            (self.dim, self.dim)
        )

    def exact_corr_mat(self) -> np.ndarray:
        psi = nk.exact.lanczos_ed(
            self.hamiltonian, first_n=1, compute_eigenvectors=True
        ).eigenvectors[0]

        psi /= np.linalg.norm(psi)

        corr_mat = np.zeros(self.n_spins, dtype=np.float32)
        for i in range(self.n_spins):
            corr_mat[i] = np.vdot(
                psi,
                self.corr_operators["{:d}-{:d}".format(self.k, i)].to_sparse().dot(psi),
            )

        return corr_mat.reshape((self.dim, self.dim))
