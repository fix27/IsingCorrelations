from typing import Dict, List, Optional

import netket as nk
import numpy as np
import pandas as pd

from python.SpinCorrelationSolver import SpinCorrelationSolver


class HeisenbergCorrelationSolver(SpinCorrelationSolver):
    def __init__(self, n_spins: int, j: float) -> None:
        self.n_spins: Optional[int] = n_spins
        self.j = j

        self.report: Optional[pd.DataFrame] = None
        self.machine: Optional[nk.machine.Machine] = None
        self.graph: Optional[nk.graph.Graph] = None
        self.hilbert: Optional[nk.hilbert.Hilbert] = None
        self.sampler: Optional[nk.sampler.MetropolisExchange] = None
        self.hamiltonian: Optional[nk.operator.GraphOperator] = None
        self.optimizer: Optional[nk.optimizer.Optimizer] = None
        self.vmc: Optional[nk.Vmc] = None
        self.corr_operators: Optional[Dict[str, nk.operator.LocalOperator]] = None
        self.correlations: List[np.ndarray] = []

        self.reset()

    def _set_graph(self):
        self.graph = nk.graph.Hypercube(length=self.n_spins, n_dim=2, pbc=False)

    def _set_operator(self):
        self.hamiltonian = nk.operator.Heisenberg(self.hilbert, self.j, False)
