"""Experimental iterative ``MultiLCA`` backend for Pathways.

This backend keeps Pathways on the standard ``bw2calc.MultiLCA`` lifecycle,
but swaps the direct sparse solve for GMRES with a Jacobi preconditioner.
Whenever available, it reuses the matrix-preparation and preconditioner
helpers from ``bw2calc.JacobiGMRESLCA``.
"""

from __future__ import annotations

import logging
from typing import Optional

import bw2calc as bc
import matrix_utils as mu
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

logger = logging.getLogger(__name__)

try:
    _BW_JACOBI_GMRES_LCA = bc.JacobiGMRESLCA
except AttributeError:  # pragma: no cover - older bw2calc versions
    _BW_JACOBI_GMRES_LCA = None


class JacobiGMRESMultiLCA(bc.MultiLCA):
    """Solve multi-demand LCI systems with GMRES and Jacobi preconditioning."""

    def __init__(
        self,
        *args,
        rtol: float = 1e-8,
        atol: float = 0.0,
        restart: Optional[int] = 50,
        maxiter: Optional[int] = 300,
        use_guess: bool = True,
        direct_fallback: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rtol = rtol
        self.atol = atol
        self.restart = restart
        self.maxiter = maxiter
        self.use_guess = use_guess
        self.direct_fallback = direct_fallback

        self._matrix_prepared = False
        self._cached_preconditioner: Optional[LinearOperator] = None
        self.guesses: dict[str, np.ndarray] = {}

    def __next__(self) -> None:
        # Matrix values can change on each Monte Carlo draw.
        self._matrix_prepared = False
        self._cached_preconditioner = None
        self.guesses = {}
        super().__next__()

    def load_lci_data(self, nonsquare_ok=False) -> None:
        super().load_lci_data(nonsquare_ok=nonsquare_ok)
        self._matrix_prepared = False
        self._cached_preconditioner = None
        self.guesses = {}

    def _prepare_matrix(self) -> None:
        # ``MappedMatrix`` updates ``technosphere_mm.matrix`` across MC draws.
        # Rebind here so GMRES always sees the current technosphere values instead
        # of a stale CSC conversion from an earlier draw.
        if hasattr(self, "technosphere_mm"):
            self.technosphere_matrix = self.technosphere_mm.matrix

        if _BW_JACOBI_GMRES_LCA is not None:
            _BW_JACOBI_GMRES_LCA._prepare_matrix(self)
            return

        if self._matrix_prepared:
            return

        self.technosphere_matrix = self.technosphere_matrix.tocsc(copy=False)
        self.technosphere_matrix.sum_duplicates()
        self.technosphere_matrix.eliminate_zeros()
        self.technosphere_matrix.sort_indices()
        self._matrix_prepared = True

    def _build_jacobi_preconditioner(self) -> Optional[LinearOperator]:
        if _BW_JACOBI_GMRES_LCA is not None:
            return _BW_JACOBI_GMRES_LCA._build_jacobi_preconditioner(self)

        if self._cached_preconditioner is not None:
            return self._cached_preconditioner

        diagonal = self.technosphere_matrix.diagonal()
        if np.any(diagonal == 0):
            return None

        inverse_diagonal = 1.0 / diagonal
        self._cached_preconditioner = LinearOperator(
            shape=self.technosphere_matrix.shape,
            matvec=lambda x: inverse_diagonal * x,
            dtype=self.technosphere_matrix.dtype,
        )
        return self._cached_preconditioner

    def _solve_with_gmres(
        self,
        demand: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        demand_name: str | None = None,
    ) -> np.ndarray:
        self._prepare_matrix()
        preconditioner = self._build_jacobi_preconditioner()

        try:
            solution, info = gmres(
                self.technosphere_matrix,
                demand,
                x0=x0,
                rtol=self.rtol,
                atol=self.atol,
                restart=self.restart,
                maxiter=self.maxiter,
                M=preconditioner,
            )
        except TypeError:  # pragma: no cover - SciPy compatibility fallback
            solution, info = gmres(
                self.technosphere_matrix,
                demand,
                x0=x0,
                tol=self.rtol,
                atol=self.atol,
                restart=self.restart,
                maxiter=self.maxiter,
                M=preconditioner,
            )

        solution = np.asarray(solution, dtype=np.float64)
        if not solution.shape:
            solution = solution.reshape((1,))

        if info != 0:
            if not self.direct_fallback:
                raise RuntimeError(
                    "GMRES failed to converge "
                    f"(demand={demand_name!r}, info={info}, rtol={self.rtol}, maxiter={self.maxiter})"
                )

            logger.warning(
                "GMRES failed to converge for demand %s; falling back to direct solve.",
                demand_name,
            )
            solution = np.asarray(bc.spsolve(self.technosphere_matrix, demand))
            if not solution.shape:
                solution = solution.reshape((1,))

        return solution

    def lci_calculation(self) -> None:
        """Calculate inventories for many demands using iterative solves."""
        count = len(self.dicts.activity)
        demand_items = list(self.demand_arrays.items())
        if not demand_items:
            self.supply_arrays = {}
            self.inventories = mu.SparseMatrixDict([])
            return

        supply_arrays: dict[str, np.ndarray] = {}
        previous_solution: np.ndarray | None = None

        for name, demand in demand_items:
            x0 = None
            if self.use_guess:
                x0 = self.guesses.get(name)
                if x0 is None:
                    x0 = previous_solution

            solution = self._solve_with_gmres(demand, x0=x0, demand_name=name)
            supply_arrays[name] = solution
            previous_solution = solution

            if self.use_guess:
                self.guesses[name] = solution

        self.supply_arrays = supply_arrays
        self.inventories = mu.SparseMatrixDict(
            [
                (
                    name,
                    self.biosphere_matrix @ sparse.spdiags([arr], [0], count, count),
                )
                for name, arr in self.supply_arrays.items()
            ]
        )
