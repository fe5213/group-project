"""
Calibration and Markov structure for the baseline fiscal RBC model.
Single source of truth for parameters used by steady state, linear solver, and PF paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Params:
    """Model parameters (quarterly calibration)."""

    beta: float = 0.99
    sigma: float = 2.0
    phi: float = 1.0
    chi: float = 15.5  # labor disutility scale; tuned for n ~ 0.33 at baseline calibration
    alpha: float = 0.36
    delta: float = 0.025
    # Productivity Markov (two states)
    z_L: float = 0.98
    z_H: float = 1.02
    pi_LL: float = 0.9
    pi_LH: float = 0.1
    pi_HL: float = 0.1
    pi_HH: float = 0.9
    # Fiscal baseline (share of steady-state output; actual g set in steady state)
    g_y_ratio: float = 0.20
    # Debt in steady state as ratio of quarterly output (0 = Ricardian baseline)
    b_y_ratio: float = 0.0

    @property
    def Pi(self) -> np.ndarray:
        return np.array(
            [[self.pi_LL, self.pi_LH], [self.pi_HL, self.pi_HH]], dtype=float
        )

    def ergodic_z(self) -> tuple[float, np.ndarray]:
        """Unconditional mean of z and ergodic distribution (pi_L, pi_H)."""
        P = self.Pi
        w, v = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, idx])
        pi = pi / pi.sum()
        z_bar = pi[0] * self.z_L + pi[1] * self.z_H
        return float(z_bar), pi

    def z_idx(self, z: float) -> int:
        return 0 if abs(z - self.z_L) < abs(z - self.z_H) else 1


def default_params() -> Params:
    return Params()
