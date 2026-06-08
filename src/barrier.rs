//! Barrier Lyapunov functions for constrained systems.
//!
//! Extends quadratic Lyapunov with barrier terms that grow to infinity
//! as the state approaches constraint boundaries. Ensures the state
//! remains within safe regions while converging to equilibrium.

use crate::quadratic::QuadraticLyapunov;

/// A single state constraint: |x_i| < bound.
#[derive(Debug, Clone)]
pub struct StateConstraint {
    /// State variable index.
    pub index: usize,
    /// Constraint bound (strictly positive).
    pub bound: f64,
}

impl StateConstraint {
    pub fn new(index: usize, bound: f64) -> Self {
        assert!(bound > 0.0, "Constraint bound must be positive");
        Self { index, bound }
    }

    /// Check if a state satisfies this constraint.
    pub fn satisfied(&self, x: &[f64]) -> bool {
        x[self.index].abs() < self.bound
    }

    /// Compute the barrier term: log(bound^2 - x_i^2).
    pub fn barrier_term(&self, x: &[f64]) -> f64 {
        let xi = x[self.index];
        let diff = self.bound * self.bound - xi * xi;
        if diff > 0.0 {
            -diff.ln()
        } else {
            f64::INFINITY
        }
    }

    /// Gradient of the barrier term with respect to the constrained variable.
    pub fn barrier_gradient(&self, x: &[f64]) -> f64 {
        let xi = x[self.index];
        let diff = self.bound * self.bound - xi * xi;
        if diff > 1e-15 {
            2.0 * xi / diff
        } else {
            f64::INFINITY
        }
    }
}

/// A barrier Lyapunov function combining quadratic and barrier terms.
///
/// V(x) = x^T P x + κ * Σ barrier(x_i, bound_i)
///
/// where κ is the barrier weight and barrier(x_i, bound_i) ensures |x_i| < bound_i.
#[derive(Debug, Clone)]
pub struct BarrierLyapunov {
    /// Quadratic term.
    pub quadratic: QuadraticLyapunov,
    /// State constraints.
    pub constraints: Vec<StateConstraint>,
    /// Barrier weight.
    pub kappa: f64,
}

impl BarrierLyapunov {
    pub fn new(quadratic: QuadraticLyapunov, constraints: Vec<StateConstraint>, kappa: f64) -> Self {
        Self { quadratic, constraints, kappa }
    }

    /// Evaluate V(x) = x^T P x + κ * Σ barrier_i(x).
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let v_quad = self.quadratic.evaluate(x);
        let v_barrier: f64 = self.constraints.iter().map(|c| self.kappa * c.barrier_term(x)).sum();
        v_quad + v_barrier
    }

    /// Compute the gradient ∇V(x).
    pub fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = self.quadratic.gradient(x);
        for c in &self.constraints {
            if c.index < grad.len() {
                grad[c.index] += self.kappa * c.barrier_gradient(x);
            }
        }
        grad
    }

    /// Time derivative dV/dt = ∇V · dx/dt.
    pub fn time_derivative(&self, x: &[f64], dxdt: &[f64]) -> f64 {
        let grad = self.gradient(x);
        grad.iter().zip(dxdt.iter()).map(|(g, d)| g * d).sum()
    }

    /// Check if all constraints are satisfied.
    pub fn constraints_satisfied(&self, x: &[f64]) -> bool {
        self.constraints.iter().all(|c| c.satisfied(x))
    }

    /// Minimum distance to any constraint boundary.
    pub fn margin(&self, x: &[f64]) -> f64 {
        self.constraints
            .iter()
            .map(|c| (c.bound - x[c.index].abs()).abs())
            .fold(f64::INFINITY, f64::min)
    }

    /// Compute a safe step size such that constraints won't be violated
    /// when moving along direction dxdt for time dt.
    pub fn safe_step_size(&self, x: &[f64], dxdt: &[f64]) -> f64 {
        let mut max_dt = f64::INFINITY;
        for c in &self.constraints {
            let xi = x[c.index];
            let vi = dxdt[c.index];
            // Solve xi + vi * dt = ±bound for dt
            if vi > 1e-15 {
                let dt = (c.bound - xi) / vi;
                if dt > 0.0 && dt < max_dt {
                    max_dt = dt;
                }
            } else if vi < -1e-15 {
                let dt = (-c.bound - xi) / vi;
                if dt > 0.0 && dt < max_dt {
                    max_dt = dt;
                }
            }
        }
        max_dt
    }
}

/// Build a barrier Lyapunov with symmetric bounds on all states.
pub fn symmetric_barrier_lyapunov(
    dim: usize,
    bounds: &[f64],
    kappa: f64,
) -> BarrierLyapunov {
    assert_eq!(bounds.len(), dim, "One bound per dimension required");
    let q = crate::quadratic::identity_lyapunov(dim);
    let constraints: Vec<StateConstraint> = (0..dim)
        .map(|i| StateConstraint::new(i, bounds[i]))
        .collect();
    BarrierLyapunov::new(q, constraints, kappa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_term_inside() {
        let c = StateConstraint::new(0, 2.0);
        let term = c.barrier_term(&[1.0]);
        assert!(term.is_finite());
        assert!(term.is_finite());
    }

    #[test]
    fn test_barrier_term_at_boundary() {
        let c = StateConstraint::new(0, 2.0);
        let term = c.barrier_term(&[2.0]);
        assert!(term.is_infinite());
    }

    #[test]
    fn test_barrier_evaluate_inside() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.1);
        let v = bl.evaluate(&[1.0, 0.5]);
        assert!(v.is_finite());
        assert!(v > 0.0);
    }

    #[test]
    fn test_barrier_evaluate_outside() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.1);
        let v = bl.evaluate(&[3.0, 0.5]);
        assert!(v.is_infinite());
    }

    #[test]
    fn test_constraints_satisfied() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 3.0], 0.1);
        assert!(bl.constraints_satisfied(&[1.0, 2.0]));
        assert!(!bl.constraints_satisfied(&[2.5, 2.0]));
    }

    #[test]
    fn test_safe_step_size() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.1);
        let x = vec![0.0, 0.0];
        let dxdt = vec![1.0, 0.0];
        let dt = bl.safe_step_size(&x, &dxdt);
        assert!((dt - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_step_size_away_from_boundary() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.1);
        let x = vec![0.0, 0.0];
        let dxdt = vec![-1.0, 0.0];
        let dt = bl.safe_step_size(&x, &dxdt);
        assert!((dt - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_barrier_gradient() {
        let c = StateConstraint::new(0, 2.0);
        let g = c.barrier_gradient(&[0.5]);
        // 2*0.5 / (4 - 0.25) = 1.0 / 3.75 = 0.2667
        assert!(g > 0.0);
    }

    #[test]
    fn test_margin() {
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 3.0], 0.1);
        let m = bl.margin(&[1.0, 2.0]);
        assert!((m - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_barrier_time_derivative_negative() {
        // Stable system inside constraints: dx/dt = -x
        let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.1);
        let x = vec![0.5, 0.5];
        let dxdt = vec![-0.5, -0.5];
        let dvdt = bl.time_derivative(&x, &dxdt);
        // Quadratic term contributes negative, barrier term contributes
        // sign(x_i) * 2*x_i/diff * (-x_i) = -2*x_i^2/diff < 0
        assert!(dvdt < 0.0);
    }
}
