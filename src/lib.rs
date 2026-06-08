//! # si-lyapunov-fleet
//!
//! Lyapunov stability theory for fleet convergence.
//!
//! This crate provides a complete toolkit for proving stability and
//! convergence of multi-agent fleets using Lyapunov methods. It includes:
//!
//! - **Quadratic Lyapunov functions** — V(x) = x^T P x with positive-definiteness checks
//! - **Barrier Lyapunov functions** — For constrained systems with state boundaries
//! - **Fleet Lyapunov functions** — Sum of agent energies with consensus coupling
//! - **LaSalle invariance principle** — For proving asymptotic (not just Lyapunov) stability
//!
//! All modules are pure Rust with zero external dependencies.

pub mod barrier;
pub mod fleet;
pub mod lasalle;
pub mod quadratic;

/// Re-export commonly used items.
pub use barrier::{BarrierLyapunov, StateConstraint, symmetric_barrier_lyapunov};
pub use fleet::{AgentState, FleetLyapunov, simulate_consensus_dynamics};
pub use lasalle::{LaSalleResult, estimate_region_of_attraction, is_asymptotically_stable, is_hurwitz, lasalle_linear};
pub use quadratic::{QuadraticLyapunov, diagonal_lyapunov, identity_lyapunov};

/// Check the fundamental Lyapunov conditions for a candidate function and dynamics.
///
/// Returns `true` if both conditions hold for all sampled points:
/// 1. V(x) > 0 for all x ≠ 0 (positive definite)
/// 2. dV/dt < 0 for all x ≠ 0 (negative definite derivative)
pub fn check_lyapunov_conditions<F>(
    v: &QuadraticLyapunov,
    dynamics: F,
    sample_points: &[Vec<f64>],
    v_tol: f64,
    dv_tol: f64,
) -> bool
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    for x in sample_points {
        let v_val = v.evaluate(x);
        let dxdt = dynamics(x);
        let dvdt = v.time_derivative(x, &dxdt);

        // Check V(0) = 0 and V(x) > 0 for x ≠ 0
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        if norm_sq > v_tol && v_val <= v_tol {
            return false;
        }

        // Check dV/dt < 0 for x ≠ 0
        if norm_sq > v_tol && dvdt >= -dv_tol {
            return false;
        }
    }
    true
}

/// Compute the convergence rate bound for a stable linear system.
///
/// For dx/dt = A x with Lyapunov V(x) = x^T P x, the convergence rate
/// satisfies V(t) ≤ V(0) * exp(-λ t) where λ = min_eigenvalue(Q) / max_eigenvalue(P)
/// and Q = -(A^T P + P A).
pub fn convergence_rate_bound(a: &[Vec<f64>], p: &QuadraticLyapunov) -> Option<f64> {
    if !is_asymptotically_stable(a, p) {
        return None;
    }

    let n = a.len();
    // Compute Q = -(A^T P + P A)
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                q[i][j] += a[k][i] * p.p[k][j] + p.p[i][k] * a[k][j];
            }
        }
    }

    let neg_q: Vec<Vec<f64>> = q.iter().map(|row| row.iter().map(|v| -v).collect()).collect();
    let q_lyap = QuadraticLyapunov::new(neg_q);
    // Bound: λ_min(Q) / λ_max(P)
    // Using Gershgorin bounds as approximations
    let lambda_q_min = gershgorin_min_eigenvalue(&q_lyap.p);
    let lambda_p_max = p.spectral_radius_bound();

    if lambda_p_max > 0.0 {
        Some(lambda_q_min / lambda_p_max)
    } else {
        None
    }
}

/// Lower bound on the minimum eigenvalue via Gershgorin circles.
fn gershgorin_min_eigenvalue(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    (0..n)
        .map(|i| {
            let center = matrix[i][i];
            let radius: f64 = (0..n).filter(|&j| j != i).map(|j| matrix[i][j].abs()).sum();
            center - radius
        })
        .fold(f64::INFINITY, f64::min)
}

/// Generate a uniform grid of points in [-range, range]^dim.
pub fn uniform_grid(dim: usize, range: f64, steps: usize) -> Vec<Vec<f64>> {
    if dim == 0 {
        return vec![vec![]];
    }
    if steps == 0 {
        return vec![vec![0.0; dim]];
    }
    let values: Vec<f64> = (0..=steps)
        .map(|i| -range + 2.0 * range * (i as f64) / (steps as f64))
        .collect();

    let mut result = vec![vec![]];
    for _ in 0..dim {
        let mut new_result = Vec::new();
        for r in &result {
            for &v in &values {
                let mut new = r.clone();
                new.push(v);
                new_result.push(new);
            }
        }
        result = new_result;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_lyapunov_stable_system() {
        let v = identity_lyapunov(2);
        let dynamics = |x: &[f64]| vec![-x[0], -2.0 * x[1]];
        let grid = uniform_grid(2, 2.0, 4);
        assert!(check_lyapunov_conditions(&v, dynamics, &grid, 1e-6, 1e-6));
    }

    #[test]
    fn test_check_lyapunov_unstable_system() {
        let v = identity_lyapunov(2);
        let dynamics = |x: &[f64]| vec![x[0], x[1]];
        let grid = uniform_grid(2, 1.0, 3);
        assert!(!check_lyapunov_conditions(&v, dynamics, &grid, 1e-6, 1e-6));
    }

    #[test]
    fn test_convergence_rate_bound() {
        let a = vec![
            vec![-1.0, 0.0],
            vec![0.0, -2.0],
        ];
        let p = identity_lyapunov(2);
        let rate = convergence_rate_bound(&a, &p);
        assert!(rate.is_some());
        assert!(rate.unwrap() > 0.0);
    }

    #[test]
    fn test_convergence_rate_unstable() {
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let p = identity_lyapunov(2);
        let rate = convergence_rate_bound(&a, &p);
        assert!(rate.is_none());
    }

    #[test]
    fn test_uniform_grid_1d() {
        let grid = uniform_grid(1, 1.0, 2);
        assert_eq!(grid.len(), 3); // -1, 0, 1
    }

    #[test]
    fn test_uniform_grid_2d() {
        let grid = uniform_grid(2, 1.0, 1);
        assert_eq!(grid.len(), 4); // 2x2 grid
    }

    #[test]
    fn test_gershgorin_min_identity() {
        let m = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let min_eig = gershgorin_min_eigenvalue(&m);
        assert!((min_eig - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reexports() {
        let _v = identity_lyapunov(2);
        let _b = symmetric_barrier_lyapunov(2, &[1.0, 1.0], 0.1);
        let _f = FleetLyapunov::new_uniform(2, 1.0);
    }
}
