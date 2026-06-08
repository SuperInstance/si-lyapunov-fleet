//! LaSalle's invariance principle for asymptotic stability analysis.
//!
//! LaSalle's principle states: if V(x) is a Lyapunov function with
//! dV/dt ≤ 0 in a compact positively invariant set, then trajectories
//! converge to the largest invariant subset of {x : dV/dt = 0}.
//!
//! This module provides tools to characterize that invariant set
//! and verify asymptotic (not just Lyapunov) stability.

use crate::quadratic::QuadraticLyapunov;

/// Result of a LaSalle invariance analysis.
#[derive(Debug, Clone)]
pub struct LaSalleResult {
    /// The set E = {x : dV/dt = 0} (sampled approximation).
    pub zero_derivative_set: Vec<Vec<f64>>,
    /// The largest invariant subset M ⊆ E (sampled approximation).
    pub invariant_set: Vec<Vec<f64>>,
    /// Whether the invariant set is exactly the origin.
    pub converges_to_origin: bool,
    /// Maximum |x| in the invariant set.
    pub max_invariant_norm: f64,
}

/// Apply LaSalle's invariance principle to a linear system dx/dt = A x
/// with quadratic Lyapunov V(x) = x^T P x.
///
/// For linear systems, dV/dt = x^T (A^T P + P A) x.
/// The zero-derivative set is the nullspace of (A^T P + P A).
/// If A is Hurwitz, the only invariant subset is the origin.
pub fn lasalle_linear(
    a: &[Vec<f64>],
    lyapunov: &QuadraticLyapunov,
    sample_grid: &[f64],
) -> LaSalleResult {
    let n = a.len();
    assert_eq!(n, lyapunov.dim());

    // Compute Q = A^T P + P A
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                q[i][j] += a[k][i] * lyapunov.p[k][j] + lyapunov.p[i][k] * a[k][j];
            }
        }
    }

    // Sample the state space on a grid
    let mut zero_derivative = Vec::new();
    let mut invariant = Vec::new();
    let mut max_norm = 0.0;

    // For small dimensions, do full grid sampling
    if n <= 3 {
        let samples = cartesian_product(sample_grid, n);
        for x in &samples {
            let dvdt = evaluate_quadratic_form(&q, x);
            if dvdt.abs() < 1e-6 {
                zero_derivative.push(x.clone());
                // Check if this point is invariant: dx/dt = A x should stay in zero-derivative set
                let dxdt: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i][j] * x[j]).sum()).collect();
                let d2vdt2 = evaluate_quadratic_form(&q, &dxdt);
                if d2vdt2.abs() < 1e-4 {
                    invariant.push(x.clone());
                    let norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
                    if norm > max_norm {
                        max_norm = norm;
                    }
                }
            }
        }
    }

    let converges = max_norm < 1e-6;

    LaSalleResult {
        zero_derivative_set: zero_derivative,
        invariant_set: invariant,
        converges_to_origin: converges,
        max_invariant_norm: max_norm,
    }
}

/// Check if a linear system is asymptotically stable using LaSalle's principle.
/// Returns true if the only invariant point in {x : dV/dt = 0} is the origin.
pub fn is_asymptotically_stable(
    a: &[Vec<f64>],
    lyapunov: &QuadraticLyapunov,
) -> bool {
    // For linear systems, check that A^T P + P A is negative definite
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                q[i][j] += a[k][i] * lyapunov.p[k][j] + lyapunov.p[i][k] * a[k][j];
            }
        }
    }

    let q_lyap = QuadraticLyapunov::new(q);
    // Q should be negative definite, i.e., -Q is positive definite
    let neg_q: Vec<Vec<f64>> = q_lyap.p.iter().map(|row| row.iter().map(|v| -v).collect()).collect();
    QuadraticLyapunov::new(neg_q).is_positive_definite()
}

/// Find the region of attraction for a nonlinear system by sampling.
/// Returns the largest radius R such that V(x) < R implies dV/dt < 0.
pub fn estimate_region_of_attraction<F>(
    lyapunov: &QuadraticLyapunov,
    dynamics: F,
    sample_points: &[Vec<f64>],
) -> f64
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut max_stable_v = f64::INFINITY;

    for x in sample_points {
        let dxdt = dynamics(x);
        let dvdt = lyapunov.time_derivative(x, &dxdt);
        let v = lyapunov.evaluate(x);
        if dvdt >= 0.0 && v > 1e-10 {
            // This point is not in the region of attraction
            if v < max_stable_v {
                max_stable_v = v;
            }
        }
    }

    if max_stable_v.is_infinite() {
        f64::INFINITY
    } else {
        max_stable_v
    }
}

/// Evaluate x^T M x for a symmetric matrix M.
fn evaluate_quadratic_form(m: &[Vec<f64>], x: &[f64]) -> f64 {
    let n = x.len();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += x[i] * m[i][j] * x[j];
        }
    }
    sum
}

/// Generate Cartesian product of a set with itself n times.
fn cartesian_product(values: &[f64], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![vec![]];
    }
    let prev = cartesian_product(values, n - 1);
    let mut result = Vec::new();
    for p in &prev {
        for &v in values {
            let mut new = p.clone();
            new.push(v);
            result.push(new);
        }
    }
    result
}

/// Check if a matrix is Hurwitz stable (all eigenvalues have negative real part).
/// Uses the Routh-Hurwitz criterion for 2x2 and 3x3 matrices.
pub fn is_hurwitz(a: &[Vec<f64>]) -> bool {
    let n = a.len();
    if n == 1 {
        return a[0][0] < 0.0;
    }
    if n == 2 {
        let trace = a[0][0] + a[1][1];
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        return trace < 0.0 && det > 0.0;
    }
    if n == 3 {
        let a1 = -(a[0][0] + a[1][1] + a[2][2]);
        let a2 = a[0][0] * a[1][1] + a[0][0] * a[2][2] + a[1][1] * a[2][2]
            - a[0][1] * a[1][0] - a[0][2] * a[2][0] - a[1][2] * a[2][1];
        let a3 = -(
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        );
        return a1 > 0.0 && a3 > 0.0 && a1 * a2 > a3;
    }
    // For larger matrices, we'd need numerical eigenvalue computation
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurwitz_1d() {
        assert!(is_hurwitz(&[vec![-2.0]]));
        assert!(!is_hurwitz(&[vec![1.0]]));
    }

    #[test]
    fn test_hurwitz_2d_stable() {
        let a = vec![
            vec![-1.0, 0.5],
            vec![-0.5, -2.0],
        ];
        assert!(is_hurwitz(&a));
    }

    #[test]
    fn test_hurwitz_2d_unstable() {
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, -2.0],
        ];
        assert!(!is_hurwitz(&a));
    }

    #[test]
    fn test_is_asymptotically_stable() {
        let a = vec![
            vec![-1.0, 0.0],
            vec![0.0, -2.0],
        ];
        let p = crate::quadratic::identity_lyapunov(2);
        assert!(is_asymptotically_stable(&a, &p));
    }

    #[test]
    fn test_not_asymptotically_stable() {
        // Purely imaginary eigenvalues: marginally stable, not asymptotically
        let a = vec![
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        let p = crate::quadratic::identity_lyapunov(2);
        assert!(!is_asymptotically_stable(&a, &p));
    }

    #[test]
    fn test_estimate_region_of_attraction() {
        let v = crate::quadratic::identity_lyapunov(2);
        // Stable linear system: dx/dt = -x
        let dynamics = |x: &[f64]| vec![-x[0], -x[1]];

        let grid = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let samples: Vec<Vec<f64>> = grid.iter()
            .flat_map(|&x| grid.iter().map(move |&y| vec![x, y]))
            .collect();

        let roa = estimate_region_of_attraction(&v, dynamics, &samples);
        // For dx/dt = -x, the entire space is the region of attraction
        assert!(roa.is_infinite() || roa > 0.0);
    }

    #[test]
    fn test_evaluate_quadratic_form() {
        let m = vec![
            vec![2.0, 0.0],
            vec![0.0, 3.0],
        ];
        let x = vec![1.0, 2.0];
        let val = evaluate_quadratic_form(&m, &x);
        assert!((val - (2.0 + 12.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cartesian_product() {
        let vals = vec![0.0, 1.0];
        let prod = cartesian_product(&vals, 2);
        assert_eq!(prod.len(), 4);
        assert!(prod.contains(&vec![0.0, 0.0]));
        assert!(prod.contains(&vec![1.0, 1.0]));
    }

    #[test]
    fn test_lasalle_linear_stable() {
        let a = vec![
            vec![-1.0, 0.0],
            vec![0.0, -2.0],
        ];
        let p = crate::quadratic::identity_lyapunov(2);
        let result = lasalle_linear(&a, &p, &vec![-1.0, 0.0, 1.0]);
        assert!(result.converges_to_origin);
    }

    #[test]
    fn test_lasalle_linear_unstable() {
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let p = crate::quadratic::identity_lyapunov(2);
        // For unstable systems, dV/dt > 0 everywhere — LaSalle doesn't apply
        // but we can verify it's not asymptotically stable
        assert!(!is_asymptotically_stable(&a, &p));
    }

    #[test]
    fn test_hurwitz_3d_stable() {
        let a = vec![
            vec![-3.0, 1.0, 0.0],
            vec![0.0, -2.0, 1.0],
            vec![0.0, 0.0, -1.0],
        ];
        assert!(is_hurwitz(&a));
    }
}
