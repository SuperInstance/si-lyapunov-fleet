//! Quadratic Lyapunov functions.
//!
//! Provides V(x) = x^T P x where P is a symmetric positive-definite matrix.
//! Includes tests for positive definiteness, evaluation, gradient, and time derivative.

/// A quadratic Lyapunov function V(x) = x^T P x.
#[derive(Debug, Clone)]
pub struct QuadraticLyapunov {
    /// The symmetric positive-definite matrix P.
    pub p: Vec<Vec<f64>>,
}

impl QuadraticLyapunov {
    /// Create a new quadratic Lyapunov from matrix P.
    /// P must be square. Positive definiteness should be checked separately.
    pub fn new(p: Vec<Vec<f64>>) -> Self {
        Self { p }
    }

    /// Dimension of the state space.
    pub fn dim(&self) -> usize {
        self.p.len()
    }

    /// Evaluate V(x) = x^T P x.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim(), "State dimension mismatch");
        let n = self.dim();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum += x[i] * self.p[i][j] * x[j];
            }
        }
        sum
    }

    /// Compute the gradient ∇V(x) = 2 P x (for symmetric P).
    pub fn gradient(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim());
        let n = self.dim();
        (0..n)
            .map(|i| {
                (0..n).map(|j| 2.0 * self.p[i][j] * x[j]).sum()
            })
            .collect()
    }

    /// Compute the time derivative dV/dt = ∇V · f(x) = 2 x^T P f(x).
    pub fn time_derivative(&self, x: &[f64], dxdt: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim());
        assert_eq!(dxdt.len(), self.dim());
        let grad = self.gradient(x);
        grad.iter().zip(dxdt.iter()).map(|(g, d)| g * d).sum()
    }

    /// Check if P is symmetric within tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                if (self.p[i][j] - self.p[j][i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if P is positive definite using Sylvester's criterion.
    /// All leading principal minors must be positive.
    pub fn is_positive_definite(&self) -> bool {
        if !self.is_symmetric(1e-10) {
            return false;
        }
        let n = self.dim();
        for k in 1..=n {
            let minor = leading_principal_minor(&self.p, k);
            if minor <= 0.0 {
                return false;
            }
        }
        true
    }

    /// Check if P is positive semidefinite.
    pub fn is_positive_semidefinite(&self) -> bool {
        if !self.is_symmetric(1e-10) {
            return false;
        }
        let n = self.dim();
        for k in 1..=n {
            let minor = leading_principal_minor(&self.p, k);
            if minor < 0.0 {
                return false;
            }
        }
        true
    }

    /// Find the largest eigenvalue bound via Gershgorin circles.
    /// Returns an upper bound on the spectral radius.
    pub fn spectral_radius_bound(&self) -> f64 {
        let n = self.dim();
        (0..n)
            .map(|i| {
                let center = self.p[i][i].abs();
                let radius: f64 = (0..n).filter(|&j| j != i).map(|j| self.p[i][j].abs()).sum();
                center + radius
            })
            .fold(0.0, f64::max)
    }
}

/// Compute the determinant of a k×k leading principal submatrix.
fn leading_principal_minor(matrix: &[Vec<f64>], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut m: Vec<Vec<f64>> = matrix[..k].iter().map(|row| row[..k].to_vec()).collect();
    determinant(&mut m)
}

/// Compute determinant via Gaussian elimination (LU decomposition style).
fn determinant(matrix: &mut [Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    let mut det = 1.0;
    for i in 0..n {
        // Partial pivoting
        let mut max_row = i;
        for r in (i + 1)..n {
            if matrix[r][i].abs() > matrix[max_row][i].abs() {
                max_row = r;
            }
        }
        if max_row != i {
            matrix.swap(i, max_row);
            det = -det;
        }
        let pivot = matrix[i][i];
        if pivot.abs() < 1e-15 {
            return 0.0;
        }
        det *= pivot;
        for r in (i + 1)..n {
            let factor = matrix[r][i] / pivot;
            for c in i..n {
                matrix[r][c] -= factor * matrix[i][c];
            }
        }
    }
    det
}

/// Build a diagonal Lyapunov function V(x) = sum(p_i * x_i^2).
pub fn diagonal_lyapunov(weights: &[f64]) -> QuadraticLyapunov {
    let n = weights.len();
    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        p[i][i] = weights[i];
    }
    QuadraticLyapunov::new(p)
}

/// Build an identity-based Lyapunov V(x) = ||x||^2.
pub fn identity_lyapunov(dim: usize) -> QuadraticLyapunov {
    let mut p = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        p[i][i] = 1.0;
    }
    QuadraticLyapunov::new(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_1d() {
        let v = diagonal_lyapunov(&[2.0]);
        assert!((v.evaluate(&[3.0]) - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_identity() {
        let v = identity_lyapunov(2);
        assert!((v.evaluate(&[3.0, 4.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_diagonal() {
        let v = diagonal_lyapunov(&[2.0, 3.0]);
        let val = v.evaluate(&[1.0, 2.0]);
        assert!((val - (2.0 + 12.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_identity() {
        let v = identity_lyapunov(2);
        let g = v.gradient(&[3.0, 4.0]);
        assert!((g[0] - 6.0).abs() < 1e-10);
        assert!((g[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_derivative_stable() {
        // System: dx/dt = -x (globally asymptotically stable)
        let v = identity_lyapunov(2);
        let x = vec![1.0, 2.0];
        let dxdt = vec![-1.0, -2.0];
        let dvdt = v.time_derivative(&x, &dxdt);
        assert!(dvdt < 0.0, "dV/dt should be negative for stable system");
    }

    #[test]
    fn test_time_derivative_unstable() {
        // System: dx/dt = x (unstable)
        let v = identity_lyapunov(2);
        let x = vec![1.0, 2.0];
        let dxdt = vec![1.0, 2.0];
        let dvdt = v.time_derivative(&x, &dxdt);
        assert!(dvdt > 0.0, "dV/dt should be positive for unstable system");
    }

    #[test]
    fn test_is_symmetric_true() {
        let v = QuadraticLyapunov::new(vec![
            vec![2.0, 1.0],
            vec![1.0, 3.0],
        ]);
        assert!(v.is_symmetric(1e-10));
    }

    #[test]
    fn test_is_symmetric_false() {
        let v = QuadraticLyapunov::new(vec![
            vec![2.0, 1.0],
            vec![2.0, 3.0],
        ]);
        assert!(!v.is_symmetric(1e-10));
    }

    #[test]
    fn test_positive_definite_identity() {
        let v = identity_lyapunov(3);
        assert!(v.is_positive_definite());
    }

    #[test]
    fn test_positive_definite_2x2() {
        let v = QuadraticLyapunov::new(vec![
            vec![4.0, 1.0],
            vec![1.0, 3.0],
        ]);
        assert!(v.is_positive_definite());
    }

    #[test]
    fn test_not_positive_definite_zero_eigenvalue() {
        let v = QuadraticLyapunov::new(vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ]);
        assert!(!v.is_positive_definite());
        assert!(v.is_positive_semidefinite());
    }

    #[test]
    fn test_not_positive_definite_negative_eigenvalue() {
        let v = QuadraticLyapunov::new(vec![
            vec![1.0, 0.0],
            vec![0.0, -1.0],
        ]);
        assert!(!v.is_positive_definite());
        assert!(!v.is_positive_semidefinite());
    }

    #[test]
    fn test_spectral_radius_bound() {
        let v = identity_lyapunov(2);
        let bound = v.spectral_radius_bound();
        assert!(bound >= 1.0);
    }

    #[test]
    fn test_zero_state_zero_value() {
        let v = identity_lyapunov(3);
        assert!(v.evaluate(&[0.0, 0.0, 0.0]).abs() < 1e-10);
    }

    #[test]
    fn test_v_positive_for_nonzero() {
        let v = identity_lyapunov(2);
        assert!(v.evaluate(&[1.0, 0.0]) > 0.0);
    }
}
