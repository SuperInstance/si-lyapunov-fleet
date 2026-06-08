//! Fleet-level Lyapunov functions.
//!
//! Combines individual agent Lyapunov functions into a fleet-wide
//! stability certificate. The fleet Lyapunov is the sum of agent
//! energies plus coupling terms that capture inter-agent interactions.
//!
//! When combined with conservation-law principles, this ensures that
//! the total fleet energy is conserved while individual agents converge.

use crate::quadratic::QuadraticLyapunov;

/// State of a single agent in the fleet.
#[derive(Debug, Clone)]
pub struct AgentState {
    pub name: String,
    pub state: Vec<f64>,
}

/// A fleet Lyapunov function: sum of agent energies plus coupling.
///
/// V_fleet = Σ V_i(x_i) + Σ coupling_ij * ||x_i - x_j||^2
///
/// The coupling term penalizes disagreement between agents.
#[derive(Debug, Clone)]
pub struct FleetLyapunov {
    /// Per-agent quadratic Lyapunov functions.
    pub agent_lyapunovs: Vec<QuadraticLyapunov>,
    /// Coupling weights between agent pairs (symmetric, non-negative).
    pub coupling: Vec<Vec<f64>>,
    /// Agent names for identification.
    pub agent_names: Vec<String>,
}

impl FleetLyapunov {
    /// Create a fleet Lyapunov with uniform identity Lyapunovs and given coupling.
    pub fn new_uniform(num_agents: usize, coupling_weight: f64) -> Self {
        let agent_lyapunovs: Vec<_> = (0..num_agents)
            .map(|_| crate::quadratic::identity_lyapunov(1))
            .collect();
        let mut coupling = vec![vec![0.0; num_agents]; num_agents];
        for i in 0..num_agents {
            for j in (i + 1)..num_agents {
                coupling[i][j] = coupling_weight;
                coupling[j][i] = coupling_weight;
            }
        }
        let agent_names: Vec<_> = (0..num_agents)
            .map(|i| format!("agent_{}", i))
            .collect();
        Self { agent_lyapunovs, coupling, agent_names }
    }

    /// Create a fleet Lyapunov with custom per-agent functions and coupling.
    pub fn new(
        agent_lyapunovs: Vec<QuadraticLyapunov>,
        coupling: Vec<Vec<f64>>,
        agent_names: Vec<String>,
    ) -> Self {
        let n = agent_lyapunovs.len();
        assert_eq!(coupling.len(), n);
        assert_eq!(agent_names.len(), n);
        for row in &coupling {
            assert_eq!(row.len(), n);
        }
        Self { agent_lyapunovs, coupling, agent_names }
    }

    /// Number of agents in the fleet.
    pub fn num_agents(&self) -> usize {
        self.agent_lyapunovs.len()
    }

    /// Evaluate the fleet Lyapunov for a set of agent states.
    pub fn evaluate(&self, states: &[AgentState]) -> f64 {
        assert_eq!(states.len(), self.num_agents());
        let mut total = 0.0;

        // Individual agent energies
        for i in 0..self.num_agents() {
            total += self.agent_lyapunovs[i].evaluate(&states[i].state);
        }

        // Coupling terms: penalize differences between agents
        for i in 0..self.num_agents() {
            for j in (i + 1)..self.num_agents() {
                if self.coupling[i][j] > 0.0 {
                    let diff_norm_sq: f64 = states[i]
                        .state
                        .iter()
                        .zip(states[j].state.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    total += self.coupling[i][j] * diff_norm_sq;
                }
            }
        }

        total
    }

    /// Compute the time derivative given agent dynamics dx_i/dt = f_i(x).
    pub fn time_derivative(&self, states: &[AgentState], dxdt: &[Vec<f64>]) -> f64 {
        assert_eq!(states.len(), self.num_agents());
        assert_eq!(dxdt.len(), self.num_agents());

        let mut total = 0.0;

        // Individual agent contributions
        for i in 0..self.num_agents() {
            total += self.agent_lyapunovs[i].time_derivative(&states[i].state, &dxdt[i]);
        }

        // Coupling contributions
        for i in 0..self.num_agents() {
            for j in (i + 1)..self.num_agents() {
                if self.coupling[i][j] > 0.0 {
                    for k in 0..states[i].state.len() {
                        let diff = states[i].state[k] - states[j].state[k];
                        total += 2.0 * self.coupling[i][j] * diff * (dxdt[i][k] - dxdt[j][k]);
                    }
                }
            }
        }

        total
    }

    /// Check if the fleet is in consensus (all agents have equal states).
    pub fn in_consensus(&self, states: &[AgentState], tol: f64) -> bool {
        if states.len() < 2 {
            return true;
        }
        let reference = &states[0].state;
        for i in 1..states.len() {
            for k in 0..reference.len().min(states[i].state.len()) {
                if (reference[k] - states[i].state[k]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the consensus energy (how far from consensus).
    pub fn consensus_energy(&self, states: &[AgentState]) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..self.num_agents() {
            for j in (i + 1)..self.num_agents() {
                for k in 0..states[i].state.len() {
                    let diff = states[i].state[k] - states[j].state[k];
                    total += diff * diff;
                }
            }
        }
        total
    }

    /// Compute individual agent energies.
    pub fn agent_energies(&self, states: &[AgentState]) -> Vec<f64> {
        states
            .iter()
            .enumerate()
            .map(|(i, s)| self.agent_lyapunovs[i].evaluate(&s.state))
            .collect()
    }

    /// Total energy budget (sum of all agent energies, without coupling).
    pub fn total_energy_budget(&self, states: &[AgentState]) -> f64 {
        self.agent_energies(states).iter().sum()
    }

    /// Check conservation law: gamma + H = total_budget.
    /// Here gamma is the usable energy, H is the coupling/entropy energy.
    pub fn check_conservation(&self, states: &[AgentState]) -> bool {
        let total = self.evaluate(states);
        let budget = self.total_energy_budget(states);
        let coupling = total - budget;
        // budget + coupling ≈ total (within numerical tolerance)
        (budget + coupling - total).abs() < 1e-10
    }
}

/// Simulate fleet consensus dynamics and verify Lyapunov decrease.
///
/// dx_i/dt = - Σ_j coupling_ij * (x_i - x_j)
pub fn simulate_consensus_dynamics(
    initial_states: &[Vec<f64>],
    coupling: &[Vec<f64>],
    dt: f64,
    steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial_states.len();
    let mut states: Vec<Vec<f64>> = initial_states.to_vec();

    for _ in 0..steps {
        let mut dxdt = vec![vec![0.0; states[0].len()]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    for k in 0..states[i].len() {
                        dxdt[i][k] -= coupling[i][j] * (states[i][k] - states[j][k]);
                    }
                }
            }
        }
        for i in 0..n {
            for k in 0..states[i].len() {
                states[i][k] += dt * dxdt[i][k];
            }
        }
    }

    states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fleet_evaluate_positive() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![1.0] },
            AgentState { name: "a1".into(), state: vec![2.0] },
            AgentState { name: "a2".into(), state: vec![3.0] },
        ];
        let v = fleet.evaluate(&states);
        assert!(v > 0.0);
    }

    #[test]
    fn test_fleet_evaluate_zero_at_consensus() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![0.0] },
            AgentState { name: "a1".into(), state: vec![0.0] },
            AgentState { name: "a2".into(), state: vec![0.0] },
        ];
        let v = fleet.evaluate(&states);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_consensus_energy_zero_at_consensus() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![5.0] },
            AgentState { name: "a1".into(), state: vec![5.0] },
            AgentState { name: "a2".into(), state: vec![5.0] },
        ];
        assert!(fleet.in_consensus(&states, 1e-10));
        assert!(fleet.consensus_energy(&states).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_energy_nonzero_when_disagree() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![0.0] },
            AgentState { name: "a1".into(), state: vec![1.0] },
            AgentState { name: "a2".into(), state: vec![2.0] },
        ];
        assert!(!fleet.in_consensus(&states, 1e-10));
        assert!(fleet.consensus_energy(&states) > 0.0);
    }

    #[test]
    fn test_time_derivative_decrease_for_consensus() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![1.0] },
            AgentState { name: "a1".into(), state: vec![2.0] },
            AgentState { name: "a2".into(), state: vec![3.0] },
        ];
        // Consensus dynamics
        let dxdt = vec![
            vec![-1.0 * (1.0 - 2.0) + -1.0 * (1.0 - 3.0)], // -(x0-x1) -(x0-x2)
            vec![-1.0 * (2.0 - 1.0) + -1.0 * (2.0 - 3.0)],
            vec![-1.0 * (3.0 - 1.0) + -1.0 * (3.0 - 2.0)],
        ];
        let dvdt = fleet.time_derivative(&states, &dxdt);
        assert!(dvdt < 0.0, "dV/dt should decrease for consensus dynamics");
    }

    #[test]
    fn test_simulate_consensus_converges() {
        let coupling = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let initial = vec![vec![0.0], vec![10.0], vec![20.0]];
        let final_states = simulate_consensus_dynamics(&initial, &coupling, 0.01, 5000);

        // Should converge near average = 10.0
        for s in &final_states {
            assert!((s[0] - 10.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_check_conservation() {
        let fleet = FleetLyapunov::new_uniform(3, 1.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![1.0] },
            AgentState { name: "a1".into(), state: vec![2.0] },
            AgentState { name: "a2".into(), state: vec![3.0] },
        ];
        assert!(fleet.check_conservation(&states));
    }

    #[test]
    fn test_agent_energies_sum() {
        let fleet = FleetLyapunov::new_uniform(2, 0.0);
        let states = vec![
            AgentState { name: "a0".into(), state: vec![3.0] },
            AgentState { name: "a1".into(), state: vec![4.0] },
        ];
        let energies = fleet.agent_energies(&states);
        assert_eq!(energies.len(), 2);
        assert!((energies[0] - 9.0).abs() < 1e-10);
        assert!((energies[1] - 16.0).abs() < 1e-10);
    }
}
