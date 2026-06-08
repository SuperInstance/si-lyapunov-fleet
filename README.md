# si-lyapunov-fleet

**Lyapunov stability theory for fleet convergence.**

This crate provides a complete mathematical toolkit for proving that multi-agent fleets converge, stay safe, and remain stable under distributed dynamics. Built on classical control theory but designed for modern agent ecosystems, it unifies quadratic Lyapunov analysis, barrier functions for constrained fleets, fleet-wide energy certificates, and LaSalle's invariance principle into a single pure-Rust library.

## Why This Matters

A fleet of agents is only as reliable as its worst-case behavior. Without stability guarantees, agents can drift, oscillate, collide, or diverge — catastrophically. Lyapunov theory gives us the language to *prove* that won't happen:

- **Quadratic Lyapunov functions** certify that a single agent's state converges to equilibrium
- **Barrier Lyapunov functions** guarantee that agents never violate safety constraints (position limits, budget boundaries, communication ranges)
- **Fleet Lyapunov functions** prove that an entire fleet converges to consensus while respecting conservation laws
- **LaSalle's invariance principle** upgrades "Lyapunov stable" to "asymptotically stable" — convergence, not just boundedness

For an AGI ecosystem, these aren't academic exercises. They're the mathematical foundation that lets you deploy a thousand agents and sleep soundly.

---

## Quick Start

```toml
[dependencies]
si-lyapunov-fleet = { git = "https://github.com/SuperInstance/si-lyapunov-fleet" }
```

```rust
use si_lyapunov_fleet::quadratic::{QuadraticLyapunov, identity_lyapunov};
use si_lyapunov_fleet::lasalle::is_asymptotically_stable;

// Define a stable linear system: dx/dt = A x
let a = vec![
    vec![-2.0, 1.0],
    vec![0.0, -3.0],
];

// Choose a Lyapunov function V(x) = ||x||^2
let v = identity_lyapunov(2);

// Verify asymptotic stability using LaSalle's principle
assert!(is_asymptotically_stable(&a, &v));

// Evaluate V at a state
let x = vec![1.0, -2.0];
println!("V(x) = {:.2}", v.evaluate(&x));

// Compute dV/dt = ∇V · dx/dt
let dxdt = vec![-2.0 * x[0] + x[1], -3.0 * x[1]];
let dvdt = v.time_derivative(&x, &dxdt);
println!("dV/dt = {:.2} (negative = stable)", dvdt);
```

---

## Architecture

| Module | Purpose |
|---|---|
| `quadratic` | Quadratic Lyapunov V(x) = x^T P x with positive-definiteness tests |
| `barrier` | Barrier Lyapunov for constrained states with safety margins |
| `fleet` | Fleet-wide Lyapunov with consensus coupling and conservation checks |
| `lasalle` | LaSalle invariance principle, Hurwitz tests, region of attraction |

---

## API Tour

### Quadratic Lyapunov (`quadratic`)

The workhorse of stability analysis. Define a positive-definite matrix P and certify stability via V(x) = x^T P x.

```rust
use si_lyapunov_fleet::quadratic::{QuadraticLyapunov, diagonal_lyapunov, identity_lyapunov};

// Identity Lyapunov: V(x) = ||x||^2
let v = identity_lyapunov(3);

// Diagonal Lyapunov: V(x) = Σ p_i x_i^2
let v_diag = diagonal_lyapunov(&[2.0, 3.0, 5.0]);

// Custom symmetric matrix
let v_custom = QuadraticLyapunov::new(vec![
    vec![4.0, 1.0],
    vec![1.0, 3.0],
]);

// Positive-definiteness via Sylvester's criterion
assert!(v_custom.is_positive_definite());

// Evaluate, gradient, time derivative
let val = v.evaluate(&[1.0, 2.0]);
let grad = v.gradient(&[1.0, 2.0]);
let dvdt = v.time_derivative(&[1.0, 2.0], &[-1.0, -2.0]);
```

### Barrier Lyapunov (`barrier`)

For agents with hard constraints — e.g., "never spend more than budget B" or "stay within region R." The barrier term grows to infinity at the boundary, making constraint violation physically impossible under the dynamics.

```rust
use si_lyapunov_fleet::barrier::{BarrierLyapunov, StateConstraint, symmetric_barrier_lyapunov};

// Agent state must stay within |x_i| < 2.0
let bl = symmetric_barrier_lyapunov(2, &[2.0, 2.0], 0.5);

// V(x) is finite inside the safe region
let v_inside = bl.evaluate(&[1.0, 0.5]);
assert!(v_inside.is_finite());

// V(x) diverges at the boundary
let v_boundary = bl.evaluate(&[2.0, 0.5]);
assert!(v_boundary.is_infinite());

// Safe step size before hitting any constraint
let dt_safe = bl.safe_step_size(&[0.0, 0.0], &[1.0, 0.0]);
assert!((dt_safe - 2.0).abs() < 1e-10);
```

### Fleet Lyapunov (`fleet`)

The fleet Lyapunov is the sum of individual agent energies plus coupling terms that penalize disagreement. When dV/dt < 0, the entire fleet converges to consensus.

```rust
use si_lyapunov_fleet::fleet::{FleetLyapunov, AgentState, simulate_consensus_dynamics};

let fleet = FleetLyapunov::new_uniform(3, 1.0);
let states = vec![
    AgentState { name: "alpha".into(), state: vec![0.0] },
    AgentState { name: "beta".into(),  state: vec![10.0] },
    AgentState { name: "gamma".into(), state: vec![20.0] },
];

let v = fleet.evaluate(&states);
assert!(v > 0.0); // Non-zero disagreement costs energy

// Simulate consensus dynamics
let coupling = vec![
    vec![0.0, 1.0, 1.0],
    vec![1.0, 0.0, 1.0],
    vec![1.0, 1.0, 0.0],
];
let final_states = simulate_consensus_dynamics(
    &vec![vec![0.0], vec![10.0], vec![20.0]],
    &coupling, 0.01, 5000,
);
// All agents converge to the average: 10.0
```

### LaSalle Invariance (`lasalle`)

LaSalle's principle turns a negative-semidefinite dV/dt into a convergence proof. It finds the largest invariant set where dV/dt = 0 and shows that trajectories must converge to it.

```rust
use si_lyapunov_fleet::lasalle::{is_asymptotically_stable, lasalle_linear, is_hurwitz};
use si_lyapunov_fleet::quadratic::identity_lyapunov;

// A Hurwitz-stable matrix
let a = vec![
    vec![-3.0, 1.0, 0.0],
    vec![0.0, -2.0, 1.0],
    vec![0.0, 0.0, -1.0],
];
let p = identity_lyapunov(3);

// Quick Hurwitz check
assert!(is_hurwitz(&a));

// Full LaSalle analysis
let result = lasalle_linear(&a, &p, &vec![-1.0, 0.0, 1.0]);
assert!(result.converges_to_origin);

// Asymptotic stability via Lyapunov equation
assert!(is_asymptotically_stable(&a, &p));
```

---

## Design Patterns

### Energy-First Thinking

Every module treats stability as an energy story. The fleet Lyapunov is literally a sum of energies. The barrier Lyapunov adds potential walls. This makes the mathematics intuitive and the code transparent.

### Pure Functions, Testable Guarantees

All Lyapunov evaluations are pure functions of state. There is no hidden mutable state, no side effects, no stochastic approximation. What you test is what you prove.

### Conservation-Aware Fleet Design

The `FleetLyapunov` includes a `check_conservation()` method that verifies the total energy budget equals the sum of individual energies plus coupling terms. This mirrors the `conservation-law-rs` philosophy at the fleet level.

---

## Ecosystem Integration

| Repository | Integration Point |
|---|---|
| `conservation-law-rs` | Energy budget conservation across fleet agents |
| `si-cli` | Capability scanning and fleet registry |
| `si-fleet-api` | Fleet budget auditing and conservation checks |
| `witness-topology-rs` | Topological shape of fleet consensus manifolds |
| `optimal-transport-agents-rs` | Wasserstein distance between fleet distributions |
| `ecosystem-dashboard` | Live monitoring of fleet Lyapunov values |

---

## Performance

- Quadratic evaluation: O(n²) for n-dimensional state
- Positive-definiteness check: O(n³) via Gaussian elimination
- Fleet evaluation: O(n² × d²) for n agents with d-dimensional states
- Consensus simulation: O(steps × n² × d)
- All pure Rust, zero allocations in hot paths when pre-allocated

---

## Ideas for Improvement

- **Numerical eigenvalue computation** — For large systems, add iterative eigenvalue solvers
- **SOS (Sum-of-Squares) programming** — Automated Lyapunov function synthesis for polynomial systems
- **Neural Lyapunov functions** — Learned V(x) for nonlinear systems with verification
- **Real-time fleet monitoring** — Streaming computation of V and dV/dt from live agent telemetry
- **GPU batch evaluation** — Evaluate Lyapunov functions for thousands of agents in parallel
- **Integration with `si-fleet-api`** — Automated stability certification before deployment

---

## License

MIT

---

## Theory Background

### Lyapunov's Direct Method

Lyapunov's second method (the direct method) allows us to prove stability without solving the differential equation. The key insight: if we can find a scalar function V(x) that acts like an "energy" — positive everywhere except at equilibrium, and decreasing along trajectories — then the equilibrium is stable.

Formally, for a system dx/dt = f(x) with equilibrium at x = 0:

1. **Lyapunov stable** if V(0) = 0, V(x) > 0 for x ≠ 0, and dV/dt ≤ 0
2. **Asymptotically stable** if additionally dV/dt < 0 for x ≠ 0
3. **Globally asymptotically stable** if the above holds for all x ∈ ℝⁿ

This crate automates the mechanical parts: evaluating V, computing dV/dt, and checking the inequalities.

### Barrier Functions

Standard Lyapunov theory assumes the state space is all of ℝⁿ. Real agents have constraints: budgets, position limits, communication ranges. Barrier Lyapunov functions encode these constraints directly into V(x):

V_barrier(x) = V_quadratic(x) + κ · Σ log(1 / (cᵢ² - xᵢ²))

As xᵢ approaches its bound cᵢ, the barrier term diverges to infinity. Since V itself would then diverge, and dV/dt < 0 ensures V decreases, the trajectory can never reach the boundary. The constraint is enforced organically by the dynamics.

### Fleet Consensus

For a fleet of n agents with states x₁, ..., xₙ, the fleet Lyapunov is:

V_fleet = Σ Vᵢ(xᵢ) + Σᵢⱼ κᵢⱼ ||xᵢ - xⱼ||²

The first term is the sum of individual agent energies. The second term is a coupling energy that penalizes disagreement. When agents run consensus dynamics:

dxᵢ/dt = - Σⱼ κᵢⱼ (xᵢ - xⱼ)

the time derivative of the coupling term is negative definite (unless all xᵢ are equal), driving the fleet to consensus. The individual terms Vᵢ(xᵢ) ensure each agent's internal state is also stable.

### LaSalle's Invariance Principle

LaSalle's principle addresses a subtlety: what if dV/dt = 0 at some points other than the origin? Then Lyapunov's theorem only gives stability, not convergence. LaSalle says: trajectories converge to the largest invariant subset of {x : dV/dt = 0}. If that subset is just the origin, we have asymptotic stability even when dV/dt is only negative semidefinite.

For linear systems dx/dt = A x with Lyapunov V(x) = x^T P x, the condition dV/dt = 0 becomes x^T (A^T P + P A) x = 0. If A^T P + P A is negative definite, the only solution is x = 0, and LaSalle guarantees global asymptotic stability.

---

## Advanced Examples

### Proving Global Asymptotic Stability

```rust
use si_lyapunov_fleet::quadratic::identity_lyapunov;
use si_lyapunov_fleet::lasalle::{is_asymptotically_stable, is_hurwitz};

// A 3D stable system
let a = vec![
    vec![-2.0, 1.0, 0.0],
    vec![0.5, -3.0, 0.5],
    vec![0.0, 1.0, -2.0],
];

// First check: is A Hurwitz?
assert!(is_hurwitz(&a));

// Second check: does there exist a Lyapunov function?
let p = identity_lyapunov(3);
assert!(is_asymptotically_stable(&a, &p));

// If both pass, the origin is globally asymptotically stable
println!("System is GAS: all trajectories converge to origin from any initial state");
```

### Constrained Fleet with Barrier Functions

```rust
use si_lyapunov_fleet::barrier::{symmetric_barrier_lyapunov, StateConstraint};
use si_lyapunov_fleet::quadratic::identity_lyapunov;

// Two agents with position constraints |x_i| < 5.0
let bl = symmetric_barrier_lyapunov(2, &[5.0, 5.0], 1.0);

// Current states
let x = vec![3.0, -2.0];
assert!(bl.constraints_satisfied(&x));

// Dynamics moving toward boundary
let dxdt = vec![1.0, -1.0];
let dt_safe = bl.safe_step_size(&x, &dxdt);
println!("Safe to advance for {:.2} time units", dt_safe);

// The barrier ensures we never exceed |x_i| = 5.0
let v = bl.evaluate(&x);
let dvdt = bl.time_derivative(&x, &dxdt);
println!("V = {:.2}, dV/dt = {:.2}", v, dvdt);
```

### Estimating Region of Attraction

```rust
use si_lyapunov_fleet::quadratic::identity_lyapunov;
use si_lyapunov_fleet::lasalle::estimate_region_of_attraction;
use si_lyapunov_fleet::uniform_grid;

let v = identity_lyapunov(2);

// A nonlinear system: dx/dt = -x + x³ (locally stable near origin)
let dynamics = |x: &[f64]| {
    vec![-x[0] + x[0].powi(3), -x[1] + x[1].powi(3)]
};

// Sample on a grid
let samples = uniform_grid(2, 1.5, 6);
let roa = estimate_region_of_attraction(&v, dynamics, &samples);
println!("Estimated region of attraction: V(x) < {:.4}", roa);
```

---

## Testing Philosophy

This crate contains 52 tests covering:

- **Positive definiteness**: Sylvester's criterion on identity, diagonal, and custom matrices
- **Stability certificates**: Stable vs. unstable linear systems, Hurwitz checks
- **Barrier constraints**: Inside/outside boundary behavior, safe step sizes, gradient correctness
- **Fleet consensus**: Convergence simulation, energy conservation, consensus detection
- **LaSalle principle**: Invariant set computation for stable and marginally stable systems
- **Edge cases**: Empty states, zero dimensions, boundary values, numerical tolerances

Every theorem has a test. Every test has a mathematical justification.

---

## Contributing

This crate is part of the SuperInstance ecosystem. Contributions should:

1. Maintain the pure-function, zero-side-effect design
2. Include tests for all new stability conditions
3. Follow the energy-first conceptual model
4. Integrate with `conservation-law-rs` budget principles where applicable

---

## References

- Khalil, H.K. *Nonlinear Systems*, 3rd Ed. Prentice Hall, 2002.
- Slotine, J.J.E. & Li, W. *Applied Nonlinear Control*. Prentice Hall, 1991.
- Tee, K.P., Ge, S.S., & Tay, E.H. "Barrier Lyapunov Functions for the Control of Output-Constrained Nonlinear Systems." *Automatica*, 2009.
- Olfati-Saber, R. & Murray, R.M. "Consensus Problems in Networks of Agents With Switching Topology and Time-Delays." *IEEE Trans. Automatic Control*, 2004.

