"""Differentiable rendering and physics simulation problems.

These problems test the ability to make non-differentiable simulations
differentiable through smooth relaxations, then compute gradients via
finite differences or analytical means.
"""

import numpy as np
from adbench.problem import Problem


# ---------------------------------------------------------------------------
# Helper: central finite differences
# ---------------------------------------------------------------------------

def _finite_diff(f, x, eps=1e-6):
    """Central finite difference for scalar function f at scalar x."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


# ===================================================================
# Problem 17: render_1d_occlusion
# ===================================================================

def _soft_color(p, c, x, tau):
    """Soft nearest-neighbor color at query point x given object positions p
    and colors c with temperature tau."""
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    # Softmax weights: w_i = exp(-|p_i - x| / tau) / sum_j exp(-|p_j - x| / tau)
    dists = np.abs(p - x)
    logits = -dists / tau
    logits -= np.max(logits)  # numerical stability
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return float(np.dot(weights, c))


def _render_1d_reference(p, x, tau=0.1):
    """Compute d(color_tau(x)) / d(p_k) for each k, via finite differences."""
    p = list(p)
    c = [0.2, 0.7, 0.9]
    grads = []
    eps = 1e-6
    for k in range(3):
        def f_k(pk, _k=k):
            pp = list(p)
            pp[_k] = pk
            return _soft_color(pp, c, x, tau)
        grad_k = _finite_diff(f_k, p[k], eps)
        grads.append(grad_k)
    return grads


def _render_1d_solve(p, x, tau=0.1):
    """Reference solver for render_1d_occlusion."""
    return _render_1d_reference(p, x, tau)


_render_1d_test_cases = []
for _p_list, _x in [
    ([0.0, 1.0, 3.0], 0.4),
    ([0.0, 1.0, 3.0], 0.5),
    ([0.0, 1.0, 3.0], 1.5),
    ([0.0, 1.0, 3.0], 0.0),
    ([-1.0, 0.0, 0.5], 0.2),
]:
    _expected = _render_1d_reference(_p_list, _x)
    _render_1d_test_cases.append({
        "inputs": {"p": _p_list, "x": _x, "tau": 0.1},
        "expected": _expected,
    })

render_1d_occlusion = Problem(
    id="render_1d_occlusion",
    category="diff_simulation",
    difficulty=3,
    description=(
        "1D differentiable rendering with occlusion: compute the gradient of a "
        "soft nearest-neighbor color function with respect to object positions."
    ),
    prompt="""\
Differentiable 1D Rendering with Occlusion
===========================================

Scene: 3 objects on a number line at positions p = [p1, p2, p3] with fixed
colors c = [0.2, 0.7, 0.9].

A "camera" queries position x. The visible color is the color of the nearest
object: color(x) = c[argmin_i |p_i - x|]. This is non-differentiable because
argmin is piecewise constant.

Differentiable relaxation using soft nearest neighbor:

    color_tau(x) = sum_i w_i * c_i

where w_i = softmax(-|p_i - x| / tau), i.e.:

    w_i = exp(-|p_i - x| / tau) / sum_j exp(-|p_j - x| / tau)

Task: Compute d(color_tau(x)) / d(p_k) for each k = 0, 1, 2 (the gradient
of the soft color with respect to each object position).

Return a list of 3 floats [d_color/d_p0, d_color/d_p1, d_color/d_p2].

You may use finite differences on the soft color function.

def solve(p, x, tau=0.1):
    \"\"\"
    Parameters
    ----------
    p : list of 3 floats — object positions [p0, p1, p2]
    x : float — camera query position
    tau : float — temperature parameter (default 0.1)

    Returns
    -------
    list of 3 floats — [d(color_tau)/d(p0), d(color_tau)/d(p1), d(color_tau)/d(p2)]
    \"\"\"
""",
    reference=_render_1d_solve,
    test_cases=_render_1d_test_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 18: collision_smooth
# ===================================================================

def _smooth_collision_sim(v1, k=1000.0, r_contact=0.05, dt=0.001, T=2.0):
    """Simulate two particles with smooth repulsive force using Verlet integration.
    Particle 1: pos=0, vel=v1. Particle 2: pos=1, vel=-0.5.
    Returns final position of particle 1."""
    x1 = 0.0
    x2 = 1.0
    v1_cur = float(v1)
    v2_cur = -0.5
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        # Compute force: repulsive when particles overlap within r_contact
        dx = x1 - x2  # signed displacement
        dist = abs(dx)
        overlap = r_contact - dist
        if overlap > 0:
            # Force magnitude: k * overlap^2
            force_mag = k * overlap * overlap
            # Direction: push particles apart
            if dx > 0:
                f1 = force_mag
                f2 = -force_mag
            elif dx < 0:
                f1 = -force_mag
                f2 = force_mag
            else:
                f1 = force_mag
                f2 = -force_mag
        else:
            f1 = 0.0
            f2 = 0.0
        # Verlet-style (velocity Verlet): update velocity, then position
        v1_cur += f1 * dt  # mass = 1
        v2_cur += f2 * dt
        x1 += v1_cur * dt
        x2 += v2_cur * dt
    return x1


def _collision_smooth_reference(v1):
    """Compute d(final_pos1)/d(v1) via finite differences."""
    eps = 1e-5
    return _finite_diff(_smooth_collision_sim, v1, eps)


def _collision_smooth_solve(v1):
    """Reference solver for collision_smooth."""
    return _collision_smooth_reference(v1)


_collision_test_cases = []
for _v1 in [0.3, 0.5, 0.8, 1.0, 1.5]:
    _expected = _collision_smooth_reference(_v1)
    _collision_test_cases.append({
        "inputs": {"v1": _v1},
        "expected": _expected,
    })

collision_smooth = Problem(
    id="collision_smooth",
    category="diff_simulation",
    difficulty=3,
    description=(
        "Differentiable 1D elastic collision: compute gradient of final position "
        "through a smooth repulsive-force simulation."
    ),
    prompt="""\
Differentiable 1D Elastic Collision
====================================

Two point particles on a line:
- Particle 1: initial position = 0, initial velocity = v1 (moving right)
- Particle 2: initial position = 1, initial velocity = -0.5 (moving left)

Both particles have mass = 1.

Instead of modeling a hard collision (which creates a non-differentiable kink),
use a smooth repulsive force:

    F = k * max(0, r_contact - |x1 - x2|)^2

with k = 1000, r_contact = 0.05. The force pushes the particles apart when
they are closer than r_contact.

Simulate using velocity Verlet integration with dt = 0.001 for T = 2.0 seconds.
At each step:
1. Compute force on each particle (equal and opposite).
2. Update velocities: v += F * dt  (mass = 1)
3. Update positions: x += v * dt

Task: Compute d(final_pos1) / d(v1) — the derivative of particle 1's final
position with respect to its initial velocity.

Return a single float.

You may use finite differences on the smooth simulation.

def solve(v1):
    \"\"\"
    Parameters
    ----------
    v1 : float — initial velocity of particle 1

    Returns
    -------
    float — d(final_position_of_particle_1) / d(v1)
    \"\"\"
""",
    reference=_collision_smooth_solve,
    test_cases=_collision_test_cases,
    atol=1e-3,
    rtol=5e-2,
)


# ===================================================================
# Problem 19: spring_contact
# ===================================================================

def _spring_contact_sim(v0, m=1.0, k_spring=10.0, L=1.0, h=2.0, g=9.8,
                         K_ground=10000.0, T=1.0, dt=0.001):
    """Simulate mass-spring system with ground contact using symplectic Euler.
    Spring fixed end at height h, mass starts at y=0.5 with upward velocity v0.
    Returns final y position of the mass."""
    y = 0.5
    v = float(v0)
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        # Spring force: F_spring = -k_spring * (stretch) in appropriate direction
        # stretch = distance_from_anchor - rest_length
        # distance from anchor at h to mass at y: h - y (assuming mass below anchor)
        # Actually: spring length = |h - y|, spring force direction towards anchor
        displacement = h - y  # vector from mass to anchor (1D)
        spring_length = abs(displacement)
        if spring_length > 0:
            # Spring force = k * (spring_length - L) * (direction toward anchor)
            # direction toward anchor = sign(displacement) = displacement / spring_length
            stretch = spring_length - L
            f_spring = k_spring * stretch * (displacement / spring_length)
        else:
            f_spring = 0.0

        # Gravity
        f_gravity = -m * g

        # Ground contact force: F_contact = K_ground * max(0, -y)^2 (upward)
        if y < 0:
            f_contact = K_ground * y * y  # y < 0 so y^2 = (-y)^2, force is upward (+)
        else:
            f_contact = 0.0

        # Total force
        f_total = f_spring + f_gravity + f_contact

        # Symplectic Euler: update velocity first, then position
        v += (f_total / m) * dt
        y += v * dt

    return y


def _spring_contact_reference(v0):
    """Compute d(final_y)/d(v0) via finite differences."""
    eps = 1e-5
    return _finite_diff(_spring_contact_sim, v0, eps)


def _spring_contact_solve(v0):
    """Reference solver for spring_contact."""
    return _spring_contact_reference(v0)


_spring_test_cases = []
for _v0 in [0.0, 2.0, 5.0, -1.0]:
    _expected = _spring_contact_reference(_v0)
    _spring_test_cases.append({
        "inputs": {"v0": _v0},
        "expected": _expected,
    })

spring_contact = Problem(
    id="spring_contact",
    category="diff_simulation",
    difficulty=3,
    description=(
        "Mass-spring system with ground contact: compute gradient of final height "
        "through a simulation with soft ground contact force."
    ),
    prompt="""\
Mass-Spring with Ground Contact
================================

A mass m = 1 is attached to a spring (stiffness k = 10, rest length L = 1).
The spring's other end is fixed at height h = 2.

Initial conditions: mass at y = 0.5, initial upward velocity = v0.

Forces:
- Gravity: F_gravity = -m * g, with g = 9.8
- Spring: The spring connects the fixed point at h = 2 to the mass at y.
  Spring length = |h - y|. Stretch = |h - y| - L.
  Spring force = k * stretch * sign(h - y)  (toward the anchor).
- Ground contact at y = 0:
  F_contact = K_ground * max(0, -y)^2  (upward, only when y < 0)
  with K_ground = 10000.

Simulate for T = 1.0 with dt = 0.001 using symplectic Euler:
1. Compute total force F
2. v += (F / m) * dt
3. y += v * dt

Task: Compute d(final_y) / d(v0) — the derivative of the mass's final
vertical position with respect to its initial upward velocity.

Return a single float.

You may use finite differences on the simulation.

def solve(v0):
    \"\"\"
    Parameters
    ----------
    v0 : float — initial upward velocity of the mass

    Returns
    -------
    float — d(final_y) / d(v0)
    \"\"\"
""",
    reference=_spring_contact_solve,
    test_cases=_spring_test_cases,
    atol=1e-3,
    rtol=5e-2,
)


# ===================================================================
# Problem 20: sdf_sphere_trace
# ===================================================================

def _sphere_trace(cx, cy=0.0, cz=5.0, r=1.0, N=50):
    """Sphere trace along ray from origin in +z direction.
    Scene: sphere of radius r centered at [cx, cy, cz].
    Returns converged distance t* (or 1e10 if miss)."""
    # Ray: origin = [0, 0, 0], direction = [0, 0, 1]
    # Point on ray at parameter t: [0, 0, t]
    t = 0.0
    for _ in range(N):
        # Current point on ray
        px, py, pz = 0.0, 0.0, t
        # SDF of sphere at center [cx, cy, cz] with radius r
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist_to_center = np.sqrt(dx * dx + dy * dy + dz * dz)
        sdf_val = dist_to_center - r
        if sdf_val < 1e-8:
            break
        t += sdf_val
        if t > 1e6:
            return 1e10  # ray missed
    return t


def _sdf_sphere_trace_reference(cx, cy=0.0):
    """Compute d(t*)/d(cx) via finite differences."""
    eps = 1e-6
    # Check if ray hits at all: ray along z-axis hits sphere at [cx, cy, 5]
    # if cx^2 + cy^2 < r^2 = 1
    t_center = _sphere_trace(cx, cy)
    if t_center > 1e8:
        return float('inf')
    def f(c):
        val = _sphere_trace(c, cy)
        if val > 1e8:
            return val
        return val
    return _finite_diff(f, cx, eps)


def _sdf_sphere_trace_solve(cx, cy=0.0):
    """Reference solver for sdf_sphere_trace."""
    return _sdf_sphere_trace_reference(cx, cy)


_sdf_test_cases = []
for _cx in [0.0, 0.3, 0.5, -0.2]:
    _expected = _sdf_sphere_trace_reference(_cx, 0.0)
    _sdf_test_cases.append({
        "inputs": {"cx": _cx, "cy": 0.0},
        "expected": _expected,
    })

sdf_sphere_trace = Problem(
    id="sdf_sphere_trace",
    category="diff_simulation",
    difficulty=3,
    description=(
        "Differentiable sphere tracing (ray marching): compute the gradient of "
        "the hit distance with respect to the sphere center position."
    ),
    prompt="""\
Differentiable Sphere Tracing (Ray Marching)
=============================================

Scene: a sphere of radius r = 1 centered at c = [cx, cy, 5] (note: center
is at z = 5, so the ray must travel forward to reach it).

Ray from origin [0, 0, 0] in direction d = [0, 0, 1] (along the z-axis).

Signed Distance Function (SDF):
    sdf(p) = ||p - c|| - r

where c = [cx, cy, 5].

Sphere tracing algorithm: start at t = 0, then iterate N = 50 times:
    p = origin + t * d = [0, 0, t]
    sdf_val = ||p - c|| - r
    if sdf_val < 1e-8: break
    t = t + sdf_val
    if t > 1e6: ray missed (return inf or very large number)

The converged t* is the hit distance.

Task: Compute d(t*) / d(cx) — the derivative of the hit distance with
respect to the sphere center's x-position.

If the ray misses (cx^2 + cy^2 > r^2), return float('inf').

Return a single float.

You may use finite differences on the sphere tracing result.

def solve(cx, cy=0.0):
    \"\"\"
    Parameters
    ----------
    cx : float — x-coordinate of sphere center
    cy : float — y-coordinate of sphere center (default 0.0)

    Returns
    -------
    float — d(t*) / d(cx), or inf if the ray misses the sphere
    \"\"\"
""",
    reference=_sdf_sphere_trace_solve,
    test_cases=_sdf_test_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# ALL problems in this module
# ===================================================================

ALL = [
    render_1d_occlusion,
    collision_smooth,
    spring_contact,
    sdf_sphere_trace,
]
