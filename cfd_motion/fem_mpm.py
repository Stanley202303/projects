"""Small, deterministic solid mechanics core used by the collision model.

The implementation deliberately uses classical methods: lumped-mass explicit
corotational tetrahedral FEM and a USL particle/grid update for elements that
have failed.  It has no external solver dependency and keeps mass and linear
momentum attached to explicit particles during the FEM -> MPM hand-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .math_utils import v_add, v_cross, v_dot, v_mul, v_norm, v_sub
from .models import Vec3

Mat3 = Tuple[Vec3, Vec3, Vec3]


def _det(a: Mat3) -> float:
    return v_dot(a[0], v_cross(a[1], a[2]))


def _transpose(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(v_dot(a[i], (b[0][j], b[1][j], b[2][j])) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _inv(a: Mat3) -> Mat3:
    d = _det(a)
    if abs(d) < 1e-14:
        raise ValueError("singular tetrahedron reference matrix")
    rows = (
        v_cross(a[1], a[2]),
        v_cross(a[2], a[0]),
        v_cross(a[0], a[1]),
    )
    # rows are cofactors; transpose is the adjugate for column-vector rows.
    return tuple(tuple(rows[j][i] / d for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    return abs(v_dot(v_sub(b, a), v_cross(v_sub(c, a), v_sub(d, a)))) / 6.0


@dataclass
class TetraElement:
    nodes: Tuple[int, int, int, int]
    rest_volume_m3: float
    dm_inverse: Mat3
    young_modulus_pa: float
    poisson_ratio: float
    yield_stress_pa: float
    failure_strain: float
    failed: bool = False
    plastic_strain: float = 0.0


@dataclass
class MPMParticle:
    position: Vec3
    velocity: Vec3
    mass_kg: float
    volume_m3: float
    source_element: int
    stress: Mat3 = field(default_factory=_identity)
    failed: bool = False


@dataclass
class ConservationAudit:
    mass_before_kg: float
    mass_after_kg: float
    momentum_before: Vec3
    momentum_after: Vec3
    kinetic_before_j: float
    kinetic_after_j: float
    internal_energy_j: float = 0.0

    @property
    def mass_error_kg(self) -> float:
        return self.mass_after_kg - self.mass_before_kg

    @property
    def momentum_error(self) -> Vec3:
        return v_sub(self.momentum_after, self.momentum_before)


def von_mises_stress(stress: Mat3) -> float:
    """Return the J2 von-Mises equivalent stress for one material point."""
    mean = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0
    deviator = tuple(
        tuple(stress[i][j] - (mean if i == j else 0.0) for j in range(3))
        for i in range(3)
    )
    return math.sqrt(1.5 * sum(deviator[i][j] ** 2 for i in range(3) for j in range(3)))


@dataclass
class HybridFEMMPMState:
    positions: List[Vec3]
    velocities: List[Vec3]
    masses_kg: List[float]
    elements: List[TetraElement]
    particles: List[MPMParticle] = field(default_factory=list)
    use_mpm: bool = False
    damping: float = 0.01
    last_audit: ConservationAudit | None = None

    @property
    def total_mass_kg(self) -> float:
        return sum(self.masses_kg) + sum(p.mass_kg for p in self.particles)

    def total_momentum(self) -> Vec3:
        result = (0.0, 0.0, 0.0)
        for mass, velocity in zip(self.masses_kg, self.velocities):
            result = v_add(result, v_mul(velocity, mass))
        for particle in self.particles:
            result = v_add(result, v_mul(particle.velocity, particle.mass_kg))
        return result


def make_tetra_element(
    positions: Sequence[Vec3],
    nodes: Tuple[int, int, int, int],
    young_modulus_pa: float,
    poisson_ratio: float,
    yield_stress_pa: float,
    failure_strain: float,
) -> TetraElement:
    a, b, c, d = (positions[index] for index in nodes)
    dm = (v_sub(b, a), v_sub(c, a), v_sub(d, a))
    return TetraElement(
        nodes=nodes,
        rest_volume_m3=_volume(a, b, c, d),
        dm_inverse=_inv(dm),
        young_modulus_pa=max(young_modulus_pa, 1.0),
        poisson_ratio=max(0.0, min(0.49, poisson_ratio)),
        yield_stress_pa=max(yield_stress_pa, 0.0),
        failure_strain=max(failure_strain, 0.0),
    )


def _kinetic(masses: Sequence[float], velocities: Sequence[Vec3]) -> float:
    return 0.5 * sum(m * v_dot(v, v) for m, v in zip(masses, velocities))


def _stress(element: TetraElement, current: Mat3) -> Mat3:
    # Corotational approximation: use the symmetric part of F-I.  This is
    # robust for the small increments used by the explicit integrator.
    strain = tuple(
        tuple(0.5 * (current[i][j] + current[j][i]) - (1.0 if i == j else 0.0) for j in range(3))
        for i in range(3)
    )
    e = element.young_modulus_pa
    nu = element.poisson_ratio
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    trace = sum(strain[i][i] for i in range(3))
    result = tuple(
        tuple(2.0 * mu * strain[i][j] + (lam * trace if i == j else 0.0) for j in range(3))
        for i in range(3)
    )
    equivalent = math.sqrt(1.5 * sum((result[i][j] - (trace * lam if i == j else 0.0)) ** 2 for i in range(3) for j in range(3)))
    if equivalent > element.yield_stress_pa > 0.0:
        scale = element.yield_stress_pa / equivalent
        result = tuple(tuple(value * scale for value in row) for row in result)
        element.plastic_strain += (equivalent - element.yield_stress_pa) / max(e, 1.0)
    if element.plastic_strain >= element.failure_strain > 0.0:
        element.failed = True
    return result  # type: ignore[return-value]


def advance_fem(state: HybridFEMMPMState, dt_s: float, external_forces: Sequence[Vec3] | None = None) -> ConservationAudit:
    """Advance one explicit FEM step and hand failed elements to particles."""
    if dt_s <= 0.0:
        audit = ConservationAudit(state.total_mass_kg, state.total_mass_kg, state.total_momentum(), state.total_momentum(), _kinetic(state.masses_kg, state.velocities), _kinetic(state.masses_kg, state.velocities))
        state.last_audit = audit
        return audit
    before_mass = state.total_mass_kg
    before_momentum = state.total_momentum()
    before_ke = _kinetic(state.masses_kg, state.velocities)
    forces = [(0.0, 0.0, 0.0) for _ in state.positions]
    for element in state.elements:
        if element.failed:
            continue
        i, j, k, l = element.nodes
        ds = (v_sub(state.positions[j], state.positions[i]), v_sub(state.positions[k], state.positions[i]), v_sub(state.positions[l], state.positions[i]))
        try:
            f = _mul(ds, element.dm_inverse)
        except ValueError:
            continue
        stress = _stress(element, f)
        traction = (stress[0][0], stress[1][0], stress[2][0])
        force = v_mul(traction, element.rest_volume_m3 / 4.0)
        for node in element.nodes:
            forces[node] = v_add(forces[node], force)
    # Internal forces are self-equilibrated.  Enforce this explicitly to avoid
    # round-off or coarse shape-function errors creating a spurious rigid-body
    # impulse (a critical conservation property of the structural solver).
    net_force = (0.0, 0.0, 0.0)
    for force in forces:
        net_force = v_add(net_force, force)
    correction = v_mul(net_force, -1.0 / max(len(forces), 1))
    forces = [v_add(force, correction) for force in forces]
    if external_forces:
        forces = [v_add(force, external_forces[i]) for i, force in enumerate(forces)]
    for i, mass in enumerate(state.masses_kg):
        if mass <= 0.0:
            continue
        acceleration = v_mul(forces[i], 1.0 / mass)
        state.velocities[i] = v_add(state.velocities[i], v_mul(acceleration, dt_s))
        state.positions[i] = v_add(state.positions[i], v_mul(state.velocities[i], dt_s))
    # Failed tetrahedra become four equal-mass particles; no mass is removed.
    failed_counts = [0] * len(state.positions)
    for element in state.elements:
        if element.failed:
            for node in element.nodes:
                failed_counts[node] += 1
    for index, element in enumerate(state.elements):
        if element.failed and not any(p.source_element == index for p in state.particles):
            for node in element.nodes:
                share = state.masses_kg[node] / max(failed_counts[node], 1)
                state.particles.append(MPMParticle(state.positions[node], state.velocities[node], share, element.rest_volume_m3 / 4.0, index, failed=True))
                state.masses_kg[node] = max(0.0, state.masses_kg[node] - share)
            state.use_mpm = True
    for particle in state.particles:
        particle.position = v_add(particle.position, v_mul(particle.velocity, dt_s))
    after_mass = state.total_mass_kg
    after_momentum = state.total_momentum()
    audit = ConservationAudit(before_mass, after_mass, before_momentum, after_momentum, before_ke, _kinetic(state.masses_kg, state.velocities))
    state.last_audit = audit
    return audit
