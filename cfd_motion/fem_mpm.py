"""Classical explicit solid FEM with conservative FEM-to-MPM failure transfer.

This module intentionally stays small and dependency-free.  Intact material is
advanced with lumped-mass, corotational constant-strain tetrahedra.  Failed
tetrahedra transfer their own nodal mass contributions to material points and
are subsequently advanced on a trilinear USL MPM grid.  The transfer is exact
for mass and linear momentum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .math_utils import v_add, v_cross, v_dot, v_mul, v_norm, v_sub
from .models import Vec3

Mat3 = Tuple[Vec3, Vec3, Vec3]
GridIndex = Tuple[int, int, int]


def mat_identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def mat_zero() -> Mat3:
    return ((0.0, 0.0, 0.0),) * 3


def mat_add(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_sub(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(a[i][j] - b[i][j] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_scale(a: Mat3, scale: float) -> Mat3:
    return tuple(
        tuple(scale * a[i][j] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_transpose(a: Mat3) -> Mat3:
    return tuple(
        tuple(a[j][i] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def mat_vec(a: Mat3, vector: Vec3) -> Vec3:
    return tuple(sum(a[i][j] * vector[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat_trace(a: Mat3) -> float:
    return a[0][0] + a[1][1] + a[2][2]


def mat_det(a: Mat3) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def mat_inverse(a: Mat3) -> Mat3:
    determinant = mat_det(a)
    if abs(determinant) <= 1e-18:
        raise ValueError("singular tetrahedron reference matrix")
    inverse = (
        (
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
        ),
        (
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
        ),
        (
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ),
    )
    return mat_scale(inverse, 1.0 / determinant)


def matrix_from_columns(a: Vec3, b: Vec3, c: Vec3) -> Mat3:
    return (
        (a[0], b[0], c[0]),
        (a[1], b[1], c[1]),
        (a[2], b[2], c[2]),
    )


def polar_rotation(deformation_gradient: Mat3) -> Mat3:
    """Compute the polar rotation with classical Newton iteration."""
    rotation = deformation_gradient
    if mat_det(rotation) <= 1e-12:
        return mat_identity()
    for _iteration in range(8):
        try:
            inverse_transpose = mat_transpose(mat_inverse(rotation))
        except ValueError:
            return mat_identity()
        updated = mat_scale(mat_add(rotation, inverse_transpose), 0.5)
        difference = max(
            abs(updated[i][j] - rotation[i][j])
            for i in range(3)
            for j in range(3)
        )
        rotation = updated
        if difference < 1e-10:
            break
    return rotation


def tetra_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    return abs(v_dot(v_sub(b, a), v_cross(v_sub(c, a), v_sub(d, a)))) / 6.0


def von_mises_stress(stress: Mat3) -> float:
    mean = mat_trace(stress) / 3.0
    deviator = tuple(
        tuple(stress[i][j] - (mean if i == j else 0.0) for j in range(3))
        for i in range(3)
    )
    return math.sqrt(
        1.5
        * sum(deviator[i][j] * deviator[i][j] for i in range(3) for j in range(3))
    )


@dataclass
class TetraElement:
    nodes: Tuple[int, int, int, int]
    rest_volume_m3: float
    dm_inverse: Mat3
    shape_gradients: Tuple[Vec3, Vec3, Vec3, Vec3]
    young_modulus_pa: float
    poisson_ratio: float
    yield_stress_pa: float
    failure_strain: float
    mass_kg: float = 0.0
    failed: bool = False
    transferred: bool = False
    equivalent_plastic_strain: float = 0.0
    strain_energy_j: float = 0.0
    stress: Mat3 = field(default_factory=mat_zero)


@dataclass
class MPMParticle:
    position: Vec3
    velocity: Vec3
    mass_kg: float
    volume_m3: float
    source_element: int
    young_modulus_pa: float
    poisson_ratio: float
    yield_stress_pa: float
    source_node: int = -1
    deformation_gradient: Mat3 = field(default_factory=mat_identity)
    stress: Mat3 = field(default_factory=mat_zero)
    damage: float = 0.0


@dataclass
class ConservationAudit:
    mass_before_kg: float
    mass_after_kg: float
    momentum_before: Vec3
    momentum_after: Vec3
    angular_momentum_before: Vec3
    angular_momentum_after: Vec3
    kinetic_before_j: float
    kinetic_after_j: float
    strain_energy_j: float = 0.0
    plastic_dissipation_j: float = 0.0
    external_work_j: float = 0.0
    momentum_projection_ns: float = 0.0
    angular_momentum_projection_nms: float = 0.0

    @property
    def mass_error_kg(self) -> float:
        return self.mass_after_kg - self.mass_before_kg

    @property
    def momentum_error(self) -> Vec3:
        return v_sub(self.momentum_after, self.momentum_before)

    @property
    def angular_momentum_error(self) -> Vec3:
        return v_sub(self.angular_momentum_after, self.angular_momentum_before)


@dataclass
class HybridFEMMPMState:
    positions: List[Vec3]
    velocities: List[Vec3]
    masses_kg: List[float]
    elements: List[TetraElement]
    reference_positions: List[Vec3] = field(default_factory=list)
    fixed_nodes: Set[int] = field(default_factory=set)
    particles: List[MPMParticle] = field(default_factory=list)
    surface_triangle_nodes: List[Tuple[int, int, int]] = field(default_factory=list)
    surface_element_indices: List[int] = field(default_factory=list)
    cfl: float = 0.35
    max_substeps: int = 512
    mpm_cell_size_m: float = 0.0
    pic_fraction: float = 0.05
    last_audit: Optional[ConservationAudit] = None

    def __post_init__(self) -> None:
        if not self.reference_positions:
            self.reference_positions = list(self.positions)

    @property
    def use_mpm(self) -> bool:
        return bool(self.particles)

    @property
    def total_mass_kg(self) -> float:
        return sum(self.masses_kg) + sum(particle.mass_kg for particle in self.particles)

    def center_of_mass(self) -> Vec3:
        total_mass = self.total_mass_kg
        if total_mass <= 1e-18:
            return (0.0, 0.0, 0.0)
        weighted = (0.0, 0.0, 0.0)
        for mass, position in zip(self.masses_kg, self.positions):
            weighted = v_add(weighted, v_mul(position, mass))
        for particle in self.particles:
            weighted = v_add(weighted, v_mul(particle.position, particle.mass_kg))
        return v_mul(weighted, 1.0 / total_mass)

    def total_momentum(self) -> Vec3:
        momentum = (0.0, 0.0, 0.0)
        for mass, velocity in zip(self.masses_kg, self.velocities):
            momentum = v_add(momentum, v_mul(velocity, mass))
        for particle in self.particles:
            momentum = v_add(momentum, v_mul(particle.velocity, particle.mass_kg))
        return momentum

    def total_angular_momentum(self) -> Vec3:
        center = self.center_of_mass()
        angular = (0.0, 0.0, 0.0)
        for mass, position, velocity in zip(
            self.masses_kg, self.positions, self.velocities
        ):
            angular = v_add(
                angular,
                v_mul(v_cross(v_sub(position, center), velocity), mass),
            )
        for particle in self.particles:
            angular = v_add(
                angular,
                v_mul(
                    v_cross(v_sub(particle.position, center), particle.velocity),
                    particle.mass_kg,
                ),
            )
        return angular


def make_tetra_element(
    positions: Sequence[Vec3],
    nodes: Tuple[int, int, int, int],
    young_modulus_pa: float,
    poisson_ratio: float,
    yield_stress_pa: float,
    failure_strain: float,
    mass_kg: float = 0.0,
) -> TetraElement:
    x0, x1, x2, x3 = (positions[index] for index in nodes)
    dm = matrix_from_columns(v_sub(x1, x0), v_sub(x2, x0), v_sub(x3, x0))
    dm_inverse = mat_inverse(dm)
    gradient_1 = dm_inverse[0]
    gradient_2 = dm_inverse[1]
    gradient_3 = dm_inverse[2]
    gradient_0 = v_mul(v_add(v_add(gradient_1, gradient_2), gradient_3), -1.0)
    return TetraElement(
        nodes=nodes,
        rest_volume_m3=tetra_volume(x0, x1, x2, x3),
        dm_inverse=dm_inverse,
        shape_gradients=(gradient_0, gradient_1, gradient_2, gradient_3),
        young_modulus_pa=max(young_modulus_pa, 1.0),
        poisson_ratio=max(0.0, min(0.49, poisson_ratio)),
        yield_stress_pa=max(yield_stress_pa, 0.0),
        failure_strain=max(failure_strain, 0.0),
        mass_kg=max(mass_kg, 0.0),
    )


def _kinetic_energy(state: HybridFEMMPMState) -> float:
    energy = 0.5 * sum(
        mass * v_dot(velocity, velocity)
        for mass, velocity in zip(state.masses_kg, state.velocities)
    )
    energy += 0.5 * sum(
        particle.mass_kg * v_dot(particle.velocity, particle.velocity)
        for particle in state.particles
    )
    return energy


def _elastic_stress(
    strain: Mat3,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> Mat3:
    lame_lambda = (
        young_modulus_pa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    return tuple(
        tuple(
            2.0 * shear * strain[i][j]
            + (lame_lambda * mat_trace(strain) if i == j else 0.0)
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def _limit_j2_stress(
    stress: Mat3,
    yield_stress_pa: float,
) -> Tuple[Mat3, float]:
    equivalent = von_mises_stress(stress)
    if yield_stress_pa <= 0.0 or equivalent <= yield_stress_pa:
        return stress, 0.0
    pressure = mat_trace(stress) / 3.0
    deviator = mat_sub(stress, mat_scale(mat_identity(), pressure))
    scale = yield_stress_pa / max(equivalent, 1e-18)
    limited = mat_add(
        mat_scale(deviator, scale),
        mat_scale(mat_identity(), pressure),
    )
    return limited, equivalent - yield_stress_pa


def element_force_and_energy(
    state: HybridFEMMPMState,
    element: TetraElement,
) -> Tuple[List[Vec3], float, float]:
    """Return four self-equilibrated corotational tetrahedral forces."""
    x0, x1, x2, x3 = (state.positions[node] for node in element.nodes)
    ds = matrix_from_columns(v_sub(x1, x0), v_sub(x2, x0), v_sub(x3, x0))
    deformation_gradient = mat_mul(ds, element.dm_inverse)
    rotation = polar_rotation(deformation_gradient)
    local_stretch = mat_mul(mat_transpose(rotation), deformation_gradient)
    strain = mat_sub(
        mat_scale(mat_add(local_stretch, mat_transpose(local_stretch)), 0.5),
        mat_identity(),
    )
    trial_stress = _elastic_stress(
        strain,
        element.young_modulus_pa,
        element.poisson_ratio,
    )
    stress, excess_stress = _limit_j2_stress(
        trial_stress,
        element.yield_stress_pa,
    )
    equivalent_increment = excess_stress / max(element.young_modulus_pa, 1.0)
    element.equivalent_plastic_strain += equivalent_increment
    element.stress = stress
    strain_norm = math.sqrt(
        sum(strain[i][j] * strain[i][j] for i in range(3) for j in range(3))
    )
    if (
        element.failure_strain > 0.0
        and max(strain_norm, element.equivalent_plastic_strain)
        >= element.failure_strain
    ):
        element.failed = True
        return [(0.0, 0.0, 0.0)] * 4, 0.0, equivalent_increment
    first_piola = mat_mul(rotation, stress)
    forces = [
        v_mul(
            mat_vec(first_piola, gradient),
            -element.rest_volume_m3,
        )
        for gradient in element.shape_gradients
    ]
    energy_density = 0.5 * sum(
        stress[i][j] * strain[i][j] for i in range(3) for j in range(3)
    )
    element.strain_energy_j = max(0.0, energy_density * element.rest_volume_m3)
    return forces, element.strain_energy_j, equivalent_increment


def stable_fem_timestep(state: HybridFEMMPMState) -> float:
    stable = math.inf
    total_volume = sum(
        element.rest_volume_m3 for element in state.elements if not element.failed
    )
    density = sum(state.masses_kg) / max(total_volume, 1e-18)
    for element in state.elements:
        if element.failed:
            continue
        wave_speed = math.sqrt(
            element.young_modulus_pa
            * (1.0 - element.poisson_ratio)
            / max(
                density
                * (1.0 + element.poisson_ratio)
                * (1.0 - 2.0 * element.poisson_ratio),
                1e-18,
            )
        )
        characteristic = (6.0 * element.rest_volume_m3) ** (1.0 / 3.0)
        stable = min(stable, state.cfl * characteristic / max(wave_speed, 1e-12))
    return stable if math.isfinite(stable) else math.inf


def _transfer_failed_elements(state: HybridFEMMPMState) -> None:
    for element_index, element in enumerate(state.elements):
        if not element.failed or element.transferred:
            continue
        if element.mass_kg > 0.0:
            shares = [0.25 * element.mass_kg] * 4
        else:
            active_incidence = [0] * len(state.positions)
            for candidate in state.elements:
                if not candidate.transferred:
                    for node in candidate.nodes:
                        active_incidence[node] += 1
            shares = [
                state.masses_kg[node] / max(active_incidence[node], 1)
                for node in element.nodes
            ]
        for local_index, node in enumerate(element.nodes):
            mass = min(shares[local_index], state.masses_kg[node])
            state.masses_kg[node] -= mass
            state.particles.append(
                MPMParticle(
                    position=state.positions[node],
                    velocity=state.velocities[node],
                    mass_kg=mass,
                    volume_m3=0.25 * element.rest_volume_m3,
                    source_element=element_index,
                    young_modulus_pa=element.young_modulus_pa,
                    poisson_ratio=element.poisson_ratio,
                    yield_stress_pa=element.yield_stress_pa,
                    source_node=node,
                    stress=mat_zero(),
                    damage=1.0,
                )
            )
        element.transferred = True


def _particle_weights(
    position: Vec3,
    cell_size: float,
) -> Iterable[Tuple[GridIndex, float, Vec3]]:
    scaled = tuple(value / cell_size for value in position)
    base = tuple(math.floor(value) for value in scaled)
    fraction = tuple(scaled[i] - base[i] for i in range(3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offsets = (dx, dy, dz)
                weight = 1.0
                gradient_values = [0.0, 0.0, 0.0]
                for axis in range(3):
                    coordinate_weight = (
                        fraction[axis] if offsets[axis] else 1.0 - fraction[axis]
                    )
                    weight *= coordinate_weight
                for axis in range(3):
                    derivative = (1.0 if offsets[axis] else -1.0) / cell_size
                    for other_axis in range(3):
                        if other_axis == axis:
                            continue
                        derivative *= (
                            fraction[other_axis]
                            if offsets[other_axis]
                            else 1.0 - fraction[other_axis]
                        )
                    gradient_values[axis] = derivative
                yield (
                    (base[0] + dx, base[1] + dy, base[2] + dz),
                    weight,
                    tuple(gradient_values),  # type: ignore[arg-type]
                )


def advance_mpm(state: HybridFEMMPMState, dt_s: float) -> None:
    if not state.particles or dt_s <= 0.0:
        return
    cell_size = state.mpm_cell_size_m
    if cell_size <= 0.0:
        mean_volume = sum(p.volume_m3 for p in state.particles) / len(state.particles)
        cell_size = max(mean_volume ** (1.0 / 3.0), 1e-6)
    grid_mass: Dict[GridIndex, float] = {}
    grid_momentum: Dict[GridIndex, Vec3] = {}
    grid_force: Dict[GridIndex, Vec3] = {}
    for particle in state.particles:
        for node, weight, gradient in _particle_weights(particle.position, cell_size):
            grid_mass[node] = grid_mass.get(node, 0.0) + weight * particle.mass_kg
            grid_momentum[node] = v_add(
                grid_momentum.get(node, (0.0, 0.0, 0.0)),
                v_mul(particle.velocity, weight * particle.mass_kg),
            )
            stress_gradient = mat_vec(particle.stress, gradient)
            grid_force[node] = v_add(
                grid_force.get(node, (0.0, 0.0, 0.0)),
                v_mul(stress_gradient, -particle.volume_m3),
            )
    old_grid_velocity: Dict[GridIndex, Vec3] = {}
    new_grid_velocity: Dict[GridIndex, Vec3] = {}
    for node, mass in grid_mass.items():
        if mass <= 1e-18:
            continue
        old_velocity = v_mul(grid_momentum[node], 1.0 / mass)
        old_grid_velocity[node] = old_velocity
        new_grid_velocity[node] = v_add(
            old_velocity,
            v_mul(grid_force.get(node, (0.0, 0.0, 0.0)), dt_s / mass),
        )
    for particle in state.particles:
        pic_velocity = (0.0, 0.0, 0.0)
        flip_delta = (0.0, 0.0, 0.0)
        velocity_gradient = mat_zero()
        for node, weight, gradient in _particle_weights(particle.position, cell_size):
            new_velocity = new_grid_velocity.get(node, (0.0, 0.0, 0.0))
            old_velocity = old_grid_velocity.get(node, (0.0, 0.0, 0.0))
            pic_velocity = v_add(pic_velocity, v_mul(new_velocity, weight))
            flip_delta = v_add(
                flip_delta,
                v_mul(v_sub(new_velocity, old_velocity), weight),
            )
            velocity_gradient = mat_add(
                velocity_gradient,
                tuple(
                    tuple(new_velocity[i] * gradient[j] for j in range(3))
                    for i in range(3)
                ),  # type: ignore[arg-type]
            )
        flip_velocity = v_add(particle.velocity, flip_delta)
        particle.velocity = v_add(
            v_mul(pic_velocity, state.pic_fraction),
            v_mul(flip_velocity, 1.0 - state.pic_fraction),
        )
        particle.position = v_add(particle.position, v_mul(particle.velocity, dt_s))
        particle.deformation_gradient = mat_mul(
            mat_add(mat_identity(), mat_scale(velocity_gradient, dt_s)),
            particle.deformation_gradient,
        )
        if particle.damage >= 1.0:
            particle.stress = mat_zero()
            continue
        strain = mat_scale(
            mat_add(
                mat_sub(particle.deformation_gradient, mat_identity()),
                mat_transpose(mat_sub(particle.deformation_gradient, mat_identity())),
            ),
            0.5,
        )
        stress = _elastic_stress(
            strain,
            particle.young_modulus_pa * max(0.0, 1.0 - particle.damage),
            particle.poisson_ratio,
        )
        particle.stress, _excess = _limit_j2_stress(
            stress,
            particle.yield_stress_pa * max(0.0, 1.0 - particle.damage),
        )


def project_free_body_momentum(
    state: HybridFEMMPMState,
    target_momentum: Vec3,
    target_angular_momentum: Vec3,
) -> Tuple[float, float]:
    """Remove grid-transfer drift without changing deformational velocity.

    The correction is the minimum rigid translation and rotation needed to
    recover the pre-step free-body momentum.  Internal FEM/MPM forces cannot
    change either invariant, so this projection removes numerical transfer
    error rather than altering the physical deformation mode.
    """
    total_mass = state.total_mass_kg
    if total_mass <= 1e-18:
        return 0.0, 0.0
    momentum_correction = v_sub(target_momentum, state.total_momentum())
    delta_velocity = v_mul(momentum_correction, 1.0 / total_mass)
    state.velocities = [
        v_add(velocity, delta_velocity) if mass > 1e-18 else velocity
        for mass, velocity in zip(state.masses_kg, state.velocities)
    ]
    for particle in state.particles:
        particle.velocity = v_add(particle.velocity, delta_velocity)

    centre = state.center_of_mass()
    inertia_rows = [[0.0, 0.0, 0.0] for _axis in range(3)]

    def accumulate_inertia(mass: float, position: Vec3) -> None:
        if mass <= 1e-18:
            return
        radius = v_sub(position, centre)
        radius_sq = v_dot(radius, radius)
        for row in range(3):
            for column in range(3):
                inertia_rows[row][column] += mass * (
                    (radius_sq if row == column else 0.0)
                    - radius[row] * radius[column]
                )

    for mass, position in zip(state.masses_kg, state.positions):
        accumulate_inertia(mass, position)
    for particle in state.particles:
        accumulate_inertia(particle.mass_kg, particle.position)
    inertia: Mat3 = tuple(tuple(row) for row in inertia_rows)  # type: ignore[assignment]
    inertia_scale = max(inertia[0][0], inertia[1][1], inertia[2][2], 0.0)
    angular_correction = v_sub(
        target_angular_momentum,
        state.total_angular_momentum(),
    )
    delta_omega = (0.0, 0.0, 0.0)
    if inertia_scale > 1e-18 and v_norm(angular_correction) > 0.0:
        normalized_inertia = mat_scale(inertia, 1.0 / inertia_scale)
        try:
            delta_omega = v_mul(
                mat_vec(mat_inverse(normalized_inertia), angular_correction),
                1.0 / inertia_scale,
            )
        except ValueError:
            delta_omega = (0.0, 0.0, 0.0)
    if v_norm(delta_omega) > 0.0:
        state.velocities = [
            v_add(
                velocity,
                v_cross(delta_omega, v_sub(position, centre)),
            )
            if mass > 1e-18
            else velocity
            for mass, position, velocity in zip(
                state.masses_kg,
                state.positions,
                state.velocities,
            )
        ]
        for particle in state.particles:
            particle.velocity = v_add(
                particle.velocity,
                v_cross(delta_omega, v_sub(particle.position, centre)),
            )
    return v_norm(momentum_correction), v_norm(angular_correction)


def advance_fem(
    state: HybridFEMMPMState,
    dt_s: float,
    external_forces: Optional[Sequence[Vec3]] = None,
) -> ConservationAudit:
    """Advance the coupled state with CFL-limited explicit substeps."""
    before_mass = state.total_mass_kg
    before_momentum = state.total_momentum()
    before_angular = state.total_angular_momentum()
    before_kinetic = _kinetic_energy(state)
    if dt_s <= 0.0:
        audit = ConservationAudit(
            before_mass,
            before_mass,
            before_momentum,
            before_momentum,
            before_angular,
            before_angular,
            before_kinetic,
            before_kinetic,
        )
        state.last_audit = audit
        return audit
    stable_dt = stable_fem_timestep(state)
    required_substeps = 1 if not math.isfinite(stable_dt) else max(
        1, math.ceil(dt_s / max(stable_dt, 1e-12))
    )
    if required_substeps > state.max_substeps:
        raise RuntimeError(
            "FEM CFL limit requires "
            f"{required_substeps} substeps but COLLISION_FEM_MAX_SUBSTEPS="
            f"{state.max_substeps}; reduce MOTION_DT or raise the limit"
        )
    substeps = required_substeps
    substep_dt = dt_s / substeps
    external_work = 0.0
    plastic_dissipation = 0.0
    for _substep in range(substeps):
        forces = [(0.0, 0.0, 0.0) for _position in state.positions]
        strain_energy = 0.0
        for element in state.elements:
            if element.failed:
                continue
            element_forces, energy, plastic_increment = element_force_and_energy(
                state, element
            )
            strain_energy += energy
            plastic_dissipation += (
                plastic_increment * element.yield_stress_pa * element.rest_volume_m3
            )
            for local_index, node in enumerate(element.nodes):
                forces[node] = v_add(forces[node], element_forces[local_index])
        if external_forces is not None:
            if len(external_forces) != len(forces):
                raise ValueError("external force count must match FEM node count")
            for node, external_force in enumerate(external_forces):
                forces[node] = v_add(forces[node], external_force)
                external_work += (
                    v_dot(external_force, state.velocities[node]) * substep_dt
                )
        for node, mass in enumerate(state.masses_kg):
            if mass <= 1e-18 or node in state.fixed_nodes:
                if node in state.fixed_nodes:
                    state.velocities[node] = (0.0, 0.0, 0.0)
                continue
            state.velocities[node] = v_add(
                state.velocities[node],
                v_mul(forces[node], substep_dt / mass),
            )
            state.positions[node] = v_add(
                state.positions[node],
                v_mul(state.velocities[node], substep_dt),
            )
        _transfer_failed_elements(state)
        advance_mpm(state, substep_dt)
    momentum_projection = 0.0
    angular_projection = 0.0
    if external_forces is None and not state.fixed_nodes:
        momentum_projection, angular_projection = project_free_body_momentum(
            state,
            before_momentum,
            before_angular,
        )
    after_mass = state.total_mass_kg
    after_momentum = state.total_momentum()
    after_angular = state.total_angular_momentum()
    audit = ConservationAudit(
        mass_before_kg=before_mass,
        mass_after_kg=after_mass,
        momentum_before=before_momentum,
        momentum_after=after_momentum,
        angular_momentum_before=before_angular,
        angular_momentum_after=after_angular,
        kinetic_before_j=before_kinetic,
        kinetic_after_j=_kinetic_energy(state),
        strain_energy_j=sum(element.strain_energy_j for element in state.elements),
        plastic_dissipation_j=plastic_dissipation,
        external_work_j=external_work,
        momentum_projection_ns=momentum_projection,
        angular_momentum_projection_nms=angular_projection,
    )
    state.last_audit = audit
    mass_tolerance = 1e-10 * max(before_mass, 1.0)
    if abs(audit.mass_error_kg) > mass_tolerance:
        raise RuntimeError(
            "FEM/MPM mass conservation failure: "
            f"before={before_mass:.12g} kg after={after_mass:.12g} kg"
        )
    if external_forces is None and not state.fixed_nodes:
        momentum_tolerance = 1e-8 * max(v_norm(before_momentum), 1.0)
        if v_norm(audit.momentum_error) > momentum_tolerance:
            raise RuntimeError(
                "FEM/MPM linear momentum conservation failure: "
                f"error={v_norm(audit.momentum_error):.12g} N s"
            )
        angular_tolerance = 1e-6 * max(v_norm(before_angular), 1.0)
        if v_norm(audit.angular_momentum_error) > angular_tolerance:
            raise RuntimeError(
                "FEM/MPM angular momentum conservation failure: "
                f"error={v_norm(audit.angular_momentum_error):.12g} N m s"
            )
    return audit
