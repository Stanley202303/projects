"""Classical explicit solid FEM with conservative FEM-to-MPM failure transfer.

Intact material is advanced with lumped-mass, corotational constant-strain
tetrahedra.  Failed tetrahedra transfer their own nodal mass contributions to
material points and are subsequently advanced on a trilinear USL MPM grid.  The
transfer is exact for mass and linear momentum.  NumPy batches the independent
tetrahedron calculations when available; the scalar implementation remains the
reference fallback.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal installations.
    np = None  # type: ignore[assignment]

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


@dataclass
class _VectorizedFEMElements:
    """Contiguous element data used by the optional NumPy execution path."""

    nodes: Any
    dm_inverse: Any
    shape_gradients: Any
    rest_volume_m3: Any
    young_modulus_pa: Any
    poisson_ratio: Any
    yield_stress_pa: Any
    failure_strain: Any
    failed: Any
    equivalent_plastic_strain: Any
    strain_energy_j: Any
    stress: Any

    @classmethod
    def from_state(cls, state: HybridFEMMPMState) -> "_VectorizedFEMElements":
        if np is None:  # pragma: no cover - guarded by the caller.
            raise RuntimeError("NumPy is unavailable")
        elements = state.elements
        return cls(
            nodes=np.asarray([element.nodes for element in elements], dtype=np.intp),
            dm_inverse=np.asarray(
                [element.dm_inverse for element in elements], dtype=np.float64
            ),
            shape_gradients=np.asarray(
                [element.shape_gradients for element in elements], dtype=np.float64
            ),
            rest_volume_m3=np.asarray(
                [element.rest_volume_m3 for element in elements], dtype=np.float64
            ),
            young_modulus_pa=np.asarray(
                [element.young_modulus_pa for element in elements], dtype=np.float64
            ),
            poisson_ratio=np.asarray(
                [element.poisson_ratio for element in elements], dtype=np.float64
            ),
            yield_stress_pa=np.asarray(
                [element.yield_stress_pa for element in elements], dtype=np.float64
            ),
            failure_strain=np.asarray(
                [element.failure_strain for element in elements], dtype=np.float64
            ),
            failed=np.asarray(
                [element.failed for element in elements], dtype=np.bool_
            ),
            equivalent_plastic_strain=np.asarray(
                [element.equivalent_plastic_strain for element in elements],
                dtype=np.float64,
            ),
            strain_energy_j=np.asarray(
                [element.strain_energy_j for element in elements], dtype=np.float64
            ),
            stress=np.asarray(
                [element.stress for element in elements], dtype=np.float64
            ),
        )

    def sync_to_state(self, state: HybridFEMMPMState) -> None:
        for index, element in enumerate(state.elements):
            element.failed = bool(self.failed[index])
            element.equivalent_plastic_strain = float(
                self.equivalent_plastic_strain[index]
            )
            element.strain_energy_j = float(self.strain_energy_j[index])
            element.stress = tuple(
                tuple(float(self.stress[index, row, column]) for column in range(3))
                for row in range(3)
            )  # type: ignore[assignment]


def _batch_determinants(matrices: Any) -> Any:
    """Return explicit 3x3 determinants with scalar-operation ordering."""
    return (
        matrices[:, 0, 0]
        * (
            matrices[:, 1, 1] * matrices[:, 2, 2]
            - matrices[:, 1, 2] * matrices[:, 2, 1]
        )
        - matrices[:, 0, 1]
        * (
            matrices[:, 1, 0] * matrices[:, 2, 2]
            - matrices[:, 1, 2] * matrices[:, 2, 0]
        )
        + matrices[:, 0, 2]
        * (
            matrices[:, 1, 0] * matrices[:, 2, 1]
            - matrices[:, 1, 1] * matrices[:, 2, 0]
        )
    )


def _batch_inverse_transposes(matrices: Any, determinants: Any) -> Any:
    """Return inverse transposes for nonsingular batches of 3x3 matrices."""
    inverse_transposes = np.empty_like(matrices)
    inverse_transposes[:, 0, 0] = (
        matrices[:, 1, 1] * matrices[:, 2, 2]
        - matrices[:, 1, 2] * matrices[:, 2, 1]
    )
    inverse_transposes[:, 0, 1] = (
        matrices[:, 1, 2] * matrices[:, 2, 0]
        - matrices[:, 1, 0] * matrices[:, 2, 2]
    )
    inverse_transposes[:, 0, 2] = (
        matrices[:, 1, 0] * matrices[:, 2, 1]
        - matrices[:, 1, 1] * matrices[:, 2, 0]
    )
    inverse_transposes[:, 1, 0] = (
        matrices[:, 0, 2] * matrices[:, 2, 1]
        - matrices[:, 0, 1] * matrices[:, 2, 2]
    )
    inverse_transposes[:, 1, 1] = (
        matrices[:, 0, 0] * matrices[:, 2, 2]
        - matrices[:, 0, 2] * matrices[:, 2, 0]
    )
    inverse_transposes[:, 1, 2] = (
        matrices[:, 0, 1] * matrices[:, 2, 0]
        - matrices[:, 0, 0] * matrices[:, 2, 1]
    )
    inverse_transposes[:, 2, 0] = (
        matrices[:, 0, 1] * matrices[:, 1, 2]
        - matrices[:, 0, 2] * matrices[:, 1, 1]
    )
    inverse_transposes[:, 2, 1] = (
        matrices[:, 0, 2] * matrices[:, 1, 0]
        - matrices[:, 0, 0] * matrices[:, 1, 2]
    )
    inverse_transposes[:, 2, 2] = (
        matrices[:, 0, 0] * matrices[:, 1, 1]
        - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    inverse_transposes *= (1.0 / determinants)[:, None, None]
    return inverse_transposes


def _batch_matrix_multiply(left: Any, right: Any) -> Any:
    """Multiply batches of 3x3 matrices in scalar summation order."""
    product = np.empty_like(left)
    for row in range(3):
        for column in range(3):
            product[:, row, column] = _batch_float_sum(
                (
                    left[:, row, 0] * right[:, 0, column],
                    left[:, row, 1] * right[:, 1, column],
                    left[:, row, 2] * right[:, 2, column],
                )
            )
    return product


def _batch_float_sum(terms: Sequence[Any]) -> Any:
    """Vector form of the interpreter's finite-float ``sum`` algorithm."""
    total = terms[0].copy()
    if sys.version_info < (3, 12):
        for term in terms[1:]:
            total += term
        return total
    compensation = np.zeros_like(total)
    for term in terms[1:]:
        updated = total + term
        compensation += np.where(
            np.abs(total) >= np.abs(term),
            (total - updated) + term,
            (term - updated) + total,
        )
        total = updated
    return total + compensation


def _batch_sum_squares(matrices: Any) -> Any:
    return _batch_float_sum(
        tuple(
            matrices[:, row, column] * matrices[:, row, column]
            for row in range(3)
            for column in range(3)
        )
    )


def _batch_inner_product(left: Any, right: Any) -> Any:
    return _batch_float_sum(
        tuple(
            left[:, row, column] * right[:, row, column]
            for row in range(3)
            for column in range(3)
        )
    )


def _batch_polar_rotations(deformation_gradients: Any) -> Any:
    """Apply the scalar Newton polar iteration independently to each matrix."""
    rotations = deformation_gradients.copy()
    identity = np.eye(3, dtype=np.float64)
    active = _batch_determinants(rotations) > 1e-12
    rotations[~active] = identity
    for _iteration in range(8):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        current = rotations[active_indices]
        determinants = _batch_determinants(current)
        singular = np.abs(determinants) <= 1e-18
        if np.any(singular):
            singular_indices = active_indices[singular]
            rotations[singular_indices] = identity
            active[singular_indices] = False
        usable_indices = active_indices[~singular]
        if usable_indices.size == 0:
            continue
        current = rotations[usable_indices]
        determinants = determinants[~singular]
        inverse_transposes = _batch_inverse_transposes(current, determinants)
        updated = 0.5 * (current + inverse_transposes)
        differences = np.max(np.abs(updated - current), axis=(1, 2))
        rotations[usable_indices] = updated
        active[usable_indices[differences < 1e-10]] = False
    return rotations


def _batched_element_forces(
    state: HybridFEMMPMState,
    elements: _VectorizedFEMElements,
    positions: Any,
) -> Tuple[Any, float, bool]:
    """Evaluate all intact tetrahedra while preserving scalar state semantics."""
    nodal_forces = np.zeros((len(state.positions), 3), dtype=np.float64)
    active_indices = np.flatnonzero(~elements.failed)
    if active_indices.size == 0:
        return nodal_forces, 0.0, False

    nodes = elements.nodes[active_indices]
    node_positions = positions[nodes]
    ds = np.empty((active_indices.size, 3, 3), dtype=np.float64)
    ds[:, :, 0] = node_positions[:, 1] - node_positions[:, 0]
    ds[:, :, 1] = node_positions[:, 2] - node_positions[:, 0]
    ds[:, :, 2] = node_positions[:, 3] - node_positions[:, 0]
    deformation_gradient = _batch_matrix_multiply(
        ds, elements.dm_inverse[active_indices]
    )
    rotation = _batch_polar_rotations(deformation_gradient)
    local_stretch = _batch_matrix_multiply(
        np.swapaxes(rotation, 1, 2), deformation_gradient
    )
    strain = 0.5 * (local_stretch + np.swapaxes(local_stretch, 1, 2))
    strain[:, 0, 0] -= 1.0
    strain[:, 1, 1] -= 1.0
    strain[:, 2, 2] -= 1.0

    young = elements.young_modulus_pa[active_indices]
    poisson = elements.poisson_ratio[active_indices]
    lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    strain_trace = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
    stress = 2.0 * shear[:, None, None] * strain
    volumetric_stress = lame_lambda * strain_trace
    stress[:, 0, 0] += volumetric_stress
    stress[:, 1, 1] += volumetric_stress
    stress[:, 2, 2] += volumetric_stress

    pressure = (stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2]) / 3.0
    deviator = stress.copy()
    deviator[:, 0, 0] -= pressure
    deviator[:, 1, 1] -= pressure
    deviator[:, 2, 2] -= pressure
    equivalent_stress = np.sqrt(1.5 * _batch_sum_squares(deviator))
    yield_stress = elements.yield_stress_pa[active_indices]
    yielded = (yield_stress > 0.0) & (equivalent_stress > yield_stress)
    excess_stress = np.zeros(active_indices.size, dtype=np.float64)
    if np.any(yielded):
        scale = yield_stress[yielded] / np.maximum(
            equivalent_stress[yielded], 1e-18
        )
        limited_stress = deviator[yielded] * scale[:, None, None]
        limited_stress[:, 0, 0] += pressure[yielded]
        limited_stress[:, 1, 1] += pressure[yielded]
        limited_stress[:, 2, 2] += pressure[yielded]
        stress[yielded] = limited_stress
        excess_stress[yielded] = (
            equivalent_stress[yielded] - yield_stress[yielded]
        )

    plastic_increment = excess_stress / np.maximum(young, 1.0)
    elements.equivalent_plastic_strain[active_indices] += plastic_increment
    elements.stress[active_indices] = stress
    strain_norm = np.sqrt(_batch_sum_squares(strain))
    failure_strain = elements.failure_strain[active_indices]
    newly_failed = (failure_strain > 0.0) & (
        np.maximum(
            strain_norm,
            elements.equivalent_plastic_strain[active_indices],
        )
        >= failure_strain
    )
    if np.any(newly_failed):
        failed_indices = active_indices[newly_failed]
        elements.failed[failed_indices] = True
        for element_index in failed_indices:
            state.elements[int(element_index)].failed = True

    surviving = ~newly_failed
    element_forces = np.zeros(
        (active_indices.size, 4, 3), dtype=np.float64
    )
    if np.any(surviving):
        surviving_indices = active_indices[surviving]
        first_piola = _batch_matrix_multiply(rotation[surviving], stress[surviving])
        gradients = elements.shape_gradients[surviving_indices]
        volume = elements.rest_volume_m3[surviving_indices]
        for local_node in range(4):
            for row in range(3):
                force_component = _batch_float_sum(
                    (
                        first_piola[:, row, 0] * gradients[:, local_node, 0],
                        first_piola[:, row, 1] * gradients[:, local_node, 1],
                        first_piola[:, row, 2] * gradients[:, local_node, 2],
                    )
                )
                element_forces[surviving, local_node, row] = -volume * force_component
        energy_density = 0.5 * _batch_inner_product(
            stress[surviving], strain[surviving]
        )
        elements.strain_energy_j[surviving_indices] = np.fmax(
            0.0, energy_density * volume
        )

    np.add.at(
        nodal_forces,
        nodes.reshape(-1),
        element_forces.reshape(-1, 3),
    )
    plastic_dissipation = float(
        np.sum(
            plastic_increment
            * yield_stress
            * elements.rest_volume_m3[active_indices]
        )
    )
    return nodal_forces, plastic_dissipation, bool(np.any(newly_failed))


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


def transfer_failed_elements(state: HybridFEMMPMState) -> None:
    """Move newly failed FEM mass to MPM particles in linear time.

    The nodal incidence count changes only when an element is transferred. The
    previous implementation rebuilt that count by scanning every element for
    every individual failure, which made a large fracture burst quadratic.
    Updating the count incrementally preserves the same deterministic mass
    shares and transfer order.
    """
    pending = [
        (element_index, element)
        for element_index, element in enumerate(state.elements)
        if element.failed and not element.transferred
    ]
    if not pending:
        return

    active_incidence = [0] * len(state.positions)
    for element in state.elements:
        if element.transferred:
            continue
        for node in element.nodes:
            active_incidence[node] += 1

    for element_index, element in pending:
        if element.mass_kg > 0.0:
            shares = [0.25 * element.mass_kg] * 4
        else:
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
        for node in element.nodes:
            active_incidence[node] = max(0, active_incidence[node] - 1)


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


def _advance_mpm_scalar(state: HybridFEMMPMState, dt_s: float) -> None:
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


@dataclass
class _VectorizedMPMParticles:
    positions: Any
    velocities: Any
    masses_kg: Any
    volumes_m3: Any
    young_modulus_pa: Any
    poisson_ratio: Any
    yield_stress_pa: Any
    damage: Any
    deformation_gradient: Any
    stress: Any

    @classmethod
    def from_state(cls, state: HybridFEMMPMState) -> "_VectorizedMPMParticles":
        particles = state.particles
        return cls(
            positions=np.asarray(
                [particle.position for particle in particles], dtype=np.float64
            ),
            velocities=np.asarray(
                [particle.velocity for particle in particles], dtype=np.float64
            ),
            masses_kg=np.asarray(
                [particle.mass_kg for particle in particles], dtype=np.float64
            ),
            volumes_m3=np.asarray(
                [particle.volume_m3 for particle in particles], dtype=np.float64
            ),
            young_modulus_pa=np.asarray(
                [particle.young_modulus_pa for particle in particles],
                dtype=np.float64,
            ),
            poisson_ratio=np.asarray(
                [particle.poisson_ratio for particle in particles], dtype=np.float64
            ),
            yield_stress_pa=np.asarray(
                [particle.yield_stress_pa for particle in particles],
                dtype=np.float64,
            ),
            damage=np.asarray(
                [particle.damage for particle in particles], dtype=np.float64
            ),
            deformation_gradient=np.asarray(
                [particle.deformation_gradient for particle in particles],
                dtype=np.float64,
            ),
            stress=np.asarray(
                [particle.stress for particle in particles], dtype=np.float64
            ),
        )

    def sync_to_state(self, state: HybridFEMMPMState) -> None:
        for index, particle in enumerate(state.particles):
            particle.position = tuple(
                float(self.positions[index, axis]) for axis in range(3)
            )  # type: ignore[assignment]
            particle.velocity = tuple(
                float(self.velocities[index, axis]) for axis in range(3)
            )  # type: ignore[assignment]
            particle.deformation_gradient = tuple(
                tuple(
                    float(self.deformation_gradient[index, row, column])
                    for column in range(3)
                )
                for row in range(3)
            )  # type: ignore[assignment]
            particle.stress = tuple(
                tuple(float(self.stress[index, row, column]) for column in range(3))
                for row in range(3)
            )  # type: ignore[assignment]


def _batch_particle_weights(
    positions: Any,
    cell_size: float,
) -> Tuple[Any, Any, Any]:
    """Return grid indices, trilinear weights and gradients in scalar order."""
    offsets = np.asarray(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        dtype=np.int64,
    )
    scaled = positions / cell_size
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    grid_indices = base[:, None, :] + offsets[None, :, :]
    coordinate_weights = np.where(
        offsets[None, :, :] == 1,
        fraction[:, None, :],
        1.0 - fraction[:, None, :],
    )
    weights = coordinate_weights[:, :, 0] * coordinate_weights[:, :, 1]
    weights *= coordinate_weights[:, :, 2]
    gradients = np.empty_like(coordinate_weights)
    for axis in range(3):
        gradients[:, :, axis] = np.where(
            offsets[None, :, axis] == 1,
            1.0 / cell_size,
            -1.0 / cell_size,
        )
        for other_axis in range(3):
            if other_axis != axis:
                gradients[:, :, axis] *= coordinate_weights[:, :, other_axis]
    return grid_indices, weights, gradients


def _advance_mpm_vectorized(state: HybridFEMMPMState, dt_s: float) -> None:
    particles = _VectorizedMPMParticles.from_state(state)
    particle_count = len(state.particles)
    cell_size = state.mpm_cell_size_m
    if cell_size <= 0.0:
        mean_volume = sum(p.volume_m3 for p in state.particles) / particle_count
        cell_size = max(mean_volume ** (1.0 / 3.0), 1e-6)

    grid_indices, weights, gradients = _batch_particle_weights(
        particles.positions, cell_size
    )
    flat_grid_indices = grid_indices.reshape(-1, 3)
    _unique_grid_indices, inverse = np.unique(
        flat_grid_indices,
        axis=0,
        return_inverse=True,
    )
    grid_node_count = int(np.max(inverse)) + 1
    flat_inverse = inverse.reshape(-1)
    grid_mass = np.zeros(grid_node_count, dtype=np.float64)
    grid_momentum = np.zeros((grid_node_count, 3), dtype=np.float64)
    grid_force = np.zeros((grid_node_count, 3), dtype=np.float64)

    weighted_mass = weights * particles.masses_kg[:, None]
    np.add.at(grid_mass, flat_inverse, weighted_mass.reshape(-1))
    momentum_contribution = particles.velocities[:, None, :] * weighted_mass[:, :, None]
    np.add.at(
        grid_momentum,
        flat_inverse,
        momentum_contribution.reshape(-1, 3),
    )
    stress_gradient = np.empty((particle_count, 8, 3), dtype=np.float64)
    for row in range(3):
        stress_gradient[:, :, row] = _batch_float_sum(
            tuple(
                particles.stress[:, None, row, column]
                * gradients[:, :, column]
                for column in range(3)
            )
        )
    force_contribution = stress_gradient * (-particles.volumes_m3[:, None, None])
    np.add.at(
        grid_force,
        flat_inverse,
        force_contribution.reshape(-1, 3),
    )

    old_grid_velocity = np.zeros_like(grid_momentum)
    new_grid_velocity = np.zeros_like(grid_momentum)
    populated = grid_mass > 1e-18
    inverse_mass = 1.0 / grid_mass[populated]
    old_grid_velocity[populated] = (
        grid_momentum[populated] * inverse_mass[:, None]
    )
    new_grid_velocity[populated] = old_grid_velocity[populated] + (
        grid_force[populated]
        * (dt_s / grid_mass[populated])[:, None]
    )

    corner_new_velocity = new_grid_velocity[inverse].reshape(particle_count, 8, 3)
    corner_old_velocity = old_grid_velocity[inverse].reshape(particle_count, 8, 3)
    pic_velocity = np.zeros((particle_count, 3), dtype=np.float64)
    flip_delta = np.zeros((particle_count, 3), dtype=np.float64)
    velocity_gradient = np.zeros((particle_count, 3, 3), dtype=np.float64)
    for corner in range(8):
        weight = weights[:, corner, None]
        new_velocity = corner_new_velocity[:, corner]
        old_velocity = corner_old_velocity[:, corner]
        pic_velocity += new_velocity * weight
        flip_delta += (new_velocity - old_velocity) * weight
        velocity_gradient += (
            new_velocity[:, :, None] * gradients[:, corner, None, :]
        )

    flip_velocity = particles.velocities + flip_delta
    particles.velocities = (
        pic_velocity * state.pic_fraction
        + flip_velocity * (1.0 - state.pic_fraction)
    )
    particles.positions += particles.velocities * dt_s
    deformation_increment = np.eye(3, dtype=np.float64)[None, :, :] + (
        dt_s * velocity_gradient
    )
    particles.deformation_gradient = _batch_matrix_multiply(
        deformation_increment,
        particles.deformation_gradient,
    )

    particles.stress[particles.damage >= 1.0] = 0.0
    active = particles.damage < 1.0
    if np.any(active):
        deformation_delta = particles.deformation_gradient[active] - np.eye(
            3, dtype=np.float64
        )
        strain = 0.5 * (
            deformation_delta + np.swapaxes(deformation_delta, 1, 2)
        )
        damage_factor = np.fmax(0.0, 1.0 - particles.damage[active])
        young = particles.young_modulus_pa[active] * damage_factor
        poisson = particles.poisson_ratio[active]
        lame_lambda = (
            young
            * poisson
            / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        )
        shear = young / (2.0 * (1.0 + poisson))
        strain_trace = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
        stress = 2.0 * shear[:, None, None] * strain
        volumetric_stress = lame_lambda * strain_trace
        stress[:, 0, 0] += volumetric_stress
        stress[:, 1, 1] += volumetric_stress
        stress[:, 2, 2] += volumetric_stress

        pressure = (
            stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2]
        ) / 3.0
        deviator = stress.copy()
        deviator[:, 0, 0] -= pressure
        deviator[:, 1, 1] -= pressure
        deviator[:, 2, 2] -= pressure
        equivalent_stress = np.sqrt(1.5 * _batch_sum_squares(deviator))
        yield_stress = particles.yield_stress_pa[active] * damage_factor
        yielded = (yield_stress > 0.0) & (equivalent_stress > yield_stress)
        if np.any(yielded):
            scale = yield_stress[yielded] / np.maximum(
                equivalent_stress[yielded], 1e-18
            )
            limited_stress = deviator[yielded] * scale[:, None, None]
            limited_stress[:, 0, 0] += pressure[yielded]
            limited_stress[:, 1, 1] += pressure[yielded]
            limited_stress[:, 2, 2] += pressure[yielded]
            stress[yielded] = limited_stress
        particles.stress[active] = stress
    particles.sync_to_state(state)


def advance_mpm(state: HybridFEMMPMState, dt_s: float) -> None:
    """Advance failed material points, using NumPy when it is available."""
    if not state.particles or dt_s <= 0.0:
        return
    if np is None:
        _advance_mpm_scalar(state, dt_s)
    else:
        _advance_mpm_vectorized(state, dt_s)


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


def _advance_fem_substeps_scalar(
    state: HybridFEMMPMState,
    substeps: int,
    substep_dt: float,
    external_forces: Optional[Sequence[Vec3]],
) -> Tuple[float, float]:
    """Dependency-free reference implementation for explicit FEM substeps."""
    external_work = 0.0
    plastic_dissipation = 0.0
    for _substep in range(substeps):
        forces = [(0.0, 0.0, 0.0) for _position in state.positions]
        for element in state.elements:
            if element.failed:
                continue
            element_forces, _energy, plastic_increment = element_force_and_energy(
                state, element
            )
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
        transfer_failed_elements(state)
        advance_mpm(state, substep_dt)
    return external_work, plastic_dissipation


def _sync_vectorized_nodes(
    state: HybridFEMMPMState,
    positions: Any,
    velocities: Any,
) -> None:
    state.positions = [
        (float(position[0]), float(position[1]), float(position[2]))
        for position in positions
    ]
    state.velocities = [
        (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        for velocity in velocities
    ]


def _advance_fem_substeps_vectorized(
    state: HybridFEMMPMState,
    substeps: int,
    substep_dt: float,
    external_forces: Optional[Sequence[Vec3]],
) -> Tuple[float, float]:
    """NumPy execution path for the same explicit FEM integration scheme."""
    elements = _VectorizedFEMElements.from_state(state)
    positions = np.asarray(state.positions, dtype=np.float64).copy()
    velocities = np.asarray(state.velocities, dtype=np.float64).copy()
    masses = np.asarray(state.masses_kg, dtype=np.float64)
    external = (
        None
        if external_forces is None
        else np.asarray(external_forces, dtype=np.float64)
    )
    fixed = np.zeros(len(state.positions), dtype=np.bool_)
    if state.fixed_nodes:
        fixed[list(state.fixed_nodes)] = True
    pending_transfer = any(
        element.failed and not element.transferred for element in state.elements
    )
    external_work = 0.0
    plastic_dissipation = 0.0

    for _substep in range(substeps):
        forces, plastic_increment, newly_failed = _batched_element_forces(
            state, elements, positions
        )
        plastic_dissipation += plastic_increment
        if external is not None:
            if len(external) != len(forces):
                elements.sync_to_state(state)
                raise ValueError("external force count must match FEM node count")
            forces += external
            external_work += float(np.sum(external * velocities)) * substep_dt

        movable = (masses > 1e-18) & ~fixed
        velocities[movable] += (
            forces[movable] * (substep_dt / masses[movable])[:, None]
        )
        positions[movable] += velocities[movable] * substep_dt
        velocities[fixed] = 0.0

        pending_transfer = pending_transfer or newly_failed
        if pending_transfer:
            _sync_vectorized_nodes(state, positions, velocities)
            elements.sync_to_state(state)
            transfer_failed_elements(state)
            masses = np.asarray(state.masses_kg, dtype=np.float64)
            pending_transfer = False
        advance_mpm(state, substep_dt)

    _sync_vectorized_nodes(state, positions, velocities)
    elements.sync_to_state(state)
    return external_work, plastic_dissipation


def advance_fem(
    state: HybridFEMMPMState,
    dt_s: float,
    external_forces: Optional[Sequence[Vec3]] = None,
) -> ConservationAudit:
    """Advance the coupled state with automatically batched CFL substeps."""
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
    # max_substeps is a work-batch size, not a reason to reject a physically
    # valid timestep.  Keep the exact CFL-required substep size and process as
    # many bounded batches as necessary.  This is numerically equivalent to a
    # single explicit loop and avoids either destabilising the solve or asking
    # the caller to tune an implementation limit for every CAD mesh.
    substep_dt = dt_s / required_substeps
    batch_limit = max(state.max_substeps, 1)
    remaining_substeps = required_substeps
    external_work = 0.0
    plastic_dissipation = 0.0
    advance_batch = (
        _advance_fem_substeps_vectorized
        if np is not None and state.elements
        else _advance_fem_substeps_scalar
    )
    while remaining_substeps > 0:
        batch_substeps = min(remaining_substeps, batch_limit)
        batch_work, batch_dissipation = advance_batch(
            state,
            batch_substeps,
            substep_dt,
            external_forces,
        )
        external_work += batch_work
        plastic_dissipation += batch_dissipation
        remaining_substeps -= batch_substeps
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
