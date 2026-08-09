from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .math_utils import (
    v_add,
    v_cross,
    v_dot,
    v_mul,
    v_norm,
    v_sub,
    v_unit,
)
from .models import AeroComponent, MotionFreedom, Triangle, Vec3
from .geometry import estimate_closed_mesh_volume
from .fem_mpm import (
    ConservationAudit,
    HybridFEMMPMState,
    advance_fem,
    make_tetra_element,
    von_mises_stress,
)


@dataclass
class ShellEdge:
    node_a: int
    node_b: int
    rest_length_m: float
    stiffness_n_m: float
    failed: bool = False


@dataclass
class ShellMembraneElement:
    """Linear constant-strain triangle in its reference tangent plane."""

    nodes: Tuple[int, int, int]
    area_m2: float
    basis_x: Vec3
    basis_y: Vec3
    b_coefficients: Tuple[float, float, float]
    c_coefficients: Tuple[float, float, float]


@dataclass
class ShellBendingSpring:
    node_a: int
    node_b: int
    rest_length_m: float
    stiffness_n_m: float


@dataclass
class ExplicitShellState:
    positions: List[Vec3]
    reference_positions: List[Vec3]
    velocities: List[Vec3]
    masses_kg: List[float]
    triangle_nodes: List[Tuple[int, int, int]]
    triangle_reference_centroids: List[Vec3]
    triangle_normals: List[Vec3]
    triangle_masses_kg: List[float]
    edges: List[ShellEdge]
    bending_springs: List[ShellBendingSpring]
    membrane_elements: List[ShellMembraneElement]
    weld_groups: List[List[int]]
    fixed_nodes: Set[int]
    contact_point: Vec3
    inward_direction: Vec3
    contact_radius_m: float
    young_modulus_pa: float
    thickness_m: float
    poisson_ratio: float
    yield_strain: float
    failure_strain: float
    damping_ratio: float
    cfl: float
    max_substeps: int
    displacement_limit_m: float
    plug_triangles: Set[int] = field(default_factory=set)
    emitted_triangles: Set[int] = field(default_factory=set)
    emitted_fragment_mass_kg: float = 0.0
    stable_dt_s: float = 1e-6
    mass_scale: float = 1.0
    max_displacement_m: float = 0.0
    permanent_displacement_m: float = 0.0
    failed_edges: int = 0
    current_hole_radius_m: float = 0.0
    # ``mpm`` selects the particle-grid update below while retaining the same
    # shell topology and fragment bookkeeping for output compatibility.
    solver_backend: str = "explicit_shell"
    mpm_cell_size_m: float = 0.0
    render_as_midsurface: bool = False


@dataclass
class DetachedFragmentBody:
    component: AeroComponent
    triangle_indices: Set[int]
    mass_kg: float
    source: str = "hybrid-shell-fragment"


@dataclass
class HybridShellCollisionState:
    shell_state: ExplicitShellState
    fragment_bodies: List[DetachedFragmentBody] = field(default_factory=list)
    emitted_triangles: Set[int] = field(default_factory=set)
    emitted_fragment_mass_kg: float = 0.0
    next_fragment_id: int = 0
    solver_backend: str = "hybrid_shell"

    def __getattr__(self, name: str) -> object:
        return getattr(self.shell_state, name)


@dataclass
class HybridFEMMPMCollisionState:
    solid_state: HybridFEMMPMState
    fragment_bodies: List[DetachedFragmentBody] = field(default_factory=list)
    emitted_elements: Set[int] = field(default_factory=set)
    next_fragment_id: int = 0
    reference_mass_kg: float = 0.0
    last_audit: Optional[ConservationAudit] = None
    solver_backend: str = "hybrid_fem_mpm"

    @property
    def max_displacement_m(self) -> float:
        return max(
            (
                v_norm(v_sub(position, reference))
                for position, reference in zip(
                    self.solid_state.positions,
                    self.solid_state.reference_positions,
                )
            ),
            default=0.0,
        )


def _triangle_area(triangle: Triangle) -> float:
    _normal, a, b, c = triangle
    return 0.5 * v_norm(v_cross(v_sub(b, a), v_sub(c, a)))


def _centroid(points: Sequence[Vec3]) -> Vec3:
    inverse = 1.0 / max(len(points), 1)
    total = (0.0, 0.0, 0.0)
    for point in points:
        total = v_add(total, point)
    return v_mul(total, inverse)


def _radial_distance(point: Vec3, origin: Vec3, axis: Vec3) -> float:
    delta = v_sub(point, origin)
    axial = v_mul(axis, v_dot(delta, axis))
    return v_norm(v_sub(delta, axial))


def _closest_point_on_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """Closest point on a triangle using the standard barycentric-region test."""
    ab = v_sub(b, a)
    ac = v_sub(c, a)
    ap = v_sub(point, a)
    d1 = v_dot(ab, ap)
    d2 = v_dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = v_sub(point, b)
    d3 = v_dot(ab, bp)
    d4 = v_dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        weight = d1 / max(d1 - d3, 1e-18)
        return v_add(a, v_mul(ab, weight))

    cp = v_sub(point, c)
    d5 = v_dot(ab, cp)
    d6 = v_dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        weight = d2 / max(d2 - d6, 1e-18)
        return v_add(a, v_mul(ac, weight))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = v_sub(c, b)
        weight = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-18)
        return v_add(b, v_mul(bc, weight))

    denominator = max(va + vb + vc, 1e-18)
    v_weight = vb / denominator
    w_weight = vc / denominator
    return v_add(a, v_add(v_mul(ab, v_weight), v_mul(ac, w_weight)))


def _closest_point_on_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3:
    ab = v_sub(b, a)
    denominator = v_dot(ab, ab)
    if denominator <= 1e-18:
        return a
    t = max(0.0, min(1.0, v_dot(v_sub(point, a), ab) / denominator))
    return v_add(a, v_mul(ab, t))


def _triangle_intersects_hole_disk(
    state: ExplicitShellState,
    triangle_index: int,
    hole_radius_m: float,
) -> bool:
    node_ids = state.triangle_nodes[triangle_index]
    points = [state.reference_positions[node] for node in node_ids]
    centroid = state.triangle_reference_centroids[triangle_index]
    centroid_radius = _radial_distance(
        centroid,
        state.contact_point,
        state.inward_direction,
    )
    if centroid_radius <= hole_radius_m:
        return True
    vertex_hits = sum(
        _radial_distance(point, state.contact_point, state.inward_direction)
        <= hole_radius_m
        for point in points
    )
    if vertex_hits >= 2:
        return True
    closest = _closest_point_on_triangle(state.contact_point, *points)
    closest_radius = _radial_distance(
        closest,
        state.contact_point,
        state.inward_direction,
    )
    return (
        closest_radius <= hole_radius_m
        and centroid_radius <= hole_radius_m + 0.5 * state.contact_radius_m
    )


def _triangle_intersects_deformed_hole_disk(
    state: ExplicitShellState,
    triangle_index: int,
    hole_radius_m: float,
) -> bool:
    node_ids = state.triangle_nodes[triangle_index]
    points = [state.positions[node] for node in node_ids]
    centroid = _centroid(points)
    centroid_radius = _radial_distance(
        centroid,
        state.contact_point,
        state.inward_direction,
    )
    if centroid_radius <= hole_radius_m:
        return True
    vertex_hits = sum(
        _radial_distance(point, state.contact_point, state.inward_direction)
        <= hole_radius_m
        for point in points
    )
    if vertex_hits >= 2:
        return True
    closest = _closest_point_on_triangle(state.contact_point, *points)
    closest_radius = _radial_distance(
        closest,
        state.contact_point,
        state.inward_direction,
    )
    return (
        closest_radius <= hole_radius_m
        and centroid_radius <= hole_radius_m + 0.5 * state.contact_radius_m
    )


def _triangle_touches_deformed_hole_disk(
    state: ExplicitShellState,
    triangle_index: int,
    hole_radius_m: float,
) -> bool:
    node_ids = state.triangle_nodes[triangle_index]
    points = [state.positions[node] for node in node_ids]
    candidates = list(points)
    candidates.append(_centroid(points))
    candidates.append(_closest_point_on_triangle(state.contact_point, *points))
    return any(
        _radial_distance(point, state.contact_point, state.inward_direction)
        <= hole_radius_m
        for point in candidates
    )


def _vertex_key(point: Vec3, tolerance_m: float) -> Tuple[int, int, int]:
    return tuple(round(value / tolerance_m) for value in point)


def _shell_source_triangles(
    component: AeroComponent,
    axis: Vec3,
    thickness_m: float,
) -> List[Triangle]:
    """Collapse opposite broad faces of a thin box-like shell to its midsurface.

    The CAD/STL input for a thin plate is often a closed thin solid with front
    face, back face, and narrow perimeter wall faces. The explicit shell model
    should operate on the plate midsurface, not on that full closed volume.
    Otherwise the perimeter wall faces produce corner spikes and unrealistically
    chunky plug fragments.
    """
    tolerance = max(component.lref * 1e-9, 1e-10)
    alignment_threshold = 0.8
    broad_faces: List[Tuple[Triangle, float]] = []
    for triangle in component.triangles:
        normal, a, b, c = triangle
        geometric_normal = v_cross(v_sub(b, a), v_sub(c, a))
        unit_normal = v_unit(
            geometric_normal,
            normal if v_norm(normal) > 0.0 else axis,
        )
        if abs(v_dot(unit_normal, axis)) < alignment_threshold:
            continue
        broad_faces.append((triangle, v_dot(_centroid((a, b, c)), axis)))

    if not broad_faces:
        return list(component.triangles)

    # A closed thin-solid STL has two broad faces.  The old approach collapsed
    # both, which only works when their triangle layouts are identical.  CAD
    # exporters frequently tessellate each face differently; keeping both then
    # produces two independently deforming sheets.  Choose one broad face and
    # project it to the physical midsurface instead.
    plane_values = [plane for _triangle, plane in broad_faces]
    lower_plane = min(plane_values)
    upper_plane = max(plane_values)
    plane_tolerance = max(
        0.25 * max(thickness_m, 0.0),
        component.lref * 1e-7,
        1e-10,
    )
    lower_faces = [
        triangle
        for triangle, plane in broad_faces
        if abs(plane - lower_plane) <= plane_tolerance
    ]
    upper_faces = [
        triangle
        for triangle, plane in broad_faces
        if abs(plane - upper_plane) <= plane_tolerance
    ]
    if upper_plane - lower_plane <= plane_tolerance:
        source_faces = [triangle for triangle, _plane in broad_faces]
        midsurface_plane = lower_plane
    elif len(upper_faces) >= len(lower_faces):
        source_faces = upper_faces
        midsurface_plane = 0.5 * (lower_plane + upper_plane)
    else:
        source_faces = lower_faces
        midsurface_plane = 0.5 * (lower_plane + upper_plane)

    orientation_sum = 0.0
    for normal, a, b, c in source_faces:
        geometric_normal = v_cross(v_sub(b, a), v_sub(c, a))
        unit_normal = v_unit(
            geometric_normal,
            normal if v_norm(normal) > 0.0 else axis,
        )
        orientation_sum += v_dot(unit_normal, axis)
    surface_normal = axis if orientation_sum >= 0.0 else v_mul(axis, -1.0)

    collapsed: Dict[
        Tuple[
            Tuple[int, int, int],
            Tuple[int, int, int],
            Tuple[int, int, int],
        ],
        Triangle,
    ] = {}
    selected: List[Triangle] = []
    for triangle in source_faces:
        _normal, a, b, c = triangle
        if v_dot(v_cross(v_sub(b, a), v_sub(c, a)), surface_normal) < 0.0:
            b, c = c, b
        face_plane = v_dot(_centroid((a, b, c)), axis)
        shift = v_mul(axis, midsurface_plane - face_plane)
        collapsed_points = (
            v_add(a, shift),
            v_add(b, shift),
            v_add(c, shift),
        )
        key = tuple(sorted(_vertex_key(point, tolerance) for point in collapsed_points))
        if key in collapsed:
            continue
        collapsed_triangle = (
            surface_normal,
            collapsed_points[0],
            collapsed_points[1],
            collapsed_points[2],
        )
        collapsed[key] = collapsed_triangle
        selected.append(collapsed_triangle)
    return selected or list(component.triangles)


def _subdivide_shell_triangle(triangle: Triangle) -> List[Triangle]:
    normal, a, b, c = triangle
    ab = v_mul(v_add(a, b), 0.5)
    bc = v_mul(v_add(b, c), 0.5)
    ca = v_mul(v_add(c, a), 0.5)
    children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
    return [
        (
            v_unit(v_cross(v_sub(child_b, child_a), v_sub(child_c, child_a)), normal),
            child_a,
            child_b,
            child_c,
        )
        for child_a, child_b, child_c in children
    ]


def _maximum_triangle_edge_m(triangle: Triangle) -> float:
    _normal, a, b, c = triangle
    return max(
        v_norm(v_sub(b, a)),
        v_norm(v_sub(c, b)),
        v_norm(v_sub(a, c)),
    )


def _resolve_small_perforation_mesh(
    triangles: Sequence[Triangle],
    perforation_radius_m: float,
    thickness_m: float,
) -> List[Triangle]:
    """Uniformly refine a coarse shell enough to represent a small opening.

    Uniform midpoint subdivision is deliberately used instead of a
    non-conforming local remesh. It preserves shared edges, so the classic
    triangular membrane FEM does not acquire artificial cracks or fixed nodes
    around the refinement boundary. The cap bounds collision runtime.
    """
    if perforation_radius_m <= 0.0 or not triangles:
        return list(triangles)
    # Keep every element edge no larger than the nominal hole diameter. The
    # failed-face intersection test then exposes the opening without an
    # expensive global remesh merely to draw a slightly rounder rim.
    target_edge_m = max(
        2.0 * perforation_radius_m,
        8.0e-7,
    )
    maximum_triangles = 16384
    refined = list(triangles)
    while (
        len(refined) * 4 <= maximum_triangles
        and max(_maximum_triangle_edge_m(triangle) for triangle in refined)
        > target_edge_m
    ):
        refined = [
            child
            for triangle in refined
            for child in _subdivide_shell_triangle(triangle)
        ]
    return refined


def build_explicit_shell_state(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    contact_radius_m: float,
    young_modulus_pa: float,
    thickness_m: float,
    poisson_ratio: float,
    yield_strength_pa: float,
    failure_strain: float,
    damping_ratio: float,
    cfl: float,
    max_substeps: int,
    displacement_limit_m: float,
    perforation_radius_m: float = 0.0,
) -> ExplicitShellState:
    """Build a lumped-mass explicit triangular shell from an STL surface.

    Every triangle keeps its own nodes. Coincident nodes are welded during
    integration, which permits the weld to separate cleanly when a perforation
    classifies one neighbouring face as plug material and the other as rim.
    """
    axis = v_unit(inward_direction)
    density = max(component.material.density_kg_m3, 1.0)
    thickness = max(thickness_m, 1e-9)
    young = max(young_modulus_pa, 1.0)
    source_triangles = _resolve_small_perforation_mesh(
        _shell_source_triangles(component, axis, thickness),
        perforation_radius_m,
        thickness,
    )
    positions: List[Vec3] = []
    masses: List[float] = []
    triangle_nodes: List[Tuple[int, int, int]] = []
    triangle_centroids: List[Vec3] = []
    triangle_normals: List[Vec3] = []
    triangle_masses: List[float] = []
    edges: List[ShellEdge] = []
    bending_springs: List[ShellBendingSpring] = []
    membrane_elements: List[ShellMembraneElement] = []
    triangle_points: List[Tuple[Vec3, Vec3, Vec3]] = []

    total_surface_area = sum(
        _triangle_area(triangle) for triangle in source_triangles
    )
    for normal, a, b, c in source_triangles:
        node_ids = (len(positions), len(positions) + 1, len(positions) + 2)
        points = (a, b, c)
        positions.extend(points)
        triangle_nodes.append(node_ids)
        triangle_points.append(points)
        triangle_centroids.append(_centroid(points))
        triangle_normals.append(
            v_unit(v_cross(v_sub(b, a), v_sub(c, a)), normal)
        )
        area = max(_triangle_area((normal, a, b, c)), 1e-18)
        basis_x = v_unit(v_sub(b, a), (1.0, 0.0, 0.0))
        basis_y = v_unit(
            v_cross(triangle_normals[-1], basis_x),
            (0.0, 1.0, 0.0),
        )
        local_points = (
            (0.0, 0.0),
            (v_dot(v_sub(b, a), basis_x), 0.0),
            (v_dot(v_sub(c, a), basis_x), v_dot(v_sub(c, a), basis_y)),
        )
        x1, y1 = local_points[0]
        x2, y2 = local_points[1]
        x3, y3 = local_points[2]
        membrane_elements.append(
            ShellMembraneElement(
                nodes=node_ids,
                area_m2=area,
                basis_x=basis_x,
                basis_y=basis_y,
                b_coefficients=(y2 - y3, y3 - y1, y1 - y2),
                c_coefficients=(x3 - x2, x1 - x3, x2 - x1),
            )
        )
        node_mass = density * thickness * area / 3.0
        masses.extend((node_mass, node_mass, node_mass))
        triangle_masses.append(
            component.mass * area / max(total_surface_area, 1e-18)
        )
        for local_a, local_b, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            edge_vector = v_sub(points[local_b], points[local_a])
            edge_length = max(v_norm(edge_vector), 1e-12)
            altitude = 2.0 * area / edge_length
            tributary_width = altitude / 3.0
            stiffness = young * thickness * tributary_width / edge_length
            edges.append(
                ShellEdge(
                    node_ids[local_a],
                    node_ids[local_b],
                    edge_length,
                    max(stiffness, 1e-6),
                )
            )

    tolerance = max(component.lref * 1e-9, 1e-10)
    groups_by_key: Dict[Tuple[int, int, int], List[int]] = {}
    for node_index, point in enumerate(positions):
        groups_by_key.setdefault(_vertex_key(point, tolerance), []).append(node_index)
    weld_groups = [group for group in groups_by_key.values() if len(group) > 1]
    canonical_by_node = {
        node_index: _vertex_key(point, tolerance)
        for node_index, point in enumerate(positions)
    }

    shared_edges: Dict[
        Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        Tuple[int, int, float, float],
    ] = {}
    flexural_rigidity = (
        young
        * thickness ** 3
        / max(12.0 * (1.0 - poisson_ratio * poisson_ratio), 1e-9)
    )
    for triangle_index, node_ids in enumerate(triangle_nodes):
        points = triangle_points[triangle_index]
        for local_a, local_b, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            key_a = _vertex_key(points[local_a], tolerance)
            key_b = _vertex_key(points[local_b], tolerance)
            edge_key = tuple(sorted((key_a, key_b)))
            edge_length = v_norm(v_sub(points[local_b], points[local_a]))
            opposite_point = points[opposite]
            closest_edge_point = _closest_point_on_segment(
                opposite_point,
                points[local_a],
                points[local_b],
            )
            edge_height = max(
                v_norm(v_sub(opposite_point, closest_edge_point)),
                thickness,
                1e-9,
            )
            record = shared_edges.get(edge_key)
            if record is None:
                shared_edges[edge_key] = (
                    triangle_index,
                    node_ids[opposite],
                    edge_length,
                    edge_height,
                )
                continue
            _other_triangle, other_opposite_node, other_edge_length, other_edge_height = record
            rest_length = max(
                v_norm(
                    v_sub(
                        positions[node_ids[opposite]],
                        positions[other_opposite_node],
                    )
                ),
                1e-9,
            )
            average_height = max(
                0.5 * (edge_height + other_edge_height),
                thickness,
                1e-9,
            )
            stiffness = max(
                flexural_rigidity
                * max(edge_length, other_edge_length, 1e-9)
                / (average_height * average_height),
                1e-6,
            )
            bending_springs.append(
                ShellBendingSpring(
                    other_opposite_node,
                    node_ids[opposite],
                    rest_length,
                    stiffness,
                )
            )

    fixed_nodes: Set[int] = set()
    if component.is_assembly_anchor:
        shared_edge_counts: Dict[
            Tuple[Tuple[int, int, int], Tuple[int, int, int]],
            int,
        ] = {}
        for node_ids in triangle_nodes:
            canonical_nodes = tuple(canonical_by_node[node] for node in node_ids)
            for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
                edge_key = tuple(
                    sorted(
                        (canonical_nodes[local_a], canonical_nodes[local_b])
                    )
                )
                shared_edge_counts[edge_key] = shared_edge_counts.get(edge_key, 0) + 1
        boundary_vertices: Set[Tuple[int, int, int]] = set()
        for edge_key, count in shared_edge_counts.items():
            if count == 1:
                boundary_vertices.update(edge_key)
        for node_index, canonical_vertex in canonical_by_node.items():
            if canonical_vertex in boundary_vertices:
                fixed_nodes.add(node_index)

    minimum_period = math.inf
    for edge in edges:
        effective_mass = min(masses[edge.node_a], masses[edge.node_b])
        minimum_period = min(
            minimum_period,
            math.sqrt(max(effective_mass, 1e-18) / edge.stiffness_n_m),
        )
    for spring in bending_springs:
        effective_mass = min(masses[spring.node_a], masses[spring.node_b])
        minimum_period = min(
            minimum_period,
            math.sqrt(max(effective_mass, 1e-18) / spring.stiffness_n_m),
        )
    membrane_stiffness = max(young * thickness, 1e-6)
    for element in membrane_elements:
        for node_index in element.nodes:
            minimum_period = min(
                minimum_period,
                math.sqrt(
                    max(masses[node_index], 1e-18) / membrane_stiffness
                ),
            )
    stable_dt = max(cfl * minimum_period, 1e-9) if math.isfinite(minimum_period) else 1e-6

    return ExplicitShellState(
        positions=list(positions),
        reference_positions=list(positions),
        velocities=[(0.0, 0.0, 0.0) for _point in positions],
        masses_kg=masses,
        triangle_nodes=triangle_nodes,
        triangle_reference_centroids=triangle_centroids,
        triangle_normals=triangle_normals,
        triangle_masses_kg=triangle_masses,
        edges=edges,
        bending_springs=bending_springs,
        membrane_elements=membrane_elements,
        weld_groups=weld_groups,
        fixed_nodes=fixed_nodes,
        contact_point=contact_point,
        inward_direction=axis,
        contact_radius_m=max(contact_radius_m, 1e-9),
        young_modulus_pa=young,
        thickness_m=thickness,
        poisson_ratio=max(0.0, min(poisson_ratio, 0.49)),
        yield_strain=max(yield_strength_pa / young, 0.0),
        failure_strain=max(failure_strain, yield_strength_pa / young),
        damping_ratio=max(damping_ratio, 0.0),
        cfl=cfl,
        max_substeps=max(max_substeps, 1),
        displacement_limit_m=max(displacement_limit_m, 1e-9),
        stable_dt_s=stable_dt,
    )


def build_hybrid_shell_collision_state(
    component: AeroComponent,
    contact_point: Vec3,
    axis: Vec3,
    contact_radius_m: float,
    young_modulus_pa: float,
    thickness_m: float,
    poisson_ratio: float,
    yield_strength_pa: float,
    failure_strain: float,
    damping_ratio: float,
    cfl: float,
    max_substeps: int,
    displacement_limit_m: float,
    perforation_radius_m: float = 0.0,
) -> HybridShellCollisionState:
    shell_state = build_explicit_shell_state(
        component,
        contact_point,
        axis,
        contact_radius_m,
        young_modulus_pa,
        thickness_m,
        poisson_ratio,
        yield_strength_pa,
        failure_strain,
        damping_ratio,
        cfl,
        max_substeps,
        displacement_limit_m,
        perforation_radius_m,
    )
    shell_state.solver_backend = "explicit_shell"
    # The imported target already carries its physical front/back faces.  Do
    # not extrude that triangulated surface a second time: doing so creates a
    # visibly thicker duplicate plate after impact.  Detached fragments are
    # thickened separately from their material thickness.
    shell_state.render_as_midsurface = True
    return HybridShellCollisionState(shell_state=shell_state)


def hybrid_fragment_components(state: HybridShellCollisionState) -> List[AeroComponent]:
    return [fragment.component for fragment in state.fragment_bodies]


def component_is_thin_for_solid_fem(component: AeroComponent) -> bool:
    declared_thickness = component.material.thickness_m
    if (
        declared_thickness is not None
        and declared_thickness > 0.0
        and declared_thickness / max(component.lref, 1e-12) < 0.1
    ):
        return True
    points = [point for triangle in component.triangles for point in triangle[1:]]
    if not points:
        return True
    extents = (
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
        max(point[2] for point in points) - min(point[2] for point in points),
    )
    positive = [extent for extent in extents if extent > 1e-10]
    if len(positive) < 3:
        return True
    return min(positive) / max(positive) < 0.08


def build_hybrid_fem_mpm_collision_state(
    component: AeroComponent,
    young_modulus_pa: float,
    poisson_ratio: float,
    yield_strength_pa: float,
    failure_strain: float,
    cfl: float,
    max_substeps: int,
) -> HybridFEMMPMCollisionState:
    """Create a star-conforming tetra mesh whose boundary is the CAD surface.

    A surface triangle and the component centre form one tetrahedron.  This is
    deterministic, preserves the exact visible boundary vertices and is robust
    for the convex or locally star-shaped CAD solids normally used as impactors.
    Thin targets are intentionally routed to the shell backend by the caller.
    """
    if not component.triangles:
        raise ValueError(f"cannot build solid FEM state for empty component {component.patch}")
    tolerance = max(component.lref * 1e-9, 1e-10)
    vertex_by_key: Dict[Tuple[int, int, int], int] = {}
    surface_positions: List[Vec3] = []
    triangle_vertices: List[Tuple[int, int, int]] = []
    for triangle in component.triangles:
        node_ids: List[int] = []
        for point in triangle[1:]:
            key = _vertex_key(point, tolerance)
            node_index = vertex_by_key.get(key)
            if node_index is None:
                node_index = len(surface_positions)
                vertex_by_key[key] = node_index
                surface_positions.append(point)
            node_ids.append(node_index)
        triangle_vertices.append(tuple(node_ids))  # type: ignore[arg-type]
    edge_counts: Dict[Tuple[int, int], int] = {}
    for a, b, c in triangle_vertices:
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(
            f"component {component.patch} is not a watertight solid; using fallback"
        )
    centre = component.cofr
    if not all(math.isfinite(value) for value in centre):
        centre = _centroid(surface_positions)
    positions = [centre, *surface_positions]
    surface_triangle_nodes = [
        (a + 1, b + 1, c + 1) for a, b, c in triangle_vertices
    ]
    raw_volumes = [
        _tetra_volume_from_nodes(positions, (0, *nodes))
        for nodes in surface_triangle_nodes
    ]
    total_volume = sum(raw_volumes)
    if total_volume <= 1e-18:
        raise ValueError(f"component {component.patch} has no usable solid volume")
    closed_volume = estimate_closed_mesh_volume(component.triangles)
    if closed_volume <= 1e-18 or total_volume > 1.05 * closed_volume:
        raise ValueError(
            f"component {component.patch} is not star-shaped about its mass centre; "
            "using fallback"
        )
    elements = []
    for nodes, volume in zip(surface_triangle_nodes, raw_volumes):
        if volume <= 1e-18:
            continue
        element_mass = component.mass * volume / total_volume
        elements.append(
            make_tetra_element(
                positions,
                (0, *nodes),
                young_modulus_pa,
                poisson_ratio,
                yield_strength_pa,
                failure_strain,
                element_mass,
            )
        )
    if not elements:
        raise ValueError(f"component {component.patch} produced no valid tetrahedra")
    # Keep surface-to-element indices in the same order, omitting degenerate
    # surface triangles from both arrays.
    valid_surface_nodes = [
        nodes
        for nodes, volume in zip(surface_triangle_nodes, raw_volumes)
        if volume > 1e-18
    ]
    masses = [0.0] * len(positions)
    for element in elements:
        for node in element.nodes:
            masses[node] += 0.25 * element.mass_kg
    velocities = []
    for position in positions:
        rotational = v_cross(component.angular_velocity, v_sub(position, component.cofr))
        velocities.append(v_add(component.linear_velocity, rotational))
    fixed_nodes = {0} if component.is_assembly_anchor else set()
    mean_element_size = (
        sum(element.rest_volume_m3 for element in elements) / len(elements)
    ) ** (1.0 / 3.0)
    solid_state = HybridFEMMPMState(
        positions=positions,
        reference_positions=list(positions),
        velocities=velocities,
        masses_kg=masses,
        elements=elements,
        fixed_nodes=fixed_nodes,
        surface_triangle_nodes=valid_surface_nodes,
        surface_element_indices=list(range(len(elements))),
        cfl=max(0.05, min(cfl, 0.8)),
        max_substeps=max(max_substeps, 1),
        mpm_cell_size_m=max(mean_element_size, 1e-6),
    )
    return HybridFEMMPMCollisionState(
        solid_state=solid_state,
        reference_mass_kg=component.mass,
    )


def _tetra_volume_from_nodes(
    positions: Sequence[Vec3],
    nodes: Tuple[int, int, int, int],
) -> float:
    a, b, c, d = (positions[node] for node in nodes)
    return abs(v_dot(v_sub(b, a), v_cross(v_sub(c, a), v_sub(d, a)))) / 6.0


def apply_fem_impact_energy(
    state: HybridFEMMPMCollisionState,
    contact_point: Vec3,
    inward_direction: Vec3,
    contact_radius_m: float,
    absorbed_energy_j: float,
    energy_fraction: float,
) -> None:
    available = max(absorbed_energy_j, 0.0) * max(0.0, min(energy_fraction, 1.0))
    if available <= 0.0:
        return
    solid = state.solid_state
    axis = v_unit(inward_direction)
    radius = max(2.5 * contact_radius_m, solid.mpm_cell_size_m, 1e-9)
    weights: List[float] = []
    free_mass = 0.0
    weighted_mass = 0.0
    for node, position in enumerate(solid.positions):
        if node in solid.fixed_nodes:
            weight = 0.0
        else:
            distance = v_norm(v_sub(position, contact_point))
            weight = max(0.0, 1.0 - distance / radius) ** 2
        weights.append(weight)
        if node not in solid.fixed_nodes:
            free_mass += solid.masses_kg[node]
            weighted_mass += solid.masses_kg[node] * weight
    mean_weight = weighted_mass / max(free_mass, 1e-18)
    centred_weights = [
        0.0 if node in solid.fixed_nodes else weight - mean_weight
        for node, weight in enumerate(weights)
    ]
    effective_mass = sum(
        solid.masses_kg[node] * weight * weight
        for node, weight in enumerate(centred_weights)
    )
    if effective_mass <= 1e-18:
        return
    speed_scale = math.sqrt(2.0 * available / effective_mass)
    for node, weight in enumerate(centred_weights):
        if abs(weight) <= 1e-18:
            continue
        solid.velocities[node] = v_add(
            solid.velocities[node],
            v_mul(axis, speed_scale * weight),
        )


def update_fem_perforation(
    state: HybridFEMMPMCollisionState,
    contact_point: Vec3,
    inward_direction: Vec3,
    hole_radius_m: float,
) -> int:
    solid = state.solid_state
    axis = v_unit(inward_direction)
    newly_failed = 0
    for triangle_nodes, element_index in zip(
        solid.surface_triangle_nodes,
        solid.surface_element_indices,
    ):
        element = solid.elements[element_index]
        if element.failed:
            continue
        centroid = _centroid([solid.reference_positions[node] for node in triangle_nodes])
        if _radial_distance(centroid, contact_point, axis) <= hole_radius_m:
            element.failed = True
            newly_failed += 1
    return newly_failed


def _solid_fragment_triangle(
    state: HybridFEMMPMCollisionState,
    element_index: int,
    triangle_nodes: Tuple[int, int, int],
) -> Optional[Triangle]:
    particles = {
        particle.source_node: particle
        for particle in state.solid_state.particles
        if particle.source_element == element_index
    }
    if not all(node in particles for node in triangle_nodes):
        return None
    a, b, c = (particles[node].position for node in triangle_nodes)
    normal = v_unit(v_cross(v_sub(b, a), v_sub(c, a)), (1.0, 0.0, 0.0))
    return normal, a, b, c


def sync_hybrid_fem_mpm_fragments(
    component: AeroComponent,
    state: HybridFEMMPMCollisionState,
) -> Tuple[int, float]:
    solid = state.solid_state
    created = 0
    emitted_mass = 0.0
    existing_by_element = {
        next(iter(fragment.triangle_indices)): fragment
        for fragment in state.fragment_bodies
        if fragment.triangle_indices
    }
    for triangle_nodes, element_index in zip(
        solid.surface_triangle_nodes,
        solid.surface_element_indices,
    ):
        element = solid.elements[element_index]
        if not element.transferred:
            continue
        triangle = _solid_fragment_triangle(state, element_index, triangle_nodes)
        if triangle is None:
            continue
        particles = [
            particle
            for particle in solid.particles
            if particle.source_element == element_index
        ]
        fragment_mass = sum(particle.mass_kg for particle in particles)
        momentum = (0.0, 0.0, 0.0)
        for particle in particles:
            momentum = v_add(momentum, v_mul(particle.velocity, particle.mass_kg))
        velocity = v_mul(momentum, 1.0 / max(fragment_mass, 1e-18))
        existing = existing_by_element.get(element_index)
        if existing is not None:
            existing.component.triangles = [triangle]
            existing.component.linear_velocity = velocity
            existing.component.cofr = _centroid(list(triangle[1:]))
            continue
        centroid = _centroid(list(triangle[1:]))
        patch = f"{component.patch}_fem_fragment_{state.next_fragment_id}"
        fragment_component = AeroComponent(
            name=f"{component.name} FEM fragment {state.next_fragment_id}",
            patch=patch,
            triangles=[triangle],
            cofr=centroid,
            lref=max(v_norm(v_sub(triangle[2], triangle[1])), 1e-9),
            aref=_triangle_area(triangle),
            freedom=MotionFreedom(
                translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                mate_type="COLLISION_FRAGMENT",
                source="hybrid-fem-mpm-fragment",
            ),
            material=component.material,
            mass=fragment_mass,
            inertia=max(fragment_mass * component.lref * component.lref / 12.0, 1e-12),
            linear_velocity=velocity,
            collision_family=component.collision_family or component.patch,
            collision_fragment_parent_state=state,
            collision_fragment_source_element=element_index,
        )
        state.fragment_bodies.append(
            DetachedFragmentBody(
                component=fragment_component,
                triangle_indices={element_index},
                mass_kg=fragment_mass,
                source="hybrid-fem-mpm-fragment",
            )
        )
        state.emitted_elements.add(element_index)
        state.next_fragment_id += 1
        created += 1
        emitted_mass += fragment_mass
    if emitted_mass > 0.0:
        previous_mass = component.mass
        component.mass = max(0.0, component.mass - emitted_mass)
        component.inertia *= component.mass / max(previous_mass, 1e-18)
    return created, emitted_mass


def refresh_hybrid_fem_mpm_geometry(
    component: AeroComponent,
    state: HybridFEMMPMCollisionState,
) -> float:
    solid = state.solid_state
    triangles: List[Triangle] = []
    for triangle_nodes, element_index in zip(
        solid.surface_triangle_nodes,
        solid.surface_element_indices,
    ):
        if solid.elements[element_index].transferred:
            continue
        a, b, c = (solid.positions[node] for node in triangle_nodes)
        normal = v_unit(v_cross(v_sub(b, a), v_sub(c, a)), (1.0, 0.0, 0.0))
        triangles.append((normal, a, b, c))
    if triangles:
        component.triangles = triangles
    maximum = state.max_displacement_m
    component.deformation_max_m = max(component.deformation_max_m, maximum)
    return maximum


def advance_hybrid_fem_mpm_collision(
    component: AeroComponent,
    state: HybridFEMMPMCollisionState,
    dt_s: float,
) -> Tuple[float, int, float]:
    state.last_audit = advance_fem(state.solid_state, dt_s)
    created, emitted_mass = sync_hybrid_fem_mpm_fragments(component, state)
    deformation = refresh_hybrid_fem_mpm_geometry(component, state)
    return deformation, created, emitted_mass


def fem_surface_von_mises_stress_pa(
    state: HybridFEMMPMCollisionState,
) -> List[float]:
    return [
        von_mises_stress(state.solid_state.elements[element_index].stress)
        for element_index in state.solid_state.surface_element_indices
        if not state.solid_state.elements[element_index].transferred
    ]


def apply_shell_impact_energy(
    state: ExplicitShellState,
    absorbed_energy_j: float,
    energy_fraction: float,
) -> None:
    """Distribute impact energy to local shell nodes with a smooth footprint."""
    available_energy = max(absorbed_energy_j, 0.0) * max(0.0, min(1.0, energy_fraction))
    if available_energy <= 0.0:
        return
    affected_radius = max(2.5 * state.contact_radius_m, 1e-9)
    weights: List[float] = []
    weighted_mass = 0.0
    for node_index, point in enumerate(state.positions):
        if node_index in state.fixed_nodes:
            weight = 0.0
        else:
            radial = _radial_distance(
                point,
                state.contact_point,
                state.inward_direction,
            )
            fraction = max(0.0, 1.0 - radial / affected_radius)
            weight = fraction * fraction * (3.0 - 2.0 * fraction)
        weights.append(weight)
        weighted_mass += state.masses_kg[node_index] * weight * weight
    if weighted_mass <= 1e-18:
        return
    velocity_scale = math.sqrt(2.0 * available_energy / weighted_mass)
    for node_index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        state.velocities[node_index] = v_add(
            state.velocities[node_index],
            v_mul(state.inward_direction, velocity_scale * weight),
        )


def apply_shell_contact_work(
    state: ExplicitShellState,
    contact_point: Vec3,
    inward_direction: Vec3,
    contact_radius_m: float,
    work_j: float,
    dt_s: float,
) -> float:
    """Apply bounded continuing contact work to the attached shell.

    A projectile does not stop acting on the target at the first contact
    sample.  This distributes a finite residual work budget into the local
    footprint and nearby plate so the dent/perforation evolves over subsequent
    frames while preserving an explicit energy accounting path.
    """
    work = max(work_j, 0.0)
    dt = max(dt_s, 0.0)
    if work <= 0.0 or dt <= 0.0:
        return 0.0

    direction = v_unit(inward_direction, state.inward_direction)
    radius = max(contact_radius_m, state.contact_radius_m, state.thickness_m)
    affected_radius = max(3.5 * radius, 8.0 * state.thickness_m)
    fragment_nodes = _emitted_fragment_nodes(state)
    weights: List[float] = []
    weighted_mass = 0.0
    for node_index, point in enumerate(state.positions):
        if node_index in state.fixed_nodes or node_index in fragment_nodes:
            weight = 0.0
        else:
            radial = _radial_distance(point, contact_point, direction)
            fraction = max(0.0, 1.0 - radial / affected_radius)
            weight = fraction * fraction * (3.0 - 2.0 * fraction)
        weights.append(weight)
        weighted_mass += state.masses_kg[node_index] * weight * weight
    if weighted_mass <= 1e-18:
        return 0.0

    velocity_scale = math.sqrt(2.0 * work / weighted_mass)
    max_applied = 0.0
    for node_index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        velocity_increment = v_mul(direction, velocity_scale * weight)
        displacement_increment = v_mul(velocity_increment, dt)
        displacement_magnitude = v_norm(displacement_increment)
        if displacement_magnitude > state.displacement_limit_m:
            displacement_increment = v_mul(
                displacement_increment,
                state.displacement_limit_m / displacement_magnitude,
            )
            displacement_magnitude = state.displacement_limit_m
        state.velocities[node_index] = v_add(
            state.velocities[node_index],
            velocity_increment,
        )
        state.positions[node_index] = v_add(
            state.positions[node_index],
            displacement_increment,
        )
        plastic_increment = v_mul(displacement_increment, 0.35)
        state.reference_positions[node_index] = v_add(
            state.reference_positions[node_index],
            plastic_increment,
        )
        state.permanent_displacement_m = max(
            state.permanent_displacement_m,
            min(
                v_norm(
                    v_sub(
                        state.positions[node_index],
                        state.reference_positions[node_index],
                    )
                ),
                state.displacement_limit_m,
            ),
        )
        max_applied = max(max_applied, displacement_magnitude)
    _apply_displacement_limits(state)
    return max_applied


def _plug_direction(state: ExplicitShellState, triangle_index: int) -> Vec3:
    normal = state.triangle_normals[triangle_index]
    return normal if v_dot(normal, state.inward_direction) >= 0.0 else v_mul(normal, -1.0)


def _material_ductility_ratio(state: ExplicitShellState) -> float:
    """Return a bounded strain-to-yield ratio for fragment sizing.

    The BOM provides elastic modulus and yield/ultimate strength.  The loader
    converts those into yield/failure strain.  A higher failure/yield ratio is
    treated as more ductile material, so the solver removes a tighter plug
    instead of spraying many small fragments.
    """
    yield_strain = max(state.yield_strain, 1e-9)
    failure_strain = max(state.failure_strain, yield_strain)
    return max(1.0, min(25.0, failure_strain / yield_strain))


def _material_brittleness_ratio(state: ExplicitShellState) -> float:
    """Return a bounded brittleness multiplier from BOM-derived strains.

    Low failure strain relative to yield strain indicates a material that
    releases damage by cracking rather than by stretching plastically.
    """
    ductility = _material_ductility_ratio(state)
    return max(1.0, min(6.0, math.sqrt(25.0 / ductility)))


def _fragment_detachment_radius_m(
    state: ExplicitShellState,
    hole_radius_m: float,
) -> float:
    brittleness = _material_brittleness_ratio(state)
    # The energy/perforation model has already computed the physical hole
    # radius.  A ductile plate (ABS, metals, most polymers) should therefore
    # eject only the plug inside that radius.  The previous 2*contact-radius
    # and 4*thickness lower bounds could exceed the hole by several times and
    # detach two enormous plate sectors.  Brittle materials may crack/chip
    # beyond the hole, but keep that extension bounded and material-driven.
    brittle_extension = max(0.0, brittleness - 1.6)
    radius_factor = 1.0 + min(1.5, 0.45 * brittle_extension)
    return max(hole_radius_m, 1.0e-9) * radius_factor


def update_shell_perforation(
    state: ExplicitShellState,
    hole_radius_m: float,
    radial_growth_m: float,
    response_dt_s: float,
) -> None:
    """Detach newly perforated faces and give the retained plug an inward speed."""
    if hole_radius_m <= 0.0:
        return
    state.current_hole_radius_m = max(state.current_hole_radius_m, hole_radius_m)
    detachment_radius_m = _fragment_detachment_radius_m(state, hole_radius_m)
    newly_detached: List[int] = []
    for triangle_index, centroid in enumerate(state.triangle_reference_centroids):
        if (
            triangle_index not in state.plug_triangles
            and _triangle_intersects_hole_disk(state, triangle_index, detachment_radius_m)
        ):
            state.plug_triangles.add(triangle_index)
            newly_detached.append(triangle_index)
    # A coarse shell can have no centroid inside a small initial hole even
    # though the impact point is on that shell face.  Leaving every face in
    # place makes the perforation invisible in the exported surface.  Detach
    # the nearest non-fixed face as a conservative finite-element element
    # failure fallback; its mass is retained in the fragment set below.
    if not newly_detached and radial_growth_m > 0.0:
        candidates = [
            (index, _radial_distance(centroid, state.contact_point, state.inward_direction))
            for index, centroid in enumerate(state.triangle_reference_centroids)
            if index not in state.plug_triangles
            and not any(node in state.fixed_nodes for node in state.triangle_nodes[index])
        ]
        if candidates:
            nearest_index, nearest_radius = min(candidates, key=lambda item: item[1])
            # Only use the fallback for an impact-scale opening; a zero-radius
            # numerical update must not fracture an unrelated shell element.
            if nearest_radius <= max(
                2.0 * detachment_radius_m,
                2.0 * state.contact_radius_m,
                state.thickness_m,
            ):
                state.plug_triangles.add(nearest_index)
                newly_detached.append(nearest_index)
    if not newly_detached:
        return
    brittleness = _material_brittleness_ratio(state)
    minimum_detached_triangles = 1 if brittleness >= 1.75 else (3 if state.emitted_triangles else 1)
    coherent_detached: List[int] = []
    for group in _connected_fragment_groups(state, set(newly_detached)):
        if len(group) < minimum_detached_triangles:
            for triangle_index in group:
                state.plug_triangles.discard(triangle_index)
            continue
        coherent_detached.extend(sorted(group))
    newly_detached = coherent_detached
    if not newly_detached:
        return
    # Never remove the complete midsurface from a coarse shell.  The detached
    # elements remain as mass-conserving fragments, while at least one element
    # must remain attached to carry the target's structural response.
    attached_candidates = [
        index
        for index in range(len(state.triangle_nodes))
        if index not in state.plug_triangles
    ]
    if not attached_candidates and len(newly_detached) > 1:
        keep_index = max(
            newly_detached,
            key=lambda index: _radial_distance(
                state.triangle_reference_centroids[index],
                state.contact_point,
                state.inward_direction,
            ),
        )
        state.plug_triangles.discard(keep_index)
        newly_detached.remove(keep_index)
    if not newly_detached:
        return
    # ``radial_growth_m`` is the change over the damage-response interval, not
    # over one structural substep.  Dividing it by the usually much smaller
    # CFL step gives the plug an artificial launch speed that is then applied
    # for the entire response interval.
    separation_speed = radial_growth_m / max(response_dt_s, 1e-9)
    launch_axis = state.inward_direction
    for triangle_index in newly_detached:
        initial_separation = max(
            state.thickness_m,
            0.25 * radial_growth_m,
            0.10 * state.contact_radius_m,
        )
        node_ids = state.triangle_nodes[triangle_index]
        total_mass = sum(state.masses_kg[node] for node in node_ids)
        inherited_velocity = (0.0, 0.0, 0.0)
        for node_index in node_ids:
            mass_fraction = state.masses_kg[node_index] / max(total_mass, 1e-18)
            inherited_velocity = v_add(
                inherited_velocity,
                v_mul(state.velocities[node_index], mass_fraction),
            )
        inherited_axial_speed = max(0.0, v_dot(inherited_velocity, launch_axis))
        inherited_tangential = v_sub(
            inherited_velocity,
            v_mul(launch_axis, v_dot(inherited_velocity, launch_axis)),
        )
        inherited_tangential_speed = v_norm(inherited_tangential)
        max_tangential_speed = 0.20 * max(
            separation_speed,
            inherited_axial_speed,
            1e-9,
        )
        if inherited_tangential_speed > max_tangential_speed:
            inherited_tangential = v_mul(
                inherited_tangential,
                max_tangential_speed / inherited_tangential_speed,
            )
        launch_velocity = v_add(
            v_mul(launch_axis, inherited_axial_speed + separation_speed),
            inherited_tangential,
        )
        for node_index in node_ids:
            state.positions[node_index] = v_add(
                state.positions[node_index],
                v_mul(launch_axis, initial_separation),
            )
            state.velocities[node_index] = launch_velocity

    fragment_nodes = _emitted_fragment_nodes(state)
    rim_outer_radius = hole_radius_m + max(2.5 * state.contact_radius_m, state.thickness_m)
    rim_displacement = min(
        state.displacement_limit_m,
        max(
            2.0 * state.thickness_m,
            0.35 * hole_radius_m,
            1.25 * state.contact_radius_m,
        ),
    )
    rim_speed = 0.35 * separation_speed
    plastically_deformed_nodes: Set[int] = set()
    for node_index, point in enumerate(state.positions):
        if node_index in fragment_nodes or node_index in state.fixed_nodes:
            continue
        radial = _radial_distance(
            point,
            state.contact_point,
            state.inward_direction,
        )
        if radial < hole_radius_m or radial > rim_outer_radius:
            continue
        fraction = 1.0 - (radial - hole_radius_m) / max(
            rim_outer_radius - hole_radius_m,
            1e-12,
        )
        weight = fraction * fraction * (3.0 - 2.0 * fraction)
        if weight <= 0.0:
            continue
        displacement = v_mul(state.inward_direction, rim_displacement * weight)
        state.positions[node_index] = v_add(state.positions[node_index], displacement)
        state.reference_positions[node_index] = v_add(
            state.reference_positions[node_index],
            displacement,
        )
        state.permanent_displacement_m = max(
            state.permanent_displacement_m,
            v_norm(displacement),
        )
        plastically_deformed_nodes.add(node_index)
        state.velocities[node_index] = v_add(
            state.velocities[node_index],
            v_mul(state.inward_direction, rim_speed * weight),
        )
    if plastically_deformed_nodes:
        for edge in state.edges:
            if (
                edge.failed
                or edge.node_a in fragment_nodes
                or edge.node_b in fragment_nodes
                or (
                    edge.node_a not in plastically_deformed_nodes
                    and edge.node_b not in plastically_deformed_nodes
                )
            ):
                continue
            edge.rest_length_m = max(
                v_norm(
                    v_sub(
                        state.reference_positions[edge.node_b],
                        state.reference_positions[edge.node_a],
                    )
                ),
                1e-12,
            )


def emit_shell_fragments(state: ExplicitShellState) -> Tuple[int, float]:
    """Detach failed plug elements from the target surface for visual export."""
    new_fragments = state.plug_triangles - state.emitted_triangles
    state.emitted_triangles.update(new_fragments)
    emitted_mass = sum(state.triangle_masses_kg[index] for index in new_fragments)
    state.emitted_fragment_mass_kg += emitted_mass
    return len(new_fragments), emitted_mass


def _connected_fragment_groups(
    state: ExplicitShellState,
    triangle_indices: Set[int],
) -> List[Set[int]]:
    if not triangle_indices:
        return []
    adjacency: Dict[int, Set[int]] = {index: set() for index in triangle_indices}
    triangles_by_vertex: Dict[Tuple[float, float, float], List[int]] = {}
    for triangle_index in triangle_indices:
        for node in state.triangle_nodes[triangle_index]:
            triangles_by_vertex.setdefault(
                state.reference_positions[node],
                [],
            ).append(triangle_index)
    for connected in triangles_by_vertex.values():
        for triangle_index in connected:
            adjacency[triangle_index].update(other for other in connected if other != triangle_index)
    groups: List[Set[int]] = []
    remaining = set(triangle_indices)
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    group.add(neighbour)
                    stack.append(neighbour)
        groups.append(group)
    return groups


def _triangle_reference_vertices(
    state: ExplicitShellState,
    triangle_indices: Set[int],
) -> Set[Tuple[float, float, float]]:
    vertices: Set[Tuple[float, float, float]] = set()
    for triangle_index in triangle_indices:
        for node in state.triangle_nodes[triangle_index]:
            vertices.add(state.reference_positions[node])
    return vertices


def _fragment_transverse_span_m(
    state: ExplicitShellState,
    triangle_indices: Set[int],
) -> float:
    radial_points: List[Vec3] = []
    for triangle_index in triangle_indices:
        for node in state.triangle_nodes[triangle_index]:
            point = state.reference_positions[node]
            relative = v_sub(point, state.contact_point)
            axial = v_dot(relative, state.inward_direction)
            radial_points.append(
                v_sub(relative, v_mul(state.inward_direction, axial))
            )
    max_span = 0.0
    for index, point in enumerate(radial_points):
        for other in radial_points[index + 1:]:
            max_span = max(max_span, v_norm(v_sub(point, other)))
    return max_span


def _fragment_merge_span_limit_m(state: ExplicitShellState) -> float:
    brittleness = _material_brittleness_ratio(state)
    if brittleness < 1.75:
        # A ductile perforation ejects one coherent plug.  Allow the complete
        # resolved hole disk to remain one body even when its boundary cuts
        # through coarse triangles.
        return max(
            3.5 * state.current_hole_radius_m,
            2.5 * state.thickness_m,
        )
    return max(
        1.5 * state.current_hole_radius_m,
        2.0 * state.thickness_m,
    )


def _fragment_component_from_triangles(
    parent: AeroComponent,
    state: HybridShellCollisionState,
    triangle_indices: Set[int],
) -> DetachedFragmentBody:
    shell_state = state.shell_state
    fragment_nodes = {
        node
        for index in triangle_indices
        for node in shell_state.triangle_nodes[index]
    }
    mean_displacement = (0.0, 0.0, 0.0)
    displacement_mass = 0.0
    for node in fragment_nodes:
        node_mass = shell_state.masses_kg[node]
        mean_displacement = v_add(
            mean_displacement,
            v_mul(
                v_sub(
                    shell_state.positions[node],
                    shell_state.reference_positions[node],
                ),
                node_mass,
            ),
        )
        displacement_mass += node_mass
    if displacement_mass > 1e-18:
        mean_displacement = v_mul(mean_displacement, 1.0 / displacement_mass)
    # Preserve the resolved plug outline.  Raw explicit-shell nodal positions
    # can contain large pre-detachment strain, which previously stretched a
    # 90 mm plug into a 170 mm pane.  The detached plug carries the mean rigid
    # displacement and velocity; subsequent collisions can deform it normally.
    mid_surface_triangles = []
    for index in sorted(triangle_indices):
        nodes = shell_state.triangle_nodes[index]
        points = tuple(
            v_add(shell_state.reference_positions[node], mean_displacement)
            for node in nodes
        )
        normal = v_unit(
            v_cross(v_sub(points[1], points[0]), v_sub(points[2], points[0])),
            shell_state.triangle_normals[index],
        )
        mid_surface_triangles.append((normal, points[0], points[1], points[2]))
    # Keep one authoritative surface for each detached element.  Extruding a
    # fragment here creates a second coincident sheet on either face and is
    # exactly the visual split seen in ParaView.  Thickness remains in the
    # structural mass/material model; it must not be represented by duplicate
    # display surfaces.
    triangles = mid_surface_triangles
    group_mass = sum(shell_state.triangle_masses_kg[index] for index in triangle_indices)
    group_velocity = (0.0, 0.0, 0.0)
    group_centroid = (0.0, 0.0, 0.0)
    counted_mass = 0.0
    counted_centroid_mass = 0.0
    for index in triangle_indices:
        node_ids = shell_state.triangle_nodes[index]
        triangle_mass = shell_state.triangle_masses_kg[index]
        node_mass = sum(shell_state.masses_kg[node] for node in node_ids)
        triangle_velocity = (0.0, 0.0, 0.0)
        for node in node_ids:
            share = shell_state.masses_kg[node] / max(node_mass, 1e-18)
            triangle_velocity = v_add(
                triangle_velocity,
                v_mul(shell_state.velocities[node], share),
            )
        group_velocity = v_add(group_velocity, v_mul(triangle_velocity, triangle_mass))
        counted_mass += triangle_mass
        triangle_centroid = _centroid(
            [shell_state.positions[node] for node in node_ids]
        )
        group_centroid = v_add(group_centroid, v_mul(triangle_centroid, triangle_mass))
        counted_centroid_mass += triangle_mass
    if counted_mass > 1e-18:
        group_velocity = v_mul(group_velocity, 1.0 / counted_mass)
    unconstrained_group_velocity = group_velocity
    brittleness = _material_brittleness_ratio(shell_state)
    axial_speed = v_dot(group_velocity, shell_state.inward_direction)
    axial_velocity = v_mul(shell_state.inward_direction, axial_speed)
    transverse_velocity = v_sub(group_velocity, axial_velocity)
    transverse_speed = v_norm(transverse_velocity)
    transverse_fraction = 0.10 if brittleness < 1.75 else 0.60
    transverse_limit = transverse_fraction * abs(axial_speed)
    if transverse_speed > transverse_limit and transverse_speed > 1e-18:
        transverse_velocity = v_mul(
            transverse_velocity,
            transverse_limit / transverse_speed,
        )
        group_velocity = v_add(axial_velocity, transverse_velocity)
    removed_fragment_velocity = v_sub(unconstrained_group_velocity, group_velocity)
    if (
        v_norm(removed_fragment_velocity) > 1e-18
        and parent.freedom.translate_axes
        and not parent.is_assembly_anchor
    ):
        remaining_parent_mass = max(parent.mass - group_mass, 1e-12)
        parent.linear_velocity = v_add(
            parent.linear_velocity,
            v_mul(
                removed_fragment_velocity,
                group_mass / remaining_parent_mass,
            ),
        )
    if counted_centroid_mass > 1e-18:
        group_centroid = v_mul(group_centroid, 1.0 / counted_centroid_mass)
    fragment_component = AeroComponent(
        name=f"{parent.name} detached fragment {state.next_fragment_id}",
        patch=f"{parent.patch}_fragment_{state.next_fragment_id}",
        triangles=triangles,
        cofr=group_centroid,
        lref=max(parent.lref, parent.material.thickness_m or parent.lref),
        aref=sum(_triangle_area(triangle) for triangle in triangles),
        freedom=MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            mate_type="COLLISION_FRAGMENT",
            source="hybrid-shell-fragment",
        ),
        material=parent.material,
        mass=group_mass,
        inertia=max(parent.inertia * group_mass / max(parent.mass + group_mass, 1e-18), 1e-12),
        linear_velocity=group_velocity,
        angular_velocity=(0.0, 0.0, 0.0),
        deformation_reference_triangles=list(triangles),
        deformation_max_m=0.0,
        deformation_mean_m=0.0,
        collision_family=parent.collision_family or parent.patch,
    )
    return DetachedFragmentBody(
        component=fragment_component,
        triangle_indices=set(triangle_indices),
        mass_kg=group_mass,
    )


def _translate_component_geometry(component: AeroComponent, delta: Vec3) -> None:
    if v_norm(delta) <= 0.0:
        return
    component.cofr = v_add(component.cofr, delta)
    translated: List[Triangle] = []
    for normal, a, b, c in component.triangles:
        translated.append(
            (
                normal,
                v_add(a, delta),
                v_add(b, delta),
                v_add(c, delta),
            )
        )
    component.triangles = translated


def sync_hybrid_shell_fragments(
    parent: AeroComponent,
    state: HybridShellCollisionState,
) -> Tuple[int, float]:
    shell_state = state.shell_state
    emit_shell_fragments(shell_state)
    new_triangles = shell_state.emitted_triangles - state.emitted_triangles
    if not new_triangles:
        return 0, 0.0
    created = 0
    emitted_mass = 0.0
    fragment_vertices = [
        _triangle_reference_vertices(shell_state, fragment.triangle_indices)
        for fragment in state.fragment_bodies
    ]
    fragment_groups = _connected_fragment_groups(shell_state, new_triangles)
    if _material_brittleness_ratio(shell_state) < 1.75:
        fragment_groups = [set(new_triangles)]
    for group in fragment_groups:
        group_vertices = _triangle_reference_vertices(shell_state, group)
        merge_index: Optional[int] = None
        for index, existing_vertices in enumerate(fragment_vertices):
            if _material_brittleness_ratio(shell_state) < 1.75:
                merge_index = index
                break
            if existing_vertices & group_vertices:
                merged_triangles = state.fragment_bodies[index].triangle_indices | group
                if (
                    _fragment_transverse_span_m(shell_state, merged_triangles)
                    <= _fragment_merge_span_limit_m(shell_state)
                ):
                    merge_index = index
                break
        if merge_index is None:
            state.fragment_bodies.append(
                _fragment_component_from_triangles(parent, state, group)
            )
            fragment_vertices.append(set(group_vertices))
            state.next_fragment_id += 1
        else:
            existing_fragment = state.fragment_bodies[merge_index]
            new_fragment = _fragment_component_from_triangles(parent, state, group)
            merged_triangles = existing_fragment.triangle_indices | group
            updated = _fragment_component_from_triangles(parent, state, merged_triangles)
            total_fragment_mass = max(
                existing_fragment.component.mass + new_fragment.component.mass,
                1e-18,
            )
            updated.component.linear_velocity = v_mul(
                v_add(
                    v_mul(
                        existing_fragment.component.linear_velocity,
                        existing_fragment.component.mass,
                    ),
                    v_mul(
                        new_fragment.component.linear_velocity,
                        new_fragment.component.mass,
                    ),
                ),
                1.0 / total_fragment_mass,
            )
            updated.component.patch = existing_fragment.component.patch
            updated.component.name = existing_fragment.component.name
            state.fragment_bodies[merge_index] = updated
            fragment_vertices[merge_index].update(group_vertices)
        created += len(group)
        emitted_mass += sum(shell_state.triangle_masses_kg[index] for index in group)
    state.emitted_triangles.update(new_triangles)
    state.emitted_fragment_mass_kg += emitted_mass
    return created, emitted_mass


def shell_fragment_triangles(
    state: ExplicitShellState | HybridShellCollisionState,
) -> List[Triangle]:
    """Return the currently detached, dynamically displaced shell elements."""
    if isinstance(state, HybridShellCollisionState):
        triangles: List[Triangle] = []
        for fragment in state.fragment_bodies:
            triangles.extend(fragment.component.triangles)
        return triangles
    fragments: List[Triangle] = []
    for triangle_index in sorted(state.emitted_triangles):
        node_ids = state.triangle_nodes[triangle_index]
        a, b, c = (state.positions[node] for node in node_ids)
        normal = v_unit(
            v_cross(v_sub(b, a), v_sub(c, a)),
            state.triangle_normals[triangle_index],
        )
        fragments.append((normal, a, b, c))
    return fragments


def shell_fragment_velocity(
    state: ExplicitShellState | HybridShellCollisionState,
) -> Vec3:
    """Mass-weighted average velocity of the emitted shell fragments."""
    if isinstance(state, HybridShellCollisionState):
        total_mass = 0.0
        velocity = (0.0, 0.0, 0.0)
        for fragment in state.fragment_bodies:
            total_mass += fragment.component.mass
            velocity = v_add(
                velocity,
                v_mul(fragment.component.linear_velocity, fragment.component.mass),
            )
        if total_mass <= 1e-18:
            return (0.0, 0.0, 0.0)
        return v_mul(velocity, 1.0 / total_mass)
    total_mass = 0.0
    velocity = (0.0, 0.0, 0.0)
    for triangle_index in sorted(state.emitted_triangles):
        for node_index in state.triangle_nodes[triangle_index]:
            mass = state.masses_kg[node_index]
            total_mass += mass
            velocity = v_add(velocity, v_mul(state.velocities[node_index], mass))
    if total_mass <= 1e-18:
        return (0.0, 0.0, 0.0)
    return v_mul(velocity, 1.0 / total_mass)


def advance_hybrid_shell_collision(
    component: AeroComponent,
    state: HybridShellCollisionState,
    dt_s: float,
) -> float:
    shell_deformation = advance_explicit_shell(component, state.shell_state, dt_s)
    for fragment in state.fragment_bodies:
        _translate_component_geometry(
            fragment.component,
            v_mul(fragment.component.linear_velocity, max(dt_s, 0.0)),
        )
    return shell_deformation


def _refresh_shell_geometry(component: AeroComponent, state: ExplicitShellState) -> float:
    if not any(index not in state.emitted_triangles for index in range(len(state.triangle_nodes))):
        # Keep a structural carrier element for extremely coarse shells.  A
        # target with zero attached elements cannot be advanced or rendered.
        keep_index = max(
            range(len(state.triangle_nodes)),
            key=lambda index: _radial_distance(
                state.triangle_reference_centroids[index],
                state.contact_point,
                state.inward_direction,
            ),
        )
        if keep_index in state.emitted_triangles:
            state.emitted_triangles.remove(keep_index)
            state.plug_triangles.discard(keep_index)
            state.emitted_fragment_mass_kg = max(
                0.0,
                state.emitted_fragment_mass_kg - state.triangle_masses_kg[keep_index],
            )
    max_displacement = 0.0
    fragment_nodes = _emitted_fragment_nodes(state)
    for triangle_index, node_ids in enumerate(state.triangle_nodes):
        if triangle_index in state.emitted_triangles:
            continue
        for node in node_ids:
            max_displacement = max(
                max_displacement,
                v_norm(v_sub(state.positions[node], state.reference_positions[node])),
            )
    if state.render_as_midsurface:
        component.triangles = _deformed_shell_mid_surface_triangles(state)
    else:
        component.triangles = _thickened_shell_triangles(state)
    reported_displacement = min(
        state.displacement_limit_m,
        max(max_displacement, state.permanent_displacement_m),
    )
    state.max_displacement_m = max(state.max_displacement_m, reported_displacement)
    component.deformation_max_m = max(component.deformation_max_m, reported_displacement)
    displacement_sum = sum(
        v_norm(v_sub(position, reference))
        for node_index, (position, reference) in enumerate(
            zip(state.positions, state.reference_positions)
        )
        if node_index not in fragment_nodes
    )
    component.deformation_mean_m = displacement_sum / max(
        len(state.positions) - len(fragment_nodes),
        1,
    )
    return max_displacement


def advance_material_point_shell(
    component: AeroComponent,
    state: ExplicitShellState,
    dt_s: float,
) -> float:
    """Advance a compact USL MPM-style particle/grid shell update.

    Shell nodes act as material points.  They transfer mass and momentum to a
    local trilinear Cartesian grid, receive elastic restoring acceleration and
    then gather the grid velocity.  This avoids the directional bias of edge
    springs while retaining explicit CFL substepping and exact lumped particle
    mass conservation.
    """
    if dt_s <= 0.0 or not state.positions:
        return 0.0
    cell = state.mpm_cell_size_m or max(2.0 * state.thickness_m, state.contact_radius_m / 2.0, 1e-4)
    state.mpm_cell_size_m = cell
    # MPM uses a grid support radius larger than a shell edge, so its practical
    # CFL limit is less restrictive.  Cap work per output interval explicitly;
    # this keeps the particle-grid solve predictable for refined impact meshes.
    substeps = min(
        state.max_substeps,
        32,
        max(1, math.ceil(dt_s / max(8.0 * state.stable_dt_s, 1e-12))),
    )
    h = dt_s / substeps
    state.mass_scale = max(
        state.mass_scale,
        (h / max(8.0 * state.stable_dt_s, 1e-12)) ** 2,
        1.0,
    )
    for _ in range(substeps):
        fragment_nodes = _emitted_fragment_nodes(state)
        nodal_forces = [(0.0, 0.0, 0.0) for _point in state.positions]
        _add_membrane_element_forces(state, nodal_forces)
        _add_bending_spring_forces(state, nodal_forces, fragment_nodes)
        grid_mass: Dict[Tuple[int, int, int], float] = {}
        grid_momentum: Dict[Tuple[int, int, int], Vec3] = {}
        grid_force: Dict[Tuple[int, int, int], Vec3] = {}
        fixed_grid_keys: Set[Tuple[int, int, int]] = set()
        for index, point in enumerate(state.positions):
            if index in fragment_nodes:
                continue
            base = tuple(math.floor(value / cell) for value in point)
            local = tuple((point[i] / cell) - base[i] for i in range(3))
            displacement = v_sub(point, state.reference_positions[index])
            displacement_norm = v_norm(displacement)
            yield_displacement = max(state.yield_strain * cell, 1e-12)
            if displacement_norm > yield_displacement:
                plastic_fraction = min(
                    0.2,
                    (displacement_norm - yield_displacement)
                    / max(displacement_norm, 1e-12),
                )
                state.reference_positions[index] = v_add(
                    state.reference_positions[index],
                    v_mul(displacement, plastic_fraction),
                )
            for mask in range(8):
                key = tuple(base[i] + ((mask >> i) & 1) for i in range(3))
                weight = 1.0
                for i in range(3):
                    fraction = local[i] if ((mask >> i) & 1) else 1.0 - local[i]
                    weight *= fraction
                if weight <= 0.0:
                    continue
                node_mass = weight * state.masses_kg[index]
                grid_mass[key] = grid_mass.get(key, 0.0) + node_mass
                grid_momentum[key] = v_add(
                    grid_momentum.get(key, (0.0, 0.0, 0.0)),
                    v_mul(state.velocities[index], node_mass),
                )
                particle_force = nodal_forces[index]
                grid_force[key] = v_add(
                    grid_force.get(key, (0.0, 0.0, 0.0)),
                    v_mul(particle_force, weight),
                )
                if index in state.fixed_nodes:
                    fixed_grid_keys.add(key)
        grid_velocity: Dict[Tuple[int, int, int], Vec3] = {}
        for key, mass in grid_mass.items():
            velocity = v_mul(grid_momentum[key], 1.0 / max(mass, 1e-18))
            velocity = v_add(
                velocity,
                v_mul(grid_force.get(key, (0.0, 0.0, 0.0)), h / max(mass, 1e-18)),
            )
            if key in fixed_grid_keys:
                velocity = (0.0, 0.0, 0.0)
            velocity = v_mul(velocity, max(0.0, 1.0 - state.damping_ratio * h * 20.0))
            grid_velocity[key] = velocity
        for index, point in enumerate(state.positions):
            if index in fragment_nodes:
                state.positions[index] = v_add(
                    state.positions[index],
                    v_mul(state.velocities[index], h),
                )
                continue
            if index in state.fixed_nodes:
                state.positions[index] = state.reference_positions[index]
                state.velocities[index] = (0.0, 0.0, 0.0)
                continue
            base = tuple(math.floor(value / cell) for value in point)
            local = tuple((point[i] / cell) - base[i] for i in range(3))
            gathered = (0.0, 0.0, 0.0)
            weight_sum = 0.0
            for mask in range(8):
                key = tuple(base[i] + ((mask >> i) & 1) for i in range(3))
                weight = 1.0
                for i in range(3):
                    fraction = local[i] if ((mask >> i) & 1) else 1.0 - local[i]
                    weight *= fraction
                if key in grid_velocity:
                    gathered = v_add(gathered, v_mul(grid_velocity[key], weight))
                    weight_sum += weight
            if weight_sum > 1e-12:
                state.velocities[index] = v_mul(gathered, 1.0 / weight_sum)
            state.positions[index] = v_add(state.positions[index], v_mul(state.velocities[index], h))
        _apply_displacement_limits(state)
    return _refresh_shell_geometry(component, state)


def _emitted_fragment_nodes(state: ExplicitShellState) -> Set[int]:
    nodes: Set[int] = set()
    for triangle_index in state.emitted_triangles:
        nodes.update(state.triangle_nodes[triangle_index])
    return nodes


def _attached_shell_vertex_positions(
    state: ExplicitShellState,
) -> Tuple[
    Dict[Tuple[float, float, float], Vec3],
    Dict[int, Tuple[float, float, float]],
]:
    fragment_nodes = _emitted_fragment_nodes(state)
    grouped_positions: Dict[Tuple[float, float, float], List[Vec3]] = {}
    node_to_key: Dict[int, Tuple[float, float, float]] = {}
    for node_index, reference in enumerate(state.reference_positions):
        if node_index in fragment_nodes:
            continue
        key = reference
        node_to_key[node_index] = key
        grouped_positions.setdefault(key, []).append(state.positions[node_index])
    averaged_positions = {
        key: _centroid(points)
        for key, points in grouped_positions.items()
    }
    return averaged_positions, node_to_key


def shell_triangle_von_mises_stress_pa(
    state: ExplicitShellState,
    triangle_index: int,
) -> float:
    """Return classic constant-strain-triangle plane-stress von Mises stress."""
    if not 0 <= triangle_index < len(state.membrane_elements):
        return 0.0
    element = state.membrane_elements[triangle_index]
    twice_area = max(2.0 * element.area_m2, 1e-18)
    local_displacements: List[Tuple[float, float]] = []
    for node in element.nodes:
        displacement = v_sub(state.positions[node], state.reference_positions[node])
        local_displacements.append(
            (
                v_dot(displacement, element.basis_x),
                v_dot(displacement, element.basis_y),
            )
        )
    strain_x = sum(
        element.b_coefficients[i] * local_displacements[i][0]
        for i in range(3)
    ) / twice_area
    strain_y = sum(
        element.c_coefficients[i] * local_displacements[i][1]
        for i in range(3)
    ) / twice_area
    shear_strain = sum(
        element.c_coefficients[i] * local_displacements[i][0]
        + element.b_coefficients[i] * local_displacements[i][1]
        for i in range(3)
    ) / twice_area
    modulus = max(state.young_modulus_pa, 1.0)
    poisson = max(0.0, min(state.poisson_ratio, 0.49))
    plane_stress_scale = modulus / max(1.0 - poisson * poisson, 1e-12)
    stress_x = plane_stress_scale * (strain_x + poisson * strain_y)
    stress_y = plane_stress_scale * (strain_y + poisson * strain_x)
    shear_stress = modulus * shear_strain / max(2.0 * (1.0 + poisson), 1e-12)
    von_mises_squared = (
        stress_x * stress_x
        - stress_x * stress_y
        + stress_y * stress_y
        + 3.0 * shear_stress * shear_stress
    )
    if not math.isfinite(von_mises_squared):
        return 0.0
    return math.sqrt(max(von_mises_squared, 0.0))


def shell_rendered_triangle_indices(state: ExplicitShellState) -> List[int]:
    """Map midsurface output triangles back to their structural elements."""
    vertex_positions, node_to_key = _attached_shell_vertex_positions(state)
    if not vertex_positions:
        return []
    rendered: List[int] = []
    seen: Set[
        Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    ] = set()
    for triangle_index, node_ids in enumerate(state.triangle_nodes):
        if triangle_index in state.emitted_triangles:
            continue
        if (
            state.current_hole_radius_m > 0.0
            and _triangle_touches_deformed_hole_disk(
                state,
                triangle_index,
                state.current_hole_radius_m,
            )
        ):
            continue
        try:
            keys = tuple(node_to_key[node] for node in node_ids)
        except KeyError:
            continue
        canonical_key = tuple(sorted(keys))
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        rendered.append(triangle_index)
    if not rendered:
        # Extremely coarse meshes can leave one attached carrier element whose
        # vertices all touch the hole disk.  Do not let the display filter erase
        # that last mass-bearing element: downstream geometry references require
        # a non-empty parent surface on the step after fragmentation.
        attached = [
            index
            for index in range(len(state.triangle_nodes))
            if index not in state.emitted_triangles
        ]
        if attached:
            rendered.append(
                max(
                    attached,
                    key=lambda index: _radial_distance(
                        state.triangle_reference_centroids[index],
                        state.contact_point,
                        state.inward_direction,
                    ),
                )
            )
    return rendered


def _deformed_shell_mid_surface_triangles(state: ExplicitShellState) -> List[Triangle]:
    """Render attached shell elements on the solved midsurface only.

    The hybrid solver tracks mass and momentum on a shell midsurface.  Rendering
    artificial front/back skins can make a highly deformed thin plate look as if
    it has gained thickness, so the default hybrid output exports the actual
    solved midsurface and omits perforated faces from the visible hole.
    """
    vertex_positions, node_to_key = _attached_shell_vertex_positions(state)
    if not vertex_positions:
        return []

    triangles: List[Triangle] = []
    for triangle_index in shell_rendered_triangle_indices(state):
        node_ids = state.triangle_nodes[triangle_index]
        keys = tuple(node_to_key[node] for node in node_ids)
        a, b, c = (vertex_positions[key] for key in keys)
        normal = v_unit(
            v_cross(v_sub(b, a), v_sub(c, a)),
            state.triangle_normals[triangle_index],
        )
        triangles.append((normal, a, b, c))
    return triangles


def _thickened_shell_triangles(state: ExplicitShellState) -> List[Triangle]:
    """Rebuild a finite-thickness shell surface from the deformed midsurface.

    The structural state evolves on a midsurface, but the visualization should
    still show a solid thin plate with visible front and back faces. Without
    this reconstruction the render shows only a local dented sheet and the rear
    side appears not to move, even when the midsurface actually bends.
    """
    vertex_positions, node_to_key = _attached_shell_vertex_positions(state)
    if not vertex_positions:
        return []

    boundary_edge_counts: Dict[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        int,
    ] = {}
    boundary_edge_order: Dict[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ] = {}
    triangle_records: List[
        Tuple[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]],
            Vec3,
        ]
    ] = []
    seen_triangles: Set[
        Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    ] = set()

    for triangle_index, node_ids in enumerate(state.triangle_nodes):
        if triangle_index in state.emitted_triangles:
            continue
        if (
            state.current_hole_radius_m > 0.0
            and _triangle_touches_deformed_hole_disk(
                state,
                triangle_index,
                state.current_hole_radius_m,
            )
        ):
            continue
        keys = tuple(node_to_key[node] for node in node_ids)
        canonical_triangle_key = tuple(sorted(keys))
        if canonical_triangle_key in seen_triangles:
            continue
        seen_triangles.add(canonical_triangle_key)
        a, b, c = (vertex_positions[key] for key in keys)
        normal = v_unit(
            v_cross(v_sub(b, a), v_sub(c, a)),
            state.triangle_normals[triangle_index],
        )
        triangle_records.append((keys, normal))
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            edge_key = tuple(sorted((keys[local_a], keys[local_b])))
            boundary_edge_counts[edge_key] = boundary_edge_counts.get(edge_key, 0) + 1
            boundary_edge_order.setdefault(
                edge_key,
                (keys[local_a], keys[local_b]),
            )

    half_thickness = 0.5 * state.thickness_m
    thickened: List[Triangle] = []
    for keys, normal in triangle_records:
        a_mid, b_mid, c_mid = (vertex_positions[key] for key in keys)
        front = [
            v_add(point, v_mul(normal, half_thickness))
            for point in (a_mid, b_mid, c_mid)
        ]
        back = [
            v_sub(point, v_mul(normal, half_thickness))
            for point in (a_mid, b_mid, c_mid)
        ]
        thickened.append((normal, front[0], front[1], front[2]))
        thickened.append((v_mul(normal, -1.0), back[2], back[1], back[0]))

    for edge_key, count in boundary_edge_counts.items():
        if count != 1:
            continue
        start_key, end_key = boundary_edge_order[edge_key]
        start = vertex_positions[start_key]
        end = vertex_positions[end_key]
        adjacent_normal = next(
            normal for keys, normal in triangle_records
            if start_key in keys and end_key in keys
        )
        front_start = v_add(start, v_mul(adjacent_normal, half_thickness))
        front_end = v_add(end, v_mul(adjacent_normal, half_thickness))
        back_start = v_sub(start, v_mul(adjacent_normal, half_thickness))
        back_end = v_sub(end, v_mul(adjacent_normal, half_thickness))
        wall_normal = v_unit(
            v_cross(v_sub(front_end, front_start), v_sub(back_start, front_start)),
            adjacent_normal,
        )
        thickened.append((wall_normal, front_start, front_end, back_end))
        thickened.append((wall_normal, front_start, back_end, back_start))
    return thickened


def _apply_weld_constraints(state: ExplicitShellState) -> None:
    for group in state.weld_groups:
        partitions: Dict[bool, List[int]] = {False: [], True: []}
        for node_index in group:
            partitions[(node_index // 3) in state.plug_triangles].append(node_index)
        for nodes in partitions.values():
            if len(nodes) <= 1:
                continue
            if any(node in state.fixed_nodes for node in nodes):
                for node in nodes:
                    state.positions[node] = state.reference_positions[node]
                    state.velocities[node] = (0.0, 0.0, 0.0)
                continue
            total_mass = sum(state.masses_kg[node] for node in nodes)
            average_position = (0.0, 0.0, 0.0)
            average_velocity = (0.0, 0.0, 0.0)
            for node in nodes:
                mass_fraction = state.masses_kg[node] / max(total_mass, 1e-18)
                average_position = v_add(
                    average_position,
                    v_mul(state.positions[node], mass_fraction),
                )
                average_velocity = v_add(
                    average_velocity,
                    v_mul(state.velocities[node], mass_fraction),
                )
            for node in nodes:
                state.positions[node] = average_position
                state.velocities[node] = average_velocity


def _apply_displacement_limits(state: ExplicitShellState) -> None:
    """Project shell nodes onto the configured explicit-solver safety bound."""
    limit = state.displacement_limit_m
    fragment_nodes = _emitted_fragment_nodes(state)
    for node_index, reference in enumerate(state.reference_positions):
        if node_index in state.fixed_nodes or node_index in fragment_nodes:
            continue
        displacement = v_sub(state.positions[node_index], reference)
        distance = v_norm(displacement)
        if distance <= limit:
            continue
        outward = v_mul(displacement, 1.0 / distance)
        state.positions[node_index] = v_add(reference, v_mul(outward, limit))
        outward_speed = v_dot(state.velocities[node_index], outward)
        if outward_speed > 0.0:
            state.velocities[node_index] = v_sub(
                state.velocities[node_index],
                v_mul(outward, outward_speed),
            )


def _add_membrane_element_forces(
    state: ExplicitShellState,
    forces: List[Vec3],
) -> None:
    """Accumulate plane-stress constant-strain triangle forces.

    The element uses a reference tangent plane and lumped nodal masses.  This
    is the classic small-strain membrane FEM contribution; the edge elements
    retain simple crack/yield handling and damping for the explicit update.
    """
    young = state.young_modulus_pa
    poisson = state.poisson_ratio
    modulus = young / max(1.0 - poisson * poisson, 1e-9)
    for triangle_index, element in enumerate(state.membrane_elements):
        if triangle_index in state.emitted_triangles:
            continue
        area = max(element.area_m2, 1e-18)
        displacement_x: List[float] = []
        displacement_y: List[float] = []
        for node in element.nodes:
            displacement = v_sub(
                state.positions[node],
                state.reference_positions[node],
            )
            displacement_x.append(v_dot(displacement, element.basis_x))
            displacement_y.append(v_dot(displacement, element.basis_y))
        scale = 1.0 / (2.0 * area)
        strain_xx = scale * sum(
            coefficient * displacement
            for coefficient, displacement in zip(
                element.b_coefficients,
                displacement_x,
            )
        )
        strain_yy = scale * sum(
            coefficient * displacement
            for coefficient, displacement in zip(
                element.c_coefficients,
                displacement_y,
            )
        )
        strain_xy = scale * sum(
            c_coefficient * x_displacement
            + b_coefficient * y_displacement
            for b_coefficient, c_coefficient, x_displacement, y_displacement in zip(
                element.b_coefficients,
                element.c_coefficients,
                displacement_x,
                displacement_y,
            )
        )
        equivalent_strain = math.sqrt(
            strain_xx * strain_xx
            + strain_yy * strain_yy
            + 0.5 * strain_xy * strain_xy
        )
        if state.yield_strain > 0.0 and equivalent_strain > state.yield_strain:
            scale_down = state.yield_strain / equivalent_strain
            strain_xx *= scale_down
            strain_yy *= scale_down
            strain_xy *= scale_down
        stress_xx = modulus * (strain_xx + poisson * strain_yy)
        stress_yy = modulus * (poisson * strain_xx + strain_yy)
        stress_xy = young * strain_xy / max(2.0 * (1.0 + poisson), 1e-9)
        for node, b_coefficient, c_coefficient in zip(
            element.nodes,
            element.b_coefficients,
            element.c_coefficients,
        ):
            force_x = 0.5 * state.thickness_m * (
                b_coefficient * stress_xx + c_coefficient * stress_xy
            )
            force_y = 0.5 * state.thickness_m * (
                c_coefficient * stress_yy + b_coefficient * stress_xy
            )
            # ``B.T @ stress`` is the element's internal resisting force.
            # The equation of motion needs its negative.  Applying this with
            # the same sign as the displacement stiffness injects energy into
            # a stretched element and produces the alternating, accordion-like
            # deformation seen in impacted plates.
            internal_force = v_add(
                v_mul(element.basis_x, force_x),
                v_mul(element.basis_y, force_y),
            )
            forces[node] = v_sub(forces[node], internal_force)


def _add_bending_spring_forces(
    state: ExplicitShellState,
    forces: List[Vec3],
    fragment_nodes: Set[int],
) -> None:
    for spring in state.bending_springs:
        if spring.node_a in fragment_nodes or spring.node_b in fragment_nodes:
            continue
        delta = v_sub(
            state.positions[spring.node_b],
            state.positions[spring.node_a],
        )
        length = v_norm(delta)
        if length <= 1e-12:
            continue
        axis = v_mul(delta, 1.0 / length)
        extension = length - spring.rest_length_m
        relative_velocity = v_dot(
            v_sub(
                state.velocities[spring.node_b],
                state.velocities[spring.node_a],
            ),
            axis,
        )
        effective_mass = (
            state.masses_kg[spring.node_a]
            * state.masses_kg[spring.node_b]
            / max(
                state.masses_kg[spring.node_a] + state.masses_kg[spring.node_b],
                1e-18,
            )
        ) * state.mass_scale
        damping = (
            2.0
            * state.damping_ratio
            * math.sqrt(spring.stiffness_n_m * max(effective_mass, 1e-18))
        )
        force = v_mul(
            axis,
            spring.stiffness_n_m * extension + damping * relative_velocity,
        )
        forces[spring.node_a] = v_add(forces[spring.node_a], force)
        forces[spring.node_b] = v_sub(forces[spring.node_b], force)


def advance_explicit_shell(
    component: AeroComponent,
    state: ExplicitShellState,
    dt_s: float,
) -> float:
    """Advance shell elastodynamics with a stable explicit central update."""
    if state.solver_backend == "mpm":
        return advance_material_point_shell(component, state, dt_s)
    if dt_s <= 0.0 or not state.positions:
        return 0.0
    stable_dt = max(state.stable_dt_s, 1e-12)
    required_substeps = max(1, math.ceil(dt_s / stable_dt))
    substeps = min(state.max_substeps, required_substeps)
    substep_dt = dt_s / substeps

    # Explicit solvers must not silently violate their CFL limit when the
    # requested interval needs more than the configured substep budget.
    # Diagonal mass scaling is the conventional, simple remedy: increase the
    # inertial mass just enough for ``substep_dt`` to be stable on the still-
    # attached shell. Detached fragments are no longer part of that stiffness
    # network, so their ballistic launch velocities must not be scaled down.
    required_mass_scale = max(1.0, (substep_dt / stable_dt) ** 2)
    if required_mass_scale > state.mass_scale:
        fragment_nodes = _emitted_fragment_nodes(state)
        velocity_scale = math.sqrt(state.mass_scale / required_mass_scale)
        for node_index, velocity in enumerate(state.velocities):
            if node_index in fragment_nodes:
                continue
            state.velocities[node_index] = v_mul(velocity, velocity_scale)
        state.mass_scale = required_mass_scale

    for _substep in range(substeps):
        fragment_nodes = _emitted_fragment_nodes(state)
        forces = [(0.0, 0.0, 0.0) for _point in state.positions]
        _add_membrane_element_forces(state, forces)
        _add_bending_spring_forces(state, forces, fragment_nodes)
        for edge in state.edges:
            if edge.failed:
                continue
            if edge.node_a in fragment_nodes or edge.node_b in fragment_nodes:
                continue
            delta = v_sub(state.positions[edge.node_b], state.positions[edge.node_a])
            length = v_norm(delta)
            if length <= 1e-12:
                continue
            axis = v_mul(delta, 1.0 / length)
            strain = (length - edge.rest_length_m) / edge.rest_length_m
            if strain > state.failure_strain:
                edge.failed = True
                state.failed_edges += 1
                continue
            if abs(strain) > state.yield_strain:
                elastic_strain = math.copysign(state.yield_strain, strain)
                edge.rest_length_m = length / max(1.0 + elastic_strain, 1e-6)
                strain = elastic_strain
            elastic_force = edge.stiffness_n_m * edge.rest_length_m * strain
            relative_velocity = v_dot(
                v_sub(state.velocities[edge.node_b], state.velocities[edge.node_a]),
                axis,
            )
            effective_mass = (
                state.masses_kg[edge.node_a]
                * state.masses_kg[edge.node_b]
                / max(
                    state.masses_kg[edge.node_a] + state.masses_kg[edge.node_b],
                    1e-18,
                )
            ) * state.mass_scale
            damping = (
                2.0
                * state.damping_ratio
                * math.sqrt(edge.stiffness_n_m * max(effective_mass, 1e-18))
            )
            force = v_mul(axis, elastic_force + damping * relative_velocity)
            forces[edge.node_a] = v_add(forces[edge.node_a], force)
            forces[edge.node_b] = v_sub(forces[edge.node_b], force)

        for node_index in range(len(state.positions)):
            if node_index in fragment_nodes:
                state.positions[node_index] = v_add(
                    state.positions[node_index],
                    v_mul(state.velocities[node_index], substep_dt),
                )
                continue
            if node_index in state.fixed_nodes:
                state.positions[node_index] = state.reference_positions[node_index]
                state.velocities[node_index] = (0.0, 0.0, 0.0)
                continue
            acceleration = v_mul(
                forces[node_index],
                1.0
                / max(
                    state.masses_kg[node_index] * state.mass_scale,
                    1e-18,
                ),
            )
            state.velocities[node_index] = v_add(
                state.velocities[node_index],
                v_mul(acceleration, substep_dt),
            )
            state.positions[node_index] = v_add(
                state.positions[node_index],
                v_mul(state.velocities[node_index], substep_dt),
            )
        _apply_weld_constraints(state)
        _apply_displacement_limits(state)

    return _refresh_shell_geometry(component, state)
