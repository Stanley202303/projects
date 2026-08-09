import math

from cfd_motion.fem_mpm import HybridFEMMPMState, advance_fem, make_tetra_element


def _state():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    element = make_tetra_element(positions, (0, 1, 2, 3), 1.0e7, 0.3, 1.0e12, 1.0)
    return HybridFEMMPMState(positions, [(2.0, 0.0, 0.0)] * 4, [1.0] * 4, [element])


def test_fem_translation_preserves_mass_and_momentum():
    state = _state()
    audit = advance_fem(state, 1.0e-4)
    assert math.isclose(audit.mass_before_kg, audit.mass_after_kg, abs_tol=1.0e-12)
    assert all(abs(value) < 1.0e-8 for value in audit.momentum_error)


def test_failed_element_transfers_mass_to_mpm_particles_without_loss():
    state = _state()
    state.elements[0].failed = True
    before = state.total_mass_kg
    advance_fem(state, 1.0e-4)
    assert state.use_mpm
    assert len(state.particles) == 4
    assert math.isclose(before, state.total_mass_kg, abs_tol=1.0e-12)


def test_elastic_tetra_responds_to_force():
    state = _state()
    forces = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    advance_fem(state, 1.0e-3, forces)
    assert state.positions[1][0] > 1.0
