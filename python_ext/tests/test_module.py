"""Tests for the FiniteDimensionalModule(Builder) bindings.

Covers basic construction/introspection, the action API, validation, and in
particular the guards that turn out-of-range degrees into ``ValueError``
instead of a Rust panic that would abort the interpreter (``set_action`` and
``check_validity``).
"""

from __future__ import annotations

import pytest

import sseq_ext as ext

P2 = 2


def build_joker() -> ext.FiniteDimensionalModule:
    """The Joker / question-mark module over the mod-2 Steenrod algebra."""
    b = ext.FiniteDimensionalModuleBuilder(
        p=2,
        graded_dimension=[1, 1, 1, 1, 1],
        algebra="adem",
        name="Joker",
    )
    for d in range(5):
        b.set_basis_element_name(d, 0, f"x{d}")
    b.add_action("Sq1 x0 = x1")
    b.add_action("Sq2 x0 = x2")
    b.add_action("Sq2 x1 = x3")
    b.add_action("Sq2 x2 = x4")
    b.add_action("Sq1 x3 = x4")
    return b.build()


# ---------------------------------------------------------------------------
# Construction and introspection
# ---------------------------------------------------------------------------


def test_builder_basics():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem", name="M"
    )
    assert b.name == "M"
    assert b.prime == 2
    assert b.min_degree() == 0
    assert b.max_degree() == 2
    assert b.dimension(0) == 1
    assert b.dimension(1) == 1
    assert b.dimension(2) == 1
    assert b.dimension(3) == 0


def test_builder_min_degree_offset():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 2], algebra="adem", min_degree=3
    )
    assert b.min_degree() == 3
    assert b.max_degree() == 4
    assert b.dimension(3) == 1
    assert b.dimension(4) == 2


def test_builder_invalid_prime():
    with pytest.raises(ValueError, match="[Pp]rime"):
        ext.FiniteDimensionalModuleBuilder(p=6, graded_dimension=[1], algebra="adem")


def test_builder_invalid_algebra():
    with pytest.raises(ValueError, match="algebra"):
        ext.FiniteDimensionalModuleBuilder(
            p=2, graded_dimension=[1], algebra="bogus"
        )


def test_default_basis_element_names():
    b = ext.FiniteDimensionalModuleBuilder(p=2, graded_dimension=[1, 2], algebra="adem")
    assert b.basis_element_to_string(0, 0) == "x0_0"
    assert b.basis_element_to_string(1, 0) == "x1_0"
    assert b.basis_element_to_string(1, 1) == "x1_1"


def test_set_basis_element_name_out_of_range():
    b = ext.FiniteDimensionalModuleBuilder(p=2, graded_dimension=[1, 1], algebra="adem")
    with pytest.raises(ValueError, match="No basis element"):
        b.set_basis_element_name(1, 5, "y")
    with pytest.raises(ValueError, match="No basis element"):
        b.set_basis_element_name(9, 0, "y")


# ---------------------------------------------------------------------------
# build() / resolution
# ---------------------------------------------------------------------------


def test_build_and_immutable_view():
    M = build_joker()
    assert M.name == "Joker"
    assert M.prime == 2
    assert M.min_degree() == 0
    assert M.max_degree() == 4
    assert [M.dimension(d) for d in range(5)] == [1, 1, 1, 1, 1]
    assert M.basis_element_to_string(0, 0) == "x0"


def test_builder_consumed_after_build():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1], algebra="adem"
    )
    b.build()
    with pytest.raises(ValueError, match="consumed"):
        b.dimension(0)
    with pytest.raises(ValueError, match="consumed"):
        b.build()


def test_resolution_of_module():
    M = build_joker()
    res = M.resolution()
    res.compute_through_stem(ext.Bidegree.n_s(8, 4))
    # The unit class lives in (n=0, s=0).
    assert res.number_of_gens_in_bidegree(ext.Bidegree.n_s(0, 0)) == 1


# ---------------------------------------------------------------------------
# add_action error paths
# ---------------------------------------------------------------------------


def test_add_action_unknown_generator():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    b.set_basis_element_name(0, 0, "x0")
    b.set_basis_element_name(1, 0, "x1")
    with pytest.raises(ValueError):
        b.add_action("Sq1 nope = x1")


def test_inconsistent_module_rejected():
    # Sq1 Sq1 = 0 is an Adem relation, but here we force Sq1 x0 = x1 and
    # Sq1 x1 = x2, making Sq1 Sq1 x0 = x2 != 0. build() must reject it.
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    for d in range(3):
        b.set_basis_element_name(d, 0, f"x{d}")
    b.add_action("Sq1 x0 = x1")
    b.add_action("Sq1 x1 = x2")
    with pytest.raises(ValueError):
        b.build()


# ---------------------------------------------------------------------------
# set_action: direct API and out-of-range guards (must raise, not panic)
# ---------------------------------------------------------------------------


def test_set_action_direct():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    b.set_basis_element_name(0, 0, "x0")
    b.set_basis_element_name(1, 0, "x1")
    # Sq1 (operation_degree=1, idx 0) x0 = x1.
    b.set_action(1, 0, 0, 0, [1])
    M = b.build()
    assert M.max_degree() == 1


def test_set_action_output_degree_out_of_range():
    """output_degree beyond max_degree must raise, not panic. Empty output
    used to slip past the length check (dimension == 0) and panic in Rust."""
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    # input_degree=2, operation_degree=2 -> output_degree=4 > max_degree=2.
    with pytest.raises(ValueError, match="output_degree"):
        b.set_action(2, 0, 2, 0, [])


def test_set_action_operation_degree_beyond_span():
    """operation_degree larger than the whole module span must raise."""
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    with pytest.raises(ValueError, match="output_degree"):
        b.set_action(4, 0, 0, 0, [])


def test_set_action_negative_operation_degree():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    with pytest.raises(ValueError, match="operation_degree"):
        b.set_action(-1, 0, 1, 0, [1])


def test_set_action_input_degree_out_of_range():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    with pytest.raises(ValueError, match="input_degree"):
        b.set_action(0, 0, 9, 0, [])


def test_set_action_bad_indices():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    # operation_idx out of range for the algebra in that degree.
    with pytest.raises(ValueError, match="operation_idx"):
        b.set_action(1, 5, 0, 0, [1])
    # input_idx out of range for the module degree.
    with pytest.raises(ValueError, match="input_idx"):
        b.set_action(1, 0, 0, 3, [1])


def test_set_action_wrong_output_length():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    # degree 1 has dimension 1, supplying 2 entries must error.
    with pytest.raises(ValueError, match="length"):
        b.set_action(1, 0, 0, 0, [1, 1])


# ---------------------------------------------------------------------------
# check_validity guard (must raise on output_deg <= input_deg, not panic)
# ---------------------------------------------------------------------------


def test_check_validity_equal_degrees():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    with pytest.raises(ValueError, match="greater"):
        b.check_validity(2, 2)


def test_check_validity_decreasing_degrees():
    b = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1, 1], algebra="adem"
    )
    with pytest.raises(ValueError, match="greater"):
        b.check_validity(3, 2)


def test_check_validity_valid_range_ok():
    M_builder = ext.FiniteDimensionalModuleBuilder(
        p=2, graded_dimension=[1, 1], algebra="adem"
    )
    M_builder.set_basis_element_name(0, 0, "x0")
    M_builder.set_basis_element_name(1, 0, "x1")
    M_builder.add_action("Sq1 x0 = x1")
    M_builder.extend_actions(0, 1)
    # A consistent action over a valid range should not raise.
    M_builder.check_validity(0, 1)
