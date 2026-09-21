#!/usr/bin/env python3
"""Check QP retry invariance, hard constraints, and an optional captured problem."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import clarabel
import numpy as np
from scipy import sparse

import smpl_surface_retarget_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path)
    args = parser.parse_args()
    original = clarabel.DefaultSolver
    J = np.eye(3)
    residual = -np.ones(3)
    lower, upper = np.full(3, -.5), np.full(3, .5)
    kwargs = dict(ineq_A=np.array([[-1., 0., 0.]]), ineq_b=np.array([-.2]),
                  l2_step_limits=[(np.arange(3), .3, "test radius")])
    expected = common.solve_clarabel_qp_step(J, residual, .01, lower, upper, **kwargs)
    calls = []

    def fail_once(*solver_args):
        calls.append(solver_args)
        if len(calls) == 1:
            return SimpleNamespace(solve=lambda: SimpleNamespace(status="InsufficientProgress"))
        return original(*solver_args)

    with patch.object(clarabel, "DefaultSolver", side_effect=fail_once):
        retried = common.solve_clarabel_qp_step(J, residual, .01, lower, upper, **kwargs)
    assert len(calls) == 2
    np.testing.assert_allclose(retried, expected, atol=1e-5)
    assert retried[0] >= .2 - 1e-7 and np.linalg.norm(retried) <= .3 + 1e-7
    assert np.all(retried >= lower - 1e-7) and np.all(retried <= upper + 1e-7)

    # Genuine infeasibility must still fail; retry never drops a constraint.
    with patch.object(clarabel, "DefaultSolver", wraps=original) as solve:
        try:
            common.solve_clarabel_qp_step(np.eye(1), np.zeros(1), .01,
                                          np.zeros(1), np.ones(1),
                                          ineq_A=np.array([[-1.]]), ineq_b=np.array([-2.]))
        except RuntimeError as exc:
            assert "Infeasible" in str(exc), exc
        else:
            raise AssertionError("Infeasible QP was accepted")
        assert solve.call_count == 1

    # Even a nominally successful retry is rejected if its original constraints fail.
    fake_results = [SimpleNamespace(status="InsufficientProgress")] + [
        SimpleNamespace(status="Solved", x=[2.]) for _ in range(3)
    ]
    with patch.object(clarabel, "DefaultSolver", side_effect=[
        SimpleNamespace(solve=lambda value=value: value) for value in fake_results
    ]):
        result = common._solve_clarabel_with_rescaling(
            sparse.eye(1, format="csc"), np.zeros(1), sparse.eye(1, format="csc"),
            np.ones(1), [clarabel.NonnegativeConeT(1)])
    assert str(result.status) == "InsufficientProgress"

    if args.problem_dir:
        directory = args.problem_dir
        for path in sorted(directory.glob("problem_*.npz")):
            index = path.stem.removeprefix("problem_")
            with np.load(path) as problem:
                q, b = problem["q"], problem["b"]
                cones = [getattr(clarabel, str(name))(int(size))
                         for name, size in zip(problem["cone_types"], problem["cone_dims"])]
            P = sparse.load_npz(directory / f"P_{index}.npz")
            A = sparse.load_npz(directory / f"A_{index}.npz")
            result = common._solve_clarabel_with_rescaling(P, q, A, b, cones)
            assert str(result.status) in {"Solved", "AlmostSolved"}, result.status
            assert common._clarabel_constraint_violation(result, A, b, cones) <= 1e-7
            print(f"PASS captured QP: {path.name}")
    print("PASS: retry preserves optimum, box/SOC bounds and hard constraints; infeasible QPs still fail")


if __name__ == "__main__":
    main()
