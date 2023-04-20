from copy import deepcopy

from .extrapolation_estimator import (
    ExtrapolatedResourceInfo,
    ExtrapolationResourceEstimator,
)
from ...data_structures import ErrorBudget, QuantumProgram


def run_resource_estimation_pipeline(
    program,
    error_budget,
    estimator,
    transformers,
):
    for transformer in transformers:
        program = transformer(program)
    return estimator.estimate(program, error_budget)


from benchq.resource_estimation.graph.transformers import count_pauli_rotations


def run_extrapolation_pipeline(
    program: QuantumProgram,
    error_budget: ErrorBudget,
    estimator: ExtrapolationResourceEstimator,
    transformers,
) -> ExtrapolatedResourceInfo:
    small_programs_resource_info = []
    total_number_of_rotations = count_pauli_rotations(program)
    total_error_budget_weight = (
        error_budget.circuit_generation_weight
        + error_budget.synthesis_weight
        + error_budget.ec_weight
    )

    for i in estimator.steps_to_extrapolate_from:
        # create copy of program for each number of steps

        small_program = deepcopy(program)
        small_program.steps = i
        current_number_of_rotations = count_pauli_rotations(small_program)

        partial_synthesis_weight = (
            error_budget.synthesis_weight
            * current_number_of_rotations
            / total_number_of_rotations
        )
        partial_total_weight = (
            error_budget.circuit_generation_weight
            + partial_synthesis_weight
            + error_budget.ec_weight
        )

        partial_ultimate_failure_rate = (
            error_budget.ultimate_failure_tolerance
            * partial_total_weight
            / total_error_budget_weight
        )

        partial_error_budget = ErrorBudget(
            ultimate_failure_tolerance=partial_ultimate_failure_rate,
            circuit_generation_weight=error_budget.circuit_generation_weight,
            synthesis_weight=partial_synthesis_weight,
            ec_weight=error_budget.ec_weight,
        )
        for transformer in transformers:
            small_program = transformer(small_program)
        resource_info = estimator.estimate(small_program, partial_error_budget)
        small_programs_resource_info.append(resource_info)

    return estimator.estimate_via_extrapolation(
        small_programs_resource_info,
        error_budget,
        small_program.delayed_gate_synthesis,
        program.steps,
    )
