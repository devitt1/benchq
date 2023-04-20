from typing import Callable

from ...compilation import (
    get_algorithmic_graph_from_graph_sim_mini,
    pyliqtr_transpile_to_clifford_t,
)
from ...compilation import simplify_rotations as _simplify_rotations
from ...data_structures import QuantumProgram, get_program_from_circuit, ErrorBudget
from .structs import GraphPartition
from orquestra.quantum.circuits import Circuit


def count_pauli_rotations(program: QuantumProgram):
    """Counts the number of Pauli rotations in a program."""
    rotation_gates_per_subroutine = [
        _count_pauli_rot_gates(circuit) for circuit in program.subroutines
    ]

    total_number_of_r_gates = sum(
        count * mult
        for count, mult in zip(rotation_gates_per_subroutine, program.multiplicities)
    )
    return total_number_of_r_gates


def _count_pauli_rot_gates(circuit: Circuit):
    return sum(op.gate.name in ["RX", "RY", "RZ"] for op in circuit.operations)


def _pauli_rot_gates_fraction_per_subroutine(program: QuantumProgram):
    rotation_gates_per_subroutine = [
        _count_pauli_rot_gates(circuit) for circuit in program.subroutines
    ]

    total_number_of_r_gates = sum(
        count * mult
        for count, mult in zip(rotation_gates_per_subroutine, program.multiplicities)
    )
    if total_number_of_r_gates == 0:
        return [0] * len(rotation_gates_per_subroutine)
    return [count / total_number_of_r_gates for count in rotation_gates_per_subroutine]


def synthesize_clifford_t(
    error_budget: ErrorBudget,
) -> Callable[[QuantumProgram], QuantumProgram]:
    """Returns function that synthesizes a Clifford T circuit from a given circuit.
    The synthesis failure tolerance is specified using the ErrorBudget object for the
    whole program.

    Args:
        error_budget: ErrorBudget object.

    """

    def _transformer(program: QuantumProgram) -> QuantumProgram:
        error_fractions_per_subroutine = _pauli_rot_gates_fraction_per_subroutine(
            program
        )
        synthesized_circuits = []
        for circuit, fraction in zip(
            program.subroutines, error_fractions_per_subroutine
        ):
            synthesis_error_budget = error_budget.synthesis_failure_tolerance * fraction
            synthesized_circuits.append(
                pyliqtr_transpile_to_clifford_t(
                    circuit, circuit_precision=synthesis_error_budget
                )
            )
        return program.replace_circuits(synthesized_circuits)

    return _transformer


def simplify_rotations(program: QuantumProgram) -> QuantumProgram:
    """Transforms a program by simplifying rotations in each subroutine.
    RX an RY rotations are simplified to RZ rotations. Then RZ rotations which
    can be expressed as simpler gates are removed.
    """
    circuits = [_simplify_rotations(circuit) for circuit in program.subroutines]
    return QuantumProgram(
        circuits,
        steps=program.steps,
        calculate_subroutine_sequence=program.calculate_subroutine_sequence,
    )


def create_graphs_for_subcircuits(
    delayed_gate_synthesis: bool,
    graph_production_method=get_algorithmic_graph_from_graph_sim_mini,
) -> Callable[[QuantumProgram], GraphPartition]:
    def _transformer(program: QuantumProgram) -> GraphPartition:
        graphs_list = [
            graph_production_method(circuit) for circuit in program.subroutines
        ]
        return GraphPartition(
            program, graphs_list, delayed_gate_synthesis=delayed_gate_synthesis
        )

    return _transformer


def create_big_graph_from_subcircuits(
    delayed_gate_synthesis: bool,
    graph_production_method=get_algorithmic_graph_from_graph_sim_mini,
) -> Callable[[QuantumProgram], GraphPartition]:
    def _transformer(program: QuantumProgram) -> GraphPartition:
        big_circuit = program.full_circuit
        new_program = get_program_from_circuit(big_circuit)
        graph = graph_production_method(big_circuit)
        return GraphPartition(
            new_program, [graph], delayed_gate_synthesis=delayed_gate_synthesis
        )

    return _transformer
