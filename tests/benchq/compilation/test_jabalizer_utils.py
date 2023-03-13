import logging

from orquestra.quantum.circuits import Circuit, H, T, X

from benchq.compilation.jabalizer_utils import get_circuit_graph, get_program_graph
from benchq.data_structures import QuantumProgram


def test_basic_case():
    logging.getLogger().setLevel(logging.INFO)
    subroutine = Circuit([H(0), T(0), X(0)])
    calculate_subroutine_sequence = lambda x: [0] * x
    program = QuantumProgram([subroutine], 3, calculate_subroutine_sequence)

    graph_from_program = get_program_graph(program)
    graph_from_circuit = get_circuit_graph(
        Circuit([H(0), T(0), X(0), H(0), T(0), X(0), H(0), T(0), X(0)])
    )

    assert graph_from_program == graph_from_circuit


if __name__ == "__main__":
    test_basic_case()
