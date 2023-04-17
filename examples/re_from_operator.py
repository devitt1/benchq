################################################################################
# © Copyright 2022-2023 Zapata Computing Inc.
################################################################################
from pprint import pprint

from benchq import BasicArchitectureModel
from benchq.algorithms.time_evolution import get_qsp_time_evolution_program
from benchq.problem_ingestion import get_vlasov_hamiltonian
from benchq.problem_ingestion.hamiltonian_generation import (
    fast_load_qubit_op,
    generate_1d_heisenberg_hamiltonian,
    generate_jw_qubit_hamiltonian_from_hdf5_file,
)
from benchq.resource_estimation.graph import (
    GraphResourceEstimator,
    create_big_graph_from_subcircuits,
    run_resource_estimation_pipeline,
    simplify_rotations,
    synthesize_clifford_t,
)
from benchq.timing import measure_time


def main(filename):

    k = 2.0
    alpha = 0.6
    nu = 0.0

    dt = 0.1  # Integration timestep
    tmax = 5  # Maximal timestep
    sclf = 1

    tolerable_logical_error_rate = 1e-3
    qsp_required_precision = (
        tolerable_logical_error_rate / 3
    )  # Allocate half the error budget to trotter precision

    error_budget = {
        "qsp_required_precision": qsp_required_precision,
        "tolerable_circuit_error_rate": tolerable_logical_error_rate,
        "total_error": 1e-2,
        "synthesis_error_rate": 0.5,
        "ec_error_rate": 0.5,
    }

    architecture_model = BasicArchitectureModel(
        physical_gate_error_rate=1e-3,
        physical_gate_time_in_seconds=1e-6,
    )

    # TA 1 part: specify the core computational capability
    with measure_time() as t_info:
        # Operator: from hdf5 file
        operator = generate_jw_qubit_hamiltonian_from_hdf5_file(filename)

        # Operator: from json file
        # operator = fast_load_qubit_op(filename)

        # Operator: Vlasov Hamiltonian
        # N = 2
        # operator = get_vlasov_hamiltonian(k, alpha, nu, N)

        # Operator: 1D Heisenberg model
        # N = 100
        # operator = generate_1d_heisenberg_hamiltonian(N)

    print("Operator generation time:", t_info.total)

    # TA 1.5 part: model algorithmic circuit
    with measure_time() as t_info:
        program = get_qsp_time_evolution_program(
            operator, qsp_required_precision, dt, tmax, sclf
        )

    print("Circuit generation time:", t_info.total)

    # TA 2 part: model hardware resources
    # with measure_time() as t_info:
    #     gsc_resource_estimates = run_resource_estimation_pipeline(
    #         program,
    #         error_budget,
    #         estimator=GraphResourceEstimator(architecture_model),
    #         transformers=[
    #             simplify_rotations,
    #             create_big_graph_from_subcircuits(delayed_gate_synthesis=True),
    #         ],
    #     )

    # print("Resource estimation time without synthesis:", t_info.total)
    # pprint(gsc_resource_estimates)

    with measure_time() as t_info:
        gsc_resource_estimates = run_resource_estimation_pipeline(
            program,
            error_budget,
            estimator=GraphResourceEstimator(architecture_model),
            transformers=[
                synthesize_clifford_t(error_budget),
                create_big_graph_from_subcircuits(delayed_gate_synthesis=False),
            ],
        )

    print("Resource estimation time with synthesis:", t_info.total)
    print(filename)
    pprint(gsc_resource_estimates)


if __name__ == "__main__":
    main(
        "wf.generate_hamiltonians.61c65af/2.0_HF_cc-pvdz_fd9f1859-c107-4b16-852e-8e1a1942e508.hdf5"
    )
    main(
        "wf.generate_hamiltonians.61c65af/2.0_HF_cc-pvdz_26cd932b-d256-456a-8d03-7bdb19290c73.hdf5"
    )
    main(
        "wf.generate_hamiltonians.61c65af/2.0_BH_cc-pvdz_7201875c-f737-47a9-8852-b77bfce5c920.hdf5"
    )
    main(
        "wf.generate_hamiltonians.61c65af/2.0_BH_cc-pvdz_558d606e-dc4c-4dbb-bcc8-3c81adebd40b.hdf5"
    )
    main(
        "wf.generate_hamiltonians.61c65af/2.0_BH_cc-pvdz_9aa41bba-65de-47ad-9714-55c3f18dcd70.hdf5"
    )
