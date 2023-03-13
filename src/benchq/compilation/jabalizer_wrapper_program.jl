include("./jabalizer_utils.jl")
using PythonCall

stim = pyimport("stim")
np = pyimport("numpy")


export tableau_partial_trace
function tableau_partial_trace(tableau::Py, qubits_to_keep::Vector{Int})
    numpy_states = tableau.to_numpy()
    # get numpy tuple from another tableau
    traced_state = []
    n_kept_qubits = length(qubits_to_keep)

    for state_num = 0:3
        traced_matrix = np.zeros((n_kept_qubits, n_kept_qubits), dtype=np.bool_)
        for (i, qubit_1) in enumerate(qubits_to_keep)
            for (j, qubit_2) in enumerate(qubits_to_keep)
                traced_matrix[i-1, j-1] = numpy_states[state_num][qubit_1-1, qubit_2-1]
            end
        end
        push!(traced_state, traced_matrix)
    end

    for state_num = 4:5
        traced_matrix = np.zeros((n_kept_qubits), dtype=np.bool_)
        for (i, qubit_1) in enumerate(qubits_to_keep)
            traced_matrix[i-1] = numpy_states[state_num][qubit_1-1]
        end
        push!(traced_state, traced_matrix)
    end

    return stim.Tableau.from_numpy(
        x2x=traced_state[1],
        z2x=traced_state[2],
        x2z=traced_state[3],
        z2z=traced_state[4],
        x_signs=traced_state[5],
        z_signs=traced_state[6],
    )
end

export append_to_graph
function append_to_graph(
    A::BitMatrix,
    program_A::BitMatrix,
    A_output_data_qubits::Vector{Int},
    program_A_output_data_qubits::Vector{Int},
)


    # avoid double counting data qubits
    new_program_A_size = length(A) + length(program_A) - length(A_output_data_qubits)
    new_program_A = BitMatrix(undef, new_program_A_size, new_program_A_size)

    # figure out where new qubits fit in the overall program adjacency matrix
    A_qubits_map = []
    for i = 1:size(A, 1)
        if i <= length(program_A_output_data_qubits)
            A_qubits_map[i] = program_A_output_data_qubits[i]
        else
            A_qubits_map = i + size(program_A, 1)
        end
    end

    # populate new_program_A with program_A's values
    for j = 1:size(program_A, 2)
        for i = 1:size(program_A, 1)
            print("($i, $j)\n\t")
            print("program_A: $program_A\n\t")
            print("new_program_A: $new_program_A\n\t")
            new_program_A[i, j]# = program_A[i, j]
        end
    end
    # populate new_program_A with A's values
    for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            new_program_A[A_qubits_map[i], A_qubits_map[j]] = A[i, j]
        end
    end

    for qubit_index in A_output_data_qubits
        new_program_A_output_data_qubits[qubit_index] = A_qubits_map[qubit_index]
    end

    return new_program_A, new_program_A_output_data_qubits
end

export make_full_program_graph

function make_full_program_graph(
    subroutine_sequence::Vector{Integer},
    step_states::Vector{StabilizerState},
    data_qubits_map_list::Vector{Vector{Integer}},
)
    n_data_qubits = length(data_qubits_map_list[1])
    tableau_on_data_qubits = zero_state(n_data_qubits).simulator.current_inverse_tableau().inverse()
    output_data_qubits_for_step = [i for i = 0:n_data_qubits-1]
    program_A = BitMatrix(undef, n_data_qubits, n_data_qubits)
    program_A_output_data_qubits = [i for i = 0:n_data_qubits-1]


    for step = 1:length(subroutine_sequence)
        state = step_states[subroutine_sequence[step]+1]
        step_tableau = state.simulator.current_inverse_tableau().inverse()

        data_qubits_tableau = tableau_partial_trace(
            tableau_on_data_qubits, output_data_qubits_for_step
        )

        # apply clifford to starting state for data qubits
        step_tableau.prepend(
            data_qubits_tableau, pylist([i for i = 0:n_data_qubits-1])
        )
        state_on_data_qubits = StabilizerState(length(data_qubits_map_list[step]))
        state_on_data_qubits.simulator.set_inverse_tableau(step_tableau.inverse())
        state_on_data_qubits.is_updated = false

        A = Jabalizer.adjacency_matrix(state_on_data_qubits)

        program_A, program_A_output_data_qubits = append_to_graph(
            A, program_A, output_data_qubits_for_step, program_A_output_data_qubits,
        )

        output_data_qubits_for_step = data_qubits_map_list[program[step]]
    end

    return program_A
end

program_info = JSON.parsefile("program_info.json")

subroutine_sequence::Vector{Integer} = program_info["subroutine_sequence"]
step_states::Vector{StabilizerState} = []
data_qubits_map_list::Vector{Vector{Integer}} = []

for subroutine = 0:program_info["n_subroutines"]-1
    print("Input ICM circuits for subroutine $subroutine\n\t")
    @time (n_qubits, circuit) = icm_input("icm_input_circuit_for_step_$subroutine.json")

    print("ICM compilation for subroutine $subroutine: qubits=$n_qubits, gates=$(length(circuit))\n\t")
    @time (icm_output, data_qubits_map) = icm_compile(circuit, n_qubits)

    print("Get total number of qubits for subroutine $subroutine\n\t")
    @time (n_qubits, qubit_map) = map_qubits(n_qubits, icm_output)

    print("Jabalizer state preparation for subroutine $subroutine: qubits=$n_qubits, gates=$(length(icm_output))\n\t")
    @time state = prepare(n_qubits, qubit_map, icm_output)

    push!(data_qubits_map_list, data_qubits_map)
    push!(step_states, state)
end

print("Creating full program graph\n\t")
@time program_A = make_full_program_graph(
    subroutine_sequence,
    step_states,
    data_qubits_map_list,
)

print("Write Adjacency List: ")
@time write_adjlist(program_A, "adjacency_list.nxl")

println("Jabalizer finished")