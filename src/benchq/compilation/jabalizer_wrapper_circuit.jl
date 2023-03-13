include("./jabalizer_utils.jl")

print("Input ICM circuit\n\t")
@time (n_qubits, circuit) = icm_input("icm_input_circuit.json")

print("ICM compilation: qubits=$n_qubits, gates=$(length(circuit))\n\t")
@time (icm_output, data_qubits_map) = icm_compile(circuit, n_qubits)

print("Output ICM circuit\n\t")
@time icm_output_circuit("icm_output.json", icm_output, data_qubits_map)

print("Get total number of qubits\n\t")
@time (n_qubits, qubit_map) = map_qubits(n_qubits, icm_output)

print("Jabalizer state preparation: qubits=$n_qubits, gates=$(length(icm_output))\n\t")
@time state = prepare(n_qubits, qubit_map, icm_output)

print("Jabalizer graph generation: $n_qubits\n\t")
@time (svec, op_seq) = graph_as_stabilizer_vector(state)

print("Write Adjacency List: ")
@time write_adjlist(svec, "adjacency_list.nxl")

println("Jabalizer finished")
