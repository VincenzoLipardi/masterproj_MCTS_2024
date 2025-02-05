from itertools import combinations
import math
import random
from typing import Callable, Dict, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import CXGate, HGate, RXGate, RYGate, RZGate
from qiskit.quantum_info import Operator


class Circuit:
    def __init__(self, variable_qubits: int, ancilla_qubits: int, initialization: str = 'h'):
        """
        Initializes the quantum circuit for the target problem.

        :param variable_qubits: Number of qubits necessary to encode the problem variables
        :param ancilla_qubits:  Number of ancilla qubits (hyperparameter)
        :param initialization: Initialization method for qubits ('h', 'equal_superposition', or 'hadamard')
        """
        # self.variable_qubits = variable_qubits
        # self.ancilla_qubits = ancilla_qubits
        # Initialization of the circuit
        variable_qubits = QuantumRegister(variable_qubits, name='v')
        ancilla_qubits = QuantumRegister(ancilla_qubits, name='a')
        qc = QuantumCircuit(variable_qubits, ancilla_qubits)
        # Initialization of the quantum circuit all qubits in the 0 states (by default) or equal superposition
        if initialization == 'h' or initialization == 'equal_superposition' or initialization == 'hadamard':
            qc.h([qubits for qubits in qc.qubits])

        # Qiskit object
        self.circuit = qc
        # NISQ CONTROL
        self.is_nisq = None

    def building_state(self, quantum_circuit: QuantumCircuit) -> 'Circuit':
        """
        Creates an instance of the Circuit class with a given quantum circuit in qiskit.

        :param quantum_circuit: A Qiskit QuantumCircuit object
        :return: The updated Circuit instance
        """
        self.circuit = quantum_circuit
        return self

    def nisq_control(self, max_depth: int) -> bool:
        """
        Checks if the circuit is executable on a NISQ device.

        :param max_depth: Max quantum circuit depth due to hardware constraints
        :return: False if the depth exceeds max_depth, else True
        """
        self.is_nisq = self.circuit.depth() < max_depth
        return self.is_nisq

    def evaluation(self, evaluation_function: Callable[[QuantumCircuit], float]) -> float:
        """
        Evaluates the circuit using a given evaluation function.

        :param evaluation_function: A function that evaluates the quantum circuit
        :return: The value of the evaluation function for the input quantum circuit
        """
        return evaluation_function(self.circuit)

    def get_legal_action(self, gate_set: 'GateSet', max_depth: int, prob_choice: Dict[str, float], stop: bool) -> Tuple[
        Callable, str]:
        """
        Determines a legal action for modifying the circuit.

        :param gate_set: The set of gates available for application
        :param max_depth: The maximum allowable depth of the quantum circuit
        :param prob_choice: Probability distribution for choosing actions
        :param stop: Boolean flag indicating whether to stop actions
        :return: A tuple containing the action and its string identifier
        """
        if stop:
            prob_choice['p'] = 0
        if self.is_nisq is None:
            self.nisq_control(max_depth)
        if not self.is_nisq:
            prob_choice['a'] = 0
            prob_choice['d'] = 50

        keys = list(prob_choice.keys())
        probabilities = np.array(list(prob_choice.values())) / sum(prob_choice.values())
        action_str = np.random.choice(keys, p=probabilities)
        action = actions_on_circuit(action_chosen=action_str, gate_set=gate_set)

        if action and callable(action):
            return action, action_str
        raise NotImplementedError("Action not implemented or callable")


class GateSet:
    def __init__(self, gate_type: str = 'continuous'):
        """
        Initializes the gate set.

        :param gate_type: Type of gate set ('discrete' (Clifford generators+T) or 'continuous'(CNOT+single qubit rotations))
        """
        self.gate_type = gate_type
        if self.gate_type == 'discrete':
            gates = ['s', 'cx', 'h', 't']
        elif self.gate_type == 'continuous':
            gates = ['cx', 'ry', 'rx', 'rz']

        else:
            raise NotImplementedError
        self.pool = gates
        
def get_add_options(quantum_circuit: QuantumCircuit, gate: str = None, max_depth: int = 10) -> list: 
    """gets a list of possible add operations from the quantum circuit

    Args:
        quantum_circuit (QuantumCircuit): the quantum circuit to be expanded
        gate (str, optional): If specified, only gives the adds for that gate. If None, 
        gives the adds for every single gate. Defaults to None.
    
    returns: 
        a non-randomized list of tuples that contain the gate to add and its quantum bit:
        [('rx',Qubit1),('cx',Qubit1,Qubit2),..., ('ry',Qubit3)]
    """
    add_options = []
    if quantum_circuit.depth() > max_depth:
        return add_options

    gates = [gate] if gate else ['cx', 'rx', 'ry', 'rz']
    for gate in gates:
        if gate == 'cx':
            add_options = add_options + [('cx',) + pair for pair in combinations(quantum_circuit.qubits, 2)]  
        else:
            for qbit in quantum_circuit.qubits:
                add_options.append((gate, qbit)) 
    return add_options

def get_delete_options(qc: QuantumCircuit) -> list:
    """gets a list of possible delete operations from the quantum circuit

    Args:
        quantum_circuit (QuantumCircuit): quantum circuit to be deleted from

    Returns:
        a nonrandom list with all possible positions of the gate to be deleted
        eg: [1,2,3,4]
    """
    return list(range(len(qc.data)))

def get_swap_options(qc: QuantumCircuit, gate: str = None) -> list:
    """
        Gets a list of possible swap operations from the quantum circuit

        Args:
            qc (QuantumCircuit): The quantum circuit to be modified
            gate (str, optional): If specified, only gives the swaps for that gate. If None, 
            gives the swaps for every single gate. Defaults to None.

        Returns:
            list: A non-randomized list of tuples that contain the gate to swap and the position to swap it with:
            [('rx', 1), ('cx', 2), ..., ('ry', 3)]
        """
    swap_options = []
    gates = [gate] if gate else ['cx', 'rx', 'ry', 'rz']
    for gate in gates:
        for position in range(len(qc.data)):
            swap_options.append((gate,position))
    return swap_options

def perform_action(qc: QuantumCircuit, action: str, instructions: list) -> QuantumCircuit:
    match action:
        case "add":
            return perform_add(qc, instructions)
        case "swap":
            return perform_swap(qc, instructions)
        case "delete":
            return perform_delete(qc, instructions)
        case "change": 
            # TODO: Implement change
            pass
        
            
            

def perform_add(qc: QuantumCircuit, instructions: list) -> QuantumCircuit:  
    """adds a gate to the quantum circuit given the instructions

    Args:
        qc (QuantumCircuit): the quantum circuit to be modified
        instructions (list): the instructions to add the gate,
                             (gate, qubit1, qubit2) for cx, (gate, qubit) for single qubit gates
    Returns:
        QuantumCircuit: the modified quantum circuit
    """
    qc = qc.copy()
    gate = instructions[0]
    qubit = instructions[1]
    
    angle = 2 * math.pi * random.random() # get a random angle for parameterized gates
    gate_map = {
        'rx': RXGate(angle),
        'ry': RYGate(angle),
        'rz': RZGate(angle)
    }
    if instructions[0] == 'cx':
        qubit2 = instructions[2]
        qc.cx(qubit,qubit2)
    else:
        qc.append(gate_map[gate], [qubit])
    return qc
        
def perform_swap(quantum_circuit: QuantumCircuit, instructions: list) -> QuantumCircuit:
    """swaps a gate in the quantum circuit with a new gate given the instructions

    Args:
        qc (QuantumCircuit): the quantum circuit to be modified
        instructions (list): the instructions to swap the gate, (new_gate, position)

    Returns:
        QuantumCircuit: the modified quantum circuit
    """
    gate = instructions[0]
    position = instructions[1]
    qc = quantum_circuit.copy()
    # TODO: I think we should discuss the qubit selection here. Should it not be the same as the gate removed?
    qubits = random.sample([i for i in range(len(qc.qubits))], k=2)
    angle = 2 * math.pi * random.random()
    if gate == 'cx':
        qc.cx(qubits[0], qubits[1])
    else:
        gate_map = {
            'ry': RYGate(angle),
            'rx': RXGate(angle),
            'rz': RZGate(angle),
        }
        gate = gate_map.get(gate)
        if gate:
            qc.append(gate, [qubits[0]])
    gate_to_add = qc.data[-1]
    qc = quantum_circuit.copy() 
    gate_to_remove = qc.data[position]
    qc.data[position] = gate_to_remove.replace(gate_to_add.operation, gate_to_add.qubits, gate_to_add.clbits)
    return qc
    
def perform_delete(qc: QuantumCircuit, position: int) -> QuantumCircuit:
    """deletes a gate from the quantum circuit given the instructions

    Args:
        qc (QuantumCircuit): the quantum circuit to be modified
        instructions (list): the instructions to delete the gate, (position)

    Returns:
        QuantumCircuit: the modified quantum circuit
    """
    qc = qc.copy()
    qc.data.remove(qc.data[position])
    return qc

def get_all_adds(quantum_circuit: QuantumCircuit, grouped: bool = False):
    """ Gets all prossible continuous additions """
    possible_circuits = {}
    for gate in ['cx', 'rx', 'ry', 'rz']:
        possible_circuits[f"add_{gate}"] = get_adds_for_gate(quantum_circuit, gate)
    return possible_circuits if grouped else [circuit for circuits in possible_circuits.values() for circuit in circuits]

def get_adds_for_gate(quantum_circuit: QuantumCircuit, gate: str):
    """ Gets all prossible continuous additions for a specific gate """
    possible_circuits = [] 

    if gate == 'cx':
        for qubit1 in quantum_circuit.qubits:
            for qubit2 in quantum_circuit.qubits:
                if qubit1 is not qubit2:
                    qc = quantum_circuit.copy()
                    qc.cx(qubit1, qubit2)
                    possible_circuits.append(qc)
    else:
        angle = 2 * math.pi * random.random()
        gate_map = {
            'rx': RXGate(angle),
            'ry': RYGate(angle),
            'rz': RZGate(angle),
        }
        for qubit in quantum_circuit.qubits:
            g = gate_map.get(gate)
            if g:
                qc = quantum_circuit.copy()
                qc.append(g, [qubit])
                possible_circuits.append(qc)
    return possible_circuits

def get_all_deletes(quantum_circuit: QuantumCircuit):
    """ Gets all prossible deletions """
    possible_circuits = []
    qc = quantum_circuit.copy()
    for position, gate in enumerate(qc.data):
        qc = quantum_circuit.copy()
        qc.data.remove(qc.data[position])
        possible_circuits.append(qc)
    return possible_circuits

def get_all_parameter_changes(quantum_circuit: QuantumCircuit):
    """ Gets one possible parameter change for each changeable quantum gate """
    possible_circuits = {}
    for position, gate in enumerate(quantum_circuit.data):
        qc = quantum_circuit.copy()
        if len(qc.data[position][0].params) == 0:
            continue
        qc.data[position][0].params[0] = qc.data[position][0].params[0] + random.gauss(0, 0.2)
        possible_circuits[f"change_{position}"] = [qc]
    return possible_circuits

def get_parameter_change(quantum_circuit: QuantumCircuit, position: int = None):
    """ Gets one possible parameter change """
    if not position:
        keys = []
        for pos, gate in enumerate(quantum_circuit.data):
            if len(quantum_circuit.data[pos][0].params) != 0:
                keys.append(pos)
        if len(keys) == 0:
            return []
        position = int(random.choice(keys))
    
    qc = quantum_circuit.copy()
    if len(qc.data[position][0].params) == 0:
        return []
    qc.data[position][0].params[0] = qc.data[position][0].params[0] + random.gauss(0, 0.2)
    return [qc]

def get_all_swaps(quantum_circuit: QuantumCircuit, gates: list = ['cx', 'rx', 'ry', 'rz'], grouped: bool = False):
    """ Gets all prossible swaps """
    possible_circuits = {}
    for gate in gates:
        possible_circuits[f"swap_{gate}"] = get_swaps(quantum_circuit, gate)
    return possible_circuits if grouped else [circuit for circuits in possible_circuits.values() for circuit in circuits]

def get_swaps(quantum_circuit: QuantumCircuit, choice: str):
    possible_circuits = []
    
    qc = quantum_circuit.copy()
    qubits = random.sample([i for i in range(len(qc.qubits))], k=2)
    angle = 2 * math.pi * random.random()
    if choice == 'cx':
        qc.cx(qubits[0], qubits[1])
    else:
        gate_map = {
            'ry': RYGate(angle),
            'rx': RXGate(angle),
            'rz': RZGate(angle),
        }
        gate = gate_map.get(choice)
        if gate:
            qc.append(gate, [qubits[0]])
    gate_to_add = qc.data[-1]

    for position, gate in enumerate(quantum_circuit.data):
        qc = quantum_circuit.copy()
        gate_to_remove = qc.data[position]
        qc.data[position] = gate_to_remove.replace(gate_to_add.operation, gate_to_add.qubits, gate_to_add.clbits)
        possible_circuits.append(qc)

    return possible_circuits
        
def actions_on_circuit(action_chosen: str, gate_set: GateSet) -> Callable[
    [QuantumCircuit], Union[QuantumCircuit, None]]:
    """
    Modifies the quantum circuit depending on the action required.

    :param action_chosen: Action chosen to apply on the circuit
    :param gate_set: Universal Gate Set chosen for the application
    :return: A function that applies the chosen action on the circuit
    """
    def add_gate(quantum_circuit):
        """ Pick a random one-qubit (two-qubit) gate to add on random qubit(s) """
        qc = quantum_circuit.copy()
        qubits = random.sample([i for i in range(len(qc.qubits))], k=2)
        angle = 2 * math.pi * random.random()
        choice = random.choice(gate_set.pool)
        if choice == 'cx':
            qc.cx(qubits[0], qubits[1])
        else:
            gate_map = {
                'ry': RYGate(angle),
                'rx': RXGate(angle),
                'rz': RZGate(angle),
                'h': HGate(),
                't': lambda: qc.t(qubits[0]),
                's': lambda: qc.s(qubits[0])
            }
            gate = gate_map.get(choice)
            if gate:
                qc.append(gate, [qubits[0]])
        return qc

    def delete_gate(quantum_circuit: QuantumCircuit) -> Union[QuantumCircuit, None]:
        """ It removes a random gate from the input quantum circuit  """
        qc = quantum_circuit.copy()
        if len(qc.data) < 4:
            return None
        position = random.randint(0, len(qc.data) - 2)
        qc.data.remove(qc.data[position])
        return qc

    def swap(quantum_circuit: QuantumCircuit) -> Union[QuantumCircuit, None]:
        """ Swaps a gate in a random position with a new randomly chosen gate """

        angle = random.random() * 2 * math.pi
        if len(quantum_circuit.data) <= 1:
            return None

        position = random.randint(0, len(quantum_circuit.data) - 2)
        gate_to_remove = quantum_circuit.data[position]
        gate_to_add_str = random.choice(gate_set.pool[1:])
        gate_to_add = get_gate(gate_to_add_str, angle=angle)
        n_qubits = len(quantum_circuit.qubits)
        qr = QuantumRegister(n_qubits, 'v')
        qc = QuantumCircuit(qr)

        instructions = []
        pos = 0
        two_qubit_gate = 0
        if gate_to_add_str == 'cx':
            two_qubit_gate = 1
        delta = len(gate_to_remove[1]) - 1 - two_qubit_gate     # difference of qubit the new gate is applied to
        for instruction, qargs, cargs in quantum_circuit:
            if pos == position:
                if delta == 1:
                    qargs = [qargs[0]]
                if delta == -1:
                    qargs.append(random.choice(quantum_circuit.qubits))
                instruction = gate_to_add

            instructions.append((instruction, qargs, cargs))
            pos += 1
        qc.data = instructions
        return qc

    def change(quantum_circuit):
        """ It changes the parameter of a gate randomly chosen"""

        qc = quantum_circuit.copy()
        n = len(qc.data)
        position = random.choice([i for i in range(n)])

        check = 0
        while len(qc.data[position][0].params) == 0:
            position = random.choice([i for i in range(n)])
            check += 1
            if check > 2*n:
                return None
        gate_to_change = qc.data[position][0]
        qc.data[position][0].params[0] = gate_to_change.params[0] + random.gauss(0, 0.2)
        return qc

    def stop(quantum_circuit)-> str:
        """ Marks the node as terminal"""
        return 'stop'

    # Define a mapping between input strings and methods
    actions = {'a': add_gate, 'd': delete_gate, 's': swap, 'c': change, 'p': stop}
    return actions.get(action_chosen, None)


def get_gate(gate_str: str, angle: float = None) -> Union[HGate, CXGate, RXGate, RYGate, RZGate]:
    """
    Returns the Qiskit gate object corresponding to the given gate string.

    :param gate_str: The gate identifier string
    :param angle: The angle parameter for rotation gates
    :return: The corresponding Qiskit gate object
    """
    gate_map = {
        'h': HGate(),
        'cx': CXGate(),
        'rx': RXGate(angle),
        'ry': RYGate(angle),
        'rz': RZGate(angle)
    }
    return gate_map.get(gate_str)


def get_action_from_str(input_string, gate_set):
    method_mapping = {
        'a': gate_set.add_gate,
        'd': gate_set.delete_gate,
        's': gate_set.swap,
        'c': gate_set.change,
        'p': gate_set.stop}

    # Choose the method based on the input string
    chosen_method = method_mapping.get(input_string, None)
    if chosen_method is not None and callable(chosen_method):
        return chosen_method
    else:
        return "Invalid method name"

def check_equivalence(qc1, qc2):
    """ It returns a boolean variable. True if the two input quantum circuits are equivalent (same matrix)eqi"""
    Op1 = Operator(qc1)
    Op2 = Operator(qc2)
    return Op1.equiv(Op2)
  