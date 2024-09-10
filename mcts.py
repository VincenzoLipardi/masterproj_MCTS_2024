import random
import numpy as np
from structure import Circuit, GateSet
from qiskit import QuantumCircuit


class Node:
    def __init__(self, state: Circuit, max_depth: int, parent=None):
        """
        A node in the tree stores information to guide the search.

        :param state: Circuit object representing the quantum circuit stored in the node.
        :param parent: Parent node. None for the root node.
        :param max_depth: Maximum depth allowed for the quantum circuit.
        """
        # Quantum circuit
        self.state = state
        # Is the circuit respecting the constraint of the hardware? boolean
        self.isTerminal = False
        # Parent node. Node object. The root node is the only one not having that.
        self.parent = parent
        # List of children of the node. list
        self.children = []
        # Number of times the node have been visited. integer
        self.visits = 0
        # Value is the total reward. float
        self.value = 0
        # Maximum quantum circuit depth
        self.max_depth = max_depth
        # Position of the node in terms of tree depth. integer
        self.tree_depth = 0 if parent is None else parent.tree_depth + 1
        # Gate set
        self.gate_set = 'continuous'
        # Control on the stop action of the node
        self.stop_is_done = False
        # Specify the action that created it, None only for root and stop nodes
        self.action = None
        # Counts the Controlled-NOT gates in the circuit
        self.counter_cx = self.state.circuit.count_ops().get('cx', 0)

    def __repr__(self):
        return f"Tree depth: {self.tree_depth}  -  Generated by Action: {self.action}  -  Number of Children: {len(self.children)}  -  Visits: {self.visits}  -  Value: {self.value}  -  Quantum Circuit (CNOT counts)= {self.counter_cx}):\n{self.state.circuit}"

    def is_fully_expanded(self, branches, pw_C, pw_alpha):
        """
        Checks if the node is fully expanded.

        :param branches: int or boolean. If false, it uses progressive widening (PW), if true it uses double (PW). If int, the maximum number of branches is fixed.
        :param pw_alpha: float (in [0,1]). Closer to 1 in strongly stochastic domains, closer to 0 otherwise.
        :param pw_C: int. Hyperparameter for progressive widening.
        :return: Boolean indicating if the node is fully expanded (leaf).
        """
        if isinstance(branches, bool):
            # Progressive Widening Techniques: adaptive number of branches
            if branches:
                raise NotImplementedError("Double progressive widening is not implemented yet.")
            t = self.visits
            k = np.ceil(pw_C * (t ** pw_alpha))
            return len(self.children) >= k
        elif isinstance(branches, int):
            # The number of the tree's branches is fixed
            return len(self.children) >= branches
        else:
            raise TypeError("The 'branches' parameter must be an int or bool.")


    def define_children(self, prob_choice, stop_deterministic, roll_out=False):
        """
        Expands the node by defining a new child node applying a modification to the circuit.

        :param prob_choice: dict. Probability to choose between possible actions.
        :param stop_deterministic: boolean. If true, the stop action is placed by default for each new node.
        :param roll_out: boolean. True if used for rollout (new nodes are temporary, not included in the tree).
        :return: Node
        """

        parent = self
        qc = parent.state.circuit.copy()
        # If we are rolling out we don't want to add the stop action in the rollout
        stop = self.stop_is_done if not roll_out else True

        new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)
        new_qc = new_qc(qc)
        temporary_prob_choice = {'a': 50, 's': 50, 'c': 0, 'd': 0}
        if new_qc is None:
            """If here, MCTS chose to change parameters when there are no parametrized gates,  or delete in a very shallow circuit. Then let's prevent this by allowing only the adding and swapping action"""
            new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, temporary_prob_choice, stop)
            new_qc = new_qc(qc)


        def det_stop():
            # Uncomment the two lines below if you want to be sure that the children node is different form its parent node
            """while check_equivalence(qc, new_qc):
                new_qc = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)(qc)"""
            new_node = node_from_qc(new_qc, parent_node=self, roll_out=roll_out)
            new_node.action = action
            self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
            return new_node

        def prob_stop():
            if new_qc == 'stop':
                # It means that get_legal_actions returned the STOP action, then we define this node as Terminal
                self.isTerminal = True
                self.stop_is_done = True
                self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
                return self
            else:
                if isinstance(new_qc, QuantumCircuit):
                    new_state = Circuit(4, 1).building_state(new_qc)
                    new_node = Node(new_state, max_depth=self.max_depth, parent=self)
                    new_node.action = action
                    self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
                    if not roll_out:
                        self.children.append(new_node)
                    return new_node
                else:
                    raise TypeError("new_qc must be a QuantumCircuit object in Qiskit")

        if stop_deterministic:
            # This is used if the move stop is not allowed
            return det_stop()
        else:
            # This is used if the move stop is allowed
            return prob_stop()

    def best_child(self, criteria):
        children_with_values = [(child, child.value, child.visits) for child in self.children]
        if isinstance(criteria, str):
            if criteria == 'value':
                best = max(children_with_values, key=lambda x: x[1])[0]
            elif criteria == 'average_value':
                best = max(children_with_values, key=lambda x: x[1]/x[2])[0]
            elif criteria == 'visits':
                best = max(children_with_values, key=lambda x: x[2])[0]
            else:
                raise ValueError("this variable can only be: visits, value, average_value")
        else:
            raise TypeError("the variable criteria must be a string")
        return best


def node_from_qc(quantum_circuit: QuantumCircuit, parent_node: Node, roll_out: bool) -> Node:
    if isinstance(quantum_circuit, QuantumCircuit):
        new_state = Circuit(4, 1).building_state(quantum_circuit)
        new_child = Node(new_state, max_depth=parent_node.max_depth, parent=parent_node)
        if not roll_out:
            parent_node.children.append(new_child)
        return new_child
    else:
        raise TypeError("The quantum_circuit must be a QuantumCircuit object in Qiskit.")


def select(node: Node, exploration: float = 0.4) -> Node:
    log_visits = np.log(node.visits)
    children_with_values = [(child, child.value / child.visits + exploration * np.sqrt(log_visits / child.visits)) for child in node.children]
    return max(children_with_values, key=lambda x: x[1])[0]


def expand(node: Node, prob_choice: dict, stop_deterministic: bool) -> Node:
    new_node = node.define_children(prob_choice=prob_choice, stop_deterministic=stop_deterministic)
    if stop_deterministic:
        stop_node = node_from_qc(new_node.state.circuit, new_node, roll_out=False)
        stop_node.isTerminal = True
        return stop_node
    return new_node


def rollout(node: Node, steps: int) -> Node:
    new_node = node
    for i in range(steps):
        new_node = new_node.define_children(prob_choice={'a': 100, 'd': 0, 's': 0, 'c': 0, 'p': 0}, roll_out=True, stop_deterministic=False)
    return new_node


def evaluate(node: Node, evaluation_function) -> float:
    return node.state.evaluation(evaluation_function)


def backpropagate(node: Node, result: float) -> None:
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


def modify_prob_choice(dictionary: dict, len_qc: int) -> dict:
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    modifications = [-40, 10, 10, 10, 10]
    modified_values = [max(0, v + m) for v, m in zip(values, modifications)]
    if len_qc < 6:
        modified_values[1] = 0
    # Normalize to ensure the sum is still 100
    modified_values = [v / sum(modified_values) * 100 for v in modified_values]
    return dict(zip(keys, modified_values))


def commit(epsilon: float, current_node: Node, criteria: str) -> Node:
    # It commits to the best action if the node has explored enough
    if epsilon is not None:
        if random.uniform(0, 1) >= epsilon:
            return current_node.best_child(criteria=criteria)
    return current_node.best_child(criteria=criteria)

def mcts(root: Node, budget: int, evaluation_function, criteria: str, rollout_type: str, roll_out_steps: int, branches, choices: dict, epsilon: float, stop_deterministic: bool, ucb_value: float = 0.4, pw_C=1, pw_alpha=0.3,
         verbose: bool = False) -> dict:

    prob_choice = {'a': 100, 'd': 0, 's': 0, 'c': 0}
    original_root = root
    if verbose:
        print('Root Node:\n', root)

    evaluate(root, evaluation_function)
    root.visits = 1

    for epoch_counter in range(budget):
        current_node = root

        if current_node.visits > budget/20 and len(current_node.children) > 2:
            root = commit(epsilon, current_node, criteria)
            if verbose:
                print('Committed to', root)
            current_node = root

        if verbose:
            print('Epoch Counter: ', epoch_counter)

        # Selection
        while not current_node.isTerminal and current_node.is_fully_expanded(branches=branches, pw_C=pw_C, pw_alpha=pw_alpha):
            current_node = select(current_node, ucb_value)
            if verbose:
                print('Selection: ', current_node)

        # Expansion
        if not current_node.isTerminal:
            current_node = expand(current_node, prob_choice=prob_choice, stop_deterministic=stop_deterministic)
            if verbose:
                print('Node Expanded:\n', current_node)


        # Simulation
        if not current_node.isTerminal:
            if isinstance(roll_out_steps, int):
                leaf_node = rollout(current_node, steps=roll_out_steps)
                result = evaluate(leaf_node, evaluation_function)

                if roll_out_steps > 1 and rollout_type != "classic":
                    node_to_evaluate = leaf_node
                    result_list = [result]
                    for _ in range(roll_out_steps):
                        result_list.append(evaluate(node_to_evaluate.parent, evaluation_function))
                        node_to_evaluate = node_to_evaluate.parent
                    if rollout_type == 'max':
                        result = max(result_list)
                    elif rollout_type == 'mean':
                        result = sum(result_list)/len(result_list)
                    else:
                        raise NotImplementedError(f"Rollout type '{rollout_type}' is not implemented.")
            else:
                raise TypeError("The variable roll_out_steps must be an integer")

        else:
            result = evaluate(current_node, evaluation_function)

        if verbose:
            print('Reward: ', result)

        # Backpropagation
        backpropagate(current_node, result)

        n_qubits = len(current_node.state.circuit.qubits)
        if current_node.tree_depth == 2*n_qubits:
            prob_choice = choices

    # Return the best
    best_node = original_root

    # path = []
    qc_path = []
    children, value, visits = [], [], []
    while not best_node.isTerminal and len(best_node.children) >= 1:
        # path.append(best_node)
        qc_path.append(best_node.state.circuit)
        children.append(len(best_node.children))
        value.append(best_node.value)
        visits.append(best_node.visits)
        best_node = best_node.best_child(criteria=criteria)
    return {'qc': qc_path, 'children': children, 'visits': visits, 'value': value}
