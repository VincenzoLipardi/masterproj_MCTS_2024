from problems.vqe import h2_class, lih_class, h2o_class
from problems.combinatorial import qaoa_class
from problems.vqls import vqls_demo, vqls_paper



def max_cut(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = qaoa_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)



def h2(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def lih(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = lih_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def h2o(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)



def vqls_0(quantum_circuit, ansatz='all', cost=False):
    # Instance shown in pennylane demo: https://pennylane.ai/qml/demos/tutorial_vqls/
    problem = vqls_demo
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_1(quantum_circuit, ansatz='all', cost=False, gradient=False):
    # Define the problem A = c_0 I + c_1 X_1 + c_2 X_2 + c_3 Z_3 Z_4
    problem = vqls_paper

    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
