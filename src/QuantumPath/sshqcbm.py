import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


total_qubits = 11  
layers = 4  
dev = qml.device("default.qubit", wires=total_qubits)


def brick_ansatz(params):
    """Standard Brick QCBM ."""
    for l in range(layers):
        for i in range(total_qubits):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        for i in range(total_qubits - 1):  
            qml.CNOT(wires=[i, i + 1])  
    return qml.probs(wires=range(total_qubits))


@qml.qnode(dev, diff_method="parameter-shift")
def quantum_model(params):
    return brick_ansatz(params)

def kl_divergence(p, q):
    p = pnp.where(p == 0, 1e-10, p)
    q = pnp.where(q == 0, 1e-10, q)
    return pnp.sum(p * pnp.log(p / q))


def cost(params):
    quantum_probs = quantum_model(params)
    empirical_probs = np.histogram(training_data, bins=len(quantum_probs), density=True)[0]
    return kl_divergence(empirical_probs, quantum_probs)



def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    prices = pd.to_numeric(data['Price'].str.replace(',', ''), errors='coerce').dropna().values 
    if len(prices) == 0:
        raise ValueError("No valid price data found in the CSV file.")
    log_returns = np.diff(np.log(prices))
    if len(log_returns) == 0:
        raise ValueError("Log returns array is empty.")
    bins = np.linspace(min(log_returns), max(log_returns), 2**total_qubits)
    return log_returns, np.digitize(log_returns, bins) - 1, bins


log_returns, training_data, bins = load_and_preprocess_data('Stoxx15_24.csv')

opt = qml.AdamOptimizer(stepsize=0.1)
params = pnp.array(np.random.uniform(-np.pi, np.pi, (layers, total_qubits, 3)), requires_grad=True)

steps_training = 200

for i in range(steps_training):
    params = opt.step(cost, params)
    if i % 5 == 0:
        print(f"Step {i}: Cost = {cost(params)}")

print("Final parameters:", params)
print("Final cost:", cost(params))

def save_params(params, file_path):
    params_df = pd.DataFrame(params.numpy().reshape(-1, 3), columns=["Theta", "Phi", "Lambda"])
    params_df.to_csv(file_path, index=False)

save_params(params, 'params.csv')


