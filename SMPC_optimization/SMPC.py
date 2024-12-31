import time
import numpy as np
import os
import torch
from torch import tensor
from phe import paillier
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from typing import List, Tuple, Dict
import random


# ====== Metric Tracking ======
class MetricsTracker:
    def __init__(self):
        self.strategies = ['Garbled Circuits', 'Secret Sharing', 'GPU', 'Homomorphic Encryption']
        self.metrics = {
            "latency": [],
            "communication": [],
            "accuracy": [],
            "gpu_speedup": [],
        }

    def track(self, latency: float, communication: int, accuracy: float, gpu_speedup: float):
        self.metrics["latency"].append(latency)
        self.metrics["communication"].append(communication)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["gpu_speedup"].append(gpu_speedup)

    def plot(self):
        print(f"Latency: {self.metrics["latency"]}")
        print(f"Communication: {self.metrics["communication"]}")
        print(f"Accuracy: {self.metrics["accuracy"]}")
        print(f"GPU Speedup: {self.metrics["gpu_speedup"]}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_config = [
            ("latency", "Latency", "Time (seconds)"),
            ("communication", "Communication", "Bytes"),
            ("accuracy", "Accuracy", "Accuracy"),
            ("gpu_speedup", "GPU Speedup", "Speedup")
        ]
        for idx, (metric, title, ylabel) in enumerate(metrics_config):
            ax = axes[idx // 2, idx % 2]
            ax.bar(self.strategies, self.metrics[metric])
            ax.set_title(title)
            ax.set_xlabel('Strategies')
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


# ====== Garbled Circuits Implementation ======
class GarbledCircuit:
    def __init__(self):
        self.backend = default_backend()

    def generate_wire_labels(self, num_wires: int) -> List[Tuple[bytes, bytes]]:
        return [(os.urandom(16), os.urandom(16)) for _ in range(num_wires)]

    def create_garbled_table(self, gate_type: str, input_labels: Tuple, output_labels: Tuple) -> List[bytes]:
        table = []
        for i in range(2):
            for j in range(2):
                input_combo = (input_labels[0][i], input_labels[1][j])
                if gate_type == 'ADD':
                    result = (i + j) % 2
                output = output_labels[result]

                cipher1 = Cipher(algorithms.AES(input_combo[0]), modes.ECB(), backend=self.backend)
                cipher2 = Cipher(algorithms.AES(input_combo[1]), modes.ECB(), backend=self.backend)

                temp = cipher1.encryptor().update(output) + cipher1.encryptor().finalize()
                entry = cipher2.encryptor().update(temp) + cipher2.encryptor().finalize()
                table.append(entry)
        return table


def garbled_circuit_average(data: np.ndarray) -> float:
    start_time = time.time()

    gc = GarbledCircuit()
    flattened_data = data.flatten()
    num_inputs = len(flattened_data)

    # Generate labels for each input bit
    input_labels = gc.generate_wire_labels(num_inputs)

    # Create addition circuit
    sum_labels = []
    current_sum = 0
    communication_cost = 0

    for i, value in enumerate(flattened_data):
        # Convert value to binary and create garbled gates
        binary_value = format(int(value), '032b')
        value_labels = []

        for bit in binary_value:
            label_pair = gc.generate_wire_labels(1)[0]
            value_labels.append(label_pair[int(bit)])
            communication_cost += 32  # Size of labels

        # Add to running sum
        current_sum += value
        sum_labels.extend(value_labels)

    average = current_sum / num_inputs

    end_time = time.time()
    metrics_tracker.track(
        latency=(end_time - start_time),
        communication=communication_cost,
        accuracy=1.0,
        gpu_speedup=0
    )

    return average


# ====== Secret Sharing Implementation ======
class ShamirSecretSharing:
    def __init__(self, prime: int):
        self.prime = prime

    def generate_polynomial(self, secret: int, degree: int) -> List[int]:
        coefficients = [secret]
        coefficients.extend(random.randrange(self.prime) for _ in range(degree))
        return coefficients

    def evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        result = 0
        for coefficient in reversed(coefficients):
            result = (result * x + coefficient) % self.prime
        return result

    def generate_shares(self, secret: int, num_shares: int, threshold: int) -> List[Tuple[int, int]]:
        coefficients = self.generate_polynomial(secret, threshold - 1)
        return [(i, self.evaluate_polynomial(coefficients, i)) for i in range(1, num_shares + 1)]

    def lagrange_interpolation(self, shares: List[Tuple[int, int]], x: int = 0) -> int:
        result = 0
        for i, (x_i, y_i) in enumerate(shares):
            numerator = denominator = 1
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (x - x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime
            result = (result + y_i * numerator * pow(denominator, self.prime - 2, self.prime)) % self.prime
        return result


def secret_sharing_average(data: np.ndarray, num_shares: int = 5, threshold: int = 3) -> float:
    start_time = time.time()

    # Use a prime larger than possible sum of values
    prime = 2 ** 31 - 1  # Mersenne prime
    ss = ShamirSecretSharing(prime)

    flattened_data = data.flatten()
    communication_cost = 0

    # Generate shares for each value
    all_shares = []
    for value in flattened_data:
        shares = ss.generate_shares(int(value), num_shares, threshold)
        all_shares.append(shares)
        communication_cost += len(shares) * 8  # 8 bytes per share

    # Sum shares at each index
    sum_shares = []
    for i in range(num_shares):
        share_sum = sum(shares[i][1] for shares in all_shares) % prime
        sum_shares.append((i + 1, share_sum))

    # Reconstruct the sum and calculate average
    total = ss.lagrange_interpolation(sum_shares[:threshold])
    average = total / len(flattened_data)

    end_time = time.time()
    metrics_tracker.track(
        latency=(end_time - start_time),
        communication=communication_cost,
        accuracy=1.0,
        gpu_speedup=0
    )

    return average


# ====== GPU-Accelerated Implementation ======
class GPUComputation:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_average(self, data: np.ndarray) -> float:
        start_time = time.time()

        # Transfer data to GPU and compute average
        data_tensor = tensor(data, dtype=torch.float32, device=self.device)
        average = torch.mean(data_tensor).item()

        end_time = time.time()
        metrics_tracker.track(
            latency=(end_time - start_time),
            communication=data.nbytes,
            accuracy=1.0,
            gpu_speedup=1.0 if torch.cuda.is_available() else 0.0
        )

        return average


# ====== Homomorphic Encryption Implementation ======
class HomomorphicEncryption:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def encrypt_vector(self, data: np.ndarray) -> List:
        return [self.public_key.encrypt(float(x)) for x in data.flatten()]

    def compute_average(self, data: np.ndarray) -> float:
        start_time = time.time()

        # Encrypt all values
        encrypted_values = self.encrypt_vector(data)
        print("Encrypted all values")
        # Sum encrypted values
        encrypted_sum = sum(encrypted_values)

        # Decrypt and compute average
        decrypted_sum = self.private_key.decrypt(encrypted_sum)
        average = decrypted_sum / len(data.flatten())

        end_time = time.time()
        metrics_tracker.track(
            latency=(end_time - start_time),
            communication=len(encrypted_values) * 256,  # Approximate size of encrypted values
            accuracy=1.0,
            gpu_speedup=0
        )

        return average


import random
import time
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ====== AI Compression Class (Autoencoder) ======
class AutoencoderCompression:
    def __init__(self, encoding_dim: int = 32):
        self.encoding_dim = encoding_dim
        self.model = self.build_model()

    def build_model(self) -> models.Model:
        input_layer = layers.Input(shape=(128,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(128, activation='sigmoid')(encoded)

        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 256):
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size)

    def compress(self, data: np.ndarray) -> np.ndarray:
        encoder = models.Model(self.model.input, self.model.layers[1].output)
        return encoder.predict(data)

    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        return self.model.predict(compressed_data)


# ====== Secret Sharing with AI Compression ======
class AICShamirSecretSharing(ShamirSecretSharing):
    def __init__(self, prime: int, encoding_dim: int = 32):
        super().__init__(prime)
        self.compression = AutoencoderCompression(encoding_dim)

    def generate_shares(self, secret: int, num_shares: int, threshold: int) -> List[Tuple[int, int]]:
        coefficients = self.generate_polynomial(secret, threshold - 1)
        shares = [(i, self.evaluate_polynomial(coefficients, i)) for i in range(1, num_shares + 1)]

        # Compress shares before transmitting
        share_values = np.array([share[1] for share in shares]).reshape(-1, 1)
        compressed_shares = self.compression.compress(share_values)

        # Return the compressed shares (in the form of (i, compressed_value))
        compressed_shares = compressed_shares.reshape(-1)
        return [(shares[i][0], compressed_shares[i]) for i in range(num_shares)]

    def decompress_shares(self, compressed_shares: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Decompress the share values
        share_values = np.array([compressed_share[1] for compressed_share in compressed_shares]).reshape(-1, 1)
        decompressed_shares = self.compression.decompress(share_values)

        decompressed_shares = decompressed_shares.reshape(-1)
        return [(compressed_shares[i][0], decompressed_shares[i]) for i in range(len(compressed_shares))]

    def lagrange_interpolation(self, shares: List[Tuple[int, int]], x: int = 0) -> int:
        # Decompress shares before interpolation
        shares = self.decompress_shares(shares)
        return super().lagrange_interpolation(shares, x)


# ====== Modified Secret Sharing Average Calculation with Compression ======
def secret_sharing_average_with_compression(data: np.ndarray, num_shares: int = 5, threshold: int = 3) -> float:
    start_time = time.time()

    # Use a prime larger than possible sum of values
    prime = 2 ** 31 - 1  # Mersenne prime
    ss = AICShamirSecretSharing(prime)

    flattened_data = data.flatten()
    communication_cost = 0

    # Generate shares for each value
    all_shares = []
    for value in flattened_data:
        shares = ss.generate_shares(int(value), num_shares, threshold)
        all_shares.append(shares)
        communication_cost += len(shares) * 8  # 8 bytes per share (before compression)

    # Sum shares at each index
    sum_shares = []
    for i in range(num_shares):
        share_sum = sum(shares[i][1] for shares in all_shares) % prime
        sum_shares.append((i + 1, share_sum))

    # Reconstruct the sum and calculate average
    total = ss.lagrange_interpolation(sum_shares[:threshold])
    average = total / len(flattened_data)

    end_time = time.time()
    metrics_tracker.track(
        latency=(end_time - start_time),
        communication=communication_cost,
        accuracy=1.0,
        gpu_speedup=0
    )

    return average

# ====== Main Execution ======
def generate_dataset(rows: int, cols: int) -> np.ndarray:
    return np.random.randint(1, 101, size=(rows, cols))


if __name__ == "__main__":
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    # Generate dataset
    rows, cols = 100, 10
    dataset = generate_dataset(rows, cols)
    true_average = np.mean(dataset)

    print("True average:", true_average)

    # Run all strategies
    print("\nRunning Garbled Circuits...")
    gc_avg = garbled_circuit_average(dataset)
    print("Garbled Circuits average:", gc_avg)

    print("\nRunning Secret Sharing...")
    ss_avg = secret_sharing_average(dataset)
    print("Secret Sharing average:", ss_avg)

    print("\nRunning GPU Computation...")
    gpu = GPUComputation()
    gpu_avg = gpu.compute_average(dataset)
    print("GPU average:", gpu_avg)

    print("\nRunning Hybrid Model...")
    hybrid_avg = secret_sharing_average_with_compression(dataset)
    print("Hybrid Model average:", hybrid_avg)

    print("\nRunning Homomorphic Encryption...")
    he = HomomorphicEncryption()
    he_avg = he.compute_average(dataset)
    print("Homomorphic Encryption average:", he_avg)


    # Plot metrics
    metrics_tracker.plot()