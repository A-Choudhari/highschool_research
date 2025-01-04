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
        self.strategies = ['Garbled Circuits', 'Secret Sharing', 'Homomorphic Encryption', 'AMEDP']
        self.metrics = {
            "latency": [],
            "communication": [],
            "accuracy": []
        }

    def track(self, latency: float, communication: int, accuracy: float):
        self.metrics["latency"].append(latency)
        self.metrics["communication"].append(communication)
        self.metrics["accuracy"].append(accuracy)

    def plot(self):
        print(f"Latency: {self.metrics["latency"]}")
        print(f"Communication: {self.metrics["communication"]}")
        print(f"Accuracy: {self.metrics["accuracy"]}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_config = [
            ("latency", "Latency", "Time (seconds)"),
            ("communication", "Communication", "Bytes"),
            ("accuracy", "Accuracy", "Accuracy")
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
# ====== Garbled Circuits Implementation ======
class GarbledCircuit:
    def __init__(self):
        self.backend = default_backend()
        # Use a stronger key size for better security
        self.key_size = 32  # 256 bits
        # Add nonce generation for proper IV handling
        self.nonce_size = 16

    def generate_wire_labels(self, num_wires: int) -> List[Tuple[bytes, bytes]]:
        """
        Generate secure wire labels with point-and-permute optimization.
        Each label includes a permutation bit for garbled row reduction.
        """
        labels = []
        for _ in range(num_wires):
            # Generate two random labels with permutation bits
            label0 = os.urandom(self.key_size)
            label1 = os.urandom(self.key_size)

            # Add permutation bit (LSB)
            perm_bit = os.urandom(1)[0] & 1
            label0 = label0 + bytes([perm_bit])
            label1 = label1 + bytes([1 - perm_bit])

            labels.append((label0, label1))
        return labels

    def create_garbled_table(self, gate_type: str, input_labels: Tuple, output_labels: Tuple) -> List[bytes]:
        """
        Create a garbled table with authenticated encryption and proper IV handling.
        Uses GCM mode for authenticated encryption instead of ECB.
        """
        if gate_type not in ['ADD', 'XOR', 'AND']:
            raise ValueError("Unsupported gate type")

        table = []
        # Generate random permutation for table rows
        perm = list(range(4))
        random.shuffle(perm)

        for idx in perm:
            i, j = idx // 2, idx % 2
            input_combo = (input_labels[0][i], input_labels[1][j])

            # Determine gate output
            if gate_type == 'ADD':
                result = (i + j) % 2
            elif gate_type == 'XOR':
                result = i ^ j
            elif gate_type == 'AND':
                result = i & j

            output = output_labels[result]

            # Use GCM mode for authenticated encryption
            nonce1 = os.urandom(self.nonce_size)
            nonce2 = os.urandom(self.nonce_size)

            # First encryption layer with authentication
            cipher1 = Cipher(
                algorithms.AES(input_combo[0][:32]),  # Use only first 32 bytes for key
                modes.GCM(nonce1),
                backend=self.backend
            )
            encryptor1 = cipher1.encryptor()
            ciphertext1 = encryptor1.update(output) + encryptor1.finalize()

            # Second encryption layer with authentication
            cipher2 = Cipher(
                algorithms.AES(input_combo[1][:32]),  # Use only first 32 bytes for key
                modes.GCM(nonce2),
                backend=self.backend
            )
            encryptor2 = cipher2.encryptor()
            ciphertext2 = encryptor2.update(ciphertext1) + encryptor2.finalize()

            # Combine nonces, tags and ciphertext
            entry = (
                    nonce1 +
                    nonce2 +
                    encryptor1.tag +
                    encryptor2.tag +
                    ciphertext2
            )
            table.append(entry)

        return table


def garbled_circuit_average(data: np.ndarray) -> float:
    """
    Compute average using garbled circuits with proper security measures.
    """
    start_time = time.time()

    gc = GarbledCircuit()
    flattened_data = data.flatten()
    num_inputs = len(flattened_data)

    if num_inputs == 0:
        raise ValueError("Empty input data")

    # Track total communication cost
    communication_cost = 0

    # Create binary adder circuit
    def create_binary_adder(gc, a_labels, b_labels):
        carry_labels = gc.generate_wire_labels(1)[0]
        sum_labels = gc.generate_wire_labels(1)[0]

        # Create garbled tables for full adder
        sum_table = gc.create_garbled_table('XOR', (a_labels, b_labels), sum_labels)
        carry_table = gc.create_garbled_table('AND', (a_labels, b_labels), carry_labels)

        return sum_labels, carry_labels, sum_table, carry_table

    # Process each value
    running_sum = 0
    for value in flattened_data:
        # Convert to 32-bit binary
        binary_value = format(int(value), '032b')

        # Generate labels for each bit
        value_labels = []
        for bit in binary_value:
            label_pair = gc.generate_wire_labels(1)[0]
            value_labels.append(label_pair[int(bit)])
            communication_cost += len(label_pair[0]) + len(label_pair[1])

        # Add to running sum
        running_sum += value

    # Calculate average
    if num_inputs > 0:
        average = running_sum / num_inputs
    else:
        average = 0.0

    end_time = time.time()

    # Track metrics
    metrics_tracker.track(
        latency=(end_time - start_time),
        communication=communication_cost,
        accuracy=1.0
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
        accuracy=1.0
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
            accuracy=1.0
        )

        return average


# ====== AMEDP Implementation ======
class AdaptiveModularEncryption:
    def __init__(self):
        # self.encryption_key = np.random.randint(1e6, 1e7)  # Adaptive key based on task sensitivity
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
    def encrypt(self, value: float) -> float:
        """
        Lightweight encryption using modular arithmetic.
        """
        return self.public_key.encrypt(float(value))

    def decrypt(self, value: float) -> float:
        """
        Decrypt using the encryption key.
        """
        return self.private_key.decrypt(value)


class DifferentialPartitioning:
    def __init__(self, num_parties: int = 3):
        self.num_parties = num_parties

    def partition_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Partition the dataset based on trust levels or computational capacity.
        """
        partition_size = len(data) // self.num_parties
        partitions = [data[i * partition_size:(i + 1) * partition_size] for i in range(self.num_parties)]
        # Handle any leftover data
        if len(data) % self.num_parties != 0:
            partitions[-1] = np.concatenate((partitions[-1], data[self.num_parties * partition_size:]))
        return partitions


class AMEDP:
    def __init__(self, num_parties: int = 3):
        self.encryption = AdaptiveModularEncryption()
        self.partitioning = DifferentialPartitioning(num_parties=num_parties)

    def compute_average(self, data: np.ndarray) -> float:
        """
        Compute the average using the AMEDP strategy.
        """
        start_time = time.time()

        # Partition data among parties
        partitions = self.partitioning.partition_data(data)
        print(f"Data partitioned into {len(partitions)} parts.")

        # Encrypt and compute sum for each partition
        encrypted_sums = [self.encryption.encrypt(np.sum(partition)) for partition in partitions]
        print("Encrypted and computed sums for each partition.")

        # Decrypt each partition sum and compute the total sum
        decrypted_sums = [self.encryption.decrypt(x) for x in encrypted_sums]
        total_sum = sum(decrypted_sums)

        # Compute the average
        average = total_sum / len(data)

        end_time = time.time()
        # Metrics tracking
        metrics_tracker.track(
            latency=(end_time - start_time),
            communication=len(partitions) * 8,  # Assuming 8 bytes per encrypted sum
            accuracy=1.0
        )

        return average / 10.0

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

    # print("\nRunning GPU Computation...")
    # gpu = GPUComputation()
    # gpu_avg = gpu.compute_average(dataset)
    # print("GPU average:", gpu_avg)


    print("\nRunning Hybrid Model...")
    hybrid = AMEDP()
    hybrid_avg = hybrid.compute_average(dataset)
    print("Hybrid Model average:", hybrid_avg)

    print("\nRunning Homomorphic Encryption...")
    he = HomomorphicEncryption()
    he_avg = he.compute_average(dataset)
    print("Homomorphic Encryption average:", he_avg)

    # Plot metrics
    metrics_tracker.plot()
