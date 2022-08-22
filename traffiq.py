import torch as t
import numpy as np  # repeat
import qiskit as q
import logging


def generate_data(
    augmentation_size: int = 100,
    scatter: float = 0.6,
    modes: dict = {"train": 0.70, "valid": 0.20, "test": 0.10},
) -> t.utils.data.TensorDataset:

    x_base = t.FloatTensor([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    y_base = t.FloatTensor([1, 1, 0, 0])

    features = 3
    datasets = {mode: None for mode in modes.keys()}

    for mode, proportion in modes.items():
        actual_augmentation = int(augmentation_size * proportion)

        x_enhanced = t.zeros((actual_augmentation * x_base.size(0), features))
        y_enhanced = np.repeat(y_base, actual_augmentation)

        for b, c in enumerate(x_base):
            for a in range(actual_augmentation):
                random_light = scatter / 2 - scatter * t.rand(features)
                scattered_feature = c + random_light
                x_enhanced[b * actual_augmentation + a] = t.FloatTensor(
                    np.interp(
                        scattered_feature,
                        (scattered_feature.min(), scattered_feature.max()),
                        (0, +1),
                    )
                )

        datasets[mode] = t.utils.data.TensorDataset(x_enhanced, y_enhanced)

        logging.info(f"{mode}-dataset size: {x_enhanced.shape}")

    return datasets


class traffic_mlp(t.nn.Module):
    """
    Standard MLP for classification with dynamic number of layers and their size
    """

    def __init__(self, input_dim, output_dim, arch=[4, 2]):
        super(traffic_mlp, self).__init__()
        self.input_layer = t.nn.Linear(input_dim, arch[0])

        self.hidden_layers = []
        for layer_it in range(len(arch) - 1):
            self.hidden_layers.append(t.nn.Linear(arch[layer_it], arch[layer_it + 1]))

        self.output_layer = t.nn.Linear(arch[-1], output_dim)

    def forward(self, x):
        x = t.nn.functional.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = t.nn.functional.relu(hidden_layer(x))

        x = t.sigmoid(self.output_layer(x))
        return x


class traffiq_pqc:
    def __init__(
        self,
        input_dim,
        output_dim,
        arch=[2, 2],
        rot_gates=["ry", "rz"],
        ent_gates="cz",
        shots=1024,
    ):
        self.shots = shots

        self.enc_qc = q.circuit.library.ZZFeatureMap(
            feature_dimension=input_dim, reps=arch[0]
        )
        self.var_qc = q.circuit.library.TwoLocal(
            input_dim, rot_gates, ent_gates, reps=arch[1]
        )

        self.qc = self.enc_qc.compose(self.var_qc)
        self.qc.measure_all()

        self.qc.draw()

    def circuit_parameters(self, data, variational):
        parameters = {}
        for i, p in enumerate(self.enc_qc.ordered_parameters):
            parameters[p] = data[i]
        for i, p in enumerate(self.var_qc.ordered_parameters):
            parameters[p] = variational[i]
        return parameters

    def assign_label(self, bitstring):
        hamming_weight = sum([int(k) for k in list(bitstring)])
        odd = hamming_weight & 1
        if odd:
            return 0
        else:
            return 1

    def label_probability(self, results):
        shots = sum(results.values())
        probabilities = {0: 0, 1: 0}
        for bitstring, counts in results.items():
            label = self.assign_label(bitstring)
            probabilities[label] += counts / shots
        return probabilities

    def classification_probability(self, data, variational):
        circuits = [
            self.qc.assign_parameters(self.circuit_parameters(d, variational))
            for d in data
        ]

        backend = q.BasicAer.get_backend("qasm_simulator")
        results = q.execute(circuits, backend).result()

        classification = [
            self.label_probability(results.get_counts(c)) for c in circuits
        ]
        return classification

    def cross_entropy_loss(self, predictions, expected):
        p = predictions.get(
            int(expected)
        )  # need int-cast here for tensor incompatibility

        try:
            return -(expected * np.log(p) + (1 - expected) * np.log(1 - p))
        except RuntimeError:
            return -(expected * np.log(p + 0.0001) + (1 - expected) * np.log(1 - p))

    def cost_function(self, data, labels, variational):
        classifications = self.classification_probability(data, variational)

        cost = 0
        for i, classification in enumerate(classifications):
            cost += self.cross_entropy_loss(classification, labels[i])
        cost /= len(data)

        return cost


class traffiqc_pqc:
    def __init__(
        self,
        input_dim,
        output_dim,
        arch=[2, 2],
        rot_gates=["ry", "rz"],
        ent_gates="cz",
        shots=1024,
    ):
        self.shots = shots

        self.enc_qc = q.circuit.library.ZZFeatureMap(
            feature_dimension=input_dim, reps=arch[0]
        )
        self.var_qc = q.circuit.library.TwoLocal(
            input_dim, rot_gates, ent_gates, reps=arch[1]
        )

        self.qc = self.enc_qc.compose(self.var_qc)
        self.qc.measure_all()

        self.qc.draw()

    def circuit_parameters(self, data, variational):
        parameters = {}
        for i, p in enumerate(self.enc_qc.ordered_parameters):
            parameters[p] = data[i]
        for i, p in enumerate(self.var_qc.ordered_parameters):
            parameters[p] = variational[i]
        return parameters

    def assign_label(self, bitstring):
        hamming_weight = sum([int(k) for k in list(bitstring)])
        odd = hamming_weight & 1
        if odd:
            return 0
        else:
            return 1

    def label_probability(self, results):
        shots = sum(results.values())
        probabilities = {0: 0, 1: 0}
        for bitstring, counts in results.items():
            label = self.assign_label(bitstring)
            probabilities[label] += counts / shots
        return probabilities

    def classification_probability(self, data, variational):
        circuits = [
            self.qc.assign_parameters(self.circuit_parameters(d, variational))
            for d in data
        ]

        backend = q.BasicAer.get_backend("qasm_simulator")
        results = q.execute(circuits, backend).result()

        classification = [
            self.label_probability(results.get_counts(c)) for c in circuits
        ]
        return classification

    def cross_entropy_loss(self, predictions, expected):
        p = predictions.get(
            int(expected)
        )  # need int-cast here for tensor incompatibility

        try:
            return -(expected * np.log(p) + (1 - expected) * np.log(1 - p))
        except RuntimeError:
            return -(expected * np.log(p + 0.0001) + (1 - expected) * np.log(1 - p))

    def cost_function(self, data, labels, variational):
        classifications = self.classification_probability(data, variational)

        cost = 0
        for i, classification in enumerate(classifications):
            cost += self.cross_entropy_loss(classification, labels[i])
        cost /= len(data)

        return cost
