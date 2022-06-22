import torch as t
import numpy as np #repeat

import logging

def generate_data(augmentation_size:int=100, scatter:float=0.6, modes:dict={'train':0.70, 'valid':0.20, 'test':0.10}
                ) -> t.utils.data.TensorDataset:

    x_base = t.FloatTensor([[0,0,1], [0,1,0], [1,1,0], [1,0,0]])
    y_base = t.FloatTensor([1, 1, 0, 0])

    features = 3
    datasets = {mode:None for mode in modes.keys()}

    for mode, proportion in modes.items():
        actual_augmentation = int(augmentation_size*proportion)

        x_enhanced = t.zeros((actual_augmentation*x_base.size(0), features))
        y_enhanced = np.repeat(y_base, augmentation_size*proportion)

        for b, c in enumerate(x_base):
            for a in range(actual_augmentation):
                random_light = scatter/2-scatter*t.rand(features)
                scattered_feature = c + random_light
                x_enhanced[b*actual_augmentation + a] = t.FloatTensor(np.interp(scattered_feature, (scattered_feature.min(), scattered_feature.max()), (0, +1) ))

        datasets[mode] = t.utils.data.TensorDataset(x_enhanced, y_enhanced)

        logging.info(f"{mode}-dataset size: {x_enhanced.shape}")

    return datasets


class traffic_mlp(t.nn.Module):
    """
    Standard MLP for classification with dynamic number of layers and their size
    """
    def __init__(self, input_dim, output_dim, arch=[4,2]):
        super(traffic_mlp, self).__init__()
        self.input_layer = t.nn.Linear(input_dim, arch[0])

        self.hidden_layers = []
        for layer_it in range(len(arch)-1):
            self.hidden_layers.append(t.nn.Linear(arch[layer_it], arch[layer_it+1]))

        self.output_layer = t.nn.Linear(arch[-1], output_dim)
        
    def forward(self, x):
        x = t.nn.functional.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = t.nn.functional.relu(hidden_layer(x))

        x = t.sigmoid(self.output_layer(x))
        return x