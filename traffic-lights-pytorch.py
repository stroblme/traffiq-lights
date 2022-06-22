# %%
import torch as t
import numpy as np #interp, repeat

# %%
x_base = t.FloatTensor([[0,0,1], [0,1,0], [1,1,0], [1,0,0]])
y_base = t.FloatTensor([1, 1, 0, 0])

modes = {'train':0.70, 'valid':0.20, 'test':0.10}
scatter = 0.6
features = 3
augmentation_size = 100
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

    print(f"{mode}-dataset size: {x_enhanced.shape}")


# %%
class MLP(t.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_layer = t.nn.Linear(input_dim, 4)
        self.hidden_layer_2 = t.nn.Linear(4, 2)
        self.output_layer = t.nn.Linear(2, output_dim)
        
    def forward(self, x):
        x = t.nn.functional.relu(self.input_layer(x))
        # x = t.nn.functional.relu(self.hidden_layer_1(x))
        x = t.nn.functional.relu(self.hidden_layer_2(x))

        x = t.sigmoid(self.output_layer(x))
        return x

# %%
learning_rate = 0.02
epochs = 100
batch_size = 10

model = MLP(features, 1)

opt = t.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = t.nn.BCELoss() 

dataloaders = {mode:None for mode in modes}
dataloaders["train"] = t.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
dataloaders["valid"] = t.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=True)
dataloaders["test"] = t.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=True)

model.eval()


# %%

for e in range(epochs):
    logs = {}
    for mode in ['train', 'valid']:
        if mode == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for x_batch, y_batch in dataloaders[mode]:
            y_pred = model(x_batch)

            loss = loss_fn(y_pred.view(-1), y_batch)

            if mode == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()

            running_loss += loss.detach() * x_batch.size(0)
            running_corrects += t.sum(y_pred >= 0.9*y_batch.data)
        epoch_loss = running_loss / len(dataloaders[mode].dataset)
        epoch_acc = running_corrects.float() / len(dataloaders[mode].dataset)
    
        if e % 10 == 0:
            print(f"{mode} loss in epoch {e}: {epoch_loss:.2}")

    


