# %%
import torch as t
import numpy as np #interp, repeat
import optuna as o

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
    def __init__(self, input_dim, output_dim, arch=[4,2]):
        super(MLP, self).__init__()
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

# %%
epochs = 100
batch_size = 10

# model = MLP(features, 1)

        
dataloaders = {mode:None for mode in modes}
dataloaders["train"] = t.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
dataloaders["valid"] = t.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=True)
dataloaders["test"] = t.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=True)


# %%


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    arch = [trial.suggest_int("n_units_l{}".format(i), 3, 6) for i in range(n_layers)]

    model = MLP(features, 1, arch)

    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_uniform("lr", 1e-5, 1e-1)
    opt = getattr(t.optim, opt_name)(model.parameters(), lr=learning_rate)

    loss_fn = t.nn.BCELoss() 

    return model, opt, loss_fn

def objective(trial):

    model, opt, loss_fn = define_model(trial)

    trial_loss = 0

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
        
            # if e % 10 == 0:
            #     logging.info(f"{mode} loss in epoch {e}: {epoch_loss:.2}")

        trial_loss += epoch_loss

    return trial_loss/epochs
    


study = o.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

pruned_trials = [t for t in study.trials if t.state == o.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == o.trial.TrialState.COMPLETE]

logging.info("Study statistics: ")
logging.info(f"  Number of finished trials: f{len(study.trials)}")
logging.info(f"  Number of pruned trials: f{len(pruned_trials)}")
logging.info(f"  Number of complete trials: f{len(complete_trials)}")

logging.info("Best trial:")
trial = study.best_trial

logging.info(f"  Value: {trial.value}")

logging.info("  Params: ")
for key, value in trial.params.items():
    logging.info(f"    {key}: {value}")