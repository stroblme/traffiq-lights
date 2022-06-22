# %%
import torch as t
import numpy as np #interp, repeat
import optuna as o

from traffiq import generate_data, traffic_mlp

import logging

augmentation_size=100 # defines how many times each of the base samples (e.g. red, red-yellow,..) is repeated
scatter=0.6 # scattering of the actual values (> 0.5 will significantly reduce the performance)
modes={'train':0.70, 'valid':0.20, 'test':0.10} # definition of available modes and their proportions

datasets = generate_data(augmentation_size, scatter, modes)

# epoch and batch size as steady parameters TODO: consider adding batch size as hyperparam
epochs = 100
batch_size = 10
        
# define dataloaders for different modes
dataloaders = {mode:t.utils.data.DataLoader(datasets[mode], batch_size=batch_size, shuffle=True) for mode in modes}

logging.info(f"Modes: {modes}\nAugmentation size: {augmentation_size}\nScatter: {scatter}\nBatch size: {batch_size}\nEpochs: {epochs}")

def define_model(trial):
    # define number of hidden layers in the mlp (input output layer is fixed)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    # generate model architecture (input and output is fixed)
    arch = [trial.suggest_int("n_units_l{}".format(i), 3, 6) for i in range(n_layers)]

    model = traffic_mlp(3, 1, arch) # 3 input features (r, ge, gr), 1 output (go, nogo)

    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_uniform("lr", 1e-5, 1e-1)
    opt = getattr(t.optim, opt_name)(model.parameters(), lr=learning_rate)

    loss_fn = t.nn.BCELoss() 

    return model, opt, loss_fn

def objective(trial):

    model, opt, loss_fn = define_model(trial)

    # result of the objective (could also use accuracy)
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