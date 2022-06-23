# %%
import torch as t
import numpy as np #interp, repeat
import optuna as o
import qiskit as q
from qiskit.algorithms import optimizers as qo

import matplotlib.pyplot as plt

from traffiq import generate_data, traffic_mlp, traffiq_pqc

import logging

augmentation_size=100 # defines how many times each of the base samples (e.g. red, red-yellow,..) is repeated
scatter=0.6 # scattering of the actual values (> 0.5 will significantly reduce the performance)
modes={'train':0.70, 'valid':0.20, 'test':0.10} # definition of available modes and their proportions

datasets = generate_data(augmentation_size, scatter, modes)

# epoch and batch size as steady parameters TODO: consider adding batch size as hyperparam
epochs = 100
batch_size = 1
        
# define dataloaders for different modes
dataloaders = {mode:t.utils.data.DataLoader(datasets[mode], batch_size=batch_size, shuffle=True) for mode in modes}

logging.info(f"Modes: {modes}\nAugmentation size: {augmentation_size}\nScatter: {scatter}\nBatch size: {batch_size}\nEpochs: {epochs}")

parameters = []
costs = []
evaluations = []

def store_intermediate_result(evaluation, parameter, cost, 
                              stepsize, accept):
    global costs, parameters, evaluations

    evaluations.append(evaluation)
    parameters.append(parameter)
    costs.append(cost)

def define_model(trial):
    # define number of hidden layers in the mlp (input output layer is fixed)
    arch = [trial.suggest_int("n_layers_enc", 1, 3), trial.suggest_int("n_layers_pqc", 1, 3)]

    model = traffiq_pqc(3, arch) # 3 input features (r, ge, gr), 1 output (go, nogo)

    opt = qo.SPSA(maxiter=100, callback=store_intermediate_result) # maxiter=100 only defines the precision of gradient approx

    loss_fn = model.cost_function

    return model, opt, loss_fn

def objective(trial):
    """
    Objective of the hyperparameter optimization.
    Here is the actual training and parameter updating
    """
    model, opt, loss_fn = define_model(trial)

    initial_point = np.random.random(model.var_qc.num_parameters)

    # result of the objective (could also use accuracy)
    trial_loss = 0

    for e in range(epochs):
        for mode in ['train']:

            running_loss = 0.0
            running_corrects = 0

            for x_batch, y_batch in dataloaders[mode]:
                objective_function = lambda variational: loss_fn(   np.array(x_batch), # need to convert to numpy here.. tensor is "non-numeric"
                                                                    np.array(y_batch),
                                                                    variational)

                opt_var, opt_value, _ = opt.optimize(len(initial_point), objective_function, initial_point=initial_point)


                running_loss += opt_value * x_batch.size(0)
                # running_corrects += t.sum(y_pred >= 0.9*y_batch.data)
            epoch_loss = running_loss / len(dataloaders[mode].dataset)
            # epoch_acc = running_corrects.float() / len(dataloaders[mode].dataset)
        
            # if e % 10 == 0:
            #     logging.info(f"{mode} loss in epoch {e}: {epoch_loss:.2}")

        trial_loss += epoch_loss

    return trial_loss/epochs

if __name__ == "__main__":
    study = o.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) # careful, this can take quite some time in the quantum regime

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

    fig = plt.figure()
    plt.plot(evaluations, costs)
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.show()