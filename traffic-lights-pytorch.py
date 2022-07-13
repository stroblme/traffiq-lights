# %%
import torch as t
import numpy as np #interp, repeat

import mlflow
import time

import matplotlib.pyplot as plt

from traffiq import generate_data, traffic_mlp

import logging
logging.basicConfig(level=logging.WARNING,
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M')

log = logging.getLogger(__name__)

augmentation_size=10 # defines how many times each of the base samples (e.g. red, red-yellow,..) is repeated
mlflow.log_param("augmentation_size", augmentation_size)
scatter=0.6 # scattering of the actual values (> 0.5 will significantly reduce the performance)
mlflow.log_param("scatter", scatter)

modes={'train':0.70, 'valid':0.20, 'test':0.10} # definition of available modes and their proportions

datasets = generate_data(augmentation_size, scatter, modes)

# epoch and batch size as steady parameters TODO: consider adding batch size as hyperparam
epochs = 10
mlflow.log_param("epochs", epochs)
batch_size = 1
mlflow.log_param("batch_size", batch_size)

# define dataloaders for different modes
dataloaders = {mode:t.utils.data.DataLoader(datasets[mode], batch_size=batch_size, shuffle=True) for mode in modes}

log.info(f"Modes: {modes}\nAugmentation size: {augmentation_size}\nScatter: {scatter}\nBatch size: {batch_size}\nEpochs: {epochs}")


def define_model():
    # define number of hidden layers in the mlp (input output layer is fixed)
    arch = [1, 1]
    mlflow.log_param("arch", arch)

    rot_gates = ['']*2
    rot_gates[0] = 'rx'
    rot_gates[1] = 'ry'
    mlflow.log_param("rot_gates", rot_gates)

    ent_gates = 'cx' # select more reasonable values here, just for demo
    mlflow.log_param("ent_gates", ent_gates)

    shots = 1024
    mlflow.log_param("shots", shots)

    model = traffic_mlp(3, 1, arch, rot_gates, ent_gates, shots) # 3 input features (r, ge, gr), 1 output (go, nogo)

    opt = t.optim.Adam(model.parameters(), lr=1e-3) # maxiter=100 only defines the precision of gradient approx

    loss_fn = model.cost_function

    return model, opt, loss_fn

if __name__ == "__main__":
    model, opt, loss_fn = define_model()

    initial_point = np.random.random(model.var_qc.num_parameters)
    mlflow.log_param("initial_point", initial_point)

    # result of the objective (could also use accuracy)
    trial_loss = 0

    start = time.time()
    log.info(f"Staring training at {start}")

    
    for e in range(epochs):
        for mode in ['train']: # no validation yet

            running_loss = 0.0
            running_corrects = 0

            for x_batch, y_batch in dataloaders[mode]:
                objective_function = lambda variational: loss_fn(   np.array(x_batch), # need to convert to numpy here.. tensor is "non-numeric"
                                                                    np.array(y_batch),
                                                                    variational)

                # opt_var, opt_value, _ = opt.optimize(len(initial_point), objective_function, initial_point=initial_point)
                initial_point, opt_value, _ = opt.optimize(len(initial_point), objective_function, initial_point=initial_point)


                running_loss += opt_value
                # running_corrects += t.sum(y_pred >= 0.9*y_batch.data)
            epoch_loss = running_loss / len(dataloaders[mode].dataset)
            # epoch_acc = running_corrects.float() / len(dataloaders[mode].dataset)
        
            # if e % 10 == 0:
            log.info(f"{mode} loss in epoch {e}: {epoch_loss:.2}")
            mlflow.log_metric(key="epoch_loss", value=epoch_loss, step=e)

        trial_loss += epoch_loss


