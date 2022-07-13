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


    model = traffic_mlp(3, 1, arch) # 3 input features (r, ge, gr), 1 output (go, nogo)

    opt = t.optim.Adam(model.parameters(), lr=1e-3) # maxiter=100 only defines the precision of gradient approx

    loss_fn = t.nn.BCELoss() 

    return model, opt, loss_fn

if __name__ == "__main__":
    model, opt, loss_fn = define_model()

    # result of the objective (could also use accuracy)
    trial_loss = 0

    start = time.time()
    log.info(f"Staring training at {start}")

    
    for e in range(epochs):
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

                running_loss += loss.detach()
                # running_corrects += t.sum(y_pred >= 0.9*y_batch.data)
            epoch_loss = running_loss / len(dataloaders[mode].dataset)
            # epoch_acc = running_corrects.float() / len(dataloaders[mode].dataset)
        
            # if e % 10 == 0:
            #     logging.info(f"{mode} loss in epoch {e}: {epoch_loss:.2}")

        trial_loss += epoch_loss
        mlflow.log_metric(key="epoch_loss", value=epoch_loss, step=e)



