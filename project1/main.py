import os
from datetime import datetime

import mlflow
import optuna
import pytorch_lightning as pl

from dataset import CIFAR10DataModule
from net import ResNet

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

STUDY_RUNS = 10

EXPERIMENT_NAME = f"ResNet_CIFAR10_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

mlflow.pytorch.autolog()

mlflow.set_experiment(EXPERIMENT_NAME)


def objective(trial):
    # define hyperparameters to tune

    num_classes = 10
    input_channels = 3
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    n_blocks = trial.suggest_int("n_blocks", 2, 5, step=1)
    block_channels = trial.suggest_int("block_channels", 64, 256, step=64)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=128)

    mlflow.start_run(run_name=f"trial_{trial.number}")

    # create data module and model
    data_module = CIFAR10DataModule(batch_size=batch_size)
    model = ResNet(
        num_classes=num_classes,
        input_channels=input_channels,
        n_blocks=n_blocks,
        block_channels=block_channels,
        hidden_dim=hidden_dim,
        lr=learning_rate,
    )

    # log hyperparameters to mlflow
    mlflow.log_params(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_blocks": n_blocks,
            "block_channels": block_channels,
            "hidden_dim": hidden_dim,
        }
    )

    # train the model using pytorch lightning
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, datamodule=data_module)

    # evaluate the model on the validation set
    val_acc = trainer.validate(model, datamodule=data_module)[0]["val_accuracy"]

    mlflow.end_run()

    return -val_acc


print("Baseline run with default hyperparameters")
batch_size = 16
learning_rate = 0.001
n_blocks = 3
block_channels = 64
hidden_dim = 128

mlflow.start_run(run_name="baseline")
# create data module and model
data_module = CIFAR10DataModule(batch_size=batch_size)
model = ResNet(
    lr=learning_rate,
    n_blocks=n_blocks,
    block_channels=block_channels,
    hidden_dim=hidden_dim,
)
# log hyperparameters to mlflow
mlflow.log_params(
    {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_blocks": n_blocks,
        "block_channels": block_channels,
        "hidden_dim": hidden_dim,
    }
)
# train the model using pytorch lightning
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, datamodule=data_module)
# evaluate the model on the validation set
baseline_val_acc = trainer.validate(model, datamodule=data_module)[0]["val_accuracy"]

print(f"Starting hyperparameter tuning with {STUDY_RUNS} trials")
# start optuna study
study = optuna.create_study(
    direction="minimize",
    study_name=EXPERIMENT_NAME,
)

# Register baseline run as a trial in the study
study.add_trial(
    optuna.trial.create_trial(
        params={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_blocks": n_blocks,
            "block_channels": block_channels,
            "hidden_dim": hidden_dim,
        },
        distributions={
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "batch_size": optuna.distributions.IntDistribution(16, 128, step=16),
            "n_blocks": optuna.distributions.IntDistribution(2, 5, step=1),
            "block_channels": optuna.distributions.IntDistribution(64, 256, step=64),
            "hidden_dim": optuna.distributions.IntDistribution(128, 512, step=128),
        },
        value=-trainer.validate(model, datamodule=data_module)[0]["val_accuracy"],
    )
)

mlflow.end_run()

study.optimize(objective, n_trials=STUDY_RUNS)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_trial.value}")

print("Improvement over baseline:")
print(f"Baseline: {baseline_val_acc}")
print(f"Best: {study.best_trial.value}")
