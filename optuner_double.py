import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import joblib
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split

from models import Autoencoder_CNN_2
from utils import get_phase, train_one_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filepath = "/lustre/ific.uv.es/ml/uv098/osvaldogramaxo/DANSur_22/data/NRHybSur3dq8_dataset.hdf"
old_filepath = "nonhyb_noprec_dataset_q8.hdf"


def validate(model, val_loader, loss_fn, device):
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vbatch in enumerate(val_loader):
            vinputs = vbatch.float().to(device)
            if vinputs.ndim == 2:
                vinputs = vinputs.unsqueeze(1)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vinputs)
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


def load_datasets():
    with h5py.File(filepath, "r") as f:
        waveforms_ds = f["Waveforms"]
        n_samples, sample_len = waveforms_ds.shape
        waveforms = np.empty((n_samples, 2, sample_len), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = waveforms_ds[start:end]
            waveforms[start:end, 0, :] = np.abs(chunk).astype(np.float32)
            waveforms[start:end, 1, :] = get_phase(chunk).astype(np.float32)

    wf_tensor = torch.from_numpy(waveforms)

    # 80/10/10 split
    train_set, val_set, _ = random_split(
        wf_tensor, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    return train_set, val_set


def make_objective(train_dataset, val_dataset, max_epochs=50):
    def objective(trial):
        # 1. Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        negative_slope = trial.suggest_float("negative_slope", 0.01, 0.3)
        lr_factor = trial.suggest_categorical("lr_factor", [0.2, 0.5])
        lr_patience = trial.suggest_int("lr_patience", 5, 15)

        # 2. Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # 3. Model, Loss, Optimizer, Scheduler
        model = Autoencoder_CNN_2(negative_slope=negative_slope).to(device)
        loss_fn = nn.L1Loss(reduction="mean")
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lr_patience, threshold=0.01, threshold_mode="rel"
        )

        best_vloss = float("inf")

        for epoch in range(max_epochs):
            _ = train_one_epoch(model, optimizer, loss_fn, device, epoch, None, train_loader)
            vloss = validate(model, val_loader, loss_fn, device)
            scheduler.step(vloss)

            if vloss < best_vloss:
                best_vloss = vloss

            # Report and check pruning
            trial.report(vloss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_vloss

    return objective


if __name__ == "__main__":
    train_set, val_set = load_datasets()

    N_TRIALS = 25
    EPOCHS_PER_TRIAL = 50

    print("\n" + "=" * 50)
    print("STUDY: DOUBLE AUTOENCODER (AMPLITUDE + PHASE)")
    print("=" * 50)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study_double = optuna.create_study(
        study_name="ae_double_study",
        direction="minimize",
        pruner=pruner,
    )
    study_double.optimize(
        make_objective(train_set, val_set, max_epochs=EPOCHS_PER_TRIAL),
        n_trials=N_TRIALS,
    )

    print("\nBest Double Autoencoder Trial:")
    print(f"  Validation Loss: {study_double.best_value:.6f}")
    for k, v in study_double.best_params.items():
        print(f"  {k}: {v}")

    # Save study and CSV report
    joblib.dump(study_double, "ae_double_study.pkl")
    study_double.trials_dataframe().to_csv("ae_double_study.csv", index=False)
    print("\nSaved study: ae_double_study.pkl, ae_double_study.csv")
