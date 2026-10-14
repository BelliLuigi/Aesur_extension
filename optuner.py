import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import joblib
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split

from models import Autoencoder_CNN_1
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
        amplitude = np.empty((n_samples, sample_len), dtype=np.float32)
        phase = np.empty((n_samples, sample_len), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            amplitude[start:end] = np.abs(waveforms_ds[start:end]).astype(np.float32)
            phase[start:end] = get_phase(waveforms_ds[start:end]).astype(np.float32)

    ampl_tensor = torch.from_numpy(amplitude)
    phase_tensor = torch.from_numpy(phase)

    # 80/10/10 split
    a_train, a_val, _ = random_split(
        ampl_tensor, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )
    p_train, p_val, _ = random_split(
        phase_tensor, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    return (a_train, a_val), (p_train, p_val)


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
        model = Autoencoder_CNN_1(negative_slope=negative_slope).to(device)
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
    (a_train, a_val), (p_train, p_val) = load_datasets()

    N_TRIALS = 25
    EPOCHS_PER_TRIAL = 50

    # 1. Study Amplitude Autoencoder
    print("\n" + "=" * 50)
    print("STUDY 1: AMPLITUDE AUTOENCODER")
    print("=" * 50)
    pruner_ampl = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study_amplitude = optuna.create_study(
        study_name="ae_amplitude_study",
        direction="minimize",
        pruner=pruner_ampl,
    )
    study_amplitude.optimize(
        make_objective(a_train, a_val, max_epochs=EPOCHS_PER_TRIAL),
        n_trials=N_TRIALS,
    )

    print("\nBest Amplitude Trial:")
    print(f"  Validation Loss: {study_amplitude.best_value:.6f}")
    for k, v in study_amplitude.best_params.items():
        print(f"  {k}: {v}")

    # 2. Study Phase Autoencoder
    print("\n" + "=" * 50)
    print("STUDY 2: PHASE AUTOENCODER")
    print("=" * 50)
    pruner_phase = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study_phase = optuna.create_study(
        study_name="ae_phase_study",
        direction="minimize",
        pruner=pruner_phase,
    )
    study_phase.optimize(
        make_objective(p_train, p_val, max_epochs=EPOCHS_PER_TRIAL),
        n_trials=N_TRIALS,
    )

    print("\nBest Phase Trial:")
    print(f"  Validation Loss: {study_phase.best_value:.6f}")
    for k, v in study_phase.best_params.items():
        print(f"  {k}: {v}")

    # Save studies and CSV reports
    joblib.dump(study_amplitude, "ae_amplitude_study.pkl")
    joblib.dump(study_phase, "ae_phase_study.pkl")
    study_amplitude.trials_dataframe().to_csv("ae_amplitude_study.csv", index=False)
    study_phase.trials_dataframe().to_csv("ae_phase_study.csv", index=False)
    print("\nSaved studies: ae_amplitude_study.pkl, ae_phase_study.pkl")
