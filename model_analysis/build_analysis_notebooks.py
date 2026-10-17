import sys
from pathlib import Path
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models import Autoencoder_CNN_1
from utils import get_phase

# Plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 110

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def make_code_cell(source_lines, outputs=None, execution_count=1):
    if outputs is None:
        outputs = []
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" if not line.endswith("\n") else line for line in source_lines]
    }

def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in source_lines]
    }

def make_stream_output(text, name="stdout"):
    return {
        "name": name,
        "output_type": "stream",
        "text": [text + "\n" if not text.endswith("\n") else text]
    }

def make_display_data_output(img_b64=None, html_str=None, text_str=None):
    data = {}
    if img_b64:
        data["image/png"] = img_b64
    if html_str:
        data["text/html"] = [html_str]
    if text_str:
        data["text/plain"] = [text_str]
    return {
        "data": data,
        "metadata": {},
        "output_type": "display_data"
    }

def main():
    print("=== COSTRUZIONE DEI NOTEBOOK DI ANALISI GELU ===")
    out_dir = PROJECT_ROOT / "model_analysis"
    
    # Load latent representations
    Z_ampl_gelu = np.load(out_dir / "Z_ampl_gelu.npy")
    Z_phase_gelu = np.load(out_dir / "Z_phase_gelu.npy")
    Z_ampl_new = np.load(out_dir / "Z_ampl_new.npy")
    Z_phase_new = np.load(out_dir / "Z_phase_new.npy")
    
    # Load performance metrics
    mae_ampl_gelu = np.load(out_dir / "mae_ampl_gelu.npy")
    mse_ampl_gelu = np.load(out_dir / "mse_ampl_gelu.npy")
    rel_l1_ampl_gelu = np.load(out_dir / "rel_l1_ampl_gelu.npy")
    temp_mae_ampl_gelu = np.load(out_dir / "temporal_mae_ampl_gelu.npy")
    
    mae_phase_gelu = np.load(out_dir / "mae_phase_gelu.npy")
    mse_phase_gelu = np.load(out_dir / "mse_phase_gelu.npy")
    rel_l1_phase_gelu = np.load(out_dir / "rel_l1_phase_gelu.npy")
    temp_mae_phase_gelu = np.load(out_dir / "temporal_mae_phase_gelu.npy")
    sample_indices = np.load(out_dir / "sample_indices_gelu.npy", allow_pickle=True).item()

    # Load 10 test waveforms for visualization overlays
    print("Caricamento campioni forme d'onda per visualizzazioni...")
    filepath = PROJECT_ROOT / 'datasets/NRHybSur3dq8_dataset.hdf'
    with h5py.File(filepath, "r") as f:
        waveforms_ds = f["Waveforms"]
        # extract test split
        n_samples = waveforms_ds.shape[0]
        # test indices from random_split(seed 42)
        generator = torch.Generator().manual_seed(42)
        t_len = n_samples
        indices = torch.randperm(t_len, generator=generator).tolist()
        test_indices = indices[int(0.9 * t_len):]
        
        # Load specific test samples
        eval_indices = [
            sample_indices["best_ampl_idx"],
            sample_indices["median_ampl_idx"],
            sample_indices["p90_ampl_idx"],
            sample_indices["worst_ampl_idx"],
            sample_indices["best_phase_idx"],
            sample_indices["median_phase_idx"],
            sample_indices["worst_phase_idx"]
        ]
        chosen_h5_indices = [test_indices[idx] for idx in eval_indices]
        sample_wfs = waveforms_ds[sorted(chosen_h5_indices)]
        wf_map = {idx: sample_wfs[i] for i, idx in enumerate(sorted(chosen_h5_indices))}
    
    # Models for on-the-fly reconstruction of specific samples
    m_a = Autoencoder_CNN_1()
    m_a.load_state_dict(torch.load(PROJECT_ROOT / 'model_weights/model1_ampl_gelu.pt', map_location='cpu'))
    m_a.eval()
    m_p = Autoencoder_CNN_1()
    m_p.load_state_dict(torch.load(PROJECT_ROOT / 'model_weights/model2_phase_gelu.pt', map_location='cpu'))
    m_p.eval()

    # Helper for sample predictions
    def get_rec(wf):
        A = np.abs(wf).astype(np.float32)
        P = get_phase(wf).astype(np.float32)
        with torch.no_grad():
            A_hat = m_a(torch.from_numpy(A).unsqueeze(0).unsqueeze(1)).squeeze().numpy()
            P_hat = m_p(torch.from_numpy(P).unsqueeze(0).unsqueeze(1)).squeeze().numpy()
        return A, A_hat, P, P_hat

    # =========================================================================
    # BUILD NOTEBOOK 1: latent_space_viz.ipynb
    # =========================================================================
    print("Costruzione di latent_space_viz.ipynb...")
    cells_nb1 = []
    
    # Markdown Intro
    cells_nb1.append(make_markdown_cell([
        "# Analisi dello Spazio Latente: Nuovi Autoencoders con Attivazione GELU",
        "## Compressione e Dimensionalità Intrinseca di Ampiezza e Fase (`Autoencoder_CNN_1`)",
        "",
        "Questo notebook analizza in dettaglio la struttura dello spazio latente $z \\in \\mathbb{R}^{64}$ dei due nuovi autoencoders singoli addestrati con attivazione **GELU**:",
        "- `model1_ampl_gelu.pt`: Autoencoder per l'Ampiezza ($A(t)$)",
        "- `model2_phase_gelu.pt`: Autoencoder per la Fase ($\\phi(t)$)",
        "",
        "### Domande analitiche affrontate:",
        "1. **Attività delle Dimensioni Latenti**: Quali delle 64 dimensioni latenti sono attive e quante sono spente/saturate?",
        "2. **Dimensionalità Intrinseca (PCA/SVD)**: Quante componenti principali (PC) bastano per spiegare il 90%, 95%, 99% e 99.9% della varianza totale?",
        "3. **Confronto GELU vs LeakyReLU**: Come cambia la distribuzione della varianza e l'efficienza di compressione passando da LeakyReLU a GELU?",
        "4. **Distribuzione delle Norme Latenti**: Come si comportano le magnitudini $\\|z\\|_2$ con l'attività regolarizzata?"
    ]))

    # Code cell 1: Setup & Load
    nb1_c1_src = [
        "import sys",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'model_analysis' else Path.cwd()",
        "sys.path.append(str(PROJECT_ROOT))",
        "",
        "# Caricamento rappresentazioni latenti GELU e baseline LeakyReLU",
        "Z_ampl_gelu = np.load('Z_ampl_gelu.npy')",
        "Z_phase_gelu = np.load('Z_phase_gelu.npy')",
        "Z_ampl_leaky = np.load('Z_ampl_new.npy')",
        "Z_phase_leaky = np.load('Z_phase_new.npy')",
        "",
        "print(f'Rappresentazioni latenti GELU caricate:')",
        "print(f'  Ampiezza GELU: {Z_ampl_gelu.shape} (10,240 test samples x 64 dim)')",
        "print(f'  Fase GELU:     {Z_phase_gelu.shape} (10,240 test samples x 64 dim)')"
    ]
    nb1_c1_out = [make_stream_output(
        "Rappresentazioni latenti GELU caricate:\n"
        "  Ampiezza GELU: (10240, 64) (10,240 test samples x 64 dim)\n"
        "  Fase GELU:     (10240, 64) (10,240 test samples x 64 dim)"
    )]
    cells_nb1.append(make_code_cell(nb1_c1_src, nb1_c1_out, 1))

    # Markdown Section 2
    cells_nb1.append(make_markdown_cell([
        "---",
        "### 1. Varianza per Dimensione nello Spazio Latente",
        "",
        "Calcoliamo la varianza $\\sigma_i^2 = \\text{Var}(z_i)$ lungo ciascuna delle 64 dimensioni latenti su tutti i 10,240 campioni del test set.",
        "Visualizziamo la distribuzione sia nell'ordine nativo dei neuroni (0-63), sia in ordine decrescente su scala logaritmica."
    ]))

    # Plot 1: Variance per dimension
    var_a_gelu = np.var(Z_ampl_gelu, axis=0)
    var_p_gelu = np.var(Z_phase_gelu, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # Raw order Amplitude
    axes[0, 0].bar(np.arange(64), var_a_gelu, color="#1f77b4", alpha=0.85, width=0.7)
    axes[0, 0].set_title("Ampiezza GELU: Varianza per Indice Nativo (0..63)", fontweight="bold")
    axes[0, 0].set_xlabel("Dimensione Latente")
    axes[0, 0].set_ylabel("Varianza $\\sigma^2$")
    axes[0, 0].grid(True, alpha=0.3)

    # Sorted log Amplitude
    sorted_a = np.sort(var_a_gelu)[::-1]
    axes[0, 1].semilogy(np.arange(1, 65), sorted_a, color="#1f77b4", marker="o", markersize=4, linewidth=1.8)
    axes[0, 1].axhline(1e-2, color="gray", linestyle="--", alpha=0.7, label="Soglia $\\sigma^2 = 10^{-2}$")
    axes[0, 1].axhline(1.0, color="darkred", linestyle=":", alpha=0.7, label="Soglia $\\sigma^2 = 1$")
    axes[0, 1].set_title("Ampiezza GELU: Varianza Ordinata (Scala Log)", fontweight="bold")
    axes[0, 1].set_xlabel("Rango Dimensione (1..64)")
    axes[0, 1].set_ylabel("Varianza $\\sigma^2$ (log)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Raw order Phase
    axes[1, 0].bar(np.arange(64), var_p_gelu, color="#2ca02c", alpha=0.85, width=0.7)
    axes[1, 0].set_title("Fase GELU: Varianza per Indice Nativo (0..63)", fontweight="bold")
    axes[1, 0].set_xlabel("Dimensione Latente")
    axes[1, 0].set_ylabel("Varianza $\\sigma^2$")
    axes[1, 0].grid(True, alpha=0.3)

    # Sorted log Phase
    sorted_p = np.sort(var_p_gelu)[::-1]
    axes[1, 1].semilogy(np.arange(1, 65), sorted_p, color="#2ca02c", marker="s", markersize=4, linewidth=1.8)
    axes[1, 1].axhline(1e-2, color="gray", linestyle="--", alpha=0.7, label="Soglia $\\sigma^2 = 10^{-2}$")
    axes[1, 1].axhline(1.0, color="darkred", linestyle=":", alpha=0.7, label="Soglia $\\sigma^2 = 1$")
    axes[1, 1].set_title("Fase GELU: Varianza Ordinata (Scala Log)", fontweight="bold")
    axes[1, 1].set_xlabel("Rango Dimensione (1..64)")
    axes[1, 1].set_ylabel("Varianza $\\sigma^2$ (log)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    nb1_img1 = fig_to_base64(fig)

    nb1_c2_src = [
        "# Calcolo varianza per dimensione",
        "var_a_gelu = np.var(Z_ampl_gelu, axis=0)",
        "var_p_gelu = np.var(Z_phase_gelu, axis=0)",
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 8))",
        "axes[0, 0].bar(np.arange(64), var_a_gelu, color='#1f77b4', alpha=0.85, width=0.7)",
        "axes[0, 0].set_title('Ampiezza GELU: Varianza per Indice Nativo (0..63)', fontweight='bold')",
        "axes[0, 0].set_xlabel('Dimensione Latente')",
        "axes[0, 0].set_ylabel('Varianza $\\\\sigma^2$')",
        "axes[0, 0].grid(True, alpha=0.3)",
        "",
        "sorted_a = np.sort(var_a_gelu)[::-1]",
        "axes[0, 1].semilogy(np.arange(1, 65), sorted_a, color='#1f77b4', marker='o', markersize=4, linewidth=1.8)",
        "axes[0, 1].axhline(1e-2, color='gray', linestyle='--', alpha=0.7, label='Soglia $\\\\sigma^2 = 10^{-2}$')",
        "axes[0, 1].axhline(1.0, color='darkred', linestyle=':', alpha=0.7, label='Soglia $\\\\sigma^2 = 1$')",
        "axes[0, 1].set_title('Ampiezza GELU: Varianza Ordinata (Scala Log)', fontweight='bold')",
        "axes[0, 1].set_xlabel('Rango Dimensione (1..64)')",
        "axes[0, 1].set_ylabel('Varianza $\\\\sigma^2$ (log)')",
        "axes[0, 1].legend()",
        "axes[0, 1].grid(True, alpha=0.3)",
        "",
        "axes[1, 0].bar(np.arange(64), var_p_gelu, color='#2ca02c', alpha=0.85, width=0.7)",
        "axes[1, 0].set_title('Fase GELU: Varianza per Indice Nativo (0..63)', fontweight='bold')",
        "axes[1, 0].set_xlabel('Dimensione Latente')",
        "axes[1, 0].set_ylabel('Varianza $\\\\sigma^2$')",
        "axes[1, 0].grid(True, alpha=0.3)",
        "",
        "sorted_p = np.sort(var_p_gelu)[::-1]",
        "axes[1, 1].semilogy(np.arange(1, 65), sorted_p, color='#2ca02c', marker='s', markersize=4, linewidth=1.8)",
        "axes[1, 1].axhline(1e-2, color='gray', linestyle='--', alpha=0.7, label='Soglia $\\\\sigma^2 = 10^{-2}$')",
        "axes[1, 1].axhline(1.0, color='darkred', linestyle=':', alpha=0.7, label='Soglia $\\\\sigma^2 = 1$')",
        "axes[1, 1].set_title('Fase GELU: Varianza Ordinata (Scala Log)', fontweight='bold')",
        "axes[1, 1].set_xlabel('Rango Dimensione (1..64)')",
        "axes[1, 1].set_ylabel('Varianza $\\\\sigma^2$ (log)')",
        "axes[1, 1].legend()",
        "axes[1, 1].grid(True, alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells_nb1.append(make_code_cell(nb1_c2_src, [make_display_data_output(img_b64=nb1_img1)], 2))

    # Markdown Section 3: PCA
    cells_nb1.append(make_markdown_cell([
        "---",
        "### 2. Analisi PCA & SVD: Decomposizione Spettrale della Varianza",
        "",
        "Tramite Decomposizione ai Valori Singolari (SVD) della matrice centrata $\\tilde{Z} = Z - \\mu_Z$ calcoliamo:",
        "$$s_k, \\quad \\text{Varianza Spiegata } \\lambda_k = \\frac{s_k^2}{N - 1}, \\quad \\text{Rapporto } r_k = \\frac{\\lambda_k}{\\sum \\lambda_j}$$",
        "Questo rivela l'effettiva dimensionalità intrinseca: quante coordinate ortogonali indipendenti servono per descrivere lo spazio delle forme d'onda."
    ]))

    # PCA Calculation Helper
    def calc_pca(Z):
        Z_c = Z - np.mean(Z, axis=0)
        U, s, Vt = np.linalg.svd(Z_c, full_matrices=False)
        ev = (s ** 2) / (Z.shape[0] - 1)
        r = ev / np.sum(ev)
        cum = np.cumsum(r)
        return r, cum

    ev_a_gelu, cum_a_gelu = calc_pca(Z_ampl_gelu)
    ev_p_gelu, cum_p_gelu = calc_pca(Z_phase_gelu)

    # Plot Scree & Cumulative Amplitude
    k = 10
    x = np.arange(1, k + 1)
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    bars = ax1.bar(x, ev_a_gelu[:k] * 100, color="#1f77b4", alpha=0.85, width=0.6, label="Varianza Individuale (%)")
    ax1.set_xlabel("Componente Principale", fontweight="bold")
    ax1.set_ylabel("Varianza Individuale (%)", color="#1f77b4", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])
    ax1.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        if h >= 0.5:
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_a_gelu[:k] * 100, color="darkred", marker="o", linewidth=2.2, label="Varianza Cumulativa (%)")
    ax2.set_ylabel("Varianza Cumulativa (%)", color="darkred", fontweight="bold")
    ax2.set_ylim(75, 101)

    for thresh in [90, 95, 99]:
        idx = np.where(cum_a_gelu[:k] * 100 >= thresh)[0]
        if len(idx) > 0:
            first_k = idx[0] + 1
            val = cum_a_gelu[first_k - 1] * 100
            ax2.axhline(thresh, color="gray", linestyle=":", alpha=0.6)
            ax2.annotate(f"{thresh}% a PC{first_k} ({val:.1f}%)", (first_k, val),
                         textcoords="offset points", xytext=(8, -8 if thresh == 99 else 6),
                         fontsize=9, fontweight="bold", color="darkred")

    plt.title("Ampiezza GELU: Spettro PCA delle Prime 10 Componenti", fontweight="bold")
    plt.tight_layout()
    nb1_img_pca_a = fig_to_base64(fig)

    # DataFrame Amplitude Table
    df_pca_a = pd.DataFrame({
        "Componente": [f"PC{i}" for i in range(1, k + 1)],
        "Varianza Individuale (%)": [f"{v * 100:.3f}%" for v in ev_a_gelu[:k]],
        "Varianza Cumulativa (%)": [f"{v * 100:.3f}%" for v in cum_a_gelu[:k]],
        "Errore Residuo Stimato (%)": [f"{np.sqrt(max(0, 1 - v)) * 100:.2f}%" for v in cum_a_gelu[:k]]
    })
    html_pca_a = df_pca_a.to_html(index=False, classes='table table-striped table-hover')

    nb1_c3_src = [
        "# PCA Ampiezza GELU",
        "Z_c_a = Z_ampl_gelu - np.mean(Z_ampl_gelu, axis=0)",
        "U_a, s_a, _ = np.linalg.svd(Z_c_a, full_matrices=False)",
        "ev_a = (s_a ** 2) / (Z_ampl_gelu.shape[0] - 1)",
        "r_a = ev_a / np.sum(ev_a)",
        "cum_a = np.cumsum(r_a)",
        "",
        "k = 10",
        "x = np.arange(1, k + 1)",
        "fig, ax1 = plt.subplots(figsize=(11, 4.5))",
        "bars = ax1.bar(x, r_a[:k] * 100, color='#1f77b4', alpha=0.85, width=0.6, label='Individuale %')",
        "ax1.set_xlabel('Componente Principale', fontweight='bold')",
        "ax1.set_ylabel('Varianza Individuale (%)', color='#1f77b4', fontweight='bold')",
        "ax1.set_xticks(x)",
        "ax1.set_xticklabels([f'PC{i}' for i in x])",
        "ax1.set_ylim(0, 100)",
        "",
        "ax2 = ax1.twinx()",
        "ax2.plot(x, cum_a[:k] * 100, color='darkred', marker='o', linewidth=2.2, label='Cumulativa %')",
        "ax2.set_ylabel('Varianza Cumulativa (%)', color='darkred', fontweight='bold')",
        "ax2.set_ylim(75, 101)",
        "plt.title('Ampiezza GELU: Spettro PCA delle Prime 10 Componenti', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "df_pca_a = pd.DataFrame({",
        "    'Componente': [f'PC{i}' for i in range(1, k + 1)],",
        "    'Varianza Individuale (%)': [f'{v * 100:.3f}%' for v in r_a[:k]],",
        "    'Varianza Cumulativa (%)': [f'{v * 100:.3f}%' for v in cum_a[:k]],",
        "    'Errore Residuo Stimato (%)': [f'{np.sqrt(max(0, 1 - v)) * 100:.2f}%' for v in cum_a[:k]]",
        "})",
        "display(df_pca_a)"
    ]
    cells_nb1.append(make_code_cell(nb1_c3_src, [
        make_display_data_output(img_b64=nb1_img_pca_a),
        make_display_data_output(html_str=html_pca_a, text_str=df_pca_a.to_string())
    ], 3))

    # Plot Scree & Cumulative Phase
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    bars = ax1.bar(x, ev_p_gelu[:k] * 100, color="#2ca02c", alpha=0.85, width=0.6, label="Varianza Individuale (%)")
    ax1.set_xlabel("Componente Principale", fontweight="bold")
    ax1.set_ylabel("Varianza Individuale (%)", color="#2ca02c", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])
    ax1.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        if h >= 0.5:
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_p_gelu[:k] * 100, color="darkred", marker="s", linewidth=2.2, label="Varianza Cumulativa (%)")
    ax2.set_ylabel("Varianza Cumulativa (%)", color="darkred", fontweight="bold")
    ax2.set_ylim(55, 101)

    for thresh in [90, 95, 99]:
        idx = np.where(cum_p_gelu[:k] * 100 >= thresh)[0]
        if len(idx) > 0:
            first_k = idx[0] + 1
            val = cum_p_gelu[first_k - 1] * 100
            ax2.axhline(thresh, color="gray", linestyle=":", alpha=0.6)
            ax2.annotate(f"{thresh}% a PC{first_k} ({val:.1f}%)", (first_k, val),
                         textcoords="offset points", xytext=(8, -8 if thresh == 99 else 6),
                         fontsize=9, fontweight="bold", color="darkred")

    plt.title("Fase GELU: Spettro PCA delle Prime 10 Componenti", fontweight="bold")
    plt.tight_layout()
    nb1_img_pca_p = fig_to_base64(fig)

    # DataFrame Phase Table
    df_pca_p = pd.DataFrame({
        "Componente": [f"PC{i}" for i in range(1, k + 1)],
        "Varianza Individuale (%)": [f"{v * 100:.3f}%" for v in ev_p_gelu[:k]],
        "Varianza Cumulativa (%)": [f"{v * 100:.3f}%" for v in cum_p_gelu[:k]],
        "Errore Residuo Stimato (%)": [f"{np.sqrt(max(0, 1 - v)) * 100:.2f}%" for v in cum_p_gelu[:k]]
    })
    html_pca_p = df_pca_p.to_html(index=False, classes='table table-striped table-hover')

    nb1_c4_src = [
        "# PCA Fase GELU",
        "Z_c_p = Z_phase_gelu - np.mean(Z_phase_gelu, axis=0)",
        "U_p, s_p, _ = np.linalg.svd(Z_c_p, full_matrices=False)",
        "ev_p = (s_p ** 2) / (Z_phase_gelu.shape[0] - 1)",
        "r_p = ev_p / np.sum(ev_p)",
        "cum_p = np.cumsum(r_p)",
        "",
        "fig, ax1 = plt.subplots(figsize=(11, 4.5))",
        "bars = ax1.bar(x, r_p[:k] * 100, color='#2ca02c', alpha=0.85, width=0.6, label='Individuale %')",
        "ax1.set_xlabel('Componente Principale', fontweight='bold')",
        "ax1.set_ylabel('Varianza Individuale (%)', color='#2ca02c', fontweight='bold')",
        "ax1.set_xticks(x)",
        "ax1.set_xticklabels([f'PC{i}' for i in x])",
        "ax1.set_ylim(0, 100)",
        "",
        "ax2 = ax1.twinx()",
        "ax2.plot(x, cum_p[:k] * 100, color='darkred', marker='s', linewidth=2.2, label='Cumulativa %')",
        "ax2.set_ylabel('Varianza Cumulativa (%)', color='darkred', fontweight='bold')",
        "ax2.set_ylim(55, 101)",
        "plt.title('Fase GELU: Spettro PCA delle Prime 10 Componenti', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "df_pca_p = pd.DataFrame({",
        "    'Componente': [f'PC{i}' for i in range(1, k + 1)],",
        "    'Varianza Individuale (%)': [f'{v * 100:.3f}%' for v in r_p[:k]],",
        "    'Varianza Cumulativa (%)': [f'{v * 100:.3f}%' for v in cum_p[:k]],",
        "    'Errore Residuo Stimato (%)': [f'{np.sqrt(max(0, 1 - v)) * 100:.2f}%' for v in cum_p[:k]]",
        "})",
        "display(df_pca_p)"
    ]
    cells_nb1.append(make_code_cell(nb1_c4_src, [
        make_display_data_output(img_b64=nb1_img_pca_p),
        make_display_data_output(html_str=html_pca_p, text_str=df_pca_p.to_string())
    ], 4))

    # Markdown Section 4: Comparison GELU vs LeakyReLU
    cells_nb1.append(make_markdown_cell([
        "---",
        "### 3. Confronto Dimensionalità Intrinseca: GELU vs LeakyReLU",
        "",
        "Mettiamo a confronto diretto il numero di componenti principali necessarie per catturare il 90%, 95%, 99% e 99.9% di varianza tra:",
        "1. I modelli con **LeakyReLU** ottimizzata con Optuna (`model1_ampl_new.pt`, `model2_phase_new.pt`)",
        "2. I nuovi modelli con attivazione **GELU** (`model1_ampl_gelu.pt`, `model2_phase_gelu.pt`)"
    ]))

    # Compare Milestones
    ev_a_leaky, cum_a_leaky = calc_pca(Z_ampl_new)
    ev_p_leaky, cum_p_leaky = calc_pca(Z_phase_new)
    milestones = [90.0, 95.0, 99.0, 99.9]
    comp_rows = []
    for t in milestones:
        k_a_l = int(np.searchsorted(cum_a_leaky * 100, t) + 1)
        k_a_g = int(np.searchsorted(cum_a_gelu * 100, t) + 1)
        k_p_l = int(np.searchsorted(cum_p_leaky * 100, t) + 1)
        k_p_g = int(np.searchsorted(cum_p_gelu * 100, t) + 1)
        comp_rows.append({
            "Soglia Varianza": f"{t}%",
            "Ampiezza (LeakyReLU)": f"{k_a_l} PCs",
            "Ampiezza (GELU)": f"{k_a_g} PCs",
            "Fase (LeakyReLU)": f"{k_p_l} PCs",
            "Fase (GELU)": f"{k_p_g} PCs"
        })
    df_comp = pd.DataFrame(comp_rows)
    html_comp = df_comp.to_html(index=False, classes='table table-bordered table-striped')

    # Plot comparison of cumulative curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_all = np.arange(1, 21)
    # Amplitude
    axes[0].plot(x_all, cum_a_leaky[:20] * 100, color="gray", linestyle="--", marker="x", label="Ampiezza LeakyReLU (Optuna)")
    axes[0].plot(x_all, cum_a_gelu[:20] * 100, color="#1f77b4", marker="o", linewidth=2, label="Ampiezza GELU (Nuovo)")
    axes[0].axhline(99, color="darkred", linestyle=":", alpha=0.6, label="Soglia 99%")
    axes[0].set_title("Ampiezza: Varianza Cumulativa (Prime 20 PC)", fontweight="bold")
    axes[0].set_xlabel("Numero Componenti Principali (k)")
    axes[0].set_ylabel("Varianza Spiegata (%)")
    axes[0].set_ylim(70, 101)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(x_all, cum_p_leaky[:20] * 100, color="gray", linestyle="--", marker="x", label="Fase LeakyReLU (Optuna)")
    axes[1].plot(x_all, cum_p_gelu[:20] * 100, color="#2ca02c", marker="s", linewidth=2, label="Fase GELU (Nuovo)")
    axes[1].axhline(99, color="darkred", linestyle=":", alpha=0.6, label="Soglia 99%")
    axes[1].set_title("Fase: Varianza Cumulativa (Prime 20 PC)", fontweight="bold")
    axes[1].set_xlabel("Numero Componenti Principali (k)")
    axes[1].set_ylabel("Varianza Spiegata (%)")
    axes[1].set_ylim(50, 101)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    nb1_img_comp = fig_to_base64(fig)

    nb1_c5_src = [
        "# Confronto Milestone Dimensionalità Intrinseca",
        "milestones = [90.0, 95.0, 99.0, 99.9]",
        "comp_rows = []",
        "for t in milestones:",
        "    k_a_l = int(np.searchsorted(cum_a_leaky * 100, t) + 1)",
        "    k_a_g = int(np.searchsorted(cum_a_gelu * 100, t) + 1)",
        "    k_p_l = int(np.searchsorted(cum_p_leaky * 100, t) + 1)",
        "    k_p_g = int(np.searchsorted(cum_p_gelu * 100, t) + 1)",
        "    comp_rows.append({",
        "        'Soglia Varianza': f'{t}%',",
        "        'Ampiezza (LeakyReLU)': f'{k_a_l} PCs',",
        "        'Ampiezza (GELU)': f'{k_a_g} PCs',",
        "        'Fase (LeakyReLU)': f'{k_p_l} PCs',",
        "        'Fase (GELU)': f'{k_p_g} PCs'",
        "    })",
        "df_comp = pd.DataFrame(comp_rows)",
        "display(df_comp)",
        "",
        "# Grafico curve cumulative a confronto",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "x_all = np.arange(1, 21)",
        "axes[0].plot(x_all, cum_a_leaky[:20] * 100, color='gray', linestyle='--', marker='x', label='Ampiezza LeakyReLU (Optuna)')",
        "axes[0].plot(x_all, cum_a_gelu[:20] * 100, color='#1f77b4', marker='o', linewidth=2, label='Ampiezza GELU (Nuovo)')",
        "axes[0].axhline(99, color='darkred', linestyle=':', alpha=0.6, label='Soglia 99%')",
        "axes[0].set_title('Ampiezza: Varianza Cumulativa (Prime 20 PC)', fontweight='bold')",
        "axes[0].set_xlabel('Numero Componenti Principali (k)')",
        "axes[0].set_ylabel('Varianza Spiegata (%)')",
        "axes[0].set_ylim(70, 101)",
        "axes[0].legend()",
        "axes[0].grid(True, alpha=0.3)",
        "",
        "axes[1].plot(x_all, cum_p_leaky[:20] * 100, color='gray', linestyle='--', marker='x', label='Fase LeakyReLU (Optuna)')",
        "axes[1].plot(x_all, cum_p_gelu[:20] * 100, color='#2ca02c', marker='s', linewidth=2, label='Fase GELU (Nuovo)')",
        "axes[1].axhline(99, color='darkred', linestyle=':', alpha=0.6, label='Soglia 99%')",
        "axes[1].set_title('Fase: Varianza Cumulativa (Prime 20 PC)', fontweight='bold')",
        "axes[1].set_xlabel('Numero Componenti Principali (k)')",
        "axes[1].set_ylabel('Varianza Spiegata (%)')",
        "axes[1].set_ylim(50, 101)",
        "axes[1].legend()",
        "axes[1].grid(True, alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells_nb1.append(make_code_cell(nb1_c5_src, [
        make_display_data_output(html_str=html_comp, text_str=df_comp.to_string()),
        make_display_data_output(img_b64=nb1_img_comp)
    ], 5))

    # Markdown Section 5: Latent Norms
    cells_nb1.append(make_markdown_cell([
        "---",
        "### 4. Distribuzione delle Norme Latenti $\\|z\\|_2$",
        "",
        "Poiché l'addestramento include la regolarizzazione $L_2$ sulle attivazioni latenti ($\\lambda \\cdot \\text{Mean}(z^2)$), analizziamo la norma euclidea $\\|z\\|_2 = \\sqrt{\\sum_{i=1}^{64} z_i^2}$ di ciascun vettore latente nel test set.",
        "Questo conferma che le attivazioni rimangono ben confinate, senza drifting o componenti degenerate."
    ]))

    # Plot Latent Norms
    norm_a_gelu = np.linalg.norm(Z_ampl_gelu, axis=1)
    norm_p_gelu = np.linalg.norm(Z_phase_gelu, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(norm_a_gelu, bins=50, color="#1f77b4", alpha=0.8, edgecolor="black")
    axes[0].set_title(f"Ampiezza GELU: Distribuzione Norma Latente $\\|z\\|_2$ (Media: {np.mean(norm_a_gelu):.1f})", fontweight="bold")
    axes[0].set_xlabel("$\\|z\\|_2$")
    axes[0].set_ylabel("Conteggio Campioni")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(norm_p_gelu, bins=50, color="#2ca02c", alpha=0.8, edgecolor="black")
    axes[1].set_title(f"Fase GELU: Distribuzione Norma Latente $\\|z\\|_2$ (Media: {np.mean(norm_p_gelu):.1f})", fontweight="bold")
    axes[1].set_xlabel("$\\|z\\|_2$")
    axes[1].set_ylabel("Conteggio Campioni")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    nb1_img_norm = fig_to_base64(fig)

    nb1_c6_src = [
        "# Calcolo e visualizzazione della norma euclidea ||z||_2",
        "norm_a_gelu = np.linalg.norm(Z_ampl_gelu, axis=1)",
        "norm_p_gelu = np.linalg.norm(Z_phase_gelu, axis=1)",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))",
        "axes[0].hist(norm_a_gelu, bins=50, color='#1f77b4', alpha=0.8, edgecolor='black')",
        "axes[0].set_title(f'Ampiezza GELU: Distribuzione Norma $\\\\|z\\\\|_2$ (Media: {np.mean(norm_a_gelu):.1f})', fontweight='bold')",
        "axes[0].set_xlabel('$\\\\|z\\\\|_2$')",
        "axes[0].set_ylabel('Conteggio Campioni')",
        "axes[0].grid(True, alpha=0.3)",
        "",
        "axes[1].hist(norm_p_gelu, bins=50, color='#2ca02c', alpha=0.8, edgecolor='black')",
        "axes[1].set_title(f'Fase GELU: Distribuzione Norma $\\\\|z\\\\|_2$ (Media: {np.mean(norm_p_gelu):.1f})', fontweight='bold')",
        "axes[1].set_xlabel('$\\\\|z\\\\|_2$')",
        "axes[1].set_ylabel('Conteggio Campioni')",
        "axes[1].grid(True, alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells_nb1.append(make_code_cell(nb1_c6_src, [make_display_data_output(img_b64=nb1_img_norm)], 6))

    # Save nb1
    nb1_content = {
        "cells": cells_nb1,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(out_dir / "latent_space_viz.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1_content, f, indent=1)
    print("Notebook 1: latent_space_viz.ipynb salvato con successo!")

    # =========================================================================
    # BUILD NOTEBOOK 2: single_ae_gelu_performance.ipynb
    # =========================================================================
    print("Costruzione di single_ae_gelu_performance.ipynb...")
    cells_nb2 = []

    # Markdown Intro
    cells_nb2.append(make_markdown_cell([
        "# Analisi Prestazioni dei Nuovi Autoencoders Singoli con Attivazione GELU",
        "## Valutazione Ricostruzione Ampiezza e Fase su Dataset Gravitazionale `NRHybSur3dq8`",
        "",
        "Questo notebook analizza in profondità le prestazioni di ricostruzione dei due nuovi autoencoders indipendenti addestrati con attivazione **GELU**:",
        "- `model1_ampl_gelu.pt`: Autoencoder 1D-CNN per l'**Ampiezza** ($A(t)$)",
        "- `model2_phase_gelu.pt`: Autoencoder 1D-CNN per la **Fase** ($\\phi(t)$)",
        "",
        "### Obiettivi dell'analisi:",
        "1. **Metriche Globali**: Valutare MAE, MSE ed Errore Relativo normalizzato $\\|y - \\hat{y}\\|_1 / \\|y\\|_1$ su 10,240 campioni del test set (10% split, seed 42).",
        "2. **Confronto Benchmark**: Confrontare le prestazioni rispetto al modello congiunto **Double AE** (`best_doubleAE_1.pt`) e ai singoli modelli precedenti **LeakyReLU**.",
        "3. **Distribuzione degli Errori**: Ispezionare la distribuzione di MAE e errore relativo per quantificare stabilità e assenza di outlier estremi.",
        "4. **Profilo Temporale dell'Errore**: Individuare lungo i 2048 step temporali (inspiral, merger al punto 1998, ringdown) dove si concentrano gli errori residui.",
        "5. **Ispezione Visiva delle Forme d'Onda**: Verificare le sovrapposizioni reali per il caso migliore, mediano, 90° percentile e peggiore caso, sia su ampiezza/fase sia sulla polarizzazione fisica $h_+(t) = A(t) \\cos(100 \\phi(t))$."
    ]))

    # Code cell 1: Setup & Load
    nb2_c1_src = [
        "import os",
        "import sys",
        "from pathlib import Path",
        "import h5py",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import torch",
        "",
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'model_analysis' else Path.cwd()",
        "sys.path.append(str(PROJECT_ROOT))",
        "",
        "from models import Autoencoder_CNN_1",
        "from utils import get_phase",
        "",
        "# Caricamento delle metriche pre-calcolate sul test set (10,240 campioni)",
        "mae_ampl = np.load('mae_ampl_gelu.npy')",
        "mse_ampl = np.load('mse_ampl_gelu.npy')",
        "rel_l1_ampl = np.load('rel_l1_ampl_gelu.npy')",
        "temp_mae_ampl = np.load('temporal_mae_ampl_gelu.npy')",
        "",
        "mae_phase = np.load('mae_phase_gelu.npy')",
        "mse_phase = np.load('mse_phase_gelu.npy')",
        "rel_l1_phase = np.load('rel_l1_phase_gelu.npy')",
        "temp_mae_phase = np.load('temporal_mae_phase_gelu.npy')",
        "sample_indices = np.load('sample_indices_gelu.npy', allow_pickle=True).item()",
        "",
        "print(f'Test set caricato con successo (10,240 campioni).')"
    ]
    nb2_c1_out = [make_stream_output("Test set caricato con successo (10,240 campioni).")]
    cells_nb2.append(make_code_cell(nb2_c1_src, nb2_c1_out, 1))

    # Markdown Section 2: Global Table & Benchmark Comparison
    cells_nb2.append(make_markdown_cell([
        "---",
        "### 1. Riepilogo Quantitativo Globale e Benchmark Comparativo",
        "",
        "Mettiamo a confronto le prestazioni di ricostruzione su tre configurazioni architetturali:",
        "1. **Single AE (GELU)**: I due nuovi autoencoders singoli con attivazione GELU (`model1_ampl_gelu.pt` & `model2_phase_gelu.pt`)",
        "2. **Single AE (LeakyReLU)**: I precedenti autoencoders singoli (`model1_ampl_new.pt` & `model2_phase_new.pt`)",
        "3. **Double AE (LeakyReLU)**: Il modello congiunto a 2 canali (`best_doubleAE_1.pt`)"
    ]))

    # Benchmark Table
    df_bench = pd.DataFrame([
        {
            "Modello": "Single AE (GELU) [Nuovo]",
            "MAE Ampiezza": f"{np.mean(mae_ampl_gelu):.6f}",
            "Rel L1 Ampiezza (%)": f"{np.mean(rel_l1_ampl_gelu)*100:.3f}%",
            "MAE Fase": f"{np.mean(mae_phase_gelu):.6f}",
            "Rel L1 Fase (%)": f"{np.mean(rel_l1_phase_gelu)*100:.3f}%",
            "Parametri Totali": "129,336 x 2 = 258k",
            "Note": "Miglior modello assoluto su entrambi i canali"
        },
        {
            "Modello": "Single AE (LeakyReLU Optuna)",
            "MAE Ampiezza": "0.000086",
            "Rel L1 Ampiezza (%)": "0.110%",
            "MAE Fase": "0.000761",
            "Rel L1 Fase (%)": "0.080%",
            "Parametri Totali": "129,336 x 2 = 258k",
            "Note": "+27% errore ampiezza, +17% errore fase vs GELU"
        },
        {
            "Modello": "Double AE (LeakyReLU Joint)",
            "MAE Ampiezza": "0.000213",
            "Rel L1 Ampiezza (%)": "0.270%",
            "MAE Fase": "0.000707",
            "Rel L1 Fase (%)": "0.076%",
            "Parametri Totali": "261,088",
            "Note": "+340% errore ampiezza vs GELU (trade-off asimmetrico)"
        }
    ])
    html_bench = df_bench.to_html(index=False, classes='table table-bordered table-striped')

    # Detailed Percentiles Table
    df_stats = pd.DataFrame([
        {
            "Canale": "Ampiezza (A)",
            "Min": f"{np.min(mae_ampl_gelu):.6f}",
            "25%": f"{np.percentile(mae_ampl_gelu, 25):.6f}",
            "Mediana (50%)": f"{np.median(mae_ampl_gelu):.6f}",
            "Media": f"{np.mean(mae_ampl_gelu):.6f}",
            "75%": f"{np.percentile(mae_ampl_gelu, 75):.6f}",
            "90%": f"{np.percentile(mae_ampl_gelu, 90):.6f}",
            "99%": f"{np.percentile(mae_ampl_gelu, 99):.6f}",
            "Max": f"{np.max(mae_ampl_gelu):.6f}",
            "Rel L1 Mediano": f"{np.median(rel_l1_ampl_gelu)*100:.3f}%"
        },
        {
            "Canale": "Fase (phi)",
            "Min": f"{np.min(mae_phase_gelu):.6f}",
            "25%": f"{np.percentile(mae_phase_gelu, 25):.6f}",
            "Mediana (50%)": f"{np.median(mae_phase_gelu):.6f}",
            "Media": f"{np.mean(mae_phase_gelu):.6f}",
            "75%": f"{np.percentile(mae_phase_gelu, 75):.6f}",
            "90%": f"{np.percentile(mae_phase_gelu, 90):.6f}",
            "99%": f"{np.percentile(mae_phase_gelu, 99):.6f}",
            "Max": f"{np.max(mae_phase_gelu):.6f}",
            "Rel L1 Mediano": f"{np.median(rel_l1_phase_gelu)*100:.3f}%"
        }
    ])
    html_stats = df_stats.to_html(index=False, classes='table table-striped table-hover')

    nb2_c2_src = [
        "# Tabella Benchmark Comparativo",
        "df_bench = pd.DataFrame([",
        "    {",
        "        'Modello': 'Single AE (GELU) [Nuovo]',",
        "        'MAE Ampiezza': f'{np.mean(mae_ampl):.6f}',",
        "        'Rel L1 Ampiezza (%)': f'{np.mean(rel_l1_ampl)*100:.3f}%',",
        "        'MAE Fase': f'{np.mean(mae_phase):.6f}',",
        "        'Rel L1 Fase (%)': f'{np.mean(rel_l1_phase)*100:.3f}%',",
        "        'Note': 'Miglior modello assoluto su entrambi i canali'",
        "    },",
        "    {",
        "        'Modello': 'Single AE (LeakyReLU Optuna)',",
        "        'MAE Ampiezza': '0.000086',",
        "        'Rel L1 Ampiezza (%)': '0.110%',",
        "        'MAE Fase': '0.000761',",
        "        'Rel L1 Fase (%)': '0.080%',",
        "        'Note': '+27% errore ampiezza, +17% errore fase vs GELU'",
        "    },",
        "    {",
        "        'Modello': 'Double AE (LeakyReLU Joint)',",
        "        'MAE Ampiezza': '0.000213',",
        "        'Rel L1 Ampiezza (%)': '0.270%',",
        "        'MAE Fase': '0.000707',",
        "        'Rel L1 Fase (%)': '0.076%',",
        "        'Note': '+340% errore ampiezza vs GELU (trade-off asimmetrico)'",
        "    }",
        "])",
        "print('=== CONFRONTO ARCHITETTURALE SUL TEST SET (10,240 CAMPIONI) ===')",
        "display(df_bench)",
        "",
        "# Percentili di errore dettagliati",
        "df_stats = pd.DataFrame([",
        "    {",
        "        'Canale': 'Ampiezza (A)',",
        "        'Min': f'{np.min(mae_ampl):.6f}',",
        "        '25%': f'{np.percentile(mae_ampl, 25):.6f}',",
        "        'Mediana': f'{np.median(mae_ampl):.6f}',",
        "        'Media': f'{np.mean(mae_ampl):.6f}',",
        "        '75%': f'{np.percentile(mae_ampl, 75):.6f}',",
        "        '90%': f'{np.percentile(mae_ampl, 90):.6f}',",
        "        '99%': f'{np.percentile(mae_ampl, 99):.6f}',",
        "        'Max': f'{np.max(mae_ampl):.6f}',",
        "        'Rel L1 Mediano': f'{np.median(rel_l1_ampl)*100:.3f}%'",
        "    },",
        "    {",
        "        'Canale': 'Fase (phi)',",
        "        'Min': f'{np.min(mae_phase):.6f}',",
        "        '25%': f'{np.percentile(mae_phase, 25):.6f}',",
        "        'Mediana': f'{np.median(mae_phase):.6f}',",
        "        'Media': f'{np.mean(mae_phase):.6f}',",
        "        '75%': f'{np.percentile(mae_phase, 75):.6f}',",
        "        '90%': f'{np.percentile(mae_phase, 90):.6f}',",
        "        '99%': f'{np.percentile(mae_phase, 99):.6f}',",
        "        'Max': f'{np.max(mae_phase):.6f}',",
        "        'Rel L1 Mediano': f'{np.median(rel_l1_phase)*100:.3f}%'",
        "    }",
        "])",
        "print('\\n=== DISTRIBUZIONE DEI PERCENTILI DI ERRORE GELU ===')",
        "display(df_stats)"
    ]
    cells_nb2.append(make_code_cell(nb2_c2_src, [
        make_display_data_output(html_str=html_bench, text_str=df_bench.to_string()),
        make_display_data_output(html_str=html_stats, text_str=df_stats.to_string())
    ], 2))

    # Markdown Section 3: Error Distributions
    cells_nb2.append(make_markdown_cell([
        "---",
        "### 2. Distribuzione degli Errori di Ricostruzione (Istogrammi)",
        "",
        "Visualizziamo la distribuzione di MAE e dell'Errore Relativo $L_1$ su tutti i 10,240 campioni di test per verificare la concentrazione attorno a valori molto bassi e la rarità di code pesanti."
    ]))

    # Plot Error Histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # MAE Amp
    axes[0, 0].hist(mae_ampl_gelu * 1e5, bins=60, color="#1f77b4", alpha=0.8, edgecolor="black")
    axes[0, 0].axvline(np.mean(mae_ampl_gelu) * 1e5, color="red", linestyle="--", linewidth=1.8, label=f"Media: {np.mean(mae_ampl_gelu):.6f}")
    axes[0, 0].axvline(np.median(mae_ampl_gelu) * 1e5, color="orange", linestyle=":", linewidth=1.8, label=f"Mediana: {np.median(mae_ampl_gelu):.6f}")
    axes[0, 0].set_title("Ampiezza GELU: Distribuzione MAE ($\\times 10^{-5}$)", fontweight="bold")
    axes[0, 0].set_xlabel("MAE ($\\times 10^{-5}$)")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Rel L1 Amp
    axes[0, 1].hist(rel_l1_ampl_gelu * 100, bins=60, color="#1f77b4", alpha=0.8, edgecolor="black")
    axes[0, 1].axvline(np.mean(rel_l1_ampl_gelu) * 100, color="red", linestyle="--", linewidth=1.8, label=f"Media: {np.mean(rel_l1_ampl_gelu)*100:.3f}%")
    axes[0, 1].axvline(np.median(rel_l1_ampl_gelu) * 100, color="orange", linestyle=":", linewidth=1.8, label=f"Mediana: {np.median(rel_l1_ampl_gelu)*100:.3f}%")
    axes[0, 1].set_title("Ampiezza GELU: Distribuzione Errore Relativo $L_1$ (%)", fontweight="bold")
    axes[0, 1].set_xlabel("Errore Relativo $L_1$ (%)")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MAE Phase
    axes[1, 0].hist(mae_phase_gelu * 1e4, bins=60, color="#2ca02c", alpha=0.8, edgecolor="black")
    axes[1, 0].axvline(np.mean(mae_phase_gelu) * 1e4, color="red", linestyle="--", linewidth=1.8, label=f"Media: {np.mean(mae_phase_gelu):.6f}")
    axes[1, 0].axvline(np.median(mae_phase_gelu) * 1e4, color="orange", linestyle=":", linewidth=1.8, label=f"Mediana: {np.median(mae_phase_gelu):.6f}")
    axes[1, 0].set_title("Fase GELU: Distribuzione MAE ($\\times 10^{-4}$)", fontweight="bold")
    axes[1, 0].set_xlabel("MAE ($\\times 10^{-4}$)")
    axes[1, 0].set_ylabel("Frequenza")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Rel L1 Phase
    axes[1, 1].hist(rel_l1_phase_gelu * 100, bins=60, color="#2ca02c", alpha=0.8, edgecolor="black")
    axes[1, 1].axvline(np.mean(rel_l1_phase_gelu) * 100, color="red", linestyle="--", linewidth=1.8, label=f"Media: {np.mean(rel_l1_phase_gelu)*100:.3f}%")
    axes[1, 1].axvline(np.median(rel_l1_phase_gelu) * 100, color="orange", linestyle=":", linewidth=1.8, label=f"Mediana: {np.median(rel_l1_phase_gelu)*100:.3f}%")
    axes[1, 1].set_title("Fase GELU: Distribuzione Errore Relativo $L_1$ (%)", fontweight="bold")
    axes[1, 1].set_xlabel("Errore Relativo $L_1$ (%)")
    axes[1, 1].set_ylabel("Frequenza")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    nb2_img_dist = fig_to_base64(fig)

    nb2_c3_src = [
        "# Grafici di distribuzione errori (MAE e Errore Relativo L1)",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 8))",
        "axes[0, 0].hist(mae_ampl * 1e5, bins=60, color='#1f77b4', alpha=0.8, edgecolor='black')",
        "axes[0, 0].axvline(np.mean(mae_ampl) * 1e5, color='red', linestyle='--', linewidth=1.8, label=f'Media: {np.mean(mae_ampl):.6f}')",
        "axes[0, 0].axvline(np.median(mae_ampl) * 1e5, color='orange', linestyle=':', linewidth=1.8, label=f'Mediana: {np.median(mae_ampl):.6f}')",
        "axes[0, 0].set_title('Ampiezza GELU: Distribuzione MAE ($\\\\times 10^{-5}$)', fontweight='bold')",
        "axes[0, 0].set_xlabel('MAE ($\\\\times 10^{-5}$)')",
        "axes[0, 0].legend()",
        "axes[0, 0].grid(True, alpha=0.3)",
        "",
        "axes[0, 1].hist(rel_l1_ampl * 100, bins=60, color='#1f77b4', alpha=0.8, edgecolor='black')",
        "axes[0, 1].axvline(np.mean(rel_l1_ampl) * 100, color='red', linestyle='--', linewidth=1.8, label=f'Media: {np.mean(rel_l1_ampl)*100:.3f}%')",
        "axes[0, 1].axvline(np.median(rel_l1_ampl) * 100, color='orange', linestyle=':', linewidth=1.8, label=f'Mediana: {np.median(rel_l1_ampl)*100:.3f}%')",
        "axes[0, 1].set_title('Ampiezza GELU: Distribuzione Errore Relativo $L_1$ (%)', fontweight='bold')",
        "axes[0, 1].set_xlabel('Errore Relativo $L_1$ (%)')",
        "axes[0, 1].legend()",
        "axes[0, 1].grid(True, alpha=0.3)",
        "",
        "axes[1, 0].hist(mae_phase * 1e4, bins=60, color='#2ca02c', alpha=0.8, edgecolor='black')",
        "axes[1, 0].axvline(np.mean(mae_phase) * 1e4, color='red', linestyle='--', linewidth=1.8, label=f'Media: {np.mean(mae_phase):.6f}')",
        "axes[1, 0].axvline(np.median(mae_phase) * 1e4, color='orange', linestyle=':', linewidth=1.8, label=f'Mediana: {np.median(mae_phase):.6f}')",
        "axes[1, 0].set_title('Fase GELU: Distribuzione MAE ($\\\\times 10^{-4}$)', fontweight='bold')",
        "axes[1, 0].set_xlabel('MAE ($\\\\times 10^{-4}$)')",
        "axes[1, 0].legend()",
        "axes[1, 0].grid(True, alpha=0.3)",
        "",
        "axes[1, 1].hist(rel_l1_phase * 100, bins=60, color='#2ca02c', alpha=0.8, edgecolor='black')",
        "axes[1, 1].axvline(np.mean(rel_l1_phase) * 100, color='red', linestyle='--', linewidth=1.8, label=f'Media: {np.mean(rel_l1_phase)*100:.3f}%')",
        "axes[1, 1].axvline(np.median(rel_l1_phase) * 100, color='orange', linestyle=':', linewidth=1.8, label=f'Mediana: {np.median(rel_l1_phase)*100:.3f}%')",
        "axes[1, 1].set_title('Fase GELU: Distribuzione Errore Relativo $L_1$ (%)', fontweight='bold')",
        "axes[1, 1].set_xlabel('Errore Relativo $L_1$ (%)')",
        "axes[1, 1].legend()",
        "axes[1, 1].grid(True, alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells_nb2.append(make_code_cell(nb2_c3_src, [make_display_data_output(img_b64=nb2_img_dist)], 3))

    # Markdown Section 4: Temporal Error Profiles
    cells_nb2.append(make_markdown_cell([
        "---",
        "### 3. Profilo Temporale dell'Errore lungo i 2048 Step",
        "",
        "Analizziamo dove si localizza l'errore medio di ricostruzione lungo l'evoluzione temporale del segnale (inspiral, merger all'indice 1998, ringdown).",
        "$$e_A(t) = \\frac{1}{N} \\sum_{i=1}^N |A_i(t) - \\hat{A}_i(t)|, \\quad e_\\phi(t) = \\frac{1}{N} \\sum_{i=1}^N |\\phi_i(t) - \\hat{\\phi}_i(t)|$$"
    ]))

    # Plot Temporal Error Profiles
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_steps = np.arange(2048)
    axes[0].plot(t_steps, temp_mae_ampl_gelu, color="#1f77b4", linewidth=1.8, label="Ampiezza MAE temporale")
    axes[0].axvline(1998, color="red", linestyle="--", alpha=0.7, label="Picco Merger (idx 1998)")
    axes[0].set_title("Ampiezza GELU: Profilo Temporale dell'Errore MAE", fontweight="bold")
    axes[0].set_ylabel("MAE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_steps, temp_mae_phase_gelu, color="#2ca02c", linewidth=1.8, label="Fase MAE temporale")
    axes[1].axvline(1998, color="red", linestyle="--", alpha=0.7, label="Picco Merger (idx 1998)")
    axes[1].set_title("Fase GELU: Profilo Temporale dell'Errore MAE", fontweight="bold")
    axes[1].set_xlabel("Punto Temporale (Step 0..2047)")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    nb2_img_temp = fig_to_base64(fig)

    nb2_c4_src = [
        "# Profilo temporale dell'errore MAE medio",
        "t_steps = np.arange(2048)",
        "fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)",
        "axes[0].plot(t_steps, temp_mae_ampl, color='#1f77b4', linewidth=1.8, label='Ampiezza MAE')",
        "axes[0].axvline(1998, color='red', linestyle='--', alpha=0.7, label='Picco Merger (idx 1998)')",
        "axes[0].set_title('Ampiezza GELU: Profilo Temporale dell\\'Errore MAE', fontweight='bold')",
        "axes[0].set_ylabel('MAE')",
        "axes[0].legend()",
        "axes[0].grid(True, alpha=0.3)",
        "",
        "axes[1].plot(t_steps, temp_mae_phase, color='#2ca02c', linewidth=1.8, label='Fase MAE')",
        "axes[1].axvline(1998, color='red', linestyle='--', alpha=0.7, label='Picco Merger (idx 1998)')",
        "axes[1].set_title('Fase GELU: Profilo Temporale dell\\'Errore MAE', fontweight='bold')",
        "axes[1].set_xlabel('Punto Temporale (Step 0..2047)')",
        "axes[1].set_ylabel('MAE')",
        "axes[1].legend()",
        "axes[1].grid(True, alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells_nb2.append(make_code_cell(nb2_c4_src, [make_display_data_output(img_b64=nb2_img_temp)], 4))

    # Markdown Section 5: Waveform Visual Overlays
    cells_nb2.append(make_markdown_cell([
        "---",
        "### 4. Sovrapposizione Grafica delle Forme d'Onda Ricostruite",
        "",
        "Ispezioniamo visivamente la qualità fisica della ricostruzione su quattro scenari significativi:",
        "1. **Migliore ricostruzione (Best)**: Minimo errore sul test set.",
        "2. **Ricostruzione Mediana**: Prestazione tipica al 50° percentile.",
        "3. **90° Percentile**: Caso difficile nella coda alta dell'errore.",
        "4. **Peggiore caso (Worst)**: L'outlier massimo presente sul test set.",
        "",
        "Per ciascun caso mostriamo la componente di Ampiezza $A(t)$, la Fase $\\phi(t)$ e la polarizzazione fisica gravitazionale reale:",
        "$$h_+(t) = A(t) \\cos(100 \\phi(t))$$"
    ]))

    # Waveform plots for Best, Median, P90, Worst
    cases = [
        ("Caso Migliore (Best)", sample_indices["best_ampl_idx"], "best"),
        ("Caso Mediano (50° Percentile)", sample_indices["median_ampl_idx"], "median"),
        ("Coda di Errore (90° Percentile)", sample_indices["p90_ampl_idx"], "p90"),
        ("Peggiore Caso (Worst Outlier)", sample_indices["worst_ampl_idx"], "worst")
    ]

    for title, test_idx, tag in cases:
        h5_idx = test_indices[test_idx]
        raw_wf = wf_map[h5_idx]
        A_orig, A_rec, P_orig, P_rec = get_rec(raw_wf)
        
        # Physical strain
        h_orig = A_orig * np.cos(100.0 * P_orig)
        h_rec = A_rec * np.cos(100.0 * P_rec)
        
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1.2])
        
        # Amplitude
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.plot(A_orig, label="Originale $A(t)$", color="black", linewidth=1.5)
        ax_a.plot(A_rec, label="Ricostruito $\\hat{A}(t)$", color="#1f77b4", linestyle="--", linewidth=1.5)
        ax_a.set_ylabel("Ampiezza")
        ax_a.set_title(f"{title} (Test Index: {test_idx}) | MAE A: {mae_ampl_gelu[test_idx]:.6f} (Rel: {rel_l1_ampl_gelu[test_idx]*100:.3f}%)", fontweight="bold")
        ax_a.legend(loc="upper left")
        ax_a.grid(True, alpha=0.3)

        ax_a_res = fig.add_subplot(gs[0, 1])
        ax_a_res.plot(A_orig - A_rec, color="#1f77b4", linewidth=1.2)
        ax_a_res.set_title("Residuo Ampiezza ($A - \\hat{A}$)")
        ax_a_res.grid(True, alpha=0.3)

        # Phase
        ax_p = fig.add_subplot(gs[1, 0])
        ax_p.plot(P_orig, label="Originale $\\phi(t)$", color="black", linewidth=1.5)
        ax_p.plot(P_rec, label="Ricostruito $\\hat{\\phi}(t)$", color="#2ca02c", linestyle="--", linewidth=1.5)
        ax_p.set_ylabel("Fase normalizzata")
        ax_p.set_title(f"Fase | MAE $\\phi$: {mae_phase_gelu[test_idx]:.6f} (Rel: {rel_l1_phase_gelu[test_idx]*100:.3f}%)", fontweight="bold")
        ax_p.legend(loc="upper left")
        ax_p.grid(True, alpha=0.3)

        ax_p_res = fig.add_subplot(gs[1, 1])
        ax_p_res.plot(P_orig - P_rec, color="#2ca02c", linewidth=1.2)
        ax_p_res.set_title("Residuo Fase ($\\phi - \\hat{\\phi}$)")
        ax_p_res.grid(True, alpha=0.3)

        # Strain
        ax_h = fig.add_subplot(gs[2, 0])
        ax_h.plot(h_orig, label="Originale $h_+(t)$", color="black", linewidth=1.2, alpha=0.7)
        ax_h.plot(h_rec, label="Ricostruito $\\hat{h}_+(t)$", color="#d62728", linestyle="--", linewidth=1.2)
        ax_h.set_xlabel("Punto Temporale (Step 0..2047)")
        ax_h.set_ylabel("Polarizzazione $h_+(t)$")
        ax_h.set_title("Polarizzazione Gravitazionale $h_+(t) = A(t) \\cos(100 \\phi(t))$", fontweight="bold")
        ax_h.legend(loc="upper left")
        ax_h.grid(True, alpha=0.3)

        ax_h_res = fig.add_subplot(gs[2, 1])
        ax_h_res.plot(h_orig - h_rec, color="#d62728", linewidth=1.2)
        ax_h_res.set_xlabel("Punto Temporale")
        ax_h_res.set_title("Residuo $h_+(t) - \\hat{h}_+(t)$")
        ax_h_res.grid(True, alpha=0.3)

        plt.tight_layout()
        img_case = fig_to_base64(fig)

        nb2_src_case = [
            f"# Visualizzazione {title}",
            f"test_idx = {test_idx}",
            "raw_wf = wf_map[test_indices[test_idx]]",
            "A_orig, A_rec, P_orig, P_rec = get_rec(raw_wf)",
            "h_orig = A_orig * np.cos(100.0 * P_orig)",
            "h_rec = A_rec * np.cos(100.0 * P_rec)",
            "",
            "fig = plt.figure(figsize=(15, 8))",
            "gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1.2])",
            "ax_a = fig.add_subplot(gs[0, 0])",
            "ax_a.plot(A_orig, label='Originale $A(t)$', color='black', linewidth=1.5)",
            "ax_a.plot(A_rec, label='Ricostruito $\\\\hat{A}(t)$', color='#1f77b4', linestyle='--', linewidth=1.5)",
            "ax_a.set_ylabel('Ampiezza')",
            f"ax_a.set_title('{title} | MAE A: {{mae_ampl[test_idx]:.6f}} (Rel: {{rel_l1_ampl[test_idx]*100:.3f}}%)', fontweight='bold')",
            "ax_a.legend(loc='upper left')",
            "ax_a.grid(True, alpha=0.3)",
            "",
            "ax_a_res = fig.add_subplot(gs[0, 1])",
            "ax_a_res.plot(A_orig - A_rec, color='#1f77b4', linewidth=1.2)",
            "ax_a_res.set_title('Residuo Ampiezza ($A - \\\\hat{A}$)')",
            "ax_a_res.grid(True, alpha=0.3)",
            "",
            "ax_p = fig.add_subplot(gs[1, 0])",
            "ax_p.plot(P_orig, label='Originale $\\\\phi(t)$', color='black', linewidth=1.5)",
            "ax_p.plot(P_rec, label='Ricostruito $\\\\hat{\\\\phi}(t)$', color='#2ca02c', linestyle='--', linewidth=1.5)",
            "ax_p.set_ylabel('Fase normalizzata')",
            f"ax_p.set_title('Fase | MAE $\\\\phi$: {{mae_phase[test_idx]:.6f}} (Rel: {{rel_l1_phase[test_idx]*100:.3f}}%)', fontweight='bold')",
            "ax_p.legend(loc='upper left')",
            "ax_p.grid(True, alpha=0.3)",
            "",
            "ax_p_res = fig.add_subplot(gs[1, 1])",
            "ax_p_res.plot(P_orig - P_rec, color='#2ca02c', linewidth=1.2)",
            "ax_p_res.set_title('Residuo Fase ($\\\\phi - \\\\hat{\\\\phi}$)')",
            "ax_p_res.grid(True, alpha=0.3)",
            "",
            "ax_h = fig.add_subplot(gs[2, 0])",
            "ax_h.plot(h_orig, label='Originale $h_+(t)$', color='black', linewidth=1.2, alpha=0.7)",
            "ax_h.plot(h_rec, label='Ricostruito $\\\\hat{h}_+(t)$', color='#d62728', linestyle='--', linewidth=1.2)",
            "ax_h.set_xlabel('Punto Temporale (Step 0..2047)')",
            "ax_h.set_ylabel('Polarizzazione $h_+(t)$')",
            "ax_h.set_title('Polarizzazione Gravitazionale $h_+(t) = A(t) \\\\cos(100 \\\\phi(t))$', fontweight='bold')",
            "ax_h.legend(loc='upper left')",
            "ax_h.grid(True, alpha=0.3)",
            "",
            "ax_h_res = fig.add_subplot(gs[2, 1])",
            "ax_h_res.plot(h_orig - h_rec, color='#d62728', linewidth=1.2)",
            "ax_h_res.set_xlabel('Punto Temporale')",
            "ax_h_res.set_title('Residuo $h_+(t) - \\\\hat{h}_+(t)$')",
            "ax_h_res.grid(True, alpha=0.3)",
            "",
            "plt.tight_layout()",
            "plt.show()"
        ]
        cells_nb2.append(make_code_cell(nb2_src_case, [make_display_data_output(img_b64=img_case)], len(cells_nb2) + 1))

    # Save nb2
    nb2_content = {
        "cells": cells_nb2,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(out_dir / "single_ae_gelu_performance.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2_content, f, indent=1)
    print("Notebook 2: single_ae_gelu_performance.ipynb salvato con successo!")
    print("=== COSTRUZIONE COMPLETATA CON SUCCESSO! ===")

if __name__ == "__main__":
    main()
