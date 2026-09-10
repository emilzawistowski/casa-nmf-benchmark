import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def bark_transform(frequencies: np.ndarray) -> np.ndarray:
    bark = 13 * np.arctan(0.00076 * frequencies) + \
           3.5 * np.arctan((frequencies / 7500)**2)
    return bark


def hz_to_bark(freq_hz: float) -> float:
    return 13 * np.arctan(0.00076 * freq_hz) + \
           3.5 * np.arctan((freq_hz / 7500)**2)


def bark_to_hz(bark: float) -> float:
    if bark < 2:
        return bark * 100
    elif bark < 13:
        return 600 * np.sinh(bark / 6)
    else:
        return 600 * ((np.exp(bark/6) - np.exp(-bark/6)) / 2)


def plot_spectrogram_with_bark(magnitude: np.ndarray, 
                                 sr: int, 
                                 hop_length: int,
                                 title: str,
                                 save_path: Path):

    fig, ax = plt.subplots(figsize=(12, 6))
    
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    img = librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap='magma'
    )
    
    bark_boundaries_hz = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                          1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                          4400, 5300, 6400, 7700, 9500, 12000]
    
    for freq in bark_boundaries_hz:
        if freq < sr/2:
            ax.axhline(y=freq, color='cyan', linestyle='--', 
                      linewidth=0.8, alpha=0.5)
    
    ax.text(0.02, 0.98, 'Critical Bands (Bark scale)', 
           transform=ax.transAxes, color='cyan', 
           fontsize=9, va='top', bbox=dict(boxstyle='round', 
           facecolor='black', alpha=0.3))
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency [Hz]', fontsize=11)
    ax.set_xlabel('Time [s]', fontsize=11)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spectrogram: {save_path}")


def plot_nmf_components(W: np.ndarray, 
                        H: np.ndarray,
                        sr: int,
                        texture_name: str,
                        save_path: Path):

    n_components = W.shape[1]
    
    fig, axes = plt.subplots(2, n_components, 
                             figsize=(4*n_components, 8))
    
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    for k in range(n_components):
        ax_w = axes[0, k] if n_components > 1 else axes[0]
        ax_w.plot(freq_axis[:len(W[:, k])], W[:, k], linewidth=1.5)
        ax_w.set_xlim(0, 8000)
        ax_w.set_xlabel('Frequency [Hz]', fontsize=9)
        ax_w.set_ylabel('Magnitude', fontsize=9)
        ax_w.set_title(f'Basis {k+1}', fontsize=10, fontweight='bold')
        ax_w.grid(alpha=0.3)
        
        for bark_hz in [500, 1000, 2000, 4000]:
            ax_w.axvline(bark_hz, color='red', linestyle=':', alpha=0.4)
        
        ax_h = axes[1, k] if n_components > 1 else axes[1]
        ax_h.plot(H[k, :], linewidth=1.0, alpha=0.8)
        ax_h.set_xlabel('Time Frame', fontsize=9)
        ax_h.set_ylabel('Activation', fontsize=9)
        ax_h.set_title(f'Activation {k+1}', fontsize=10, fontweight='bold')
        ax_h.grid(alpha=0.3)
    
    plt.suptitle(f'NMF Components: {texture_name}', 
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved NMF components: {save_path}")


def compute_basis_redundancy(W: np.ndarray) -> Dict[str, float]:

    n_components = W.shape[1]
    
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
    
    similarity_matrix = W_norm.T @ W_norm
    
    upper_tri_indices = np.triu_indices(n_components, k=1)
    similarities = similarity_matrix[upper_tri_indices]
    
    return {
        'mean_similarity': np.mean(similarities),
        'max_similarity': np.max(similarities),
        'redundancy_score': np.mean(similarities > 0.7)
    }


def export_best_worst_configs(df: pd.DataFrame, 
                               textures: list,
                               results_dir: Path):

    summary = []
    
    for texture in df['texture'].unique():
        subset = df[df['texture'] == texture].copy()
        
        best = subset.nlargest(1, 'sdr').iloc[0]
        worst = subset.nsmallest(1, 'sdr').iloc[0]
        
        summary.append({
            'texture': texture,
            'category': 'BEST',
            'rank': int(best['rank']),
            'beta': int(best['beta']),
            'alpha': best['alpha'],
            'max_iter': int(best['max_iter']),
            'phase_method': best['phase_method'],
            'sdr': best['sdr'],
            'mci': best['mci'],
            'bark_entropy': best['bark_entropy']
        })
        
        summary.append({
            'texture': texture,
            'category': 'WORST',
            'rank': int(worst['rank']),
            'beta': int(worst['beta']),
            'alpha': worst['alpha'],
            'max_iter': int(worst['max_iter']),
            'phase_method': worst['phase_method'],
            'sdr': worst['sdr'],
            'mci': worst['mci'],
            'bark_entropy': worst['bark_entropy']
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / 'best_worst_configs.csv', index=False)
    
    print("\n" + "="*80)
    print("BEST & WORST CONFIGURATIONS PER TEXTURE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    return summary_df


def hypothesis_testing(df: pd.DataFrame, results_dir: Path):

    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)
    
    from scipy.stats import spearmanr, pearsonr
    
    df_clean = df.dropna(subset=['bark_entropy', 'sdr', 'mci', 'sar'])
    
    corr_bark_sdr, p_bark_sdr = spearmanr(df_clean['bark_entropy'], df_clean['sdr'])
    
    print(f"\nH1: Bark-scale overlap predicts separability")
    print(f"    Spearman ρ(Bark Entropy, SDR) = {corr_bark_sdr:.3f}, p = {p_bark_sdr:.4f}")
    
    if corr_bark_sdr < -0.3 and p_bark_sdr < 0.05:
        print(f"    ✓ SUPPORTED: Higher spectral spread → Lower SDR (significant)")
    else:
        print(f"    ✗ NOT SUPPORTED: Weak or non-significant correlation")
    
    mci_std = df_clean.groupby('texture')['mci'].std().mean()
    sdr_std = df_clean.groupby('texture')['sdr'].std().mean()
    
    print(f"\nH2: MCI captures identifiability better than SDR")
    print(f"    Average std(MCI) across textures: {mci_std:.2f}")
    print(f"    Average std(SDR) across textures: {sdr_std:.2f}")
    print(f"    Ratio std(SDR)/std(MCI): {sdr_std/mci_std:.2f}")
    
    if mci_std < sdr_std:
        print(f"    ✓ SUPPORTED: MCI more stable than SDR")
    else:
        print(f"    ✗ NOT SUPPORTED: SDR more stable")
    
    reverb_subset = df_clean[df_clean['texture'] == 'reverberant_speech']
    
    if len(reverb_subset) > 0:
        gl_sar = reverb_subset[reverb_subset['phase_method'] == 'griffin_lim']['sar'].mean()
        fast_gl_sar = reverb_subset[reverb_subset['phase_method'] == 'fast_griffin_lim']['sar'].mean()
        
        print(f"\nH3: Phase recovery dominates SAR on reverberant textures")
        print(f"    Mean SAR (Griffin-Lim): {gl_sar:.2f} dB")
        print(f"    Mean SAR (Fast GL):     {fast_gl_sar:.2f} dB")
        print(f"    Difference: {abs(fast_gl_sar - gl_sar):.2f} dB")
        
        if abs(fast_gl_sar - gl_sar) > 2.0:
            print(f"    ✓ SUPPORTED: Phase method significantly affects SAR (>2 dB)")
        else:
            print(f"    ✗ NOT SUPPORTED: Minor difference (<2 dB)")
    
    print("="*80 + "\n")
    
    hypothesis_results = {
        'H1_bark_sdr_correlation': float(corr_bark_sdr),
        'H1_p_value': float(p_bark_sdr),
        'H2_mci_stability': float(mci_std),
        'H2_sdr_stability': float(sdr_std),
        'H3_gl_sar': float(gl_sar) if (len(reverb_subset) > 0 and not np.isnan(gl_sar)) else None,
        'H3_fast_gl_sar': float(fast_gl_sar) if (len(reverb_subset) > 0 and not np.isnan(fast_gl_sar)) else None
    }
    
    with open(results_dir / 'hypothesis_test_results.json', 'w') as f:
        import json
        json.dump(hypothesis_results, f, indent=2)


def generate_summary_report(df: pd.DataFrame, results_dir: Path):

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CASA NMF BENCHMARK - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(f"Total experiments: {len(df)}")
    report_lines.append(f"Textures analyzed: {df['texture'].nunique()}")
    report_lines.append(f"Parameter configurations tested: {len(df) // df['texture'].nunique()}")
    report_lines.append("")
    
    report_lines.append("PERFORMANCE BY TEXTURE")
    report_lines.append("-"*80)
    
    for texture in df['texture'].unique():
        subset = df[df['texture'] == texture]
        report_lines.append(f"\n{texture.upper()}")
        report_lines.append(f"  Mean SDR: {subset['sdr'].mean():.2f} ± {subset['sdr'].std():.2f} dB")
        report_lines.append(f"  Mean MCI: {subset['mci'].mean():.1f} ± {subset['mci'].std():.1f} %")
        report_lines.append(f"  Mean SAR: {subset['sar'].mean():.2f} ± {subset['sar'].std():.2f} dB")
        report_lines.append(f"  Best SDR: {subset['sdr'].max():.2f} dB")
        report_lines.append(f"  Worst SDR: {subset['sdr'].min():.2f} dB")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    with open(results_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)


if __name__ == "__main__":
    print("CASA NMF Utilities Module")
    print("Import this module in main.py for psychoacoustic analysis")
    
    test_freqs = np.array([100, 500, 1000, 2000, 4000, 8000])
    test_barks = bark_transform(test_freqs)
    
    print("\nBark Scale Conversion Test:")
    for freq, bark in zip(test_freqs, test_barks):
        print(f"  {freq:5.0f} Hz → {bark:5.2f} Bark")
