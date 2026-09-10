import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings
from tqdm import tqdm
import itertools

warnings.filterwarnings('ignore')

CONFIG = {
    'sample_rate': 22050,
    'n_fft': 2048,
    'hop_length': 512,
    'audio_duration': 20,


    'rank_K': [2, 4, 8, 16],
    'beta_divergence': [0, 1, 2],
    'regularization': [0.0, 0.1],
    'max_iter': [100, 300],
    'phase_recovery_iter': [50],

    'bark_boundaries': [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                        1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                        4400, 5300, 6400, 7700, 9500, 12000, 15500],

    'data_dir': Path('audio_data/audio_data_ready_to_go'),
    'results_dir': Path('results'),
    'figures_dir': Path('figures'),
}

for dir_path in [CONFIG['data_dir'], CONFIG['results_dir'], CONFIG['figures_dir']]:
    dir_path.mkdir(exist_ok=True)


class AudioTexture:

    def __init__(self, name: str, audio_path: Optional[Path] = None):
        self.name = name
        self.audio_path = audio_path
        self.waveform = None
        self.magnitude = None
        self.phase = None
        self.sr = CONFIG['sample_rate']

    def load(self):
        if not self.audio_path or not self.audio_path.exists():
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"❌ ERROR: Audio file not found!\n"
                f"{'='*80}\n"
                f"Expected file: {self.audio_path}\n"
                f"Texture name: {self.name}\n\n"
                f"Please ensure your audio files are in the correct location:\n"
                f"  {CONFIG['data_dir']}/\n\n"
                f"Supported formats: .wav (mono or stereo, any sample rate)\n"
                f"Minimum duration: 10 seconds (20 seconds recommended)\n"
                f"{'='*80}\n"
            )

        self._load_from_file()
        self._compute_stft()
        print(f"✓ Loaded: {self.name} | Shape: {self.magnitude.shape} | "
            f"Duration: {len(self.waveform)/self.sr:.1f}s")

    def _load_from_file(self):
        self.waveform, self.sr = librosa.load(
            self.audio_path,
            sr=CONFIG['sample_rate'],
            duration=CONFIG['audio_duration']
        )
        print(f"Loaded {self.name} from {self.audio_path}")

    def _compute_stft(self):
        stft = librosa.stft(
            self.waveform,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length']
        )
        self.magnitude = np.abs(stft)
        self.phase = np.angle(stft)
        print(f"STFT computed: {self.magnitude.shape}")


class NMFDecomposer:

    def __init__(self, rank: int, beta: float, alpha: float, max_iter: int):
        self.rank = rank
        self.beta = beta
        self.alpha = alpha
        self.max_iter = max_iter
        self.W = None
        self.H = None

    def decompose(self, magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.decomposition import NMF

        loss_map = {0: 'kullback-leibler', 1: 'kullback-leibler', 2: 'frobenius'}
        solver_map = {0: 'mu', 1: 'mu', 2: 'cd'}

        nmf = NMF(
            n_components=self.rank,
            init='random',
            solver=solver_map[self.beta],
            beta_loss=loss_map[self.beta],
            max_iter=self.max_iter,
            alpha_W=self.alpha,
            alpha_H=self.alpha,
            l1_ratio=1.0,
            random_state=42
        )

        self.W = nmf.fit_transform(magnitude)
        self.H = nmf.components_

        return self.W, self.H

    def reconstruct(self) -> np.ndarray:
        return self.W @ self.H


class PhaseRecovery:

    def griffin_lim(magnitude: np.ndarray, n_iter: int = 50) -> np.ndarray:

        phase = np.random.randn(*magnitude.shape) * 2 * np.pi

        for _ in range(n_iter):
            stft_complex = magnitude * np.exp(1j * phase)

            waveform = librosa.istft(
                stft_complex,
                hop_length=CONFIG['hop_length'],
                n_fft=CONFIG['n_fft']
            )

            stft_reconstructed = librosa.stft(
                waveform,
                n_fft=CONFIG['n_fft'],
                hop_length=CONFIG['hop_length']
            )

            phase = np.angle(stft_reconstructed)

        return magnitude * np.exp(1j * phase)

    def fast_griffin_lim(magnitude: np.ndarray, n_iter: int = 50) -> np.ndarray:

        phase = np.random.randn(*magnitude.shape) * 2 * np.pi
        momentum = 0.99
        previous = None

        for i in range(n_iter):
            stft_complex = magnitude * np.exp(1j * phase)
            waveform = librosa.istft(stft_complex, hop_length=CONFIG['hop_length'])
            stft_reconstructed = librosa.stft(waveform, n_fft=CONFIG['n_fft'],
                                              hop_length=CONFIG['hop_length'])

            if previous is not None and i > 0:
                stft_reconstructed = stft_reconstructed + momentum * (stft_reconstructed - previous)

            previous = stft_reconstructed.copy()
            phase = np.angle(stft_reconstructed)

        return magnitude * np.exp(1j * phase)


class MetricsComputer:

    def compute_sdr_sir_sar(reference: np.ndarray, estimated: np.ndarray) -> Dict[str, float]:

        ref_norm = reference / (np.linalg.norm(reference) + 1e-10)
        est_norm = estimated / (np.linalg.norm(estimated) + 1e-10)

        distortion = ref_norm - est_norm
        sdr = 10 * np.log10((np.sum(ref_norm**2) + 1e-10) / (np.sum(distortion**2) + 1e-10))

        sir = sdr

        artifacts = distortion
        sar = 10 * np.log10((np.sum(est_norm**2) + 1e-10) / (np.sum(artifacts**2) + 1e-10))

        return {'sdr': sdr, 'sir': sir, 'sar': sar}

    def mix_clarity_index(mixture_mag: np.ndarray, separated_mags: List[np.ndarray]) -> float:

        def entropy(spec: np.ndarray) -> float:
            spec_flat = spec.flatten()
            spec_norm = spec_flat / (np.sum(spec_flat) + 1e-10)
            spec_norm = spec_norm[spec_norm > 1e-10]
            h = -np.sum(spec_norm * np.log2(spec_norm))
            return h

        h_mix = entropy(mixture_mag)

        h_separated = np.mean([entropy(s) for s in separated_mags])

        mci = ((h_mix - h_separated) / (h_mix + 1e-10)) * 100

        return mci

    def bark_scale_overlap(magnitude: np.ndarray) -> Dict[str, float]:

        n_freq_bins = magnitude.shape[0]
        freq_axis = librosa.fft_frequencies(sr=CONFIG['sample_rate'], n_fft=CONFIG['n_fft'])

        bark_axis = 13 * np.arctan(0.00076 * freq_axis) + 3.5 * np.arctan((freq_axis / 7500)**2)

        bark_bands = CONFIG['bark_boundaries']
        bark_energy = []

        for i in range(len(bark_bands) - 1):
            low_bark = bark_bands[i]
            high_bark = bark_bands[i + 1]

            low_freq = 600 * (np.sinh(low_bark / 6) if low_bark < 13 else
                              (np.exp(low_bark / 6) - np.exp(-low_bark / 6)) / 2)
            high_freq = 600 * (np.sinh(high_bark / 6) if high_bark < 13 else
                               (np.exp(high_bark / 6) - np.exp(-high_bark / 6)) / 2)

            mask = (freq_axis >= low_freq) & (freq_axis < high_freq)
            band_energy = np.sum(magnitude[mask, :])
            bark_energy.append(band_energy)

        total_energy = np.sum(magnitude)
        bark_occupancy = np.array(bark_energy) / (total_energy + 1e-10)

        overlap = -np.sum(bark_occupancy * np.log2(bark_occupancy + 1e-10))

        return {
            'bark_entropy': overlap,
            'bark_occupancy': bark_occupancy.tolist()
        }


def run_single_experiment(texture: AudioTexture, params: Dict) -> Dict:

    decomposer = NMFDecomposer(
        rank=params['rank'],
        beta=params['beta'],
        alpha=params['alpha'],
        max_iter=params['max_iter']
    )

    W, H = decomposer.decompose(texture.magnitude)
    reconstructed_mag = decomposer.reconstruct()

    if params['phase_method'] == 'griffin_lim':
        reconstructed_complex = PhaseRecovery.griffin_lim(
            reconstructed_mag,
            n_iter=params['phase_iter']
        )
    else:
        reconstructed_complex = PhaseRecovery.fast_griffin_lim(
            reconstructed_mag,
            n_iter=params['phase_iter']
        )

    metrics_comp = MetricsComputer()

    quality_metrics = metrics_comp.compute_sdr_sir_sar(
        texture.magnitude,
        reconstructed_mag
    )

    mci = metrics_comp.mix_clarity_index(
        texture.magnitude,
        [reconstructed_mag]
    )

    bark_metrics = metrics_comp.bark_scale_overlap(texture.magnitude)

    result = {
        'texture': texture.name,
        **params,
        **quality_metrics,
        'mci': mci,
        'bark_entropy': bark_metrics['bark_entropy'],
        'reconstruction_error': np.mean((texture.magnitude - reconstructed_mag)**2)
    }

    return result


def run_grid_search(textures: List[AudioTexture]) -> pd.DataFrame:

    results = []

    param_combinations = list(itertools.product(
        CONFIG['rank_K'],
        CONFIG['beta_divergence'],
        CONFIG['regularization'],
        CONFIG['max_iter'],
        CONFIG['phase_recovery_iter'],
        ['griffin_lim', 'fast_griffin_lim']
    ))

    print(f"\nStarting grid search: {len(param_combinations)} configs × {len(textures)} textures")
    print(f"Total experiments: {len(param_combinations) * len(textures)}\n")

    total_experiments = len(param_combinations) * len(textures)
    pbar = tqdm(total=total_experiments, desc="Grid search progress")

    for texture in textures:
        for rank, beta, alpha, max_iter, phase_iter, phase_method in param_combinations:
            params = {
                'rank': rank,
                'beta': beta,
                'alpha': alpha,
                'max_iter': max_iter,
                'phase_iter': phase_iter,
                'phase_method': phase_method
            }

            try:
                result = run_single_experiment(texture, params)
                results.append(result)
            except Exception as e:
                print(f"\nError in experiment: {e}")
                results.append({
                    'texture': texture.name,
                    **params,
                    'sdr': np.nan,
                    'sir': np.nan,
                    'sar': np.nan,
                    'mci': np.nan,
                    'bark_entropy': np.nan,
                    'reconstruction_error': np.nan
                })

            pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)


def visualize_results(df: pd.DataFrame):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, texture in enumerate(df['texture'].unique()):
        subset = df[df['texture'] == texture]
        pivot = subset.pivot_table(
            values='sdr',
            index='rank',
            columns='beta',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=axes[idx], cbar_kws={'label': 'SDR (dB)'})
        axes[idx].set_title(f'{texture}: SDR vs. Rank × Beta')
        axes[idx].set_xlabel('Beta Divergence (0=IS, 1=KL, 2=Euclidean)')
        axes[idx].set_ylabel('Rank K')

    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'heatmap_sdr_rank_beta.png', dpi=150)
    print(f"Saved: {CONFIG['figures_dir'] / 'heatmap_sdr_rank_beta.png'}")

    fig, ax = plt.subplots(figsize=(10, 6))

    for texture in df['texture'].unique():
        subset = df[df['texture'] == texture]
        ax.scatter(subset['mci'], subset['sdr'], alpha=0.6,
                   label=texture, s=30)

    ax.set_xlabel('Mix Clarity Index (MCI) [%]', fontsize=12)
    ax.set_ylabel('SDR [dB]', fontsize=12)
    ax.set_title('Correlation: MCI vs. SDR', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'scatter_mci_vs_sdr.png', dpi=150)
    print(f"Saved: {CONFIG['figures_dir'] / 'scatter_mci_vs_sdr.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = ['sdr', 'sar', 'mci', 'bark_entropy']
    titles = ['SDR Distribution', 'SAR Distribution',
              'MCI Distribution', 'Bark Entropy (Spectral Spread)']

    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        df.boxplot(column=metric, by='texture', ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Texture', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        plt.sca(ax)
        plt.xticks(rotation=15)

    plt.suptitle('Metric Distributions Across Textures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'boxplots_metrics.png', dpi=150)
    print(f"Saved: {CONFIG['figures_dir'] / 'boxplots_metrics.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, texture in enumerate(df['texture'].unique()):
        subset = df[df['texture'] == texture]
        grouped = subset.groupby('max_iter')[['sdr', 'mci']].mean()

        ax = axes[idx]
        ax.plot(grouped.index, grouped['sdr'], marker='o', label='SDR', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(grouped.index, grouped['mci'], marker='s', linewidth=2)

        ax.set_xlabel('Max Iterations', fontsize=11)
        ax.set_ylabel('SDR [dB]', fontsize=11)
        ax2.set_ylabel('MCI [%]', fontsize=11)
        ax.set_title(f'{texture}: Convergence', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'convergence_iterations.png', dpi=150)
    print(f"Saved: {CONFIG['figures_dir'] / 'convergence_iterations.png'}")


def analyze_correlations(df: pd.DataFrame):
    metrics_cols = ['sdr', 'sir', 'sar', 'mci', 'bark_entropy', 'reconstruction_error']

    corr_matrix = df[metrics_cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax,
                cbar_kws={'label': 'Spearman Correlation'})
    ax.set_title('Metric Correlations (Spearman ρ)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'correlation_matrix.png', dpi=150)
    print(f"Saved: {CONFIG['figures_dir'] / 'correlation_matrix.png'}")

    print("\n" + "="*60)
    print("KEY CORRELATION FINDINGS")
    print("="*60)
    print(f"MCI vs. SDR:   ρ = {corr_matrix.loc['mci', 'sdr']:.3f}")
    print(f"MCI vs. SAR:   ρ = {corr_matrix.loc['mci', 'sar']:.3f}")
    print(f"SDR vs. SAR:   ρ = {corr_matrix.loc['sdr', 'sar']:.3f}")
    print(f"Bark Entropy vs. SDR: ρ = {corr_matrix.loc['bark_entropy', 'sdr']:.3f}")
    print("="*60 + "\n")


def main():
    print("="*80)
    print("CASA NMF BENCHMARK - GRID SEARCH ON EXTREME TEXTURES")
    print("="*80)
    print(f"Sample rate: {CONFIG['sample_rate']} Hz")
    print(f"Audio duration: {CONFIG['audio_duration']} seconds")
    print(f"Grid search space: {len(CONFIG['rank_K'])} ranks × {len(CONFIG['beta_divergence'])} betas × ...")
    print("="*80 + "\n")

    print("STEP 1: Loading Audio Textures")
    print("-"*80)

    textures = [
        AudioTexture('polyphonic_choir',
                    audio_path=CONFIG['data_dir'] / 'choir.wav'),
        AudioTexture('rainforest',
                    audio_path=CONFIG['data_dir'] / 'rainforest.wav'),
        AudioTexture('reverberant_speech',
                    audio_path=CONFIG['data_dir'] / 'reverb_speech.wav')
    ]

    for texture in textures:
        texture.load()

    print()

    print("STEP 2: Grid Search Execution")
    print("-"*80)

    results_df = run_grid_search(textures)

    results_path = CONFIG['results_dir'] / 'grid_search_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")
    print(f"Total experiments: {len(results_df)}")
    print("\n")

    print("STEP 3: Generating Visualizations")
    print("-"*80)

    visualize_results(results_df)

    print("\n")

    print("STEP 4: Metric Correlation Analysis")
    print("-"*80)

    analyze_correlations(results_df)

    print("STEP 5: Identifying Optimal Configurations")
    print("-"*80)

    from utils import export_best_worst_configs, hypothesis_testing, generate_summary_report

    export_best_worst_configs(results_df, textures, CONFIG['results_dir'])

    print("STEP 6: Statistical Hypothesis Testing")
    print("-"*80)

    hypothesis_testing(results_df, CONFIG['results_dir'])

    print("STEP 7: Generating Summary Report")
    print("-"*80)

    generate_summary_report(results_df, CONFIG['results_dir'])

    print("\nSTEP 8: (Optional) Detailed Spectrogram Analysis")
    print("-"*80)
    print("To generate per-texture spectrograms with Bark overlays, run:")
    print("  python generate_spectrograms.py")
    print("\n")

    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {CONFIG['results_dir']}")
    print(f"All figures saved to: {CONFIG['figures_dir']}")
    print("\nNext steps:")
    print("  1. Review results/summary_report.txt")
    print("  2. Examine figures/ for visualizations")
    print("  3. Check results/best_worst_configs.csv for optimal parameters")
    print("  4. Read results/hypothesis_test_results.json for statistical findings")
    print("="*80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("RUNNING IN TEST MODE (minimal grid search)")
        CONFIG['rank_K'] = [2, 4]
        CONFIG['beta_divergence'] = [1, 2]
        CONFIG['regularization'] = [0.0, 0.1]
        CONFIG['max_iter'] = [100]
        CONFIG['phase_recovery_iter'] = [50]
        CONFIG['audio_duration'] = 5

    main()
