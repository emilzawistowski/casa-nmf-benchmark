
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from main import AudioTexture, NMFDecomposer, CONFIG
from utils import plot_spectrogram_with_bark, plot_nmf_components


def generate_all_spectrograms():
    
    print("="*80)
    print("GENERATING DETAILED SPECTROGRAMS")
    print("="*80)
    
    spec_dir = CONFIG['figures_dir'] / 'spectrograms'
    spec_dir.mkdir(exist_ok=True)

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
    
    best_configs = pd.read_csv(CONFIG['results_dir'] / 'best_worst_configs.csv')
    
    print("\n")
    
    for texture in textures:
        print(f"Processing: {texture.name}")
        print("-"*80)
        
        plot_spectrogram_with_bark(
            texture.magnitude,
            CONFIG['sample_rate'],
            CONFIG['hop_length'],
            title=f'{texture.name.replace("_", " ").title()} - Original Mixture',
            save_path=spec_dir / f'{texture.name}_original.png'
        )
        
        best_row = best_configs[
            (best_configs['texture'] == texture.name) & 
            (best_configs['category'] == 'BEST')
        ].iloc[0]
        
        print(f"  Best config: Rank={int(best_row['rank'])}, Beta={int(best_row['beta'])}, "
              f"α={best_row['alpha']}, Iter={int(best_row['max_iter'])}")
        
        decomposer_best = NMFDecomposer(
            rank=int(best_row['rank']),
            beta=int(best_row['beta']),
            alpha=best_row['alpha'],
            max_iter=int(best_row['max_iter'])
        )
        
        W_best, H_best = decomposer_best.decompose(texture.magnitude)
        
        plot_nmf_components(
            W_best,
            H_best,
            CONFIG['sample_rate'],
            f'{texture.name} (BEST: SDR={best_row["sdr"]:.2f} dB)',
            save_path=spec_dir / f'{texture.name}_best_components.png'
        )
        
        reconstructed_best = decomposer_best.reconstruct()
        
        plot_spectrogram_with_bark(
            reconstructed_best,
            CONFIG['sample_rate'],
            CONFIG['hop_length'],
            title=f'{texture.name.replace("_", " ").title()} - Best Reconstruction '
                  f'(SDR={best_row["sdr"]:.2f} dB, MCI={best_row["mci"]:.1f}%)',
            save_path=spec_dir / f'{texture.name}_best_reconstruction.png'
        )
        
        worst_row = best_configs[
            (best_configs['texture'] == texture.name) & 
            (best_configs['category'] == 'WORST')
        ].iloc[0]
        
        print(f"  Worst config: Rank={int(worst_row['rank'])}, Beta={int(worst_row['beta'])}, "
              f"α={worst_row['alpha']}, Iter={int(worst_row['max_iter'])}")
        
        decomposer_worst = NMFDecomposer(
            rank=int(worst_row['rank']),
            beta=int(worst_row['beta']),
            alpha=worst_row['alpha'],
            max_iter=int(worst_row['max_iter'])
        )
        
        W_worst, H_worst = decomposer_worst.decompose(texture.magnitude)
        
        plot_nmf_components(
            W_worst,
            H_worst,
            CONFIG['sample_rate'],
            f'{texture.name} (WORST: SDR={worst_row["sdr"]:.2f} dB)',
            save_path=spec_dir / f'{texture.name}_worst_components.png'
        )
        
        reconstructed_worst = decomposer_worst.reconstruct()
        
        plot_spectrogram_with_bark(
            reconstructed_worst,
            CONFIG['sample_rate'],
            CONFIG['hop_length'],
            title=f'{texture.name.replace("_", " ").title()} - Worst Reconstruction '
                  f'(SDR={worst_row["sdr"]:.2f} dB, MCI={worst_row["mci"]:.1f}%)',
            save_path=spec_dir / f'{texture.name}_worst_reconstruction.png'
        )

        error_best = np.abs(texture.magnitude - reconstructed_best)
        error_worst = np.abs(texture.magnitude - reconstructed_worst)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        img0 = librosa.display.specshow(
            librosa.amplitude_to_db(error_best, ref=np.max),
            sr=CONFIG['sample_rate'],
            hop_length=CONFIG['hop_length'],
            x_axis='time',
            y_axis='hz',
            ax=axes[0],
            cmap='Reds'
        )
        axes[0].set_title(f'Reconstruction Error (BEST)', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 8000)
        fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')
        
        img1 = librosa.display.specshow(
            librosa.amplitude_to_db(error_worst, ref=np.max),
            sr=CONFIG['sample_rate'],
            hop_length=CONFIG['hop_length'],
            x_axis='time',
            y_axis='hz',
            ax=axes[1],
            cmap='Reds'
        )
        axes[1].set_title(f'Reconstruction Error (WORST)', fontsize=12, fontweight='bold')
        axes[1].set_ylim(0, 8000)
        fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
        
        plt.suptitle(f'{texture.name.replace("_", " ").title()} - Error Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(spec_dir / f'{texture.name}_error_comparison.png', dpi=150)
        plt.close()
        
        print(f"  Saved 6 figures for {texture.name}")
        print()
    
    print("="*80)
    print(f"All spectrograms saved to: {spec_dir}")
    print("="*80)


if __name__ == '__main__':
    # Check if results exist
    if not (CONFIG['results_dir'] / 'best_worst_configs.csv').exists():
        print("ERROR: Run main.py first to generate best_worst_configs.csv")
        print("Usage: python main.py")
        exit(1)
    
    generate_all_spectrograms()
