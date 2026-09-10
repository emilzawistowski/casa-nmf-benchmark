import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import AudioTexture, NMFDecomposer, PhaseRecovery, MetricsComputer, CONFIG

def quick_test():
    print("="*80)
    print("CASA NMF BENCHMARK - QUICK TEST")
    print("="*80)
    print("This script runs a minimal experiment to verify your installation.")
    print("Expected runtime: ~5 minutes\n")
    
    print("Step 1: Loading test audio")
    print("-"*80)
    
    texture = AudioTexture('test_audio', 
                          audio_path=CONFIG['data_dir'] / 'choir.wav')
    texture.load()
    
    print(f"✓ Audio loaded: {texture.magnitude.shape}")
    print(f"  Frequency bins: {texture.magnitude.shape[0]}")
    print(f"  Time frames: {texture.magnitude.shape[1]}")
    print()
    
    print("Step 2: Running NMF decomposition")
    print("-"*80)
    print("Configuration: Rank=4, Beta=1 (KL), Alpha=0.1, Max_iter=100")
    
    decomposer = NMFDecomposer(
        rank=4,
        beta=1,
        alpha=0.1,
        max_iter=100
    )
    
    W, H = decomposer.decompose(texture.magnitude)
    reconstructed = decomposer.reconstruct()
    
    print(f"✓ NMF completed")
    print(f"  Basis matrix (W): {W.shape}")
    print(f"  Activation matrix (H): {H.shape}")
    print(f"  Reconstruction: {reconstructed.shape}")
    print()
    
    print("Step 3: Phase recovery (Griffin-Lim, 50 iterations)")
    print("-"*80)
    
    reconstructed_complex = PhaseRecovery.griffin_lim(reconstructed, n_iter=50)
    
    print(f"✓ Phase recovery completed")
    print()
    
    print("Step 4: Computing evaluation metrics")
    print("-"*80)
    
    metrics_comp = MetricsComputer()
    
    quality = metrics_comp.compute_sdr_sir_sar(texture.magnitude, reconstructed)
    
    mci = metrics_comp.mix_clarity_index(texture.magnitude, [reconstructed])
    
    bark_metrics = metrics_comp.bark_scale_overlap(texture.magnitude)
    
    print(f"✓ Metrics computed:")
    print(f"  SDR (Signal-to-Distortion Ratio): {quality['sdr']:.2f} dB")
    print(f"  SIR (Source-to-Interference):     {quality['sir']:.2f} dB")
    print(f"  SAR (Source-to-Artifacts):        {quality['sar']:.2f} dB")
    print(f"  MCI (Mix Clarity Index):          {mci:.1f} %")
    print(f"  Bark Entropy (Spectral Spread):   {bark_metrics['bark_entropy']:.2f}")
    print()
    
    print("Step 5: Analyzing NMF component quality")
    print("-"*80)
    
    from utils import compute_basis_redundancy
    
    redundancy = compute_basis_redundancy(W)
    
    print(f"✓ Component analysis:")
    print(f"  Mean similarity between components: {redundancy['mean_similarity']:.3f}")
    print(f"  Max similarity (redundancy check):  {redundancy['max_similarity']:.3f}")
    
    if redundancy['max_similarity'] > 0.8:
        print(f"  ⚠ WARNING: High component redundancy (>{0.8:.1f})")
        print(f"    → Components may represent mixed sources")
    else:
        print(f"  ✓ Components appear distinct")
    
    print()
    
    print("Step 6: Generating test visualization")
    print("-"*80)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    img0 = axes[0, 0].imshow(
        20 * np.log10(texture.magnitude + 1e-10),
        aspect='auto',
        origin='lower',
        cmap='magma',
        interpolation='nearest'
    )
    axes[0, 0].set_title('Original Mixture', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Frequency Bin')
    plt.colorbar(img0, ax=axes[0, 0], label='Magnitude [dB]')
    
    img1 = axes[0, 1].imshow(
        20 * np.log10(reconstructed + 1e-10),
        aspect='auto',
        origin='lower',
        cmap='magma',
        interpolation='nearest'
    )
    axes[0, 1].set_title('NMF Reconstruction', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Frequency Bin')
    plt.colorbar(img1, ax=axes[0, 1], label='Magnitude [dB]')
    
    for k in range(W.shape[1]):
        axes[1, 0].plot(W[:, k], label=f'Component {k+1}', alpha=0.7)
    axes[1, 0].set_title('NMF Basis Vectors (Spectral Templates)', 
                        fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    img3 = axes[1, 1].imshow(
        H,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    axes[1, 1].set_title('NMF Activations (Temporal Patterns)', 
                        fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Component')
    plt.colorbar(img3, ax=axes[1, 1], label='Activation')
    
    plt.suptitle(f'Quick Test Results (SDR={quality["sdr"]:.2f} dB, MCI={mci:.1f}%)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    test_dir = Path('test_results')
    test_dir.mkdir(exist_ok=True)
    
    save_path = test_dir / 'quick_test_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"✓ Visualization saved: {save_path}")
    print()
    
    print("="*80)
    print("✓ QUICK TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nYour installation is working correctly!")
    print("\nNext steps:")
    print("  1. Run full pipeline: python main.py")
    print("     (Estimated time: 2-4 hours for complete grid search)")
    print()
    print("  2. Or run minimal grid search: python main.py --test")
    print("     (Estimated time: ~30 minutes)")
    print()
    print("  3. Generate detailed spectrograms: python generate_spectrograms.py")
    print("="*80)
    
    return True


if __name__ == '__main__':
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print("\n" + "="*80)
        print("✗ ERROR DURING QUICK TEST")
        print("="*80)
        print(f"Error message: {e}")
        print("\nPlease check:")
        print("  1. All dependencies installed: pip install -r requirements.txt")
        print("  2. Python version >= 3.7")
        print("  3. Sufficient memory (>2GB RAM recommended)")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
