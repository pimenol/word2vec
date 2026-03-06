import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training(loss_h, tok_h, lr_h, save_dir, prefix=''):
    """Save a two-panel loss + learning-rate curve to *save_dir*."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    t = np.array(tok_h) / 1e6

    a1.plot(t, loss_h, lw=0.4, alpha=0.3, color='blue')
    if len(loss_h) > 50:
        w = max(1, len(loss_h) // 80)
        sm = np.convolve(loss_h, np.ones(w) / w, mode='valid')
        a1.plot(t[:len(sm)], sm, lw=2, color='red', label='Smoothed')
        a1.legend()
    a1.set_xlabel('Pairs processed (M)')
    a1.set_ylabel('Loss')
    a1.set_title('Training Loss')
    a1.grid(True, alpha=0.3)

    a2.plot(t, lr_h, lw=1, color='green')
    a2.set_xlabel('Pairs processed (M)')
    a2.set_ylabel('LR')
    a2.set_title('Learning Rate')
    a2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}training_curves.png'), dpi=150)
    plt.close()


def plot_comparison(loss_h, tok_h, loss_h2, tok_h2, save_dir):
    """Overlay smoothed loss curves for baseline vs AFWS."""
    fig, ax = plt.subplots(figsize=(10, 5))

    def smooth_plot(h, t, label):
        if len(h) > 50:
            w = max(1, len(h) // 80)
            s = np.convolve(h, np.ones(w) / w, mode='valid')
            ax.plot(np.array(t[:len(s)]) / 1e6, s, lw=2, label=label)

    smooth_plot(loss_h,  tok_h,  'Baseline')
    smooth_plot(loss_h2, tok_h2, 'AFWS')

    ax.set_xlabel('Pairs (M)')
    ax.set_ylabel('Loss')
    ax.set_title('Baseline vs AFWS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    plt.close()
