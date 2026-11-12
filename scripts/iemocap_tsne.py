#!/usr/bin/env python3
"""
Aggregate IEMOCAP embeddings across folds and produce a 2D t-SNE plot.
Loads embeddings from `datasets/IEMOCAP_KFOLD/embeddings` (W-PRO by default).
If embeddings are missing, optionally runs `extract_iemocap_embeddings.py` to create them.
"""
import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_embeddings(emb_dir, run_extractor=True, device='cuda:0'):
    wpro_file = os.path.join(emb_dir, 'wpro_embeddings.npy')
    wavlm_file = os.path.join(emb_dir, 'wavlm_base_plus_embeddings.npy')
    roberta_file = os.path.join(emb_dir, 'roberta_embeddings.npy')

    # If any embedding file exists already, consider embeddings available.
    if os.path.exists(wpro_file) or os.path.exists(wavlm_file) or os.path.exists(roberta_file):
        return True

    if not run_extractor:
        return False

    print('Embeddings not found — running extractor (this may take time)...')
    cmd = [
        'python3', 'extract_iemocap_embeddings.py',
        '--csv', 'datasets/IEMOCAP_KFOLD/all.csv',
        '--output_dir', emb_dir,
        '--device', device,
        '--batch_size_audio', '8',
        '--batch_size_text', '32'
    ]
    subprocess.check_call(cmd)
    return os.path.exists(wpro_file)


def load_metadata(emb_dir):
    meta_file = os.path.join(emb_dir, 'metadata.npz')
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata not found: {meta_file}")
    meta = np.load(meta_file, allow_pickle=True)
    return meta


def run_tsne(X, n_components=2, pca_dim=50, perplexity=30, n_iter=1000, seed=42):
    if X.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, X.shape[1]))
        Xp = pca.fit_transform(X)
    else:
        Xp = X

    # Use `max_iter` for compatibility across sklearn versions
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                init='pca',
                learning_rate='auto',
                random_state=seed,
                metric='cosine',
                max_iter=n_iter)
    Y = tsne.fit_transform(Xp)
    return Y


def plot_tsne(Y, labels, folds, out_png, title='IEMOCAP t-SNE (W-PRO)'):
    label_map = {'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral'}
    colors = {'ang': '#d62728', 'hap': '#1f77b4', 'sad': '#2ca02c', 'neu': '#9467bd'}

    plt.figure(figsize=(10, 8))
    for em in np.unique(labels):
        idx = labels == em
        plt.scatter(Y[idx, 0], Y[idx, 1], c=colors.get(em, '#7f7f7f'), label=label_map.get(em, em), s=12, alpha=0.8)

    plt.legend(title='Emotion')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', default='datasets/IEMOCAP_KFOLD/embeddings')
    parser.add_argument('--use', choices=['wpro', 'wavlm', 'roberta'], default='wpro')
    parser.add_argument('--out', default='outputs/iemocap_tsne_wpro.png')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--pca_dim', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(args.emb_dir, exist_ok=True)

    ok = ensure_embeddings(args.emb_dir, run_extractor=not args.skip_extract, device=args.device)
    if not ok:
        raise SystemExit('Embeddings not available. Re-run with extraction enabled or generate embeddings first.')

    meta = load_metadata(args.emb_dir)
    if args.use == 'wpro':
        emb_file = os.path.join(args.emb_dir, 'wpro_embeddings.npy')
    elif args.use == 'wavlm':
        emb_file = os.path.join(args.emb_dir, 'wavlm_base_plus_embeddings.npy')
    else:
        emb_file = os.path.join(args.emb_dir, 'roberta_embeddings.npy')

    X = np.load(emb_file)
    labels = np.array(meta['emotion'])
    folds = np.array(meta['fold'])

    print(f'Loaded embeddings: {X.shape}, labels: {labels.shape}')

    print('Running t-SNE (this can take a minute)...')
    Y = run_tsne(X, pca_dim=args.pca_dim, perplexity=args.perplexity)

    np.savez(args.out.replace('.png', '_coords.npz'), tsne=Y, labels=labels, folds=folds)
    print(f'  ✓ Saved coords to {args.out.replace(".png","_coords.npz")}')

    plot_tsne(Y, labels, folds, args.out, title=f'IEMOCAP t-SNE ({args.use.upper()})')
    print(f'  ✓ Saved t-SNE figure to {args.out}')


if __name__ == '__main__':
    main()
