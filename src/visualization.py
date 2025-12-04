"""
TAMV Visualization - PCA and plotting utilities

Provides visualization of TAMV distributions across documents and corpora
using dimensionality reduction and various plotting techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import json
import os

from .tamv_aggregator import TAMVAggregator, TAMVProfile


class TAMVVisualizer:
    """
    Visualizes TAMV distributions using PCA and other techniques.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save output figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color palettes for different corpora/genres
        self.corpus_colors = {
            'brown': '#2ecc71',
            'cga_wiki': '#e74c3c',
            'cga_cmv': '#9b59b6'
        }

        self.genre_colors = {
            # Brown genres
            'news': '#1f77b4',
            'editorial': '#ff7f0e',
            'reviews': '#2ca02c',
            'religion': '#d62728',
            'hobbies': '#9467bd',
            'lore': '#8c564b',
            'belles_lettres': '#e377c2',
            'government': '#7f7f7f',
            'learned': '#bcbd22',
            'fiction': '#17becf',
            'mystery': '#aec7e8',
            'science_fiction': '#ffbb78',
            'adventure': '#98df8a',
            'romance': '#ff9896',
            'humor': '#c5b0d5',
            # CGA genres
            'cga_derailed': '#e74c3c',
            'cga_civil': '#3498db'
        }

    def fit_pca(self, aggregator: TAMVAggregator,
                n_components: int = 2,
                scale: bool = True) -> Tuple[np.ndarray, PCA, List[str]]:
        """
        Fit PCA on TAMV profiles.

        Args:
            aggregator: TAMVAggregator with processed profiles
            n_components: Number of PCA components
            scale: Whether to standardize features

        Returns:
            Tuple of (transformed_data, fitted_pca, document_ids)
        """
        matrix, doc_ids, labels = aggregator.to_matrix()

        if scale:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix)
            self.scaler = scaler
        else:
            self.scaler = None

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(matrix)

        self.pca = pca
        self.feature_names = labels

        return transformed, pca, doc_ids

    def plot_pca_2d(self, aggregator: TAMVAggregator,
                    color_by: str = 'genre',
                    title: str = "TAMV Distribution (PCA)",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create 2D PCA scatter plot of TAMV profiles.

        Args:
            aggregator: TAMVAggregator with processed profiles
            color_by: 'genre' or 'corpus' for coloring points
            title: Plot title
            save_path: Path to save figure (optional)
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        transformed, pca, doc_ids = self.fit_pca(aggregator, n_components=2)
        df = aggregator.to_dataframe()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get color mapping
        if color_by == 'genre':
            categories = df['genre'].values
            colors = self.genre_colors
        else:
            categories = df['corpus'].values
            colors = self.corpus_colors

        # Plot each category
        unique_cats = sorted(set(categories))
        for cat in unique_cats:
            mask = categories == cat
            color = colors.get(cat, '#333333')
            ax.scatter(transformed[mask, 0], transformed[mask, 1],
                      c=color, label=cat, alpha=0.6, s=50)

        # Add labels and legend
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_pca_loadings(self, n_top: int = 10,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot top PCA loadings to show which TAMV labels drive variation.

        Args:
            n_top: Number of top loadings to show per component
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        if not hasattr(self, 'pca'):
            raise ValueError("Run fit_pca or plot_pca_2d first")

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for i, ax in enumerate(axes):
            pc_col = f'PC{i+1}'
            # Get top positive and negative loadings
            top_pos = loadings[pc_col].nlargest(n_top // 2)
            top_neg = loadings[pc_col].nsmallest(n_top // 2)
            top_loadings = pd.concat([top_pos, top_neg]).sort_values()

            colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_loadings.values]
            ax.barh(range(len(top_loadings)), top_loadings.values, color=colors)
            ax.set_yticks(range(len(top_loadings)))
            ax.set_yticklabels([self._format_label(l) for l in top_loadings.index],
                              fontsize=8)
            ax.set_xlabel('Loading')
            ax.set_title(f'{pc_col} Loadings')
            ax.axvline(0, color='black', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def _format_label(self, label: str) -> str:
        """Format TAMV label for display."""
        # Abbreviate for readability
        abbrev = {
            'present': 'pres',
            'past': 'past',
            'future': 'fut',
            'simple': 'simp',
            'progressive': 'prog',
            'perfect': 'perf',
            'perfect-progressive': 'perf-prog',
            'indicative': 'ind',
            'subjunctive': 'subj',
            'imperative': 'imp',
            'conditional': 'cond',
            'modal': 'mod',
            'active': 'act',
            'passive': 'pass'
        }
        parts = label.split('-')
        return '-'.join(abbrev.get(p, p) for p in parts)

    def plot_genre_heatmap(self, aggregator: TAMVAggregator,
                           n_top_labels: int = 15,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create heatmap of top TAMV labels by genre.

        Args:
            aggregator: TAMVAggregator with processed profiles
            n_top_labels: Number of top TAMV labels to show
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        genre_means = aggregator.get_genre_summary()

        # Get top labels by variance across genres
        label_variance = genre_means.var(axis=0)
        top_labels = label_variance.nlargest(n_top_labels).index.tolist()

        # Subset and format
        heatmap_data = genre_means[top_labels].T
        heatmap_data.index = [self._format_label(l) for l in heatmap_data.index]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_title('TAMV Distribution by Genre')
        ax.set_xlabel('Genre')
        ax.set_ylabel('TAMV Label')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_tamv_comparison(self, aggregator: TAMVAggregator,
                             groups: Dict[str, List[str]],
                             n_top_labels: int = 10,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Compare TAMV distributions between groups of genres.

        Args:
            aggregator: TAMVAggregator with processed profiles
            groups: Dict mapping group names to lists of genres
            n_top_labels: Number of labels to compare
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        df = aggregator.to_dataframe()
        tamv_cols = aggregator.all_labels

        # Compute group means
        group_means = {}
        for group_name, genres in groups.items():
            mask = df['genre'].isin(genres)
            group_means[group_name] = df.loc[mask, tamv_cols].mean()

        group_df = pd.DataFrame(group_means)

        # Get top labels by difference
        if len(groups) == 2:
            names = list(groups.keys())
            diff = abs(group_df[names[0]] - group_df[names[1]])
            top_labels = diff.nlargest(n_top_labels).index.tolist()
        else:
            variance = group_df.var(axis=1)
            top_labels = variance.nlargest(n_top_labels).index.tolist()

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(top_labels))
        width = 0.8 / len(groups)

        for i, (group_name, means) in enumerate(group_means.items()):
            values = [means[l] for l in top_labels]
            ax.bar(x + i * width, values, width, label=group_name)

        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels([self._format_label(l) for l in top_labels],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Proportion')
        ax.set_title('TAMV Comparison Between Groups')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_toxicity_tamv_correlation(self, df: pd.DataFrame,
                                       tamv_labels: List[str],
                                       n_top: int = 10,
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot correlation between TAMV labels and toxicity scores.

        Args:
            df: DataFrame with TAMV profiles and toxicity metadata
            tamv_labels: List of TAMV label columns
            n_top: Number of top correlations to show
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        if 'meta_max_toxicity' not in df.columns:
            raise ValueError("DataFrame must contain 'meta_max_toxicity' column")

        # Compute correlations
        correlations = {}
        toxicity = df['meta_max_toxicity'].dropna()
        valid_idx = toxicity.index

        for label in tamv_labels:
            if label in df.columns:
                tamv_values = df.loc[valid_idx, label]
                if tamv_values.std() > 0:
                    corr = np.corrcoef(toxicity, tamv_values)[0, 1]
                    correlations[label] = corr

        # Sort and get top correlations
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_corr = sorted_corr[:n_top]

        labels = [self._format_label(l) for l, _ in top_corr]
        values = [c for _, c in top_corr]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Correlation with Toxicity')
        ax.set_title('TAMV-Toxicity Correlations')
        ax.axvline(0, color='black', linewidth=0.5)

        # Add correlation values as text
        for i, v in enumerate(values):
            ax.text(v + 0.01 if v > 0 else v - 0.01, i,
                   f'{v:.3f}', va='center',
                   ha='left' if v > 0 else 'right', fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def export_for_3d_visualization(self, aggregator: TAMVAggregator,
                                    similarity_threshold: float = 0.7,
                                    output_path: Optional[str] = None) -> Dict:
        """
        Export data for external 3D visualization tools.

        Prepares node and edge data suitable for D3.js, Three.js, or similar.

        Args:
            aggregator: TAMVAggregator with processed profiles
            similarity_threshold: Minimum similarity for edges
            output_path: Path to save JSON file

        Returns:
            Dict with graph data
        """
        graph_data = aggregator.prepare_graph_data(similarity_threshold)

        # Add PCA coordinates for initial positioning
        transformed, pca, _ = self.fit_pca(aggregator, n_components=3)

        for i, node in enumerate(graph_data['nodes']):
            node['x'] = float(transformed[i, 0])
            node['y'] = float(transformed[i, 1])
            node['z'] = float(transformed[i, 2]) if transformed.shape[1] > 2 else 0.0

        # Add PCA metadata
        graph_data['pca_variance_explained'] = pca.explained_variance_ratio_.tolist()

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            print(f"Exported graph data to {output_path}")

        return graph_data


if __name__ == "__main__":
    # Test visualization with sample data
    from .corpus_loaders import BrownCorpusLoader
    from .tamv_aggregator import TAMVAggregator

    print("Testing TAMV Visualization...")

    # Load sample documents
    loader = BrownCorpusLoader(genres=['news', 'fiction', 'learned'])
    docs = list(loader.load())[:30]  # Limit for testing

    # Process
    aggregator = TAMVAggregator()
    aggregator.process_documents(docs)

    # Create visualizations
    viz = TAMVVisualizer(output_dir='output')

    print("\nCreating PCA plot...")
    viz.plot_pca_2d(aggregator, color_by='genre',
                   save_path='output/pca_by_genre.png')

    print("\nCreating loadings plot...")
    viz.plot_pca_loadings(save_path='output/pca_loadings.png')

    print("\nCreating heatmap...")
    viz.plot_genre_heatmap(aggregator, save_path='output/genre_heatmap.png')

    print("\nExporting 3D data...")
    viz.export_for_3d_visualization(aggregator,
                                    output_path='output/graph_data.json')

    print("\nDone!")
