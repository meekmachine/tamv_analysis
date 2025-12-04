"""
Toxicity Analysis - Correlating TAMV profiles with toxicity in CGA

Analyzes relationships between TAMV distributions and toxicity scores
in the Conversations Gone Awry datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import warnings

from .tamv_aggregator import TAMVAggregator, TAMVProfile, get_all_tamv_labels
from .corpus_loaders import CGALoader, Conversation


class ToxicityAnalyzer:
    """
    Analyzes relationships between TAMV profiles and toxicity/derailment.
    """

    def __init__(self, aggregator: TAMVAggregator):
        """
        Initialize with a TAMVAggregator containing CGA profiles.

        Args:
            aggregator: TAMVAggregator with processed CGA documents
        """
        self.aggregator = aggregator
        self.all_labels = get_all_tamv_labels()
        self.df = aggregator.to_dataframe()

    def compute_correlations(self, toxicity_col: str = 'meta_max_toxicity') -> pd.DataFrame:
        """
        Compute correlations between TAMV labels and toxicity.

        Args:
            toxicity_col: Column name for toxicity scores

        Returns:
            DataFrame with correlation statistics for each TAMV label
        """
        if toxicity_col not in self.df.columns:
            raise ValueError(f"Column '{toxicity_col}' not found in data")

        results = []
        toxicity = self.df[toxicity_col].dropna()
        valid_idx = toxicity.index

        for label in self.all_labels:
            if label not in self.df.columns:
                continue

            tamv_values = self.df.loc[valid_idx, label]

            # Skip if no variance
            if tamv_values.std() == 0:
                continue

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(toxicity, tamv_values)

            # Spearman correlation (more robust to outliers)
            spearman_r, spearman_p = stats.spearmanr(toxicity, tamv_values)

            results.append({
                'tamv_label': label,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significant': pearson_p < 0.05
            })

        return pd.DataFrame(results).sort_values('pearson_r', key=abs, ascending=False)

    def compare_derailed_vs_civil(self) -> pd.DataFrame:
        """
        Compare TAMV distributions between derailed and civil conversations.

        Returns:
            DataFrame with comparison statistics
        """
        if 'meta_has_derailed' not in self.df.columns:
            raise ValueError("Data must contain 'meta_has_derailed' column")

        derailed = self.df[self.df['meta_has_derailed'] == True]
        civil = self.df[self.df['meta_has_derailed'] == False]

        results = []
        for label in self.all_labels:
            if label not in self.df.columns:
                continue

            derailed_vals = derailed[label].dropna()
            civil_vals = civil[label].dropna()

            if len(derailed_vals) == 0 or len(civil_vals) == 0:
                continue

            # Means
            derailed_mean = derailed_vals.mean()
            civil_mean = civil_vals.mean()

            # T-test
            t_stat, p_value = stats.ttest_ind(derailed_vals, civil_vals)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((derailed_vals.std()**2 + civil_vals.std()**2) / 2)
            if pooled_std > 0:
                cohens_d = (derailed_mean - civil_mean) / pooled_std
            else:
                cohens_d = 0

            results.append({
                'tamv_label': label,
                'derailed_mean': derailed_mean,
                'civil_mean': civil_mean,
                'difference': derailed_mean - civil_mean,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            })

        return pd.DataFrame(results).sort_values('cohens_d', key=abs, ascending=False)

    def toxicity_bands_analysis(self,
                                 toxicity_col: str = 'meta_max_toxicity',
                                 n_bands: int = 4) -> pd.DataFrame:
        """
        Analyze TAMV distributions across toxicity bands.

        Args:
            toxicity_col: Column for toxicity scores
            n_bands: Number of toxicity bands to create

        Returns:
            DataFrame with mean TAMV per toxicity band
        """
        if toxicity_col not in self.df.columns:
            raise ValueError(f"Column '{toxicity_col}' not found")

        # Create toxicity bands
        df_with_tox = self.df[self.df[toxicity_col].notna()].copy()
        df_with_tox['toxicity_band'] = pd.qcut(
            df_with_tox[toxicity_col],
            q=n_bands,
            labels=[f'Q{i+1}' for i in range(n_bands)]
        )

        # Compute means per band
        tamv_cols = [c for c in self.all_labels if c in df_with_tox.columns]
        band_means = df_with_tox.groupby('toxicity_band')[tamv_cols].mean()

        return band_means

    def predict_derailment(self, cv_folds: int = 5) -> Dict:
        """
        Train a simple classifier to predict derailment from TAMV features.

        Args:
            cv_folds: Number of cross-validation folds

        Returns:
            Dict with model performance metrics
        """
        if 'meta_has_derailed' not in self.df.columns:
            raise ValueError("Data must contain 'meta_has_derailed' column")

        # Prepare features and target
        tamv_cols = [c for c in self.all_labels if c in self.df.columns]
        X = self.df[tamv_cols].values
        y = self.df['meta_has_derailed'].values

        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(int)

        if len(X) < cv_folds * 2:
            return {'error': 'Not enough data for cross-validation'}

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='f1')
            cv_auc = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='roc_auc')

        # Fit final model to get coefficients
        model.fit(X_scaled, y)

        # Get top predictive features
        coef_df = pd.DataFrame({
            'feature': tamv_cols,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        return {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'auc_mean': cv_auc.mean(),
            'auc_std': cv_auc.std(),
            'top_positive_features': coef_df[coef_df['coefficient'] > 0].head(5).to_dict('records'),
            'top_negative_features': coef_df[coef_df['coefficient'] < 0].head(5).to_dict('records'),
            'n_samples': len(y),
            'n_derailed': int(y.sum()),
            'n_civil': int(len(y) - y.sum())
        }

    def plot_toxicity_distribution(self, save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot toxicity distribution for derailed vs civil conversations.

        Args:
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if 'meta_max_toxicity' not in self.df.columns:
            raise ValueError("Data must contain toxicity scores")

        fig, ax = plt.subplots(figsize=figsize)

        derailed = self.df[self.df['meta_has_derailed'] == True]['meta_max_toxicity'].dropna()
        civil = self.df[self.df['meta_has_derailed'] == False]['meta_max_toxicity'].dropna()

        ax.hist(civil, bins=30, alpha=0.5, label=f'Civil (n={len(civil)})', color='#3498db')
        ax.hist(derailed, bins=30, alpha=0.5, label=f'Derailed (n={len(derailed)})', color='#e74c3c')

        ax.set_xlabel('Maximum Toxicity Score')
        ax.set_ylabel('Count')
        ax.set_title('Toxicity Distribution: Derailed vs Civil Conversations')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_tamv_by_toxicity_band(self, n_top_labels: int = 8,
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot TAMV distributions across toxicity bands.

        Args:
            n_top_labels: Number of TAMV labels to show
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        band_means = self.toxicity_bands_analysis()

        # Get labels with highest variance across bands
        label_var = band_means.var()
        top_labels = label_var.nlargest(n_top_labels).index.tolist()

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(band_means.index))
        width = 0.8 / n_top_labels

        for i, label in enumerate(top_labels):
            values = band_means[label].values
            # Abbreviate label
            short_label = '-'.join(label.split('-')[:2])  # Just tense-aspect
            ax.bar(x + i * width, values, width, label=short_label)

        ax.set_xticks(x + width * (n_top_labels - 1) / 2)
        ax.set_xticklabels([f'{band}\n(low→high tox)' for band in band_means.index])
        ax.set_ylabel('Mean Proportion')
        ax.set_title('TAMV Distribution Across Toxicity Bands')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_derailed_vs_civil_comparison(self, n_top: int = 10,
                                           save_path: Optional[str] = None,
                                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison of TAMV profiles between derailed and civil conversations.

        Args:
            n_top: Number of top differentiating labels to show
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        comparison = self.compare_derailed_vs_civil()
        top_diff = comparison.head(n_top)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(top_diff))
        width = 0.35

        # Abbreviate labels
        labels = ['-'.join(l.split('-')[:2]) for l in top_diff['tamv_label']]

        ax.bar(x - width/2, top_diff['civil_mean'], width, label='Civil', color='#3498db')
        ax.bar(x + width/2, top_diff['derailed_mean'], width, label='Derailed', color='#e74c3c')

        # Add significance markers
        for i, (_, row) in enumerate(top_diff.iterrows()):
            if row['significant']:
                max_val = max(row['civil_mean'], row['derailed_mean'])
                ax.text(i, max_val + 0.01, '*', ha='center', fontsize=14)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Proportion')
        ax.set_title('TAMV Comparison: Derailed vs Civil Conversations\n(* p < 0.05)')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_report(self, output_dir: str = 'output') -> str:
        """
        Generate a comprehensive toxicity analysis report.

        Args:
            output_dir: Directory to save report and figures

        Returns:
            Report text
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TAMV-TOXICITY ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Dataset summary
        n_total = len(self.df)
        n_derailed = self.df['meta_has_derailed'].sum() if 'meta_has_derailed' in self.df.columns else 0
        n_civil = n_total - n_derailed

        report_lines.append("DATASET SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total conversations: {n_total}")
        report_lines.append(f"Derailed: {n_derailed}")
        report_lines.append(f"Civil: {n_civil}")
        report_lines.append("")

        # Correlation analysis
        if 'meta_max_toxicity' in self.df.columns:
            report_lines.append("TAMV-TOXICITY CORRELATIONS")
            report_lines.append("-" * 40)
            corr_df = self.compute_correlations()
            sig_corr = corr_df[corr_df['significant']]
            report_lines.append(f"Significant correlations (p < 0.05): {len(sig_corr)}")
            report_lines.append("\nTop 5 positive correlations:")
            for _, row in corr_df[corr_df['pearson_r'] > 0].head(5).iterrows():
                report_lines.append(f"  {row['tamv_label']}: r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})")
            report_lines.append("\nTop 5 negative correlations:")
            for _, row in corr_df[corr_df['pearson_r'] < 0].head(5).iterrows():
                report_lines.append(f"  {row['tamv_label']}: r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})")
            report_lines.append("")

        # Derailed vs Civil comparison
        if 'meta_has_derailed' in self.df.columns:
            report_lines.append("DERAILED VS CIVIL COMPARISON")
            report_lines.append("-" * 40)
            comparison = self.compare_derailed_vs_civil()
            sig_diff = comparison[comparison['significant']]
            report_lines.append(f"Significant differences (p < 0.05): {len(sig_diff)}")
            report_lines.append("\nTop 5 differentiating TAMV labels (by effect size):")
            for _, row in comparison.head(5).iterrows():
                direction = "higher in derailed" if row['difference'] > 0 else "higher in civil"
                report_lines.append(f"  {row['tamv_label']}: d={row['cohens_d']:.3f} ({direction})")
            report_lines.append("")

        # Prediction model
        if 'meta_has_derailed' in self.df.columns:
            report_lines.append("DERAILMENT PREDICTION MODEL")
            report_lines.append("-" * 40)
            pred_results = self.predict_derailment()
            if 'error' not in pred_results:
                report_lines.append(f"Accuracy: {pred_results['accuracy_mean']:.3f} ± {pred_results['accuracy_std']:.3f}")
                report_lines.append(f"F1 Score: {pred_results['f1_mean']:.3f} ± {pred_results['f1_std']:.3f}")
                report_lines.append(f"AUC-ROC: {pred_results['auc_mean']:.3f} ± {pred_results['auc_std']:.3f}")
                report_lines.append("\nTop predictive features (positive = predicts derailment):")
                for feat in pred_results['top_positive_features'][:3]:
                    report_lines.append(f"  {feat['feature']}: coef={feat['coefficient']:.3f}")
            else:
                report_lines.append(f"Could not train model: {pred_results['error']}")
            report_lines.append("")

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = os.path.join(output_dir, 'toxicity_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"Report saved to {report_path}")

        return report_text


if __name__ == "__main__":
    print("Toxicity Analyzer module - run via main.py")
