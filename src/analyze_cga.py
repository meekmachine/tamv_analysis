#!/usr/bin/env python3
"""
CGA (Conversations Gone Awry) TAMV Analysis

Simple comparison of TAMV rates between civil and derailed conversations.

Usage:
    python -m src.analyze_cga --dataset wiki --sample-rate 10
    python -m src.analyze_cga --dataset cmv --sample-rate 5 --output output/cga_analysis
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .corpus_loaders import CGALoader, Conversation
from .tamv_extractor import TAMVExtractor, get_all_tamv_labels


@dataclass
class ConversationProfile:
    """TAMV profile for a conversation."""
    conversation_id: str
    word_count: int
    tamv_counts: Dict[str, int] = field(default_factory=dict)
    tamv_rates: Dict[str, float] = field(default_factory=dict)  # per 100 words
    derailed: bool = False

    def compute_rates(self):
        """Compute TAMV rates per 100 words."""
        if self.word_count > 0:
            for label, count in self.tamv_counts.items():
                self.tamv_rates[label] = (count / self.word_count) * 100


class CGAAnalyzer:
    """Analyzes TAMV patterns in CGA conversations."""

    def __init__(self, dataset: str = 'wiki', sample_rate: int = 1):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.extractor = TAMVExtractor()
        self.profiles: List[ConversationProfile] = []
        self.all_labels = get_all_tamv_labels()

    def load_and_process(self) -> None:
        """Load conversations and extract TAMV profiles."""
        loader = CGALoader(dataset=self.dataset)

        print(f"Loading {self.dataset} dataset with 1/{self.sample_rate} sampling...")

        for i, convo in enumerate(loader.load()):
            if i % self.sample_rate != 0:
                continue

            # Get full conversation text
            text = convo.full_text
            word_count = len(text.split())

            if word_count < 20:  # Skip very short conversations
                continue

            # Extract TAMV
            tamv_counts: Dict[str, int] = defaultdict(int)
            try:
                labels = self.extractor.extract_from_text(text)
                for label in labels:
                    tamv_counts[label.to_string()] += 1
            except Exception:
                continue

            profile = ConversationProfile(
                conversation_id=convo.id,
                word_count=word_count,
                tamv_counts=dict(tamv_counts),
                derailed=convo.has_derailed
            )
            profile.compute_rates()
            self.profiles.append(profile)

            if (len(self.profiles)) % 100 == 0:
                print(f"  Processed {len(self.profiles)} conversations...")

        print(f"Loaded {len(self.profiles)} conversations")

    def compute_comparison(self) -> Dict:
        """Compare TAMV rates between civil and derailed conversations."""
        civil_rates: Dict[str, List[float]] = defaultdict(list)
        derailed_rates: Dict[str, List[float]] = defaultdict(list)

        for profile in self.profiles:
            target = derailed_rates if profile.derailed else civil_rates
            for label in self.all_labels:
                rate = profile.tamv_rates.get(label, 0.0)
                target[label].append(rate)

        # Compute statistics for each label
        results = {}
        for label in self.all_labels:
            civil = civil_rates[label]
            derailed = derailed_rates[label]

            if len(civil) < 5 or len(derailed) < 5:
                continue

            civil_mean = np.mean(civil)
            derailed_mean = np.mean(derailed)
            civil_std = np.std(civil, ddof=1)
            derailed_std = np.std(derailed, ddof=1)

            # T-test
            t_stat, p_value = stats.ttest_ind(civil, derailed)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((civil_std**2 + derailed_std**2) / 2)
            cohens_d = (derailed_mean - civil_mean) / pooled_std if pooled_std > 0 else 0

            # Overall mean and std (baseline)
            all_rates = civil + derailed
            overall_mean = np.mean(all_rates)
            overall_std = np.std(all_rates, ddof=1)

            # Z-scores: how far each group deviates from overall baseline
            civil_z = (civil_mean - overall_mean) / overall_std if overall_std > 0 else 0
            derailed_z = (derailed_mean - overall_mean) / overall_std if overall_std > 0 else 0

            results[label] = {
                'civil_mean': civil_mean,
                'civil_std': civil_std,
                'civil_n': len(civil),
                'derailed_mean': derailed_mean,
                'derailed_std': derailed_std,
                'derailed_n': len(derailed),
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'civil_z': civil_z,  # deviation from baseline
                'derailed_z': derailed_z,  # deviation from baseline
                'difference': derailed_mean - civil_mean,
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }

        return results

    def compute_feature_comparison(self) -> Dict[str, Dict]:
        """Compare individual TAMV features (tense, aspect, mood, voice)."""
        # Aggregate by individual features
        features = {
            'tense': ['present', 'past', 'future', 'none'],
            'aspect': ['simple', 'progressive', 'perfect', 'perfect-progressive'],
            'mood': ['indicative', 'subjunctive', 'imperative', 'conditional', 'modal', 'infinitival', 'participle'],
            'voice': ['active', 'passive']
        }

        results = {}
        for feature_name, values in features.items():
            results[feature_name] = {}

            for value in values:
                civil_rates = []
                derailed_rates = []

                for profile in self.profiles:
                    # Sum rates for all TAMV combos containing this feature value
                    rate = sum(
                        profile.tamv_rates.get(label, 0.0)
                        for label in self.all_labels
                        if value in label.split('-')
                    )
                    if profile.derailed:
                        derailed_rates.append(rate)
                    else:
                        civil_rates.append(rate)

                if len(civil_rates) < 5 or len(derailed_rates) < 5:
                    continue

                civil_mean = np.mean(civil_rates)
                derailed_mean = np.mean(derailed_rates)
                civil_std = np.std(civil_rates, ddof=1)
                derailed_std = np.std(derailed_rates, ddof=1)

                # Baseline
                all_rates = civil_rates + derailed_rates
                overall_mean = np.mean(all_rates)
                overall_std = np.std(all_rates, ddof=1)

                civil_z = (civil_mean - overall_mean) / overall_std if overall_std > 0 else 0
                derailed_z = (derailed_mean - overall_mean) / overall_std if overall_std > 0 else 0

                t_stat, p_value = stats.ttest_ind(civil_rates, derailed_rates)

                results[feature_name][value] = {
                    'civil_mean': civil_mean,
                    'civil_std': civil_std,
                    'derailed_mean': derailed_mean,
                    'derailed_std': derailed_std,
                    'overall_mean': overall_mean,
                    'civil_z': civil_z,
                    'derailed_z': derailed_z,
                    'difference': derailed_mean - civil_mean,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        return results

    def generate_report(self, comparison: Dict) -> str:
        """Generate text report."""
        n_civil = sum(1 for p in self.profiles if not p.derailed)
        n_derailed = sum(1 for p in self.profiles if p.derailed)

        lines = [
            "=" * 70,
            f"CGA TAMV ANALYSIS REPORT ({self.dataset.upper()})",
            "=" * 70,
            "",
            "DATASET SUMMARY",
            "-" * 40,
            f"Dataset:              {self.dataset}",
            f"Sample rate:          1/{self.sample_rate}",
            f"Total conversations:  {len(self.profiles)}",
            f"Civil:                {n_civil}",
            f"Derailed:             {n_derailed}",
            f"TAMV categories:      {len(comparison)}",
            "",
        ]

        # Top differences (sorted by absolute Cohen's d)
        sorted_labels = sorted(
            comparison.items(),
            key=lambda x: abs(x[1]['cohens_d']),
            reverse=True
        )

        lines.extend([
            "TOP 20 TAMV DIFFERENCES (by effect size)",
            "-" * 70,
            f"{'Label':<40} {'Civil':>8} {'Derailed':>10} {'Diff':>8} {'d':>6} {'p':>8}",
            "-" * 70,
        ])

        for label, s in sorted_labels[:20]:
            sig = "*" if s['significant'] else ""
            lines.append(
                f"{label:<40} {s['civil_mean']:>8.3f} {s['derailed_mean']:>10.3f} "
                f"{s['difference']:>+8.3f} {s['cohens_d']:>+6.2f} {s['p_value']:>7.4f}{sig}"
            )

        # Significant differences only
        sig_labels = [(l, s) for l, s in sorted_labels if s['significant']]
        lines.extend([
            "",
            f"SIGNIFICANT DIFFERENCES (p < 0.05): {len(sig_labels)}",
            "-" * 40,
        ])

        # Higher in derailed
        higher_derailed = [(l, s) for l, s in sig_labels if s['difference'] > 0]
        lines.append(f"\nHigher in DERAILED conversations ({len(higher_derailed)}):")
        for label, s in sorted(higher_derailed, key=lambda x: x[1]['difference'], reverse=True)[:10]:
            lines.append(f"  {label}: +{s['difference']:.3f} (d={s['cohens_d']:+.2f})")

        # Higher in civil
        higher_civil = [(l, s) for l, s in sig_labels if s['difference'] < 0]
        lines.append(f"\nHigher in CIVIL conversations ({len(higher_civil)}):")
        for label, s in sorted(higher_civil, key=lambda x: x[1]['difference'])[:10]:
            lines.append(f"  {label}: {s['difference']:.3f} (d={s['cohens_d']:+.2f})")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def save_data(self, comparison: Dict, output_path: str) -> None:
        """Save comparison data to TSV."""
        with open(output_path, 'w') as f:
            header = [
                'tamv_label', 'civil_mean', 'civil_std', 'civil_n',
                'derailed_mean', 'derailed_std', 'derailed_n',
                'difference', 'cohens_d', 't_stat', 'p_value', 'significant'
            ]
            f.write('\t'.join(header) + '\n')

            for label, s in sorted(comparison.items()):
                row = [
                    label,
                    f"{s['civil_mean']:.6f}",
                    f"{s['civil_std']:.6f}",
                    str(s['civil_n']),
                    f"{s['derailed_mean']:.6f}",
                    f"{s['derailed_std']:.6f}",
                    str(s['derailed_n']),
                    f"{s['difference']:.6f}",
                    f"{s['cohens_d']:.4f}",
                    f"{s['t_stat']:.4f}",
                    f"{s['p_value']:.6f}",
                    '1' if s['significant'] else '0'
                ]
                f.write('\t'.join(row) + '\n')

    def run(self, output_dir: str = 'output') -> Tuple[Dict, Dict]:
        """Run full analysis."""
        self.load_and_process()

        if not self.profiles:
            print("No conversations processed.")
            return {}, {}

        comparison = self.compute_comparison()
        feature_comparison = self.compute_feature_comparison()

        # Generate report
        report = self.generate_report(comparison)
        print(report)

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, f'cga_{self.dataset}_analysis.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        data_path = os.path.join(output_dir, f'cga_{self.dataset}_comparison.tsv')
        self.save_data(comparison, data_path)
        print(f"Data saved to: {data_path}")

        return comparison, feature_comparison


class CGAVisualizer:
    """Visualizations for CGA TAMV analysis."""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.derailed_color = '#e74c3c'
        self.civil_color = '#3498db'

    def _format_label(self, label: str) -> str:
        """Abbreviate TAMV label."""
        abbrev = {
            'present': 'pres', 'past': 'past', 'future': 'fut', 'none': 'none',
            'simple': 'simp', 'progressive': 'prog', 'perfect': 'perf',
            'perfect-progressive': 'p-prog', 'indicative': 'ind',
            'subjunctive': 'subj', 'imperative': 'imp', 'conditional': 'cond',
            'modal': 'mod', 'infinitival': 'inf', 'participle': 'part',
            'active': 'act', 'passive': 'pass'
        }
        parts = label.split('-')
        return '-'.join(abbrev.get(p, p) for p in parts)

    def plot_comparison(self, comparison: Dict, n_top: int = 15,
                        save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Grouped bar chart comparing civil vs derailed for top TAMV categories."""
        # Sort by frequency (sum of means)
        sorted_labels = sorted(
            comparison.items(),
            key=lambda x: x[1]['civil_mean'] + x[1]['derailed_mean'],
            reverse=True
        )[:n_top]

        labels = [self._format_label(l) for l, _ in sorted_labels]
        civil_means = [s['civil_mean'] for _, s in sorted_labels]
        derailed_means = [s['derailed_mean'] for _, s in sorted_labels]
        civil_stds = [s['civil_std'] for _, s in sorted_labels]
        derailed_stds = [s['derailed_std'] for _, s in sorted_labels]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, civil_means, width, yerr=civil_stds,
                       label='Civil', color=self.civil_color, capsize=3, alpha=0.8)
        bars2 = ax.bar(x + width/2, derailed_means, width, yerr=derailed_stds,
                       label='Derailed', color=self.derailed_color, capsize=3, alpha=0.8)

        # Mark significant differences
        for i, (label, s) in enumerate(sorted_labels):
            if s['significant']:
                max_val = max(s['civil_mean'] + s['civil_std'],
                              s['derailed_mean'] + s['derailed_std'])
                ax.text(i, max_val + 0.1, '*', ha='center', fontsize=14, fontweight='bold')

        ax.set_ylabel('Rate per 100 words', fontsize=12)
        ax.set_xlabel('TAMV Category', fontsize=12)
        ax.set_title('TAMV Rates: Civil vs Derailed Conversations (* p < 0.05)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_effect_sizes(self, comparison: Dict, n_top: int = 20,
                          save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Horizontal bar chart of effect sizes (Cohen's d)."""
        # Sort by absolute effect size, filter significant
        sorted_labels = sorted(
            [(l, s) for l, s in comparison.items() if s['significant']],
            key=lambda x: abs(x[1]['cohens_d']),
            reverse=True
        )[:n_top]

        if not sorted_labels:
            print("No significant differences to plot")
            return None

        # Sort by value for display
        sorted_labels.sort(key=lambda x: x[1]['cohens_d'])

        labels = [self._format_label(l) for l, _ in sorted_labels]
        effects = [s['cohens_d'] for _, s in sorted_labels]
        colors = [self.derailed_color if d > 0 else self.civil_color for d in effects]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, effects, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
        ax.set_title('Significant TAMV Differences by Effect Size\n(+ = higher in derailed, - = higher in civil)', fontsize=12)
        ax.axvline(0, color='black', linewidth=1)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.derailed_color, label='Higher in Derailed'),
            Patch(facecolor=self.civil_color, label='Higher in Civil')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_feature_comparison(self, feature_comparison: Dict[str, Dict],
                                  save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Bar chart showing civil vs derailed for each individual TAMV feature."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        feature_order = ['tense', 'aspect', 'mood', 'voice']

        for idx, feature_name in enumerate(feature_order):
            ax = axes[idx]
            data = feature_comparison.get(feature_name, {})

            if not data:
                ax.set_title(f'{feature_name.title()}: No data')
                continue

            labels = list(data.keys())
            civil_means = [data[l]['civil_mean'] for l in labels]
            derailed_means = [data[l]['derailed_mean'] for l in labels]
            civil_stds = [data[l]['civil_std'] for l in labels]
            derailed_stds = [data[l]['derailed_std'] for l in labels]
            significant = [data[l]['significant'] for l in labels]

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, civil_means, width, yerr=civil_stds,
                          label='Civil', color=self.civil_color, capsize=3, alpha=0.8)
            bars2 = ax.bar(x + width/2, derailed_means, width, yerr=derailed_stds,
                          label='Derailed', color=self.derailed_color, capsize=3, alpha=0.8)

            # Mark significant differences
            for i, sig in enumerate(significant):
                if sig:
                    max_val = max(civil_means[i] + civil_stds[i], derailed_means[i] + derailed_stds[i])
                    ax.text(i, max_val + 0.3, '*', ha='center', fontsize=14, fontweight='bold')

            ax.set_ylabel('Rate per 100 words')
            ax.set_title(f'{feature_name.title()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.legend(loc='upper right', fontsize=8)

        plt.suptitle('Individual TAMV Features: Civil vs Derailed (* p < 0.05)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_deviation_from_baseline(self, comparison: Dict, n_top: int = 20,
                                      save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Show how far above/below baseline each group is for top TAMV categories.
        This visualizes the 'predictive power' - categories where civil and derailed
        deviate in opposite directions from the baseline are most predictive.
        """
        # Filter to significant differences and sort by absolute z-difference
        sig_data = [
            (label, s) for label, s in comparison.items()
            if s.get('significant', False) and abs(s.get('civil_z', 0)) + abs(s.get('derailed_z', 0)) > 0
        ]

        if not sig_data:
            # Fall back to top by effect size
            sig_data = sorted(comparison.items(), key=lambda x: abs(x[1].get('cohens_d', 0)), reverse=True)[:n_top]

        # Sort by the difference in z-scores (predictive power)
        sig_data.sort(key=lambda x: abs(x[1].get('derailed_z', 0) - x[1].get('civil_z', 0)), reverse=True)
        sig_data = sig_data[:n_top]

        if not sig_data:
            print("No data to plot for deviation analysis")
            return None

        labels = [self._format_label(l) for l, _ in sig_data]
        civil_z = [s['civil_z'] for _, s in sig_data]
        derailed_z = [s['derailed_z'] for _, s in sig_data]

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(labels))
        height = 0.35

        bars1 = ax.barh(y_pos - height/2, civil_z, height, label='Civil', color=self.civil_color, alpha=0.8)
        bars2 = ax.barh(y_pos + height/2, derailed_z, height, label='Derailed', color=self.derailed_color, alpha=0.8)

        ax.axvline(0, color='black', linewidth=1, linestyle='-')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Z-score (deviation from baseline)', fontsize=11)
        ax.set_title('TAMV Deviation from Baseline\\n(Opposite directions = predictive of outcome)', fontsize=12)
        ax.legend(loc='lower right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_all(self, comparison: Dict, feature_comparison: Dict, dataset: str, show: bool = True) -> None:
        """Generate all visualizations."""
        prefix = f'cga_{dataset}_'

        print("\nGenerating visualizations...")

        self.plot_comparison(
            comparison,
            save_path=os.path.join(self.output_dir, f'{prefix}comparison.png'),
            show=show
        )

        self.plot_effect_sizes(
            comparison,
            save_path=os.path.join(self.output_dir, f'{prefix}effect_sizes.png'),
            show=show
        )

        self.plot_feature_comparison(
            feature_comparison,
            save_path=os.path.join(self.output_dir, f'{prefix}features.png'),
            show=show
        )

        self.plot_deviation_from_baseline(
            comparison,
            save_path=os.path.join(self.output_dir, f'{prefix}deviation.png'),
            show=show
        )

        print(f"Visualizations saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TAMV patterns between civil and derailed conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset wiki --sample-rate 10
  %(prog)s --dataset cmv --sample-rate 5
  %(prog)s --dataset wiki --no-show
        """
    )

    parser.add_argument('--dataset', '-d', choices=['wiki', 'cmv'], default='wiki',
                        help='Dataset: wiki (Wikipedia) or cmv (Reddit)')
    parser.add_argument('--sample-rate', '-s', type=int, default=10,
                        help='Take 1 out of every N conversations (default: 10)')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display interactive plots')

    args = parser.parse_args()

    analyzer = CGAAnalyzer(dataset=args.dataset, sample_rate=args.sample_rate)
    comparison, feature_comparison = analyzer.run(output_dir=args.output)

    if comparison:
        visualizer = CGAVisualizer(output_dir=args.output)
        visualizer.plot_all(comparison, feature_comparison, args.dataset, show=not args.no_show)


if __name__ == '__main__':
    main()
