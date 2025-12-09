#!/usr/bin/env python3
"""
Utterance-Level TAMV Analysis

Simple correlation between TAMV features and toxicity scores at the utterance level.

Usage:
    python -m src.utterance_analysis --dataset wiki --sample-rate 1
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .corpus_loaders import CGALoader
from .tamv_extractor import TAMVExtractor


@dataclass
class UtteranceProfile:
    """TAMV profile for a single utterance."""
    utterance_id: str
    conversation_id: str
    speaker: str
    text: str
    word_count: int
    toxicity: float
    tamv_counts: Dict[str, int] = field(default_factory=dict)
    # Individual features
    tense_counts: Dict[str, int] = field(default_factory=dict)
    aspect_counts: Dict[str, int] = field(default_factory=dict)
    mood_counts: Dict[str, int] = field(default_factory=dict)
    voice_counts: Dict[str, int] = field(default_factory=dict)


class UtteranceAnalyzer:
    """Analyzes TAMV patterns at utterance level with toxicity correlation."""

    def __init__(self, dataset: str = 'wiki', sample_rate: int = 1):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.extractor = TAMVExtractor()
        self.profiles: List[UtteranceProfile] = []

    def load_and_process(self) -> None:
        """Load utterances and extract TAMV profiles."""
        loader = CGALoader(dataset=self.dataset)

        print(f"Loading {self.dataset} dataset with 1/{self.sample_rate} sampling...")

        convo_count = 0
        for i, convo in enumerate(loader.load()):
            if i % self.sample_rate != 0:
                continue

            convo_count += 1

            for utt in convo.utterances:
                text = utt.get('text', '')
                word_count = len(text.split())

                if word_count < 5:  # Skip very short utterances
                    continue

                toxicity = utt.get('toxicity', 0.0)
                if toxicity is None:
                    toxicity = 0.0

                # Extract TAMV
                try:
                    labels = self.extractor.extract_from_text(text)
                except Exception:
                    continue

                if not labels:
                    continue

                # Count TAMV combos and individual features
                tamv_counts: Dict[str, int] = defaultdict(int)
                tense_counts: Dict[str, int] = defaultdict(int)
                aspect_counts: Dict[str, int] = defaultdict(int)
                mood_counts: Dict[str, int] = defaultdict(int)
                voice_counts: Dict[str, int] = defaultdict(int)

                for label in labels:
                    tamv_counts[label.to_string()] += 1
                    tense_counts[label.tense.value] += 1
                    aspect_counts[label.aspect.value] += 1
                    mood_counts[label.mood.value] += 1
                    voice_counts[label.voice.value] += 1

                profile = UtteranceProfile(
                    utterance_id=utt.get('id', ''),
                    conversation_id=convo.id,
                    speaker=utt.get('speaker', ''),
                    text=text,
                    word_count=word_count,
                    toxicity=toxicity,
                    tamv_counts=dict(tamv_counts),
                    tense_counts=dict(tense_counts),
                    aspect_counts=dict(aspect_counts),
                    mood_counts=dict(mood_counts),
                    voice_counts=dict(voice_counts)
                )
                self.profiles.append(profile)

            if convo_count % 100 == 0:
                print(f"  Processed {convo_count} conversations, {len(self.profiles)} utterances...")

        print(f"Loaded {len(self.profiles)} utterances from {convo_count} conversations")

    def compute_combo_correlations(self) -> Dict[str, Dict]:
        """Compute correlations for all TAMV combinations with toxicity."""
        from collections import defaultdict

        # Get all toxicity scores
        toxicities = np.array([p.toxicity for p in self.profiles])
        n_profiles = len(self.profiles)

        # First pass: find all unique TAMV combos
        all_combos = set()
        for p in self.profiles:
            all_combos.update(p.tamv_counts.keys())

        # Second pass: build rate arrays for each combo
        combo_rates: Dict[str, np.ndarray] = {}
        for label in all_combos:
            rates = []
            for p in self.profiles:
                count = p.tamv_counts.get(label, 0)
                rate = count / p.word_count * 100
                rates.append(rate)
            combo_rates[label] = np.array(rates)

        results = {}
        for label, rates_arr in combo_rates.items():
            # Need enough non-zero values
            if np.sum(rates_arr > 0) < 10:
                continue

            if np.std(rates_arr) > 0:
                r, p_value = stats.pearsonr(rates_arr, toxicities)

                # High vs low toxicity means
                median_tox = np.median(toxicities)
                high_mask = toxicities > median_tox
                high_mean = np.mean(rates_arr[high_mask])
                low_mean = np.mean(rates_arr[~high_mask])

                results[label] = {
                    'correlation': r,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'high_tox_mean': high_mean,
                    'low_tox_mean': low_mean,
                    'difference': high_mean - low_mean,
                    'n_occurrences': int(np.sum(rates_arr > 0))
                }

        return results

    def compute_correlations(self) -> Dict[str, Dict]:
        """Compute correlations between TAMV features and toxicity."""
        results = {}

        # Get all toxicity scores
        toxicities = np.array([p.toxicity for p in self.profiles])

        # Individual features
        for feature_name, getter in [
            ('tense', lambda p: p.tense_counts),
            ('aspect', lambda p: p.aspect_counts),
            ('mood', lambda p: p.mood_counts),
            ('voice', lambda p: p.voice_counts),
        ]:
            results[feature_name] = {}

            # Get all unique values for this feature
            all_values = set()
            for p in self.profiles:
                all_values.update(getter(p).keys())

            for value in sorted(all_values):
                # Get counts for this value (normalized by word count)
                rates = np.array([
                    getter(p).get(value, 0) / p.word_count * 100
                    for p in self.profiles
                ])

                # Pearson correlation
                if np.std(rates) > 0 and np.std(toxicities) > 0:
                    r, p_value = stats.pearsonr(rates, toxicities)

                    # Also compute means for high vs low toxicity
                    high_tox_mask = toxicities > np.median(toxicities)
                    low_tox_mask = ~high_tox_mask

                    high_tox_mean = np.mean(rates[high_tox_mask])
                    low_tox_mean = np.mean(rates[low_tox_mask])

                    results[feature_name][value] = {
                        'correlation': r,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'high_tox_mean': high_tox_mean,
                        'low_tox_mean': low_tox_mean,
                        'difference': high_tox_mean - low_tox_mean
                    }

        return results

    def generate_report(self, correlations: Dict) -> str:
        """Generate text report."""
        lines = [
            "=" * 70,
            f"UTTERANCE-LEVEL TAMV-TOXICITY ANALYSIS ({self.dataset.upper()})",
            "=" * 70,
            "",
            "DATASET SUMMARY",
            "-" * 40,
            f"Dataset:              {self.dataset}",
            f"Sample rate:          1/{self.sample_rate}",
            f"Total utterances:     {len(self.profiles)}",
            f"Mean toxicity:        {np.mean([p.toxicity for p in self.profiles]):.4f}",
            f"Median toxicity:      {np.median([p.toxicity for p in self.profiles]):.4f}",
            "",
            "TAMV-TOXICITY CORRELATIONS",
            "-" * 70,
            f"{'Feature':<15} {'Value':<20} {'r':>8} {'p':>10} {'HighTox':>10} {'LowTox':>10}",
            "-" * 70,
        ]

        for feature_name in ['tense', 'aspect', 'mood', 'voice']:
            feature_data = correlations.get(feature_name, {})
            for value, stats_dict in sorted(feature_data.items(), key=lambda x: abs(x[1]['correlation']), reverse=True):
                sig = "*" if stats_dict['significant'] else ""
                lines.append(
                    f"{feature_name:<15} {value:<20} {stats_dict['correlation']:>+8.3f} "
                    f"{stats_dict['p_value']:>9.4f}{sig} {stats_dict['high_tox_mean']:>10.3f} "
                    f"{stats_dict['low_tox_mean']:>10.3f}"
                )
            lines.append("")

        # Summary of significant correlations
        sig_positive = []
        sig_negative = []
        for feature_name, feature_data in correlations.items():
            for value, stats_dict in feature_data.items():
                if stats_dict['significant']:
                    label = f"{feature_name}:{value}"
                    if stats_dict['correlation'] > 0:
                        sig_positive.append((label, stats_dict['correlation']))
                    else:
                        sig_negative.append((label, stats_dict['correlation']))

        lines.extend([
            "SIGNIFICANT CORRELATIONS SUMMARY",
            "-" * 40,
            f"\nPositive (higher rate = higher toxicity): {len(sig_positive)}"
        ])
        for label, r in sorted(sig_positive, key=lambda x: x[1], reverse=True):
            lines.append(f"  {label}: r = {r:+.3f}")

        lines.append(f"\nNegative (higher rate = lower toxicity): {len(sig_negative)}")
        for label, r in sorted(sig_negative, key=lambda x: x[1]):
            lines.append(f"  {label}: r = {r:+.3f}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def run(self, output_dir: str = 'output') -> Tuple[Dict, Dict, List[UtteranceProfile]]:
        """Run full analysis."""
        self.load_and_process()

        if not self.profiles:
            print("No utterances processed.")
            return {}, {}, []

        correlations = self.compute_correlations()
        combo_correlations = self.compute_combo_correlations()

        # Generate report
        report = self.generate_report(correlations)
        print(report)

        # Print top combo correlations
        print("\nTOP 10 TAMV COMBINATIONS BY TOXICITY CORRELATION")
        print("-" * 60)
        sorted_combos = sorted(combo_correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:10]
        for label, stats in sorted_combos:
            sig = "*" if stats['significant'] else ""
            print(f"  {label}: r = {stats['correlation']:+.3f}{sig} (n={stats['n_occurrences']})")

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, f'utterance_{self.dataset}_analysis.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        return correlations, combo_correlations, self.profiles


class UtteranceVisualizer:
    """Visualizations for utterance-level analysis."""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.high_tox_color = '#e74c3c'
        self.low_tox_color = '#3498db'

    def plot_feature_comparison(self, correlations: Dict, save_path: Optional[str] = None,
                                 show: bool = True) -> plt.Figure:
        """2x2 grid showing high vs low toxicity for each TAMV feature."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        feature_order = ['tense', 'aspect', 'mood', 'voice']

        for idx, feature_name in enumerate(feature_order):
            ax = axes[idx]
            data = correlations.get(feature_name, {})

            if not data:
                ax.set_title(f'{feature_name.title()}: No data')
                continue

            labels = list(data.keys())
            high_means = [data[l]['high_tox_mean'] for l in labels]
            low_means = [data[l]['low_tox_mean'] for l in labels]
            significant = [data[l]['significant'] for l in labels]

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, low_means, width,
                          label='Low Toxicity', color=self.low_tox_color, alpha=0.8)
            bars2 = ax.bar(x + width/2, high_means, width,
                          label='High Toxicity', color=self.high_tox_color, alpha=0.8)

            # Mark significant differences
            for i, sig in enumerate(significant):
                if sig:
                    max_val = max(high_means[i], low_means[i])
                    ax.text(i, max_val + 0.2, '*', ha='center', fontsize=14, fontweight='bold')

            ax.set_ylabel('Rate per 100 words')
            ax.set_title(f'{feature_name.title()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.legend(loc='upper right', fontsize=8)

        plt.suptitle('TAMV Features: Low vs High Toxicity Utterances (* p < 0.05)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_top_tamv_combos(self, profiles: List[UtteranceProfile], n_top: int = 10,
                              save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Bar chart showing top TAMV combinations with high vs low toxicity comparison."""
        from collections import Counter

        # Get toxicity threshold (median)
        toxicities = [p.toxicity for p in profiles]
        median_tox = np.median(toxicities)

        # Count all TAMV combos
        all_counts = Counter()
        high_tox_rates = {}  # label -> list of rates
        low_tox_rates = {}

        for p in profiles:
            is_high_tox = p.toxicity > median_tox
            for label, count in p.tamv_counts.items():
                all_counts[label] += count
                rate = count / p.word_count * 100

                if label not in high_tox_rates:
                    high_tox_rates[label] = []
                    low_tox_rates[label] = []

                if is_high_tox:
                    high_tox_rates[label].append(rate)
                else:
                    low_tox_rates[label].append(rate)

        # Get top N by frequency
        top_labels = [label for label, _ in all_counts.most_common(n_top)]

        # Compute means
        high_means = []
        low_means = []
        for label in top_labels:
            high_means.append(np.mean(high_tox_rates[label]) if high_tox_rates[label] else 0)
            low_means.append(np.mean(low_tox_rates[label]) if low_tox_rates[label] else 0)

        # Abbreviate labels
        def abbrev(label):
            abbr = {
                'present': 'pres', 'past': 'past', 'future': 'fut', 'none': 'none',
                'simple': 'simp', 'progressive': 'prog', 'perfect': 'perf',
                'perfect-progressive': 'p-prog', 'indicative': 'ind',
                'subjunctive': 'subj', 'imperative': 'imp', 'conditional': 'cond',
                'modal': 'mod', 'infinitival': 'inf', 'participle': 'part',
                'active': 'act', 'passive': 'pass'
            }
            return '-'.join(abbr.get(p, p) for p in label.split('-'))

        labels = [abbrev(l) for l in top_labels]

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, low_means, width,
                      label='Low Toxicity', color=self.low_tox_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, high_means, width,
                      label='High Toxicity', color=self.high_tox_color, alpha=0.8)

        ax.set_ylabel('Rate per 100 words', fontsize=11)
        ax.set_xlabel('TAMV Combination', fontsize=11)
        ax.set_title(f'Top {n_top} Most Common TAMV Combinations:\nLow vs High Toxicity', fontsize=12)
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

    def plot_top_correlated_combos(self, combo_correlations: Dict, n_top: int = 10,
                                    save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Bar chart showing TAMV combinations with highest toxicity correlation."""
        # Sort by absolute correlation
        sorted_combos = sorted(
            combo_correlations.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )[:n_top]

        if not sorted_combos:
            print("No combo correlations to plot")
            return None

        # Abbreviate labels
        def abbrev(label):
            abbr = {
                'present': 'pres', 'past': 'past', 'future': 'fut', 'none': 'none',
                'simple': 'simp', 'progressive': 'prog', 'perfect': 'perf',
                'perfect-progressive': 'p-prog', 'indicative': 'ind',
                'subjunctive': 'subj', 'imperative': 'imp', 'conditional': 'cond',
                'modal': 'mod', 'infinitival': 'inf', 'participle': 'part',
                'active': 'act', 'passive': 'pass'
            }
            return '-'.join(abbr.get(p, p) for p in label.split('-'))

        labels = [abbrev(l) for l, _ in sorted_combos]
        correlations = [s['correlation'] for _, s in sorted_combos]
        significant = [s['significant'] for _, s in sorted_combos]
        colors = [self.high_tox_color if r > 0 else self.low_tox_color for r in correlations]

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, correlations, color=colors, alpha=0.8)

        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars, significant)):
            if sig:
                x_pos = correlations[i]
                offset = 0.005 if x_pos > 0 else -0.005
                ax.text(x_pos + offset, i, '*', ha='left' if x_pos > 0 else 'right',
                       va='center', fontsize=14, fontweight='bold')

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Correlation with Toxicity (r)', fontsize=11)
        ax.set_title(f'Top {n_top} TAMV Combinations by Toxicity Correlation\n(* p < 0.05)', fontsize=12)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.high_tox_color, label='Positive (more toxic)'),
            Patch(facecolor=self.low_tox_color, label='Negative (less toxic)')
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

    def plot_correlations(self, correlations: Dict, save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
        """Bar chart of correlation coefficients."""
        # Collect all correlations
        all_corrs = []
        for feature_name, feature_data in correlations.items():
            for value, stats_dict in feature_data.items():
                all_corrs.append({
                    'label': f"{feature_name}:{value}",
                    'r': stats_dict['correlation'],
                    'significant': stats_dict['significant']
                })

        # Sort by absolute correlation
        all_corrs.sort(key=lambda x: abs(x['r']), reverse=True)

        labels = [c['label'] for c in all_corrs]
        values = [c['r'] for c in all_corrs]
        colors = ['#e74c3c' if c['r'] > 0 else '#3498db' for c in all_corrs]
        alphas = [0.9 if c['significant'] else 0.4 for c in all_corrs]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)

        # Dim non-significant bars
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Correlation with Toxicity (r)', fontsize=11)
        ax.set_title('TAMV Features Correlated with Toxicity\n(solid = significant, faded = not significant)', fontsize=12)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_tamv_combo_distribution(self, profiles: List[UtteranceProfile], n_top: int = 30,
                                       save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot mean and standard deviation of occurrence rates for TAMV combinations.
        Shows error bars representing the SD of rates across utterances.
        """
        from collections import Counter

        # Collect rates for each TAMV combo
        combo_rates: Dict[str, List[float]] = {}
        combo_counts = Counter()

        for p in profiles:
            for label, count in p.tamv_counts.items():
                combo_counts[label] += count
                rate = count / p.word_count * 100
                if label not in combo_rates:
                    combo_rates[label] = []
                combo_rates[label].append(rate)

        # Get top N by total occurrence
        top_labels = [label for label, _ in combo_counts.most_common(n_top)]

        # Compute mean and SD for each
        means = []
        stds = []
        for label in top_labels:
            rates = combo_rates[label]
            # Include zeros for utterances that don't have this combo
            all_rates = rates + [0.0] * (len(profiles) - len(rates))
            means.append(np.mean(all_rates))
            stds.append(np.std(all_rates))

        # Abbreviate labels
        def abbrev(label):
            abbr = {
                'present': 'pres', 'past': 'past', 'future': 'fut', 'none': 'none',
                'simple': 'simp', 'progressive': 'prog', 'perfect': 'perf',
                'perfect-progressive': 'p-prog', 'indicative': 'ind',
                'subjunctive': 'subj', 'imperative': 'imp', 'conditional': 'cond',
                'modal': 'mod', 'infinitival': 'inf', 'participle': 'part',
                'active': 'act', 'passive': 'pass'
            }
            return '-'.join(abbr.get(p, p) for p in label.split('-'))

        labels = [abbrev(l) for l in top_labels]

        fig, ax = plt.subplots(figsize=(14, 8))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, means, xerr=stds, capsize=3,
                       color='#2ecc71', alpha=0.8, ecolor='#27ae60')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Rate per 100 words (mean ± SD)', fontsize=11)
        ax.set_title(f'Top {n_top} TAMV Combinations: Mean Rate with Standard Deviation', fontsize=12)
        ax.invert_yaxis()  # Most common at top

        # Add count annotations
        for i, (label, mean, std) in enumerate(zip(top_labels, means, stds)):
            count = combo_counts[label]
            ax.annotate(f'n={count}', xy=(mean + std + 0.1, i),
                       va='center', fontsize=8, color='#666')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_scatter(self, profiles: List[UtteranceProfile], feature: str, value: str,
                     save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """Scatter plot of a specific feature vs toxicity."""
        getter_map = {
            'tense': lambda p: p.tense_counts,
            'aspect': lambda p: p.aspect_counts,
            'mood': lambda p: p.mood_counts,
            'voice': lambda p: p.voice_counts,
        }
        getter = getter_map.get(feature)

        rates = [getter(p).get(value, 0) / p.word_count * 100 for p in profiles]
        toxicities = [p.toxicity for p in profiles]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(rates, toxicities, alpha=0.3, s=10)
        ax.set_xlabel(f'{feature}:{value} rate (per 100 words)', fontsize=11)
        ax.set_ylabel('Toxicity Score', fontsize=11)
        ax.set_title(f'{feature}:{value} vs Toxicity', fontsize=12)

        # Add trend line
        z = np.polyfit(rates, toxicities, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(rates), max(rates), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.8, linewidth=2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Correlate TAMV features with toxicity at utterance level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset', '-d', choices=['wiki', 'cmv'], default='wiki',
                        help='Dataset: wiki (Wikipedia) or cmv (Reddit)')
    parser.add_argument('--sample-rate', '-s', type=int, default=1,
                        help='Take 1 out of every N conversations (default: 1 = all)')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display interactive plots')

    args = parser.parse_args()

    analyzer = UtteranceAnalyzer(dataset=args.dataset, sample_rate=args.sample_rate)
    correlations, combo_correlations, profiles = analyzer.run(output_dir=args.output)

    if correlations:
        visualizer = UtteranceVisualizer(output_dir=args.output)
        visualizer.plot_correlations(
            correlations,
            save_path=os.path.join(args.output, f'utterance_{args.dataset}_correlations.png'),
            show=not args.no_show
        )
        visualizer.plot_feature_comparison(
            correlations,
            save_path=os.path.join(args.output, f'utterance_{args.dataset}_features.png'),
            show=not args.no_show
        )
        visualizer.plot_top_tamv_combos(
            profiles,
            save_path=os.path.join(args.output, f'utterance_{args.dataset}_top_combos.png'),
            show=not args.no_show
        )
        visualizer.plot_top_correlated_combos(
            combo_correlations,
            save_path=os.path.join(args.output, f'utterance_{args.dataset}_combo_correlations.png'),
            show=not args.no_show
        )
        visualizer.plot_tamv_combo_distribution(
            profiles,
            save_path=os.path.join(args.output, f'utterance_{args.dataset}_combo_sd.png'),
            show=not args.no_show
        )


if __name__ == '__main__':
    main()
