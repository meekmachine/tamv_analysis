#!/usr/bin/env python3
"""
Phrase-level TAMV validation.

This module validates TAMV extraction at the phrase level, handling:
- Multi-verb sentences with multiple expected TAMV labels
- Span matching (full verbal complex vs. main verb)
- Filtering of non-finite verbs when expected output only includes finite verbs

Usage:
    python -m tests.validate_phrases                    # Run with default data
    python -m tests.validate_phrases --verbose          # Show all test details
    python -m tests.validate_phrases --no-show          # Don't show interactive plots
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.tamv_extractor import TAMVExtractor, TAMVLabel, Tense, Aspect, Mood, Voice


def build_confusion_matrix(matches: List['PhraseMatch'],
                           get_expected, get_extracted,
                           enum_class: type) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix for a single dimension.

    Args:
        matches: List of PhraseMatch objects
        get_expected: Function to extract expected value from ExpectedPhrase
        get_extracted: Function to extract actual value from TAMVLabel
        enum_class: The enum class (Tense, Aspect, Mood, Voice)

    Returns:
        Nested dict: {expected_label: {predicted_label: count}}
    """
    # Get all possible labels from enum
    all_labels = [e.value for e in enum_class]

    # Initialize matrix with zeros
    matrix = {exp: {pred: 0 for pred in all_labels} for exp in all_labels}

    for m in matches:
        if not m.matched or not m.extracted:
            continue
        expected = get_expected(m.expected).value
        extracted = get_extracted(m.extracted).value
        if expected in matrix and extracted in matrix[expected]:
            matrix[expected][extracted] += 1

    return matrix


def compute_metrics(matrix: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1, and support for each label.

    Args:
        matrix: Confusion matrix from build_confusion_matrix

    Returns:
        Dict mapping label -> {precision, recall, f1, support}
    """
    # Get all labels that appear in the data
    active_labels = set()
    for exp, preds in matrix.items():
        for pred, count in preds.items():
            if count > 0:
                active_labels.add(exp)
                active_labels.add(pred)

    metrics = {}
    for label in active_labels:
        # True positives: predicted label correctly
        tp = matrix.get(label, {}).get(label, 0)

        # False positives: predicted this label but was actually something else
        fp = sum(matrix.get(other, {}).get(label, 0)
                 for other in active_labels if other != label)

        # False negatives: was this label but predicted something else
        fn = sum(matrix.get(label, {}).get(other, 0)
                 for other in active_labels if other != label)

        # Support: total actual instances of this label
        support = tp + fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    return metrics


def format_confusion_matrix(matrix: Dict[str, Dict[str, int]],
                           title: str,
                           min_width: int = 8) -> str:
    """
    Format a confusion matrix as a readable string with metrics.

    Args:
        matrix: Nested dict from build_confusion_matrix
        title: Title for the matrix
        min_width: Minimum column width

    Returns:
        Formatted string representation
    """
    # Filter to only show labels that appear in the data
    active_labels = set()
    for exp, preds in matrix.items():
        for pred, count in preds.items():
            if count > 0:
                active_labels.add(exp)
                active_labels.add(pred)

    if not active_labels:
        return f"\n{title}\n{'-' * 40}\nNo data to display\n"

    # Sort labels for consistent ordering
    labels = sorted(active_labels)

    # Calculate column widths
    col_width = max(min_width, max(len(l) for l in labels) + 2)

    lines = [
        f"\n{title}",
        "-" * 60,
        "(rows = expected, columns = predicted)",
        ""
    ]

    # Header row
    header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    lines.append(header)

    # Data rows
    for exp in labels:
        row_data = "".join(f"{matrix.get(exp, {}).get(pred, 0):>{col_width}}" for pred in labels)
        lines.append(f"{exp:<{col_width}}{row_data}")

    # Add metrics table
    metrics = compute_metrics(matrix)
    lines.append("")
    lines.append(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)

    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in labels:
        m = metrics.get(label, {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0})
        lines.append(f"{label:<20} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%} {m['support']:>10}")
        total_support += m['support']
        weighted_precision += m['precision'] * m['support']
        weighted_recall += m['recall'] * m['support']
        weighted_f1 += m['f1'] * m['support']

    # Weighted averages
    if total_support > 0:
        lines.append("-" * 60)
        lines.append(f"{'Weighted Avg':<20} {weighted_precision/total_support:>10.2%} {weighted_recall/total_support:>10.2%} {weighted_f1/total_support:>10.2%} {total_support:>10}")

    lines.append("")
    return "\n".join(lines)


def plot_confusion_matrix(matrix: Dict[str, Dict[str, int]],
                          title: str,
                          output_path: str = None) -> plt.Figure:
    """
    Create a matplotlib heatmap visualization of a confusion matrix.

    Args:
        matrix: Nested dict from build_confusion_matrix
        title: Title for the plot
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Filter to only show labels that appear in the data
    active_labels = set()
    for exp, preds in matrix.items():
        for pred, count in preds.items():
            if count > 0:
                active_labels.add(exp)
                active_labels.add(pred)

    if not active_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
        ax.set_title(title)
        return fig

    # Sort labels for consistent ordering
    labels = sorted(active_labels)

    # Build numpy array
    n = len(labels)
    data = np.zeros((n, n), dtype=int)
    for i, exp in enumerate(labels):
        for j, pred in enumerate(labels):
            data[i, j] = matrix.get(exp, {}).get(pred, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    # Create heatmap
    im = ax.imshow(data, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add text annotations
    thresh = data.max() / 2.
    for i in range(n):
        for j in range(n):
            color = 'white' if data[i, j] > thresh else 'black'
            ax.text(j, i, data[i, j], ha='center', va='center', color=color, fontsize=10)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Expected')
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_confusion_matrices(matrices: Dict[str, Dict[str, Dict[str, int]]],
                                 output_dir: str,
                                 prefix: str = 'phrase_',
                                 show: bool = True) -> None:
    """
    Create a combined figure with all four confusion matrices.

    Args:
        matrices: Dict with 'tense', 'aspect', 'mood', 'voice' confusion matrices
        output_dir: Directory to save the figures
        prefix: Prefix for output filenames
        show: If True, display the plot interactively
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create individual plots and save them
    for name, matrix in matrices.items():
        fig = plot_confusion_matrix(matrix, f'{name.upper()} Confusion Matrix')
        fig.savefig(os.path.join(output_dir, f'{prefix}confusion_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, matrix) in enumerate(matrices.items()):
        ax = axes[idx]

        # Filter active labels
        active_labels = set()
        for exp, preds in matrix.items():
            for pred, count in preds.items():
                if count > 0:
                    active_labels.add(exp)
                    active_labels.add(pred)

        if not active_labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{name.upper()}')
            continue

        labels = sorted(active_labels)
        n = len(labels)
        data = np.zeros((n, n), dtype=int)
        for i, exp in enumerate(labels):
            for j, pred in enumerate(labels):
                data[i, j] = matrix.get(exp, {}).get(pred, 0)

        im = ax.imshow(data, cmap='Blues')

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Add text annotations
        thresh = data.max() / 2. if data.max() > 0 else 0
        for i in range(n):
            for j in range(n):
                color = 'white' if data[i, j] > thresh else 'black'
                ax.text(j, i, data[i, j], ha='center', va='center', color=color, fontsize=8)

        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Expected', fontsize=9)
        ax.set_title(f'{name.upper()}', fontsize=11, fontweight='bold')

    plt.suptitle('Phrase-Level TAMV Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}confusion_matrices_combined.png'), dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


@dataclass
class ExpectedPhrase:
    """Expected TAMV label for a verb phrase."""
    span: str  # e.g., "has been writing"
    tense: Tense
    aspect: Aspect
    mood: Mood
    voice: Voice

    @classmethod
    def from_label_string(cls, span: str, label: str) -> 'ExpectedPhrase':
        """Parse a label string like 'present-perfect-indicative-active'.

        Also handles compound aspects like 'present-perfect-progressive-indicative-active'.
        """
        parts = label.lower().split('-')

        # Standard format: tense-aspect-mood-voice (4 parts)
        # Extended format: tense-aspect1-aspect2-mood-voice (5 parts for perfect-progressive)
        if len(parts) == 5 and parts[1] == 'perfect' and parts[2] == 'progressive':
            # Compound aspect: perfect-progressive
            tense_str = parts[0].upper()
            aspect_str = 'PERFECT_PROGRESSIVE'
            mood_str = parts[3].upper()
            voice_str = parts[4].upper()
        elif len(parts) == 4:
            tense_str = parts[0].upper()
            aspect_str = parts[1].upper()
            mood_str = parts[2].upper()
            voice_str = parts[3].upper()
        else:
            raise ValueError(f"Invalid label format: {label}")

        return cls(
            span=span,
            tense=Tense[tense_str],
            aspect=Aspect[aspect_str],
            mood=Mood[mood_str],
            voice=Voice[voice_str]
        )


@dataclass
class PhraseTestCase:
    """Test case with one or more expected verb phrases."""
    id: int
    text: str
    expected: List[ExpectedPhrase]
    notes: str = ""


@dataclass
class PhraseMatch:
    """Result of matching an extracted label to an expected phrase."""
    expected: ExpectedPhrase
    extracted: Optional[TAMVLabel]
    matched: bool
    tense_match: bool = False
    aspect_match: bool = False
    mood_match: bool = False
    voice_match: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class PhraseTestResult:
    """Result of validating a single test case."""
    test_case: PhraseTestCase
    matches: List[PhraseMatch]

    @property
    def all_matched(self) -> bool:
        return all(m.matched for m in self.matches)

    @property
    def all_correct(self) -> bool:
        return all(
            m.tense_match and m.aspect_match and m.mood_match and m.voice_match
            for m in self.matches if m.extracted
        )


class PhraseValidator:
    """Validates TAMV extraction at the phrase level."""

    def __init__(self):
        self.extractor = TAMVExtractor()
        self.results: List[PhraseTestResult] = []

    def load_test_cases(self, json_path: str) -> List[PhraseTestCase]:
        """Load test cases from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            expected = [
                ExpectedPhrase.from_label_string(t['span'], t['label'])
                for t in item['tamv']
            ]
            test_cases.append(PhraseTestCase(
                id=item['id'],
                text=item['text'],
                expected=expected,
                notes=item.get('notes', '')
            ))

        return test_cases

    def _normalize_verb(self, text: str) -> str:
        """Normalize verb text for matching."""
        return text.lower().strip()

    def _get_verbal_complex_text(self, label: TAMVLabel) -> str:
        """Get the full verbal complex text from a TAMVLabel."""
        parts = []
        if label.auxiliary_chain:
            parts.extend(label.auxiliary_chain)
        parts.append(label.verb_text)
        return ' '.join(parts)

    def _match_span(self, expected_span: str, label: TAMVLabel) -> bool:
        """Check if an extracted label matches the expected span."""
        expected_norm = self._normalize_verb(expected_span)

        # Try matching full verbal complex
        full_complex = self._get_verbal_complex_text(label)
        if self._normalize_verb(full_complex) == expected_norm:
            return True

        # Try matching just main verb text
        if self._normalize_verb(label.verb_text) == expected_norm:
            return True

        # Try matching main verb to last word of expected span
        expected_parts = expected_span.lower().split()
        if expected_parts and self._normalize_verb(label.verb_text) == expected_parts[-1]:
            return True

        return False

    def _compare_labels(self, expected: ExpectedPhrase, extracted: TAMVLabel) -> PhraseMatch:
        """Compare expected and extracted TAMV labels."""
        errors = []

        tense_match = expected.tense == extracted.tense
        if not tense_match:
            errors.append(f"Tense: expected {expected.tense.value}, got {extracted.tense.value}")

        aspect_match = expected.aspect == extracted.aspect
        if not aspect_match:
            errors.append(f"Aspect: expected {expected.aspect.value}, got {extracted.aspect.value}")

        mood_match = expected.mood == extracted.mood
        if not mood_match:
            errors.append(f"Mood: expected {expected.mood.value}, got {extracted.mood.value}")

        voice_match = expected.voice == extracted.voice
        if not voice_match:
            errors.append(f"Voice: expected {expected.voice.value}, got {extracted.voice.value}")

        return PhraseMatch(
            expected=expected,
            extracted=extracted,
            matched=True,
            tense_match=tense_match,
            aspect_match=aspect_match,
            mood_match=mood_match,
            voice_match=voice_match,
            errors=errors
        )

    def validate_case(self, test_case: PhraseTestCase) -> PhraseTestResult:
        """Validate a single test case."""
        # Extract all labels from the sentence
        extracted = self.extractor.extract_from_text(test_case.text)

        # Track which extracted labels have been matched
        used_extracted: Set[int] = set()
        matches: List[PhraseMatch] = []

        # For each expected phrase, find the best matching extracted label
        for expected in test_case.expected:
            best_match = None
            best_idx = None

            for idx, label in enumerate(extracted):
                if idx in used_extracted:
                    continue

                if self._match_span(expected.span, label):
                    comparison = self._compare_labels(expected, label)
                    if best_match is None or len(comparison.errors) < len(best_match.errors):
                        best_match = comparison
                        best_idx = idx

            if best_match:
                used_extracted.add(best_idx)
                matches.append(best_match)
            else:
                # No match found for this expected phrase
                matches.append(PhraseMatch(
                    expected=expected,
                    extracted=None,
                    matched=False,
                    errors=[f"Verb '{expected.span}' not found in extraction"]
                ))

        return PhraseTestResult(
            test_case=test_case,
            matches=matches
        )

    def run_validation(self, test_cases: List[PhraseTestCase]) -> None:
        """Run validation on all test cases."""
        self.results = [self.validate_case(tc) for tc in test_cases]

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_phrases = sum(len(r.matches) for r in self.results)
        matched_phrases = sum(
            sum(1 for m in r.matches if m.matched)
            for r in self.results
        )
        correct_labels = sum(
            sum(1 for m in r.matches if m.matched and not m.errors)
            for r in self.results
        )

        # Per-dimension accuracy (only for matched phrases)
        tense_correct = sum(
            sum(1 for m in r.matches if m.matched and m.tense_match)
            for r in self.results
        )
        aspect_correct = sum(
            sum(1 for m in r.matches if m.matched and m.aspect_match)
            for r in self.results
        )
        mood_correct = sum(
            sum(1 for m in r.matches if m.matched and m.mood_match)
            for r in self.results
        )
        voice_correct = sum(
            sum(1 for m in r.matches if m.matched and m.voice_match)
            for r in self.results
        )

        return {
            'total_cases': len(self.results),
            'total_phrases': total_phrases,
            'matched_phrases': matched_phrases,
            'correct_labels': correct_labels,
            'phrase_detection_rate': matched_phrases / total_phrases if total_phrases > 0 else 0,
            'label_accuracy': correct_labels / matched_phrases if matched_phrases > 0 else 0,
            'tense_accuracy': tense_correct / matched_phrases if matched_phrases > 0 else 0,
            'aspect_accuracy': aspect_correct / matched_phrases if matched_phrases > 0 else 0,
            'mood_accuracy': mood_correct / matched_phrases if matched_phrases > 0 else 0,
            'voice_accuracy': voice_correct / matched_phrases if matched_phrases > 0 else 0,
        }

    def get_confusion_matrices(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Generate confusion matrices for all four TAMV dimensions."""
        # Flatten all matches from all results
        all_matches = [m for r in self.results for m in r.matches]

        return {
            'tense': build_confusion_matrix(
                all_matches,
                lambda exp: exp.tense,
                lambda ext: ext.tense,
                Tense
            ),
            'aspect': build_confusion_matrix(
                all_matches,
                lambda exp: exp.aspect,
                lambda ext: ext.aspect,
                Aspect
            ),
            'mood': build_confusion_matrix(
                all_matches,
                lambda exp: exp.mood,
                lambda ext: ext.mood,
                Mood
            ),
            'voice': build_confusion_matrix(
                all_matches,
                lambda exp: exp.voice,
                lambda ext: ext.voice,
                Voice
            ),
        }

    def generate_report(self, verbose: bool = False) -> str:
        """Generate validation report."""
        summary = self.get_summary()

        lines = [
            "=" * 70,
            "PHRASE-LEVEL TAMV VALIDATION REPORT",
            "=" * 70,
            "",
            "OVERALL RESULTS",
            "-" * 40,
            f"Total test cases:     {summary['total_cases']}",
            f"Total verb phrases:   {summary['total_phrases']}",
            f"Phrases detected:     {summary['matched_phrases']} ({summary['phrase_detection_rate']:.1%})",
            f"Labels fully correct: {summary['correct_labels']} ({summary['label_accuracy']:.1%})",
            "",
            "PER-DIMENSION ACCURACY (where phrase was detected)",
            "-" * 40,
            f"Tense:   {summary['tense_accuracy']:.1%}",
            f"Aspect:  {summary['aspect_accuracy']:.1%}",
            f"Mood:    {summary['mood_accuracy']:.1%}",
            f"Voice:   {summary['voice_accuracy']:.1%}",
            "",
        ]

        # Confusion matrices
        matrices = self.get_confusion_matrices()
        lines.append(format_confusion_matrix(matrices['tense'], "TENSE CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['aspect'], "ASPECT CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['mood'], "MOOD CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['voice'], "VOICE CONFUSION MATRIX"))

        # Failed cases
        failed_results = [r for r in self.results if not r.all_correct or not r.all_matched]

        if failed_results:
            lines.extend([
                "FAILED TEST CASES",
                "-" * 40,
                ""
            ])

            for result in failed_results:
                lines.append(f"  ID {result.test_case.id}: \"{result.test_case.text}\"")

                for match in result.matches:
                    if not match.matched:
                        lines.append(f"    - '{match.expected.span}' NOT FOUND")
                    elif match.errors:
                        expected_label = f"{match.expected.tense.value}-{match.expected.aspect.value}-{match.expected.mood.value}-{match.expected.voice.value}"
                        extracted = match.extracted
                        got_label = f"{extracted.tense.value}-{extracted.aspect.value}-{extracted.mood.value}-{extracted.voice.value}"
                        lines.append(f"    - '{match.expected.span}':")
                        lines.append(f"        Expected: {expected_label}")
                        lines.append(f"        Got:      {got_label}")
                        for err in match.errors:
                            lines.append(f"        * {err}")

                if result.test_case.notes:
                    lines.append(f"    Note: {result.test_case.notes}")

                lines.append("")

        if verbose:
            lines.extend([
                "ALL TEST CASES",
                "-" * 40,
                ""
            ])

            for result in self.results:
                status = "PASS" if result.all_correct and result.all_matched else "FAIL"
                lines.append(f"  [{status}] ID {result.test_case.id}: \"{result.test_case.text}\"")

                for match in result.matches:
                    if match.matched and not match.errors:
                        lines.append(f"    - '{match.expected.span}' OK")
                    elif match.matched:
                        lines.append(f"    - '{match.expected.span}' MISMATCH: {match.errors}")
                    else:
                        lines.append(f"    - '{match.expected.span}' NOT FOUND")

                lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def get_default_data_dir() -> Path:
    """Get the default data directory."""
    return Path(__file__).parent.parent / 'data'


def main():
    parser = argparse.ArgumentParser(
        description="Phrase-level TAMV validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This validator tests TAMV extraction on multi-verb sentences,
checking that each verb phrase is correctly detected and labeled.

Examples:
  %(prog)s                        # Run with default data
  %(prog)s --verbose              # Show all test details
  %(prog)s --data FILE.json       # Use custom test data
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output for all tests')
    parser.add_argument('--data', type=str,
                        help='Path to JSON test data file')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory for report')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display interactive plots (just save to files)')

    args = parser.parse_args()

    print("=" * 70)
    print("PHRASE-LEVEL TAMV VALIDATION")
    print("=" * 70)

    # Load test data
    if args.data:
        data_path = args.data
    else:
        data_path = get_default_data_dir() / 'phrase_test_expected.json'

    print(f"Loading test data from: {data_path}")

    validator = PhraseValidator()
    test_cases = validator.load_test_cases(str(data_path))

    print(f"Loaded {len(test_cases)} test cases with {sum(len(tc.expected) for tc in test_cases)} verb phrases")

    validator.run_validation(test_cases)

    report = validator.generate_report(verbose=args.verbose)
    print(report)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'phrase_validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Generate confusion matrix plots
    matrices = validator.get_confusion_matrices()
    plot_all_confusion_matrices(matrices, args.output, prefix='phrase_', show=not args.no_show)
    print(f"Confusion matrix plots saved to: {args.output}/")

    # Exit with status based on accuracy
    summary = validator.get_summary()
    if summary['label_accuracy'] < 0.80:
        print(f"\nWARNING: Accuracy ({summary['label_accuracy']:.1%}) is below 80% threshold")
        sys.exit(1)
    else:
        print(f"\nValidation completed with {summary['label_accuracy']:.1%} accuracy")
        sys.exit(0)


if __name__ == '__main__':
    main()
