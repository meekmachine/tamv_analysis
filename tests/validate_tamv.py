#!/usr/bin/env python3
"""
TAMV Extraction Validation Script

Validates the TAMV extractor against linguistically-grounded test cases.
Test cases are derived from:
- Standard English grammar references (Biber et al. 1999, Palmer 2001)
- TMV-annotator documentation (Ramm et al. 2017)
- Universal Dependencies morphological guidelines

Usage:
    python -m tests.validate_tamv                  # Run all tests
    python -m tests.validate_tamv --verbose        # Show detailed output
    python -m tests.validate_tamv --no-show        # Don't show interactive plots

Output:
    - Console summary of pass/fail rates
    - Detailed error report for failed cases
    - output/validation_report.txt (full report)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from src.tamv_extractor import TAMVExtractor, Tense, Aspect, Mood, Voice


def build_confusion_matrix(results: List['ValidationResult'],
                           get_expected, get_extracted,
                           enum_class: type) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix for a single dimension.

    Args:
        results: List of ValidationResult objects
        get_expected: Function to extract expected value from test case
        get_extracted: Function to extract actual value from result
        enum_class: The enum class (Tense, Aspect, Mood, Voice)

    Returns:
        Nested dict: {expected_label: {predicted_label: count}}
    """
    # Get all possible labels from enum
    all_labels = [e.value for e in enum_class]

    # Initialize matrix with zeros
    matrix = {exp: {pred: 0 for pred in all_labels} for exp in all_labels}

    for r in results:
        if not r.verb_found:
            continue
        expected = get_expected(r.test_case).value
        extracted = get_extracted(r).value if get_extracted(r) else "NONE"
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
                                 show: bool = True) -> None:
    """
    Create a combined figure with all four confusion matrices.

    Args:
        matrices: Dict with 'tense', 'aspect', 'mood', 'voice' confusion matrices
        output_dir: Directory to save the figures
        show: If True, display the plot interactively
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create individual plots and save them
    for name, matrix in matrices.items():
        fig = plot_confusion_matrix(matrix, f'{name.upper()} Confusion Matrix')
        fig.savefig(os.path.join(output_dir, f'confusion_{name}.png'), dpi=150, bbox_inches='tight')
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

    plt.suptitle('TAMV Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrices_combined.png'), dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


@dataclass
class TestCase:
    """A single TAMV validation test case."""
    sentence: str
    verb: str  # The main verb to check
    expected_tense: Tense
    expected_aspect: Aspect
    expected_mood: Mood
    expected_voice: Voice
    category: str  # For grouping (e.g., 'tense', 'aspect', 'mood', 'voice', 'complex')
    source: str = ""  # Reference source
    notes: str = ""  # Additional notes


def load_test_cases_from_files(sentences_file: str, expected_file: str) -> List[TestCase]:
    """
    Load test cases from separate sentence and expected label files.

    Args:
        sentences_file: Path to file with one sentence per line
        expected_file: Path to TSV file with expected labels
                      Format: index \t verb \t tense \t aspect \t mood \t voice \t category \t source

    Returns:
        List of TestCase objects
    """
    # Load sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Load expected labels
    test_cases = []
    with open(expected_file, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 7:
                continue

            idx = int(parts[0]) - 1  # Convert 1-indexed to 0-indexed
            verb = parts[1]
            tense_str = parts[2].upper()
            aspect_str = parts[3].upper()
            mood_str = parts[4].upper()
            voice_str = parts[5].upper()
            category = parts[6]
            source = parts[7] if len(parts) > 7 else ""

            # Map string values to enums
            tense = Tense[tense_str] if tense_str in Tense.__members__ else Tense.NONE
            aspect = Aspect[aspect_str] if aspect_str in Aspect.__members__ else Aspect.SIMPLE
            mood = Mood[mood_str] if mood_str in Mood.__members__ else Mood.INDICATIVE
            voice = Voice[voice_str] if voice_str in Voice.__members__ else Voice.ACTIVE

            if idx < len(sentences):
                test_cases.append(TestCase(
                    sentence=sentences[idx],
                    verb=verb,
                    expected_tense=tense,
                    expected_aspect=aspect,
                    expected_mood=mood,
                    expected_voice=voice,
                    category=category,
                    source=source
                ))

    return test_cases


def get_default_data_dir() -> Path:
    """Get the default data directory relative to this file."""
    return Path(__file__).parent.parent / 'data'


# =============================================================================
# TEST CASES - Loaded from data files
# =============================================================================
# Test cases are stored in data/synthetic_sentences.txt and data/synthetic_expected.tsv
# This ensures separation between test inputs and expected outputs (no cheating)


def load_default_test_cases() -> List[TestCase]:
    """Load test cases from the default data directory."""
    data_dir = get_default_data_dir()
    sentences_file = data_dir / 'synthetic_sentences.txt'
    expected_file = data_dir / 'synthetic_expected.tsv'

    if not sentences_file.exists() or not expected_file.exists():
        raise FileNotFoundError(
            f"Test data files not found. Expected:\n"
            f"  {sentences_file}\n"
            f"  {expected_file}\n"
            "Run the data extraction scripts first."
        )

    return load_test_cases_from_files(str(sentences_file), str(expected_file))


@dataclass
class ValidationResult:
    """Result of validating a single test case."""
    test_case: TestCase
    passed: bool
    extracted_tense: Optional[Tense]
    extracted_aspect: Optional[Aspect]
    extracted_mood: Optional[Mood]
    extracted_voice: Optional[Voice]
    errors: List[str]
    verb_found: bool


class TAMVValidator:
    """Validates TAMV extraction against ground truth test cases."""

    def __init__(self):
        self.extractor = TAMVExtractor()
        self.results: List[ValidationResult] = []

    def validate_test_case(self, test_case: TestCase) -> ValidationResult:
        """Validate a single test case."""
        labels = self.extractor.extract_from_text(test_case.sentence)

        # Find the label for the target verb
        target_label = None
        verb_lower = test_case.verb.lower()

        for label in labels:
            # Check if this label matches the target verb
            if (verb_lower in label.verb_text.lower() or
                verb_lower in label.verb_lemma.lower() or
                any(verb_lower in aux.lower() for aux in label.auxiliary_chain)):
                target_label = label
                break

        errors = []
        verb_found = target_label is not None

        if not verb_found:
            errors.append(f"Verb '{test_case.verb}' not found in extraction")
            return ValidationResult(
                test_case=test_case,
                passed=False,
                extracted_tense=None,
                extracted_aspect=None,
                extracted_mood=None,
                extracted_voice=None,
                errors=errors,
                verb_found=False
            )

        # Check each dimension
        if target_label.tense != test_case.expected_tense:
            errors.append(
                f"Tense: expected {test_case.expected_tense.value}, "
                f"got {target_label.tense.value}"
            )

        if target_label.aspect != test_case.expected_aspect:
            errors.append(
                f"Aspect: expected {test_case.expected_aspect.value}, "
                f"got {target_label.aspect.value}"
            )

        if target_label.mood != test_case.expected_mood:
            errors.append(
                f"Mood: expected {test_case.expected_mood.value}, "
                f"got {target_label.mood.value}"
            )

        if target_label.voice != test_case.expected_voice:
            errors.append(
                f"Voice: expected {test_case.expected_voice.value}, "
                f"got {target_label.voice.value}"
            )

        return ValidationResult(
            test_case=test_case,
            passed=len(errors) == 0,
            extracted_tense=target_label.tense,
            extracted_aspect=target_label.aspect,
            extracted_mood=target_label.mood,
            extracted_voice=target_label.voice,
            errors=errors,
            verb_found=True
        )

    def run_validation(self, test_cases: List[TestCase] = None,
                       category: str = None) -> List[ValidationResult]:
        """Run validation on test cases."""
        if test_cases is None:
            test_cases = TEST_CASES

        if category:
            test_cases = [tc for tc in test_cases if category in tc.category]

        self.results = []
        for tc in test_cases:
            result = self.validate_test_case(tc)
            self.results.append(result)

        return self.results

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        verb_found = sum(1 for r in self.results if r.verb_found)

        # Per-dimension accuracy (only for cases where verb was found)
        found_results = [r for r in self.results if r.verb_found]
        tense_correct = sum(1 for r in found_results
                           if r.extracted_tense == r.test_case.expected_tense)
        aspect_correct = sum(1 for r in found_results
                            if r.extracted_aspect == r.test_case.expected_aspect)
        mood_correct = sum(1 for r in found_results
                          if r.extracted_mood == r.test_case.expected_mood)
        voice_correct = sum(1 for r in found_results
                           if r.extracted_voice == r.test_case.expected_voice)

        n_found = len(found_results) or 1  # Avoid division by zero

        # Per-category breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
        for r in self.results:
            cat = r.test_case.category
            category_stats[cat]['total'] += 1
            if r.passed:
                category_stats[cat]['passed'] += 1

        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'accuracy': passed / total if total > 0 else 0,
            'verb_detection_rate': verb_found / total if total > 0 else 0,
            'tense_accuracy': tense_correct / n_found,
            'aspect_accuracy': aspect_correct / n_found,
            'mood_accuracy': mood_correct / n_found,
            'voice_accuracy': voice_correct / n_found,
            'by_category': dict(category_stats)
        }

    def get_confusion_matrices(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Generate confusion matrices for all four TAMV dimensions."""
        return {
            'tense': build_confusion_matrix(
                self.results,
                lambda tc: tc.expected_tense,
                lambda r: r.extracted_tense,
                Tense
            ),
            'aspect': build_confusion_matrix(
                self.results,
                lambda tc: tc.expected_aspect,
                lambda r: r.extracted_aspect,
                Aspect
            ),
            'mood': build_confusion_matrix(
                self.results,
                lambda tc: tc.expected_mood,
                lambda r: r.extracted_mood,
                Mood
            ),
            'voice': build_confusion_matrix(
                self.results,
                lambda tc: tc.expected_voice,
                lambda r: r.extracted_voice,
                Voice
            ),
        }

    def generate_report(self, verbose: bool = False) -> str:
        """Generate a validation report."""
        summary = self.get_summary()

        lines = [
            "=" * 70,
            "TAMV EXTRACTION VALIDATION REPORT",
            "=" * 70,
            "",
            "OVERALL RESULTS",
            "-" * 40,
            f"Total test cases:     {summary['total']}",
            f"Passed:               {summary['passed']}",
            f"Failed:               {summary['failed']}",
            f"Overall accuracy:     {summary['accuracy']:.1%}",
            "",
            f"Verb detection rate:  {summary['verb_detection_rate']:.1%}",
            "",
            "PER-DIMENSION ACCURACY (where verb was found)",
            "-" * 40,
            f"Tense:   {summary['tense_accuracy']:.1%}",
            f"Aspect:  {summary['aspect_accuracy']:.1%}",
            f"Mood:    {summary['mood_accuracy']:.1%}",
            f"Voice:   {summary['voice_accuracy']:.1%}",
            "",
            "ACCURACY BY CATEGORY",
            "-" * 40,
        ]

        for cat, stats in sorted(summary['by_category'].items()):
            acc = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            lines.append(f"  {cat:35} {stats['passed']:2}/{stats['total']:2} ({acc:.0%})")

        # Confusion matrices
        matrices = self.get_confusion_matrices()
        lines.append(format_confusion_matrix(matrices['tense'], "TENSE CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['aspect'], "ASPECT CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['mood'], "MOOD CONFUSION MATRIX"))
        lines.append(format_confusion_matrix(matrices['voice'], "VOICE CONFUSION MATRIX"))

        # Failed cases
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.extend([
                "",
                "FAILED TEST CASES",
                "-" * 40,
            ])
            for r in failed:
                lines.append(f"\n  Sentence: \"{r.test_case.sentence}\"")
                lines.append(f"  Verb: {r.test_case.verb}")
                lines.append(f"  Category: {r.test_case.category}")
                lines.append(f"  Expected: {r.test_case.expected_tense.value}-"
                           f"{r.test_case.expected_aspect.value}-"
                           f"{r.test_case.expected_mood.value}-"
                           f"{r.test_case.expected_voice.value}")
                if r.verb_found:
                    lines.append(f"  Got:      {r.extracted_tense.value}-"
                               f"{r.extracted_aspect.value}-"
                               f"{r.extracted_mood.value}-"
                               f"{r.extracted_voice.value}")
                for error in r.errors:
                    lines.append(f"    - {error}")
                if r.test_case.notes:
                    lines.append(f"  Notes: {r.test_case.notes}")

        if verbose:
            lines.extend([
                "",
                "ALL TEST CASES",
                "-" * 40,
            ])
            for r in self.results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(f"\n  [{status}] \"{r.test_case.sentence}\"")
                lines.append(f"    Verb: {r.test_case.verb} | Category: {r.test_case.category}")
                lines.append(f"    Expected: {r.test_case.expected_tense.value}-"
                           f"{r.test_case.expected_aspect.value}-"
                           f"{r.test_case.expected_mood.value}-"
                           f"{r.test_case.expected_voice.value}")
                if r.verb_found:
                    lines.append(f"    Got:      {r.extracted_tense.value}-"
                               f"{r.extracted_aspect.value}-"
                               f"{r.extracted_mood.value}-"
                               f"{r.extracted_voice.value}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Validate TAMV extraction against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories available:
  tense_present, tense_past, tense_future
  aspect_progressive, aspect_perfect, aspect_perfect_progressive
  mood_modal, mood_imperative, mood_conditional, mood_subjunctive
  voice_passive
  complex, edge_cases

Examples:
  %(prog)s                        # Run all tests from data/
  %(prog)s --verbose              # Show all test details
  %(prog)s --category mood_modal  # Test only modal mood cases
  %(prog)s --sentences FILE --expected FILE  # Custom data files
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for all tests')
    parser.add_argument('--category', '-c', type=str,
                       help='Test only specific category')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for report')
    parser.add_argument('--sentences', type=str,
                       help='Path to sentences file (overrides default)')
    parser.add_argument('--expected', type=str,
                       help='Path to expected labels file (overrides default)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display interactive plots (just save to files)')

    args = parser.parse_args()

    print("=" * 70)
    print("TAMV EXTRACTION VALIDATION")
    print("=" * 70)

    # Load test cases from files (no hardcoded data)
    if args.sentences and args.expected:
        print(f"Loading test cases from:")
        print(f"  Sentences: {args.sentences}")
        print(f"  Expected:  {args.expected}")
        test_cases = load_test_cases_from_files(args.sentences, args.expected)
    else:
        print(f"Loading test cases from default data directory: {get_default_data_dir()}")
        test_cases = load_default_test_cases()

    print(f"Loaded {len(test_cases)} test cases")

    validator = TAMVValidator()
    validator.run_validation(test_cases=test_cases, category=args.category)

    report = validator.generate_report(verbose=args.verbose)
    print(report)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Generate confusion matrix plots
    matrices = validator.get_confusion_matrices()
    plot_all_confusion_matrices(matrices, args.output, show=not args.no_show)
    print(f"Confusion matrix plots saved to: {args.output}/")

    # Return exit code based on results
    summary = validator.get_summary()
    if summary['accuracy'] < 0.8:
        print(f"\nWARNING: Accuracy ({summary['accuracy']:.1%}) is below 80% threshold")
        sys.exit(1)
    else:
        print(f"\nValidation passed with {summary['accuracy']:.1%} accuracy")
        sys.exit(0)


if __name__ == '__main__':
    main()
