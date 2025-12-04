#!/usr/bin/env python3
"""
TAMV Extraction Validation Script

Validates the TAMV extractor against linguistically-grounded test cases.
Test cases are derived from:
- Standard English grammar references (Biber et al. 1999, Palmer 2001)
- TMV-annotator documentation (Ramm et al. 2017)
- Universal Dependencies morphological guidelines

Usage:
    python validate_tamv.py                  # Run all tests
    python validate_tamv.py --verbose        # Show detailed output
    python validate_tamv.py --category tense # Test specific category

Output:
    - Console summary of pass/fail rates
    - Detailed error report for failed cases
    - output/validation_report.txt (full report)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tamv_extractor import TAMVExtractor, Tense, Aspect, Mood, Voice


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


# =============================================================================
# GROUND TRUTH TEST CASES
# =============================================================================
# Organized by grammatical category, with sources cited

TEST_CASES = [
    # =========================================================================
    # TENSE: Present
    # =========================================================================
    TestCase(
        sentence="She writes novels.",
        verb="writes",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_present",
        source="Biber et al. 1999, Ch. 6",
        notes="Simple present, habitual"
    ),
    TestCase(
        sentence="The sun rises in the east.",
        verb="rises",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_present",
        source="Biber et al. 1999",
        notes="Simple present, general truth"
    ),
    TestCase(
        sentence="I know the answer.",
        verb="know",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_present",
        source="Biber et al. 1999",
        notes="Simple present, stative verb"
    ),

    # =========================================================================
    # TENSE: Past
    # =========================================================================
    TestCase(
        sentence="She wrote a novel.",
        verb="wrote",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_past",
        source="Biber et al. 1999, Ch. 6",
        notes="Simple past"
    ),
    TestCase(
        sentence="They arrived yesterday.",
        verb="arrived",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_past",
        source="Biber et al. 1999",
        notes="Simple past with time adverb"
    ),
    TestCase(
        sentence="The committee decided unanimously.",
        verb="decided",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_past",
        source="Biber et al. 1999",
        notes="Simple past"
    ),

    # =========================================================================
    # TENSE: Future (via will/shall)
    # TMV-annotator: will/shall don't change mood to subjunctive
    # =========================================================================
    TestCase(
        sentence="She will write a novel.",
        verb="write",
        expected_tense=Tense.FUTURE,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_future",
        source="Biber et al. 1999, Ch. 6",
        notes="Future with will (indicative per TMV-annotator)"
    ),
    TestCase(
        sentence="They will arrive tomorrow.",
        verb="arrive",
        expected_tense=Tense.FUTURE,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="tense_future",
        source="Biber et al. 1999",
        notes="Future with will (indicative per TMV-annotator)"
    ),

    # =========================================================================
    # ASPECT: Progressive
    # =========================================================================
    TestCase(
        sentence="She is writing a novel.",
        verb="writing",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_progressive",
        source="Biber et al. 1999, Ch. 6",
        notes="Present progressive"
    ),
    TestCase(
        sentence="They were discussing the proposal.",
        verb="discussing",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_progressive",
        source="Biber et al. 1999",
        notes="Past progressive"
    ),
    TestCase(
        sentence="He will be waiting for us.",
        verb="waiting",
        expected_tense=Tense.FUTURE,
        expected_aspect=Aspect.PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_progressive",
        source="Biber et al. 1999",
        notes="Future progressive (indicative per TMV-annotator)"
    ),

    # =========================================================================
    # ASPECT: Perfect
    # =========================================================================
    TestCase(
        sentence="She has written three novels.",
        verb="written",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_perfect",
        source="Biber et al. 1999, Ch. 6",
        notes="Present perfect"
    ),
    TestCase(
        sentence="They had finished before noon.",
        verb="finished",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_perfect",
        source="Biber et al. 1999",
        notes="Past perfect (pluperfect)"
    ),
    TestCase(
        sentence="By next week, she will have completed the project.",
        verb="completed",
        expected_tense=Tense.FUTURE,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_perfect",
        source="Biber et al. 1999",
        notes="Future perfect (indicative per TMV-annotator)"
    ),

    # =========================================================================
    # ASPECT: Perfect Progressive
    # =========================================================================
    TestCase(
        sentence="She has been writing for hours.",
        verb="writing",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT_PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_perfect_progressive",
        source="Biber et al. 1999, Ch. 6",
        notes="Present perfect progressive"
    ),
    TestCase(
        sentence="They had been waiting since morning.",
        verb="waiting",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.PERFECT_PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="aspect_perfect_progressive",
        source="Biber et al. 1999",
        notes="Past perfect progressive"
    ),

    # =========================================================================
    # MOOD: Modal / Subjunctive (per TMV-annotator terminology)
    # TMV-annotator: would/could/should/might → subjunctive
    # TMV-annotator: will/shall/can/may/must → indicative
    # =========================================================================
    TestCase(
        sentence="She can write novels.",
        verb="write",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_modal",
        source="Palmer 2001, Ch. 4",
        notes="Modal: ability (can) - indicative per TMV-annotator"
    ),
    TestCase(
        sentence="You must submit the report.",
        verb="submit",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_modal",
        source="Palmer 2001",
        notes="Modal: obligation (must) - indicative per TMV-annotator"
    ),
    TestCase(
        sentence="He might arrive late.",
        verb="arrive",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_modal",
        source="Palmer 2001",
        notes="Modal: possibility (might) - subjunctive per TMV-annotator"
    ),
    TestCase(
        sentence="They should consider the alternatives.",
        verb="consider",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_modal",
        source="Palmer 2001",
        notes="Modal: advice (should) - subjunctive per TMV-annotator"
    ),
    TestCase(
        sentence="She could have written a better ending.",
        verb="written",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_modal",
        source="Palmer 2001",
        notes="Modal perfect: counterfactual (could) - subjunctive per TMV-annotator"
    ),

    # =========================================================================
    # MOOD: Imperative
    # =========================================================================
    TestCase(
        sentence="Write the report now.",
        verb="Write",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.IMPERATIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_imperative",
        source="Biber et al. 1999, Ch. 8",
        notes="Imperative"
    ),
    TestCase(
        sentence="Please submit your application.",
        verb="submit",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.IMPERATIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_imperative",
        source="Biber et al. 1999",
        notes="Imperative with please"
    ),
    TestCase(
        sentence="Don't forget the meeting.",
        verb="forget",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.IMPERATIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_imperative",
        source="Biber et al. 1999",
        notes="Negative imperative"
    ),

    # =========================================================================
    # MOOD: Conditional
    # =========================================================================
    TestCase(
        sentence="If she writes the novel, it will be published.",
        verb="writes",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.CONDITIONAL,
        expected_voice=Voice.ACTIVE,
        category="mood_conditional",
        source="Palmer 2001, Ch. 6",
        notes="Conditional: protasis (if-clause)"
    ),
    TestCase(
        sentence="Unless they agree, we cannot proceed.",
        verb="agree",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.CONDITIONAL,
        expected_voice=Voice.ACTIVE,
        category="mood_conditional",
        source="Palmer 2001",
        notes="Conditional: unless-clause"
    ),

    # =========================================================================
    # MOOD: Subjunctive
    # =========================================================================
    TestCase(
        sentence="I suggest that he leave immediately.",
        verb="leave",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_subjunctive",
        source="Biber et al. 1999, Ch. 8",
        notes="Mandative subjunctive"
    ),
    TestCase(
        sentence="The committee recommends that she be appointed.",
        verb="appointed",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.PASSIVE,
        category="mood_subjunctive",
        source="Biber et al. 1999",
        notes="Mandative subjunctive, passive"
    ),
    TestCase(
        sentence="It is essential that he attend the meeting.",
        verb="attend",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.ACTIVE,
        category="mood_subjunctive",
        source="Biber et al. 1999",
        notes="Subjunctive after adjective"
    ),

    # =========================================================================
    # VOICE: Passive
    # =========================================================================
    TestCase(
        sentence="The novel was written by her.",
        verb="written",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999, Ch. 6",
        notes="Simple past passive"
    ),
    TestCase(
        sentence="The report is being reviewed.",
        verb="reviewed",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Present progressive passive"
    ),
    TestCase(
        sentence="The decision has been made.",
        verb="made",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Present perfect passive"
    ),
    TestCase(
        sentence="The proposal will be discussed tomorrow.",
        verb="discussed",
        expected_tense=Tense.FUTURE,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Future passive (indicative per TMV-annotator)"
    ),
    TestCase(
        sentence="The work had been completed before the deadline.",
        verb="completed",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Past perfect passive"
    ),
    TestCase(
        sentence="Mistakes can be corrected.",
        verb="corrected",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Modal passive (can = indicative per TMV-annotator)"
    ),
    TestCase(
        sentence="The house got destroyed in the storm.",
        verb="destroyed",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="voice_passive",
        source="Biber et al. 1999",
        notes="Get-passive"
    ),

    # =========================================================================
    # COMPLEX COMBINATIONS
    # =========================================================================
    TestCase(
        sentence="The report has been being written for weeks.",
        verb="written",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT_PROGRESSIVE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.PASSIVE,
        category="complex",
        source="Biber et al. 1999",
        notes="Perfect progressive passive (rare)"
    ),
    TestCase(
        sentence="She might have been being watched.",
        verb="watched",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.PERFECT_PROGRESSIVE,
        expected_mood=Mood.SUBJUNCTIVE,
        expected_voice=Voice.PASSIVE,
        category="complex",
        source="Biber et al. 1999",
        notes="Modal perfect progressive passive (might = subjunctive per TMV-annotator)"
    ),
    TestCase(
        sentence="If the data had been analyzed correctly, we would have found the error.",
        verb="analyzed",
        expected_tense=Tense.PAST,
        expected_aspect=Aspect.PERFECT,
        expected_mood=Mood.CONDITIONAL,
        expected_voice=Voice.PASSIVE,
        category="complex",
        source="Palmer 2001",
        notes="Counterfactual conditional passive"
    ),

    # =========================================================================
    # EDGE CASES
    # =========================================================================
    TestCase(
        sentence="There is a problem.",
        verb="is",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="edge_cases",
        source="Biber et al. 1999",
        notes="Existential there"
    ),
    TestCase(
        sentence="It seems that he left.",
        verb="seems",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="edge_cases",
        source="Biber et al. 1999",
        notes="Raising verb"
    ),
    TestCase(
        sentence="She wants to leave.",
        verb="wants",
        expected_tense=Tense.PRESENT,
        expected_aspect=Aspect.SIMPLE,
        expected_mood=Mood.INDICATIVE,
        expected_voice=Voice.ACTIVE,
        category="edge_cases",
        source="Biber et al. 1999",
        notes="Matrix verb (not infinitive)"
    ),
]


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
  %(prog)s                        # Run all tests
  %(prog)s --verbose              # Show all test details
  %(prog)s --category mood_modal  # Test only modal mood cases
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for all tests')
    parser.add_argument('--category', '-c', type=str,
                       help='Test only specific category')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for report')

    args = parser.parse_args()

    print("=" * 70)
    print("TAMV EXTRACTION VALIDATION")
    print("=" * 70)

    validator = TAMVValidator()
    validator.run_validation(category=args.category)

    report = validator.generate_report(verbose=args.verbose)
    print(report)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

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
