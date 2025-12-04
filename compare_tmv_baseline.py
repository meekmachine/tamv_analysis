#!/usr/bin/env python3
"""
TMV-Annotator Baseline Comparison

Compares our TAMV extractor against the original TMV-annotator (Ramm et al. 2017)
to establish a baseline and identify areas for improvement.

This script:
1. Converts spaCy parses to CoNLL format compatible with TMV-annotator
2. Runs TMV-annotator on the converted data
3. Compares results with our extractor
4. Reports accuracy differences

Usage:
    python compare_tmv_baseline.py
    python compare_tmv_baseline.py --sentences "She has been writing." "He would go."
"""

import sys
import os
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spacy
from src.tamv_extractor import TAMVExtractor, Tense, Aspect, Mood, Voice


@dataclass
class TMVResult:
    """Result from TMV-annotator."""
    verb_ids: str
    verbs: str
    finite: str
    main_verb: str
    tense: str
    mood: str
    voice: str
    progressive: str
    negation: str


def spacy_to_mate_conll(doc) -> str:
    """
    Convert spaCy Doc to Mate parser CoNLL format.

    Mate CoNLL format (14 columns):
    ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED

    We map spaCy output to approximate this format.
    """
    lines = []

    # Map spaCy dependency labels to Mate-style labels
    dep_map = {
        'nsubj': 'SBJ',
        'nsubjpass': 'SBJ',
        'dobj': 'OBJ',
        'iobj': 'OBJ',
        'aux': 'VC',  # TMV-annotator uses VC for verb complexes
        'auxpass': 'VC',
        'ROOT': 'ROOT',
        'xcomp': 'VC',
        'ccomp': 'OBJ',
        'advcl': 'ADV',
        'amod': 'NMOD',
        'det': 'NMOD',
        'prep': 'ADV',
        'pobj': 'PMOD',
        'mark': 'SUB',
        'cc': 'COORD',
        'conj': 'CONJ',
        'punct': 'P',
        'advmod': 'ADV',
        'neg': 'ADV',
        'prt': 'PRT',
        'compound': 'NMOD',
        'attr': 'PRD',
        'acomp': 'PRD',
    }

    for token in doc:
        # ID (1-indexed)
        tok_id = token.i + 1

        # FORM
        form = token.text

        # LEMMA (use _ as placeholder for PLEMMA)
        lemma = token.lemma_

        # POS tags (use spaCy's Penn Treebank tags)
        pos = token.tag_

        # FEAT (morphological features - use _ as placeholder)
        feat = '_'

        # HEAD (1-indexed, 0 for root)
        if token.dep_ == 'ROOT':
            head = 0
        else:
            head = token.head.i + 1

        # DEPREL
        deprel = dep_map.get(token.dep_, token.dep_.upper())
        if token.dep_ == 'ROOT':
            deprel = 'ROOT'

        # Mate format: ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL _ _
        # TMV-EN.py reads columns: 0=ID, 1=FORM, 3=LEMMA, 5=POS, 7=MORPH, 9=HEAD, 11=DEPREL
        line = f"{tok_id}\t{form}\t_\t{lemma}\t_\t{pos}\t_\t{feat}\t_\t{head}\t_\t{deprel}\t_\t_"
        lines.append(line)

    return '\n'.join(lines) + '\n'


def run_tmv_annotator(sentences: List[str], tmv_script_path: str) -> List[List[TMVResult]]:
    """
    Run TMV-annotator on a list of sentences.

    Returns list of TMVResult for each sentence.
    """
    nlp = spacy.load('en_core_web_sm')

    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parsed', delete=False) as f:
        parsed_file = f.name

        for sent in sentences:
            doc = nlp(sent)
            conll = spacy_to_mate_conll(doc)
            f.write(conll)
            f.write('\n')  # Empty line between sentences

    # Create output directory if needed
    output_dir = os.path.join(os.path.dirname(tmv_script_path), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Run TMV-annotator
    # The script expects: python TMV-EN.py <parsed_file> <output_name>
    output_name = 'temp_comparison'

    try:
        # Change to TMV-annotator directory and run
        orig_dir = os.getcwd()
        os.chdir(os.path.dirname(tmv_script_path))

        result = subprocess.run(
            [sys.executable, os.path.basename(tmv_script_path), parsed_file, output_name],
            capture_output=True,
            text=True,
            timeout=60
        )

        os.chdir(orig_dir)

        if result.returncode != 0:
            print(f"TMV-annotator stderr: {result.stderr}")
            return []

        # Read output
        output_file = os.path.join(output_dir, f"{output_name}.verbs")
        results = parse_tmv_output(output_file)

        # Cleanup
        os.unlink(parsed_file)
        if os.path.exists(output_file):
            os.unlink(output_file)

        return results

    except Exception as e:
        print(f"Error running TMV-annotator: {e}")
        os.chdir(orig_dir)
        return []


def parse_tmv_output(output_file: str) -> List[List[TMVResult]]:
    """Parse TMV-annotator output file."""
    results_by_sent = {}

    if not os.path.exists(output_file):
        return []

    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 10:
                sent_nr = int(parts[0])
                result = TMVResult(
                    verb_ids=parts[1],
                    verbs=parts[2],
                    finite=parts[3],
                    main_verb=parts[4],
                    tense=parts[5],
                    mood=parts[6],
                    voice=parts[7],
                    progressive=parts[8] if len(parts) > 8 else '-',
                    negation=parts[9] if len(parts) > 9 else '-'
                )

                if sent_nr not in results_by_sent:
                    results_by_sent[sent_nr] = []
                results_by_sent[sent_nr].append(result)

    # Convert to list ordered by sentence number
    return [results_by_sent.get(i, []) for i in range(1, max(results_by_sent.keys()) + 1)] if results_by_sent else []


def map_our_result_to_tmv(tense: Tense, aspect: Aspect, mood: Mood, voice: Voice) -> Dict[str, str]:
    """Map our TAMV result to TMV-annotator terminology."""

    # Tense mapping
    tense_map = {
        (Tense.PRESENT, Aspect.SIMPLE): 'pres',
        (Tense.PRESENT, Aspect.PROGRESSIVE): 'presProg',
        (Tense.PRESENT, Aspect.PERFECT): 'presPerf',
        (Tense.PRESENT, Aspect.PERFECT_PROGRESSIVE): 'presPerfProg',
        (Tense.PAST, Aspect.SIMPLE): 'past',
        (Tense.PAST, Aspect.PROGRESSIVE): 'pastProg',
        (Tense.PAST, Aspect.PERFECT): 'pastPerf',
        (Tense.PAST, Aspect.PERFECT_PROGRESSIVE): 'pastPerfProg',
        (Tense.FUTURE, Aspect.SIMPLE): 'futureI',
        (Tense.FUTURE, Aspect.PROGRESSIVE): 'futureIProg',
        (Tense.FUTURE, Aspect.PERFECT): 'futureII',
        (Tense.FUTURE, Aspect.PERFECT_PROGRESSIVE): 'futureIIProg',
    }

    # Mood mapping
    # TMV-annotator: indicative, subjunctive (for would/could/should/might)
    # Ours: indicative, subjunctive, imperative, conditional, modal
    mood_map = {
        Mood.INDICATIVE: 'indicative',
        Mood.SUBJUNCTIVE: 'subjunctive',  # True subjunctive
        Mood.IMPERATIVE: 'indicative',  # TMV-annotator doesn't have imperative
        Mood.CONDITIONAL: 'subjunctive',  # TMV-annotator calls conditionals "subjunctive"
        Mood.MODAL: 'indicative',  # TMV-annotator: regular modals are indicative
    }

    # Handle conditional as subjunctive in TMV-annotator terms
    tmv_tense = tense_map.get((tense, aspect), 'err')

    # If it's conditional mood, adjust tense name
    if mood == Mood.CONDITIONAL or mood == Mood.MODAL:
        if tense == Tense.PRESENT and aspect == Aspect.SIMPLE:
            if mood == Mood.CONDITIONAL:
                tmv_tense = 'condI'
        elif tense == Tense.PRESENT and aspect == Aspect.PERFECT:
            if mood == Mood.CONDITIONAL:
                tmv_tense = 'condII'

    return {
        'tense': tmv_tense,
        'mood': mood_map.get(mood, 'err'),
        'voice': voice.value,
        'progressive': 'yes' if aspect in [Aspect.PROGRESSIVE, Aspect.PERFECT_PROGRESSIVE] else 'no'
    }


def compare_results(sentences: List[str], our_extractor: TAMVExtractor,
                   tmv_results: List[List[TMVResult]]) -> Dict:
    """Compare our results with TMV-annotator results."""

    comparison = {
        'total_sentences': len(sentences),
        'total_verbs_ours': 0,
        'total_verbs_tmv': 0,
        'matches': {'tense': 0, 'mood': 0, 'voice': 0, 'progressive': 0},
        'mismatches': [],
        'details': []
    }

    for i, sent in enumerate(sentences):
        our_labels = our_extractor.extract_from_text(sent)
        tmv_sent_results = tmv_results[i] if i < len(tmv_results) else []

        comparison['total_verbs_ours'] += len(our_labels)
        comparison['total_verbs_tmv'] += len(tmv_sent_results)

        sent_detail = {
            'sentence': sent,
            'ours': [],
            'tmv': [],
            'comparison': []
        }

        for label in our_labels:
            our_mapped = map_our_result_to_tmv(label.tense, label.aspect, label.mood, label.voice)
            sent_detail['ours'].append({
                'verb': label.verb_text,
                'original': label.to_string(),
                'mapped': our_mapped
            })

        for tmv_res in tmv_sent_results:
            sent_detail['tmv'].append({
                'verbs': tmv_res.verbs,
                'tense': tmv_res.tense,
                'mood': tmv_res.mood,
                'voice': tmv_res.voice,
                'progressive': tmv_res.progressive
            })

        comparison['details'].append(sent_detail)

    return comparison


def run_direct_comparison(test_sentences: List[str] = None):
    """
    Run a direct comparison without TMV-annotator (since format conversion is complex).
    Instead, manually compare against TMV-annotator's documented behavior.
    """

    if test_sentences is None:
        test_sentences = [
            # Simple tenses
            ("She writes novels.", "pres", "indicative", "active"),
            ("She wrote a novel.", "past", "indicative", "active"),

            # Progressive
            ("She is writing a novel.", "presProg", "indicative", "active"),
            ("She was writing a novel.", "pastProg", "indicative", "active"),

            # Perfect
            ("She has written a novel.", "presPerf", "indicative", "active"),
            ("She had written a novel.", "pastPerf", "indicative", "active"),

            # Perfect Progressive
            ("She has been writing.", "presPerfProg", "indicative", "active"),
            ("She had been writing.", "pastPerfProg", "indicative", "active"),

            # Future (with will)
            ("She will write a novel.", "futureI", "indicative", "active"),
            ("She will be writing.", "futureIProg", "indicative", "active"),
            ("She will have written.", "futureII", "indicative", "active"),

            # Conditional/Subjunctive (TMV-annotator calls these subjunctive)
            ("She would write a novel.", "condI", "subjunctive", "active"),
            ("She would have written.", "condII", "subjunctive", "active"),
            ("She could write a novel.", "condI", "subjunctive", "active"),

            # Passive
            ("The novel was written.", "past", "indicative", "passive"),
            ("The novel is being written.", "presProg", "indicative", "passive"),
            ("The novel has been written.", "presPerf", "indicative", "passive"),
            ("The novel will be written.", "futureI", "indicative", "passive"),

            # Modals (non-conditional)
            ("She can write novels.", "pres", "indicative", "active"),
            ("She must write a novel.", "pres", "indicative", "active"),

            # Imperative (TMV-annotator doesn't have this category)
            ("Write the novel!", "pres", "indicative", "active"),
        ]

    extractor = TAMVExtractor()

    results = {
        'total': len(test_sentences),
        'tense_match': 0,
        'mood_match': 0,
        'voice_match': 0,
        'all_match': 0,
        'details': []
    }

    print("=" * 80)
    print("COMPARISON: Our Extractor vs TMV-Annotator Expected Output")
    print("=" * 80)
    print()

    for item in test_sentences:
        if isinstance(item, tuple):
            sent, expected_tense, expected_mood, expected_voice = item
        else:
            sent = item
            expected_tense = expected_mood = expected_voice = None

        labels = extractor.extract_from_text(sent)

        if not labels:
            print(f"SENTENCE: {sent}")
            print(f"  ERROR: No verbs extracted")
            print()
            continue

        # Get the main verb (usually the last one or the ROOT)
        label = labels[0]
        our_mapped = map_our_result_to_tmv(label.tense, label.aspect, label.mood, label.voice)

        tense_ok = our_mapped['tense'] == expected_tense if expected_tense else True
        mood_ok = our_mapped['mood'] == expected_mood if expected_mood else True
        voice_ok = our_mapped['voice'] == expected_voice if expected_voice else True

        if tense_ok:
            results['tense_match'] += 1
        if mood_ok:
            results['mood_match'] += 1
        if voice_ok:
            results['voice_match'] += 1
        if tense_ok and mood_ok and voice_ok:
            results['all_match'] += 1

        status = "✓" if (tense_ok and mood_ok and voice_ok) else "✗"

        print(f"{status} SENTENCE: {sent}")
        print(f"  Our result:      {label.to_string()}")
        print(f"  Mapped to TMV:   tense={our_mapped['tense']}, mood={our_mapped['mood']}, voice={our_mapped['voice']}")
        if expected_tense:
            print(f"  TMV expected:    tense={expected_tense}, mood={expected_mood}, voice={expected_voice}")
            if not (tense_ok and mood_ok and voice_ok):
                mismatches = []
                if not tense_ok:
                    mismatches.append(f"tense ({our_mapped['tense']} vs {expected_tense})")
                if not mood_ok:
                    mismatches.append(f"mood ({our_mapped['mood']} vs {expected_mood})")
                if not voice_ok:
                    mismatches.append(f"voice ({our_mapped['voice']} vs {expected_voice})")
                print(f"  MISMATCH: {', '.join(mismatches)}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {results['total']}")
    print(f"Tense accuracy:   {results['tense_match']}/{results['total']} ({100*results['tense_match']/results['total']:.1f}%)")
    print(f"Mood accuracy:    {results['mood_match']}/{results['total']} ({100*results['mood_match']/results['total']:.1f}%)")
    print(f"Voice accuracy:   {results['voice_match']}/{results['total']} ({100*results['voice_match']/results['total']:.1f}%)")
    print(f"All match:        {results['all_match']}/{results['total']} ({100*results['all_match']/results['total']:.1f}%)")
    print()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare TAMV extractor with TMV-annotator baseline')
    parser.add_argument('--sentences', nargs='+', help='Custom sentences to test')
    parser.add_argument('--run-tmv', action='store_true', help='Actually run TMV-annotator (requires setup)')

    args = parser.parse_args()

    if args.run_tmv:
        # Try to run actual TMV-annotator
        tmv_script = os.path.join(
            os.path.dirname(__file__),
            'tmv_annotator_reference/tmv-annotator-tool/TMV-EN.py'
        )

        if not os.path.exists(tmv_script):
            print("TMV-annotator not found. Run: git clone https://github.com/aniramm/tmv-annotator.git tmv_annotator_reference")
            sys.exit(1)

        sentences = args.sentences or [
            "She has been writing for hours.",
            "The novel was written by the author.",
            "He would have gone if he could.",
        ]

        print("Running TMV-annotator...")
        results = run_tmv_annotator(sentences, tmv_script)

        if results:
            print("\nTMV-annotator results:")
            for i, sent_results in enumerate(results):
                print(f"\nSentence {i+1}:")
                for r in sent_results:
                    print(f"  {r.verbs}: tense={r.tense}, mood={r.mood}, voice={r.voice}")
    else:
        # Run comparison against expected TMV-annotator behavior
        run_direct_comparison()


if __name__ == '__main__':
    main()
