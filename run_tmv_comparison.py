#!/usr/bin/env python3
"""
Direct comparison between our TAMV extractor and TMV-annotator.

Creates CoNLL-format input for TMV-annotator, runs it, and compares results.
"""

import os
import sys
import tempfile
import spacy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tamv_extractor import TAMVExtractor


def spacy_to_mate_conll(doc) -> str:
    """
    Convert spaCy Doc to Mate parser CoNLL format.

    TMV-EN.py reads columns at these indices:
    - 0: ID
    - 1: FORM (token)
    - 3: LEMMA
    - 5: POS (Penn Treebank tag)
    - 7: MORPH (morphological features)
    - 9: HEAD
    - 11: DEPREL
    """
    lines = []

    # Map spaCy dep labels to Mate-style
    dep_map = {
        'ROOT': 'ROOT',
        'nsubj': 'SBJ',
        'nsubjpass': 'SBJ',
        'aux': 'VC',
        'auxpass': 'VC',
        'dobj': 'OBJ',
        'xcomp': 'VC',
        'ccomp': 'OBJ',
        'advcl': 'ADV',
        'punct': 'P',
        'det': 'NMOD',
        'amod': 'NMOD',
        'advmod': 'ADV',
        'prep': 'ADV',
        'pobj': 'PMOD',
        'mark': 'SUB',
        'neg': 'ADV',
        'attr': 'PRD',
        'acomp': 'PRD',
    }

    for token in doc:
        tok_id = token.i + 1
        form = token.text
        lemma = token.lemma_
        pos = token.tag_
        morph = str(token.morph) if token.morph else '_'

        if token.dep_ == 'ROOT':
            head = 0
        else:
            head = token.head.i + 1

        deprel = dep_map.get(token.dep_, token.dep_.upper())
        if token.dep_ == 'ROOT':
            deprel = '__'  # TMV-annotator uses __ for ROOT

        # 14-column format: ID FORM _ LEMMA _ POS _ MORPH _ HEAD _ DEPREL _ _
        line = f"{tok_id}\t{form}\t_\t{lemma}\t_\t{pos}\t_\t{morph}\t_\t{head}\t_\t{deprel}\t_\t_"
        lines.append(line)

    return '\n'.join(lines)


def run_comparison():
    """Run comparison on test sentences."""

    nlp = spacy.load('en_core_web_sm')
    extractor = TAMVExtractor()

    # Test sentences with expected TMV-annotator output
    test_cases = [
        # (sentence, expected_tense, expected_mood, expected_voice)
        ("She writes novels.", "pres", "indicative", "active"),
        ("She wrote a novel.", "past", "indicative", "active"),
        ("She is writing.", "presProg", "indicative", "active"),
        ("She was writing.", "pastProg", "indicative", "active"),
        ("She has written.", "presPerf", "indicative", "active"),
        ("She had written.", "pastPerf", "indicative", "active"),
        ("She has been writing.", "presPerfProg", "indicative", "active"),
        ("She had been writing.", "pastPerfProg", "indicative", "active"),
        ("She will write.", "futureI", "indicative", "active"),
        ("She will be writing.", "futureIProg", "indicative", "active"),
        ("She will have written.", "futureII", "indicative", "active"),
        ("She would write.", "condI", "subjunctive", "active"),
        ("She would have written.", "condII", "subjunctive", "active"),
        ("She could write.", "condI", "subjunctive", "active"),
        ("She might write.", "condI", "subjunctive", "active"),
        ("She should write.", "condI", "subjunctive", "active"),
        ("The novel was written.", "past", "indicative", "passive"),
        ("The novel is being written.", "presProg", "indicative", "passive"),
        ("The novel has been written.", "presPerf", "indicative", "passive"),
        ("She can write.", "pres", "indicative", "active"),
        ("She must write.", "pres", "indicative", "active"),
        ("She may write.", "pres", "indicative", "active"),
    ]

    # Create CoNLL file for TMV-annotator
    tmv_tool_dir = os.path.join(os.path.dirname(__file__),
                                 'tmv_annotator_reference/tmv-annotator-tool')

    conll_lines = []
    for sent, _, _, _ in test_cases:
        doc = nlp(sent)
        conll_lines.append(spacy_to_mate_conll(doc))
        conll_lines.append('')  # Empty line between sentences

    # Write CoNLL file
    conll_path = os.path.join(tmv_tool_dir, 'data', 'test_comparison.parsed')
    with open(conll_path, 'w') as f:
        f.write('\n'.join(conll_lines))

    # Run TMV-annotator
    orig_dir = os.getcwd()
    os.chdir(tmv_tool_dir)
    os.system(f'{sys.executable} TMV-EN.py data/test_comparison.parsed comparison_output 2>/dev/null')
    os.chdir(orig_dir)

    # Read TMV-annotator output
    tmv_output_path = os.path.join(tmv_tool_dir, 'output', 'comparison_output.verbs')
    tmv_results = {}

    if os.path.exists(tmv_output_path):
        with open(tmv_output_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    sent_id = int(parts[0])
                    tmv_results[sent_id] = {
                        'verbs': parts[2],
                        'tense': parts[5],
                        'mood': parts[6],
                        'voice': parts[7]
                    }

    # Compare results
    print("=" * 90)
    print("COMPARISON: Our TAMV Extractor vs TMV-Annotator (Ramm et al. 2017)")
    print("=" * 90)
    print()
    print(f"{'Sentence':<40} {'TMV-Ann':<25} {'Ours':<25} {'Match'}")
    print("-" * 90)

    our_correct = 0
    tmv_found = 0

    for i, (sent, exp_tense, exp_mood, exp_voice) in enumerate(test_cases, 1):
        # Our result
        labels = extractor.extract_from_text(sent)
        if labels:
            our_label = labels[0]
            our_str = our_label.to_string()
        else:
            our_str = "NO VERB"

        # TMV-annotator result
        if i in tmv_results:
            tmv = tmv_results[i]
            tmv_str = f"{tmv['tense']}/{tmv['mood']}/{tmv['voice']}"
            tmv_found += 1

            # Check if TMV matches expected
            tmv_match = (tmv['tense'] == exp_tense and
                        tmv['mood'] == exp_mood and
                        tmv['voice'] == exp_voice)
        else:
            tmv_str = "NOT FOUND"
            tmv_match = False

        # Check our match to expected (using TMV terminology mapping)
        our_tense_map = {
            'present-simple': 'pres',
            'present-progressive': 'presProg',
            'present-perfect': 'presPerf',
            'present-perfect-progressive': 'presPerfProg',
            'past-simple': 'past',
            'past-progressive': 'pastProg',
            'past-perfect': 'pastPerf',
            'past-perfect-progressive': 'pastPerfProg',
            'future-simple': 'futureI',
            'future-progressive': 'futureIProg',
            'future-perfect': 'futureII',
        }

        if labels:
            our_tense_aspect = f"{our_label.tense.value}-{our_label.aspect.value}"
            our_mapped_tense = our_tense_map.get(our_tense_aspect, our_label.tense.value)

            # Map our mood to TMV mood
            our_mapped_mood = 'indicative'
            if our_label.mood.value in ['conditional']:
                our_mapped_mood = 'subjunctive'
            elif our_label.mood.value == 'modal':
                # Check if it's would/could/should/might (subjunctive in TMV)
                if any(aux in ['would', 'could', 'should', 'might']
                       for aux in our_label.auxiliary_chain):
                    our_mapped_mood = 'subjunctive'
                    if our_label.aspect.value == 'simple':
                        our_mapped_tense = 'condI'
                    elif our_label.aspect.value == 'perfect':
                        our_mapped_tense = 'condII'

            our_match = (our_mapped_tense == exp_tense and
                        our_mapped_mood == exp_mood and
                        our_label.voice.value == exp_voice)
            if our_match:
                our_correct += 1
        else:
            our_match = False

        match_str = ""
        if tmv_match and our_match:
            match_str = "Both ✓"
        elif tmv_match:
            match_str = "TMV ✓"
        elif our_match:
            match_str = "Ours ✓"
        else:
            match_str = "Neither"

        print(f"{sent:<40} {tmv_str:<25} {our_str:<25} {match_str}")

    print("-" * 90)
    print()
    print("SUMMARY")
    print("-" * 40)
    print(f"TMV-annotator found verbs: {tmv_found}/{len(test_cases)}")
    print(f"Our accuracy vs expected:  {our_correct}/{len(test_cases)} ({100*our_correct/len(test_cases):.1f}%)")
    print()

    # Cleanup
    if os.path.exists(conll_path):
        os.unlink(conll_path)
    if os.path.exists(tmv_output_path):
        os.unlink(tmv_output_path)


if __name__ == '__main__':
    run_comparison()
