# TAMV Analysis Tool

A Python pipeline for analyzing **Tense, Aspect, Mood, and Voice (TAMV)** distributions across English corpora. This tool extracts clause-level TAMV annotations using rule-based parsing, aggregates them into document-level profiles, and enables analysis of grammatical patterns across genres and conversational contexts.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run validation tests
python -m tests.validate_tamv --no-show      # Single-verb validation (80% accuracy)
python -m tests.validate_phrases --no-show   # Phrase-level validation (87.5% accuracy)

# Extract TAMV from a corpus
python -m src extract --corpus brown --max-docs 50

# Analyze CGA conversations
python -m src cga --dataset wiki --sample-rate 10 --no-show
```

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Validation](#validation)
- [Technical Approach](#technical-approach)
- [TAMV Categories](#tamv-categories)
- [Project Structure](#project-structure)
- [References](#references)

## Overview

This project investigates how TAMV is distributed across two complementary corpora:

- **Brown Corpus**: 500 documents of edited written American English across 15 genres (news, fiction, academic, etc.)
- **Conversations Gone Awry (CGA)**: 4,188 online conversations from Wikipedia Talk Pages that either remain civil or derail into personal attacks

The tool provides:
- Rule-based TAMV extraction using spaCy dependency parsing
- Document and corpus-level aggregation with statistical analysis
- Speaker-level CGA analysis with z-score deviations and derailment correlation
- Validation against linguistic ground truth (80% single-verb, 87.5% phrase-level accuracy)
- Extended mood detection beyond TMV-annotator (MODAL, CONDITIONAL, INFINITIVAL, PARTICIPLE)

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone or download the repository
cd tamv_analysis

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

## Usage

### Validation Tests

Run the validation suite to verify extraction accuracy:

```bash
# Single-verb validation (65 test cases from Brown corpus + synthetic examples)
python -m tests.validate_tamv --no-show

# Phrase-level validation (32 test cases with 43 verb phrases)
python -m tests.validate_phrases --no-show

# Show interactive confusion matrix plots
python -m tests.validate_tamv
python -m tests.validate_phrases

# Verbose output showing all test details
python -m tests.validate_tamv --verbose --no-show
```

### Data Extraction

Extract TAMV distributions from corpora:

```bash
# Extract from both corpora (Brown + CGA)
python -m src extract

# Brown corpus only
python -m src extract --corpus brown

# CGA only (Wikipedia version)
python -m src extract --corpus cga --dataset wiki

# Quick test with limited documents
python -m src extract --max-docs 50 --output output/test
```

This produces:
- `tamv_profiles.csv` - Document-level TAMV frequencies
- `tamv_summary_by_genre.csv` - Aggregated means by genre
- `tamv_summary_by_corpus.csv` - Aggregated means by corpus
- `extraction_report.txt` - Summary statistics

### CGA Speaker-Level Analysis

Analyze TAMV patterns per speaker in conversations, with z-score deviations and derailment correlation:

```bash
# Sample 1/10 of Wikipedia conversations
python -m src cga --dataset wiki --sample-rate 10

# Sample 1/5 of Reddit CMV
python -m src cga --dataset cmv --sample-rate 5

# Run without interactive plots (just save files)
python -m src cga --dataset wiki --no-show
```

This produces:
- `cga_wiki_analysis.txt` - Full analysis report
- `cga_wiki_speakers.tsv` - Speaker-level z-scores
- `cga_wiki_derailment_comparison.png` - Bar chart of mean |z| by outcome
- `cga_wiki_label_differences.png` - TAMV labels with largest derailment differences
- `cga_wiki_zscore_distribution.png` - Histogram of z-scores by outcome
- `cga_wiki_zscore_boxplot.png` - Box plot comparison
- `cga_wiki_top_tamv.png` - Top TAMV combinations by frequency

### Command Line Options

**Extract command** (`python -m src extract`):

| Option | Description | Default |
|--------|-------------|---------|
| `--corpus` | Corpus to extract: `brown`, `cga`, or `both` | `both` |
| `--genres` | Specific Brown genres (space-separated) | All genres |
| `--dataset` | CGA dataset: `wiki` or `cmv` | `wiki` |
| `--output` | Output directory | `output/` |
| `--max-docs` | Maximum documents per corpus | None (all) |

**CGA command** (`python -m src cga`):

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | CGA dataset: `wiki` or `cmv` | `wiki` |
| `--sample-rate` | Take 1 out of every N conversations | `10` |
| `--output` | Output directory | `output/` |
| `--no-show` | Don't display interactive plots | False |

## Validation

### Test Data

Test data is stored in `data/` with separate input and expected output files:

| File | Description |
|------|-------------|
| `synthetic_sentences.txt` | 50 sentences for single-verb validation |
| `synthetic_expected.tsv` | Expected TAMV labels (65 verb instances) |
| `phrase_test_sentences.txt` | 32 sentences for phrase-level validation |
| `phrase_test_expected.json` | Expected labels with span matching (43 verb phrases) |

### Current Results

**Single-verb validation** (`python -m tests.validate_tamv --no-show`):

| Metric | Value |
|--------|-------|
| Overall accuracy | 80.0% |
| Verb detection | 95.4% |
| Tense accuracy | 93.5% |
| Aspect accuracy | 98.4% |
| Mood accuracy | 88.7% |
| Voice accuracy | 100% |

**Phrase-level validation** (`python -m tests.validate_phrases --no-show`):

| Metric | Value |
|--------|-------|
| Labels correct | 87.5% |
| Phrases detected | 93.0% |
| Tense accuracy | 97.5% |
| Aspect accuracy | 97.5% |
| Mood accuracy | 90.0% |
| Voice accuracy | 100% |

### Known Limitations

Failures occur for:
- **Mandative subjunctive**: "I suggest that he leave" — requires semantic analysis
- **Negated imperatives**: "Don't forget the meeting" — spaCy parses "don't" as main verb
- **Inverted conditionals**: "Should you need help" — detected as MODAL not CONDITIONAL
- **Adjective vs passive**: "got tired" vs "got promoted" — requires lexical knowledge

## Technical Approach

### Pipeline Architecture

```
Raw Text → spaCy Parsing → Rule-Based TAMV Extraction → TAMV Labels
              │                       │
              ├─ Tokenization         ├─ Auxiliary Chain Collection
              ├─ POS Tagging          ├─ Tense Determination
              ├─ Dependency Parsing   ├─ Aspect Determination
              └─ Morphological        ├─ Mood Determination
                 Analysis             └─ Voice Determination
```

### spaCy Model

We use `en_core_web_sm`, which provides:
- **POS Tags**: Penn Treebank tags (VBD, VBG, VBN, VBP, VBZ, VB, MD)
- **Dependency Relations**: Universal Dependencies (aux, auxpass, nsubj, nsubjpass)
- **Morphological Features**: Tense, Number, Person

### Comparison with TMV-Annotator

Our extractor extends beyond TMV-annotator (Ramm et al. 2017):

| Feature | Our Extractor | TMV-Annotator |
|---------|---------------|---------------|
| **Parser** | spaCy (Python) | Mate (Java) |
| **Input** | Raw text | CoNLL format |
| **Mood values** | 7 (indicative, subjunctive, imperative, conditional, modal, infinitival, participle) | 2 (indicative, subjunctive) |
| **Non-finite support** | Yes (infinitives, participles) | No |
| **Python version** | Python 3 | Python 2 |

## TAMV Categories

### Tense
- `present` - "She writes novels."
- `past` - "She wrote a novel."
- `future` - "She will write a novel."
- `none` - Non-finite phrases ("to write", "having written")

### Aspect
- `simple` - "She writes." / "She wrote."
- `progressive` - "She is writing."
- `perfect` - "She has written."
- `perfect-progressive` - "She has been writing."

### Mood
- `indicative` - Default declarative statements
- `subjunctive` - would/could/might constructions
- `imperative` - "Write the report now."
- `conditional` - In conditional clauses with would/might
- `modal` - can/may/must/should constructions
- `infinitival` - "to write", "to have been told"
- `participle` - "Having been warned", "Written by hand"

### Voice
- `active` - "She wrote the novel."
- `passive` - "The novel was written."

Labels combine into strings like `past-simple-indicative-active` or `none-perfect-infinitival-passive`.

## Project Structure

```
tamv_analysis/
├── setup.py                   # Package installation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/                       # Main source code
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # CLI entry point (python -m src)
│   ├── tamv_extractor.py      # Rule-based TAMV extraction
│   ├── corpus_loaders.py      # Brown and CGA data loading
│   ├── tamv_aggregator.py     # Profile aggregation
│   ├── visualization.py       # Matplotlib visualizations
│   ├── extract_tamv.py        # Data extraction script
│   └── analyze_cga.py         # CGA speaker-level analysis
├── tests/                     # Validation tests
│   ├── validate_tamv.py       # Single-verb validation (65 cases)
│   └── validate_phrases.py    # Phrase-level validation (43 phrases)
├── data/                      # Test data files
│   ├── synthetic_sentences.txt    # Single-verb test sentences
│   ├── synthetic_expected.tsv     # Expected TAMV labels
│   ├── phrase_test_sentences.txt  # Phrase test sentences
│   ├── phrase_test_expected.json  # Expected phrase labels
│   ├── europarl_sentences.txt     # TMV-annotator comparison data
│   └── europarl_expected.tsv      # TMV-annotator expected output
├── docs/                      # Documentation
│   └── implementation_notes.md    # Detailed technical notes
├── output/                    # Generated results
└── tmv_annotator_reference/   # Reference implementation
    ├── tmv-annotator-tool/    # Original TMV-annotator code
    └── example-outputs/       # Mate-parsed Europarl samples
```

## Programmatic Usage

```python
from src import TAMVExtractor

# Extract TAMV from text
extractor = TAMVExtractor()
labels = extractor.extract_from_text("She had been writing the report.")
for label in labels:
    print(f"{label.verb_text}: {label.to_string()}")
# Output: writing: past-perfect_progressive-indicative-active

# Get TAMV profile for a document
profile = extractor.get_tamv_profile("The committee has approved the proposal.")
print(profile)
# Output: {'present-perfect-indicative-active': 1}

# Analyze modal constructions
labels = extractor.extract_from_text("She might have written a better ending.")
print(labels[0].mood.value)  # Output: subjunctive
```

## References

### Core References

- **Ramm et al. (2017)**. "Annotating tense, mood and voice for English, French and German." *Proceedings of ACL 2017*.
- **Biber et al. (1999)**. *Longman Grammar of Spoken and Written English*. Longman.
- **Palmer, F.R. (2001)**. *Mood and Modality* (2nd ed.). Cambridge University Press.

### Corpora

- **Francis, W.N. & Kucera, H. (1979)**. Brown Corpus Manual. Brown University.
- **Zhang et al. (2018)**. "Conversations Gone Awry: Detecting Early Signs of Conversational Failure." *Proceedings of ACL 2018*.

### NLP Tools

- **spaCy** (Honnibal & Montani, 2017). Industrial-strength NLP. [spacy.io](https://spacy.io)

## License

MIT License
