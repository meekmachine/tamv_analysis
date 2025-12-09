# TAMV Extraction: Implementation Notes

## Overview

This document describes the design decisions, theoretical background, and validation approach for our TAMV (Tense, Aspect, Mood, Voice) extraction system. It explains why we reimplemented TMV-annotator's logic for Universal Dependencies rather than using the original tool directly.

---

## 1. Background: Two Parsing Traditions

### 1.1 Chomskyan Phrase Structure (Government & Binding)

- **Origin:** Chomsky (1957, 1981)
- **Core idea:** Sentences are hierarchical phrase structures (S → NP VP)
- **Auxiliary treatment:** The finite verb (carrying tense/agreement) "governs" the clause
- **Influence on NLP:** Penn Treebank (1993), PropBank (2005), CoNLL-2008 shared task

### 1.2 Dependency Grammar (Tesnière)

- **Origin:** Tesnière (1959), *Éléments de syntaxe structurale*
- **Core idea:** Sentences are words connected by directed dependencies; no intermediate phrase nodes
- **Auxiliary treatment:** The lexical/content verb is the head; auxiliaries depend on it
- **Influence on NLP:** Universal Dependencies (2014), spaCy, Stanza

### 1.3 Why This Matters

TMV-annotator (Ramm et al. 2017) was built for **Mate parser**, which follows the CoNLL-2008 scheme derived from phrase-structure conventions. Modern Python NLP tools (spaCy, Stanza) follow **Universal Dependencies**, which uses the opposite head direction for auxiliary chains.

---

## 2. Parser Comparison: Mate vs. spaCy

### 2.1 Example Sentence: *"She has been writing"*

**Mate parser (CoNLL-2008 style):**
```
ID  TOKEN    LEMMA  POS  HEAD  DEPREL
1   She      she    PRP  2     SBJ
2   has      have   VBZ  0     ROOT    ← finite verb is ROOT
3   been     be     VBN  2     VC      ← chains DOWN via VC
4   writing  write  VBG  3     VC      ← chains DOWN via VC
```

**spaCy (Universal Dependencies):**
```
ID  TOKEN    LEMMA  POS   HEAD  DEPREL
1   She      she    PRON  4     nsubj
2   has      have   AUX   4     aux     ← points UP to main verb
3   been     be     AUX   4     aux     ← points UP to main verb
4   writing  write  VERB  0     ROOT    ← main verb is ROOT
```

### 2.2 Key Structural Differences

| Feature | Mate (CoNLL-2008) | spaCy (UD) |
|---------|-------------------|------------|
| **ROOT** | Finite verb ("has") | Main verb ("writing") |
| **Head direction** | Finite → auxiliaries → main (downward) | Main ← auxiliaries (upward) |
| **Verb complex label** | `VC` (verb complex) | `aux` / `auxpass` |
| **Subject attachment** | To finite verb | To main verb |

### 2.3 Theoretical Rationale

**CoNLL-2008 (Mate):** Follows Government & Binding theory. The inflected element carries grammatical features (tense, agreement) and syntactically governs the clause.

**Universal Dependencies:** Follows Tesnière's principle that the **content word** (carrying semantic meaning) should be the syntactic head. Auxiliaries are "functional" elements that modify the main verb.

### 2.4 Visual Comparison: Step-by-Step Dependency Building

The following jsSyntaxTree bracket notation diagrams show how each parsing tradition incrementally builds the dependency structure for *"She has been writing"*.

**Stanford Dependencies (SD) — Finite verb as root, VC chain downward:**

```
Step 1: [ROOT has]

Step 2: [ROOT [SBJ She] has]

Step 3: [ROOT [SBJ She] has [VC been]]

Step 4: [ROOT [SBJ She] has [VC been [VC writing]]]
```

The finite verb "has" is established as ROOT. The subject attaches to the finite verb. Auxiliaries chain downward via `VC` (verb complex) relations, with the main lexical verb "writing" at the bottom of the chain.

**Universal Dependencies (UD) — Main verb as root, aux points upward:**

```
Step 1: [ROOT writing]

Step 2: [ROOT [nsubj She] writing]

Step 3: [ROOT [nsubj She] [aux has] writing]

Step 4: [ROOT [nsubj She] [aux has] [aux been] writing]
```

The main lexical verb "writing" is established as ROOT. Both the subject and auxiliaries attach directly to the main verb via `nsubj` and `aux` relations. All dependencies point upward to the content word.

**Key insight:** In SD, to find the main verb you must traverse *down* the VC chain. In UD, the main verb is already the ROOT — auxiliaries are its children.

### 2.5 Foundational References

| Framework | Key Paper | Description |
|-----------|-----------|-------------|
| **Stanford Dependencies** | de Marneffe & Manning (2008). "The Stanford Typed Dependencies Representation." *COLING Workshop on Cross-Framework and Cross-Domain Parser Evaluation*. | Defines SD scheme used by CoNLL-2008 and Mate parser. |
| **Universal Dependencies** | Nivre et al. (2016). "Universal Dependencies v1: A Multilingual Treebank Collection." *LREC 2016*. | Defines UD scheme used by spaCy and modern NLP tools. |

---

## 3. TMV-Annotator Algorithm

### 3.1 Input Requirements

TMV-annotator expects CoNLL format with:
- Column 6: POS tag (VBZ, VBN, VBG, MD, etc.)
- Column 10: HEAD (1-indexed)
- Column 12: DEPREL (must include `VC`, `IM` for verb complexes)

### 3.2 Algorithm Steps

1. **Find finite verbs:** Look for ROOT or finite POS tags (VBD, VBZ, VBP, MD, TO)
2. **Build verb chain:** Traverse `VC` (verb complex) and `IM` (infinitive marker) dependencies downward
3. **Set boolean flags:** Based on what's in the chain:
   - `aux_pres`, `aux_past` (have/has/had)
   - `aux_be_pres`, `aux_be_past` (am/is/are/was/were)
   - `aux_part` (been)
   - `ger` (VBG main verb)
   - `part` (VBN main verb)
   - `fut_mod` (will/shall)
   - `subj_mod` (would/could/should/might)
4. **Match pattern:** Large if/elif cascade maps flag combinations to output labels

### 3.3 Output Format

TMV-annotator produces labels like:
- `presPerf` (present perfect)
- `pastPerfProg` (past perfect progressive)
- `futureI` (simple future)
- `condII` (conditional type II)

**Note:** Aspect is encoded implicitly within the tense label, not as a separate dimension.

### 3.4 Why spaCy Input Fails

Running TMV-annotator on spaCy-parsed CoNLL produces **0 results** because:
1. spaCy uses `aux` label, not `VC`
2. TMV-annotator's `extractVerbalDepDictEN` function explicitly checks for `VC`:
   ```python
   elif curr_pos in ["VBN", "VB"] and curr_rel in ["VC", "IM", "CONJ", "PRD"]:
   ```
3. Without `VC` labels, no verb complexes are found

---

## 4. Our TAMV Extractor Design

### 4.1 Design Principles

1. **UD-native:** Works with spaCy's dependency structure directly
2. **Explicit dimensions:** Tense, Aspect, Mood, Voice as separate enums
3. **End-to-end:** Raw text → TAMV labels (no pre-parsing required)
4. **Terminology alignment:** Uses TMV-annotator's definitions for comparability

### 4.2 Algorithm Steps

1. **Find main verbs:** Tokens with `pos_ == "VERB"` or `AUX` at ROOT
2. **Collect auxiliaries:** Children with `dep_ == "aux"` or `dep_ == "auxpass"`
3. **Determine each dimension separately:**
   - **Tense:** Check for will/shall (future), morphology of finite verb (present/past)
   - **Aspect:** Check for have+VBN (perfect), be+VBG (progressive)
   - **Mood:** Check for modals, imperatives, conditionals, subjunctive triggers
   - **Voice:** Check for be/get+VBN (passive), auxpass dependency

### 4.3 Algorithm Statechart

The following statechart shows how our extractor iterates over UD-parsed tokens, identifies verbal complexes, and determines TAMV labels.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TAMV EXTRACTION ALGORITHM                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │    START     │
                              └──────┬───────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Parse text with spaCy │
                         │  (get UD dependencies) │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Initialize:           │
                         │ • labels = []         │
                         │ • processed_verbs = {}│
                         └───────────┬───────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────────┐
              │           FOR EACH TOKEN IN DOC             │◄────────────┐
              └──────────────────────┬──────────────────────┘             │
                                     │                                    │
                                     ▼                                    │
                        ┌────────────────────────┐                        │
                        │ token.i in processed?  │───YES──────────────────┤
                        └────────────┬───────────┘                        │
                                     │ NO                                 │
                                     ▼                                    │
                   ┌─────────────────────────────────┐                    │
                   │      IS MAIN VERB?              │                    │
                   │  • pos_ == "VERB"               │                    │
                   │  • OR (pos_ == "AUX" AND        │                    │
                   │        dep_ == "ROOT")          │                    │
                   └────────────────┬────────────────┘                    │
                                    │                                     │
                         NO ────────┴────────── YES                       │
                         │                      │                         │
                         │                      ▼                         │
                         │    ┌──────────────────────────────────┐        │
                         │    │     COLLECT AUXILIARIES          │        │
                         │    │  For each child of main_verb:    │        │
                         │    │    if dep_ in {aux, auxpass}     │        │
                         │    │      → add to aux_chain          │        │
                         │    │  Sort aux_chain by position      │        │
                         │    └────────────────┬─────────────────┘        │
                         │                     │                          │
                         │                     ▼                          │
                         │    ┌──────────────────────────────────┐        │
                         │    │   MARK AS PROCESSED              │        │
                         │    │  • processed_verbs.add(main.i)   │        │
                         │    │  • for aux: processed.add(aux.i) │        │
                         │    └────────────────┬─────────────────┘        │
                         │                     │                          │
                         │                     ▼                          │
                         │    ╔══════════════════════════════════╗        │
                         │    ║   DETERMINE TAMV FEATURES        ║        │
                         │    ╚════════════════╤═════════════════╝        │
                         │                     │                          │
                         │         ┌───────────┴───────────┐              │
                         │         ▼           ▼           ▼              │
                         │    ┌─────────┐ ┌─────────┐ ┌─────────┐         │
                         │    │ TENSE   │ │ ASPECT  │ │  MOOD   │         │
                         │    └────┬────┘ └────┬────┘ └────┬────┘         │
                         │         │           │           │              │
                         │         ▼           ▼           ▼              │
                         │    ┌─────────┐ ┌─────────┐ ┌─────────┐         │
                         │    │ VOICE   │ │         │ │         │         │
                         │    └────┬────┘ └─────────┘ └─────────┘         │
                         │         │                                      │
                         │         └───────────┬───────────┘              │
                         │                     │                          │
                         │                     ▼                          │
                         │    ┌──────────────────────────────────┐        │
                         │    │  CREATE TAMVLabel                │        │
                         │    │  labels.append(label)            │        │
                         │    └────────────────┬─────────────────┘        │
                         │                     │                          │
                         └─────────────────────┴──────────────────────────┘
                                               │
                                               ▼ (no more tokens)
                                     ┌───────────────────┐
                                     │   RETURN labels   │
                                     └───────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                        FEATURE DETERMINATION SUB-CHARTS
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETERMINE TENSE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────────┐
                    │  Check aux_chain for modals   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │ Any aux.lemma in {will, shall}?         │
              └──────────────────┬──────────────────────┘
                                 │
                      YES ───────┴─────── NO
                       │                   │
                       ▼                   ▼
              ┌─────────────┐    ┌──────────────────────┐
              │   FUTURE    │    │ Get finite verb:     │
              └─────────────┘    │ aux_chain[0] or main │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │ Check morph.Tense    │
                                 └──────────┬───────────┘
                                            │
                         ┌──────────────────┼──────────────────┐
                         │                  │                  │
                         ▼                  ▼                  ▼
                   "Past" in           "Pres" in         (not found)
                     tense              tense                 │
                         │                  │                  │
                         ▼                  ▼                  ▼
                   ┌─────────┐       ┌─────────────┐   ┌──────────────┐
                   │  PAST   │       │   PRESENT   │   │ Fallback to  │
                   └─────────┘       └─────────────┘   │ POS tag      │
                                                       └──────┬───────┘
                                                              │
                                     ┌────────────────────────┼────────┐
                                     │                        │        │
                                     ▼                        ▼        ▼
                               VBD, VBN               VBZ, VBP, VB    VBG
                                     │                        │        │
                                     ▼                        ▼        ▼
                               ┌─────────┐           ┌─────────────┐  ┌─────────────┐
                               │  PAST   │           │   PRESENT   │  │   PRESENT   │
                               └─────────┘           └─────────────┘  └─────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETERMINE ASPECT                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────────┐
                    │  Initialize flags:            │
                    │  has_perfect = False          │
                    │  has_progressive = False      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │ Any aux.lemma in {have, has, had}?      │──YES──► has_perfect = True
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────────┐
              │ main_verb.tag_ == "VBG" AND             │
              │ any aux.lemma == "be"?                  │──YES──► has_progressive = True
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────────┐
              │ Any aux.text == "being"?                │──YES──► has_progressive = True
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
                    ┌────────────┴────────────┐
                    │                         │
         has_perfect AND           has_perfect only    has_progressive only    neither
         has_progressive                  │                    │                  │
                    │                     │                    │                  │
                    ▼                     ▼                    ▼                  ▼
         ┌──────────────────┐      ┌───────────┐      ┌─────────────┐      ┌────────┐
         │ PERFECT-         │      │  PERFECT  │      │ PROGRESSIVE │      │ SIMPLE │
         │ PROGRESSIVE      │      └───────────┘      └─────────────┘      └────────┘
         └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETERMINE MOOD                                     │
└─────────────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────┐
              │ Any aux.lemma in MODAL_VERBS?           │
              └──────────────────┬──────────────────────┘
                                 │
                      YES ───────┴─────── NO
                       │                   │
                       ▼                   │
       ┌───────────────────────────┐       │
       │ Modal in {would, could,   │       │
       │         should, might}?   │       │
       └─────────────┬─────────────┘       │
                     │                     │
          YES ───────┴─────── NO           │
           │                  │            │
           ▼                  │            ▼
    ┌─────────────┐           │  ┌──────────────────────────────┐
    │ SUBJUNCTIVE │           │  │ tag_ == "VB" AND no aux?     │
    └─────────────┘           │  └──────────────┬───────────────┘
                              │                 │
                              │      YES ───────┴─────── NO
                              │       │                   │
                              │       ▼                   │
                              │  ┌────────────────────┐   │
                              │  │ Has nsubj child?   │   │
                              │  └─────────┬──────────┘   │
                              │            │              │
                              │   NO ──────┴────── YES    │
                              │    │                │     │
                              │    ▼                │     │
                              │ ┌───────────┐       │     │
                              │ │ IMPERATIVE│       │     │
                              │ └───────────┘       │     │
                              │                     │     │
                              │                     ▼     ▼
                              │           ┌────────────────────────┐
                              │           │ In conditional context?│
                              │           │ (if, unless, etc.)     │
                              │           └─────────┬──────────────┘
                              │                     │
                              │          YES ───────┴─────── NO
                              │           │                   │
                              │           ▼                   ▼
                              │    ┌─────────────┐     ┌─────────────┐
                              └───►│ CONDITIONAL │     │  INDICATIVE │
                                   └─────────────┘     └─────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETERMINE VOICE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────┐
              │ main_verb.tag_ == "VBN" (past part.)?   │
              └──────────────────┬──────────────────────┘
                                 │
                      YES ───────┴─────── NO
                       │                   │
                       ▼                   │
       ┌───────────────────────────┐       │
       │ Any aux.lemma in {be,get}?│       │
       └─────────────┬─────────────┘       │
                     │                     │
          YES ───────┴─────── NO           │
           │                  │            │
           ▼                  │            ▼
    ┌─────────┐               │  ┌──────────────────────────────┐
    │ PASSIVE │               │  │ Any aux.dep_ == "auxpass"?   │
    └─────────┘               │  └──────────────┬───────────────┘
                              │                 │
                              │      YES ───────┴─────── NO
                              │       │                   │
                              │       ▼                   │
                              │  ┌─────────┐              │
                              │  │ PASSIVE │              │
                              │  └─────────┘              ▼
                              │              ┌────────────────────────┐
                              │              │ main has nsubjpass     │
                              │              │ child dependency?      │
                              │              └─────────┬──────────────┘
                              │                        │
                              │             YES ───────┴─────── NO
                              │              │                   │
                              │              ▼                   ▼
                              │        ┌─────────┐        ┌────────┐
                              └───────►│ PASSIVE │        │ ACTIVE │
                                       └─────────┘        └────────┘
```

### 4.4 Output Format

Our extractor produces labels like:
- `present-perfect-indicative-active`
- `past-perfect_progressive-indicative-active`
- `future-simple-indicative-passive`

### 4.5 Key Code Locations

- Main extractor: `src/tamv_extractor.py`
- Validation tests: `validate_tamv.py`
- Test cases: 42 examples from Biber et al. (1999) and Palmer (2001)

---

## 5. Label Mapping: TMV-Annotator → Our TAMV

To compare outputs, we need to map TMV-annotator's compound labels to our four-dimensional labels:

| TMV-Annotator Label | Tense | Aspect | Mood | Voice | Notes |
|---------------------|-------|--------|------|-------|-------|
| `pres` | present | simple | indicative | active | |
| `past` | past | simple | indicative | active | |
| `futureI` | future | simple | indicative | active | will + VB |
| `futureII` | future | perfect | indicative | active | will have + VBN |
| `presPerf` | present | perfect | indicative | active | have/has + VBN |
| `pastPerf` | past | perfect | indicative | active | had + VBN |
| `presProg` | present | progressive | indicative | active | am/is/are + VBG |
| `pastProg` | past | progressive | indicative | active | was/were + VBG |
| `presPerfProg` | present | perfect-progressive | indicative | active | have been + VBG |
| `pastPerfProg` | past | perfect-progressive | indicative | active | had been + VBG |
| `futureIProg` | future | progressive | indicative | active | will be + VBG |
| `futureIIProg` | future | perfect-progressive | indicative | active | will have been + VBG |
| `condI` | present | simple | subjunctive | active | would + VB |
| `condII` | present | perfect | subjunctive | active | would have + VBN |
| `condIProg` | present | progressive | subjunctive | active | would be + VBG |
| `condIIProg` | present | perfect-progressive | subjunctive | active | would have been + VBG |

**Voice handling:** TMV-annotator outputs voice as a separate column. We combine it with the above.

**Mood mapping:**
- TMV-annotator: `indicative` for will/shall/can/may/must
- TMV-annotator: `subjunctive` for would/could/should/might
- We follow the same convention

---

## 6. Validation Approach

### 6.1 Ground Truth Source

We validate against **linguistic literature**, not against TMV-annotator output:

- **Biber et al. (1999)** *Longman Grammar of Spoken and Written English* — 40M word corpus, TAMV frequency norms
- **Palmer (2001)** *Mood and Modality* — theoretical framework for mood

### 6.2 Test Cases

42 hand-crafted test cases covering:
- All tense values (present, past, future)
- All aspect values (simple, progressive, perfect, perfect-progressive)
- All mood values (indicative, subjunctive, imperative, conditional)
- Both voice values (active, passive)
- Complex combinations (e.g., "might have been being watched")
- Edge cases (existential there, raising verbs, get-passive)

### 6.3 Results

| Dimension | Accuracy |
|-----------|----------|
| Tense | 97.6% |
| Aspect | 100% |
| Mood | 90.5% |
| Voice | 100% |
| **Overall** | **90.5%** (38/42) |

### 6.4 Failure Cases

| Case | Expected | Issue |
|------|----------|-------|
| "Don't forget the meeting." | imperative | spaCy parses "don't" as main verb |
| "I suggest that he leave." (×3) | subjunctive | Mandative subjunctive is rare; TMV-annotator also misses this |

---

## 7. Comparison Test Design

### 7.1 Goal

Measure alignment between our spaCy-based extractor and TMV-annotator on the same input sentences.

### 7.2 Process

1. **Input:** Mate-parsed Europarl sentences (CoNLL format)
2. **Strip tags:** Extract raw tokens, reconstruct sentences
3. **Parse with spaCy:** Get UD-style dependencies
4. **Extract TAMV:** Run our extractor
5. **Map labels:** Convert TMV-annotator output to our 4-dimensional format
6. **Compare:** Calculate alignment percentage per dimension and overall

### 7.3 Expected Outcomes

- **High alignment:** Tense, Voice (clear morphological signals)
- **Moderate alignment:** Aspect (both systems use similar rules)
- **Variable alignment:** Mood (depends on parser's ability to identify contexts)

### 7.4 Caveats

1. **Tokenization differences:** spaCy may tokenize differently than Mate
2. **Sentence boundary differences:** May affect verb count
3. **Parser accuracy:** Both parsers make errors; we're comparing system outputs, not gold standard

### 7.5 Alignment Test Results (2025-12-07)

**Test corpus:** 25 Europarl sentences, 35 verb complexes identified by TMV-annotator

| Metric | Result |
|--------|--------|
| **TMV-annotator verb complexes** | 35 |
| **Verb complexes we matched** | 27 (77.1%) |
| **Tense alignment** | 100% (27/27) |
| **Aspect alignment** | 100% (27/27) |
| **Mood alignment** | 100% (27/27) |
| **Voice alignment** | 96.3% (26/27) |
| **Full TAMV match** | **96.3%** (26/27) |

**Interpretation:**
- 77% verb detection rate — 8 missed verbs due to tokenization differences and non-finite verb handling
- 96.3% alignment on matched verbs — when we find the same verb, we almost always agree
- One voice mismatch was actually **our extractor being correct** (see Section 7.6)

### 7.6 Case Study: Copular vs. Passive (Our Advantage)

**Sentence:** *"It has never been easy to have a rational conversation about the value of gold."*

| System | Voice Label | Correct? |
|--------|-------------|----------|
| TMV-annotator | passive | **NO** |
| Our extractor | active | **YES** |

**Analysis:**

TMV-annotator sees `has + been (VBN)` and pattern-matches to passive:
```python
# From TMV-EN.py lines 367-370
elif vp['aux_pres'] and vp['aux_part'] and (vp['part'] or vp['aux_part']):
    voice = "passive"
```

But this is a **copular construction**, not a passive:
```
It has been easy
        ↓
    [copula "be" + adjective complement]
```

**The linguistic distinction:**

| Construction | Example | Structure | Voice |
|--------------|---------|-----------|-------|
| **Passive** | "The book has been written" | has + been + **VBN** (past participle) | Passive |
| **Copular** | "It has been easy" | has + been + **ADJ** (adjective) | Active |

**spaCy parse shows this clearly:**
```
Token        POS    TAG    DEP          HEAD
It           PRON   PRP    nsubj        been
has          AUX    VBZ    aux          been
never        ADV    RB     neg          been
been         AUX    VBN    ROOT         been
easy         ADJ    JJ     acomp        been      ← adjective complement!
```

The `acomp` (adjectival complement) dependency reveals this is copular, not passive. TMV-annotator's pattern-matching approach cannot make this distinction because it only looks at POS tags, not the complement type.

**Conclusion:** Our UD-based approach provides richer structural information that enables more accurate voice detection in copular constructions.

---

## 8. Known Limitations

### 8.1 Our Extractor's Limitations

| Issue | Example | Cause |
|-------|---------|-------|
| Negative imperatives | "Don't forget the meeting." | spaCy parses "don't" as main verb |
| Mandative subjunctive | "I suggest that he leave." | Rare pattern; requires semantic analysis |
| Coordinated verbs | "She wrote and edited the report." | May miss second verb in coordination |

### 8.2 TMV-Annotator's Limitations

| Issue | Example | Cause |
|-------|---------|-------|
| Copular constructions | "It has been easy." | Pattern-matches to passive incorrectly |
| Mandative subjunctive | "I suggest that he leave." | Same as our limitation |
| Requires Mate parser | N/A | Cannot use modern Python parsers |
| Python 2 only | N/A | `.keys()[0]` syntax fails in Python 3 |

### 8.3 Shared Limitations (Hard Problems)

Both systems struggle with:
- **Mandative subjunctive** — requires understanding that "suggest/recommend/demand" trigger subjunctive in that-clauses
- **Ambiguous participles** — "The door was closed" (passive action vs. stative result)
- **Ellipsis** — "She can and will help" (modal scope ambiguity)

---

## 9. References

### Foundational Linguistics

- Comrie, B. (1976). *Aspect: An Introduction to the Study of Verbal Aspect and Related Problems*. Cambridge University Press.
- Comrie, B. (1985). *Tense*. Cambridge University Press.
- Smith, C. S. (1997). *The Parameter of Aspect* (2nd ed.). Kluwer Academic Publishers.
- Palmer, F. R. (2001). *Mood and Modality* (2nd ed.). Cambridge University Press.
- Tesnière, L. (1959). *Éléments de syntaxe structurale*. Klincksieck.

### Grammar References

- Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). *Longman Grammar of Spoken and Written English*. Longman.

### Computational Linguistics

- Bohnet, B. (2010). Very High Accuracy and Fast Dependency Parsing is not a Contradiction. *COLING 2010*.
- de Marneffe, M.-C., et al. (2014). Universal Stanford Dependencies: A cross-linguistic typology. *LREC 2014*.
- Ramm, A., Loáiciga, S., Friedrich, A., & Fraser, A. (2017). Annotating tense, mood and voice for English, French and German. *ACL 2017*.
- Surdeanu, M., et al. (2008). The CoNLL-2008 Shared Task on Joint Parsing of Syntactic and Semantic Dependencies. *CoNLL 2008*.

### NLP Tools

- **Mate parser:** https://code.google.com/archive/p/mate-tools/
- **spaCy:** https://spacy.io/ (uses Universal Dependencies)
- **Universal Dependencies:** https://universaldependencies.org/

---

## 10. Project Structure and File Documentation

### 10.1 Core Library (`src/`)

All code is consolidated in the `src/` package. Run commands using `python -m src <command>`.

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/__init__.py` | Package exports | Re-exports all public classes |
| `src/__main__.py` | CLI entry point | `python -m src main/extract/validate` |
| `src/tamv_extractor.py` | **Core TAMV extraction** | `TAMVExtractor`, `TAMVLabel`, `Tense`, `Aspect`, `Mood`, `Voice` enums |
| `src/tamv_aggregator.py` | Profile aggregation | `TAMVAggregator`, `TAMVProfile`, `ConversationTAMVAnalyzer` |
| `src/corpus_loaders.py` | Data loading | `BrownCorpusLoader`, `CGALoader`, `Document`, `Conversation` |
| `src/visualization.py` | Plotting & PCA | `TAMVVisualizer` — PCA projections, heatmaps, bar charts |
| `src/toxicity_analysis.py` | Toxicity correlation | `ToxicityAnalyzer` — correlates TAMV with toxicity scores |
| `src/main.py` | **Analysis pipeline** | `analyze_brown()`, `analyze_cga()`, `analyze_combined()` |
| `src/extract_tamv.py` | **Data extraction** | `run_extraction()` — exports TAMV to CSV |
| `src/validate_tamv.py` | **Validation tests** | 42 test cases from Biber/Palmer |

**Usage:**
```bash
python -m src main --corpus brown       # Full analysis pipeline
python -m src extract --corpus cga      # Extract TAMV to CSV
python -m src validate                  # Run validation (90.5% accuracy)
```

### 10.2 Root Files

| File | Purpose |
|------|---------|
| `setup.py` | Package installation (`pip install -e .`) |

### 10.3 Analysis Scripts (`scripts/`)

Scripts for specific experiments and comparisons.

| File | Purpose | Status |
|------|---------|--------|
| `scripts/tmv_alignment_test.py` | Compare our extractor to TMV-annotator output | ✅ Active — produces 96.3% alignment |
| `scripts/conll_spacy_comparison.py` | Compare Mate vs spaCy CoNLL output | ✅ Active — tokenization/parse comparison |

### 10.4 Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `docs/implementation_notes.md` | This document — design decisions, validation, comparison |
| `docs/presentation.md` | Slides for project presentation |

### 10.5 Output (`output/`)

Generated artifacts from running scripts.

| File | Purpose |
|------|---------|
| `output/tmv_alignment_report.txt` | Detailed alignment test results |
| `output/*.csv` | Extracted TAMV profiles |
| `output/*.png` | Visualization plots |

### 10.6 Reference Data (`tmv_annotator_reference/`)

External reference materials for comparison.

| Path | Purpose |
|------|---------|
| `tmv_annotator_reference/tmv-annotator-tool/TMV-EN.py` | Original TMV-annotator code (Python 2) |
| `tmv_annotator_reference/example-outputs/en.parsed` | Mate-parsed Europarl sample (CoNLL format) |
| `tmv_annotator_reference/tmv-annotator-tool/output/en.parsed.verbs` | TMV-annotator reference output |
| `tmv_annotator_reference/europarl/README.md` | Link to full annotated Europarl corpus |

---

## 11. Summary

Our spaCy-based TAMV extractor achieves:

1. **80.0% accuracy** on 65 test cases from Brown corpus and user-provided sentences
2. **96.3% alignment** with TMV-annotator on matched verbs from Europarl
3. **Better handling** of copular constructions than TMV-annotator
4. **Extended mood detection** — MODAL mood for deontic modals (can/may/must)
5. **Non-finite phrase support** — INFINITIVAL and PARTICIPLE moods for complex constructions
6. **Modern tooling** — works with Python 3, spaCy, raw text input

The reimplementation was necessary because:
- TMV-annotator requires Mate parser (Java) with CoNLL-2008 `VC` labels
- Modern parsers (spaCy, Stanza) use Universal Dependencies with different head direction
- Converting UD → CoNLL-2008 would require non-trivial structural transformations

Our approach extends beyond TMV-annotator (Ramm et al. 2017) to include modal mood detection and non-finite phrases, similar to Doenicke's thesis work.

---

*Last updated: 2025-12-08*

---

## 12. TMV-Annotator Mood Handling: No MODAL Category

**Important:** TMV-annotator (Ramm et al. 2017) does **not** have a separate MODAL mood category.

### 12.1 How TMV-Annotator Handles Modals

Looking at the TMV-annotator output (`en.parsed.verbs`), the mood column only contains two values:
- `indicative`
- `subjunctive`

**Modal verb treatment:**
| Modal | TMV-annotator Tense | TMV-annotator Mood | Our Current Output |
|-------|--------------------|--------------------|-------------------|
| can/may/must | `pres` | `indicative` | indicative |
| will/shall | `futureI` | `indicative` | indicative |
| would/could/should/might | `condI` or `condII` | `subjunctive` | subjunctive |

### 12.2 Evidence from TMV-Annotator Output

```
Line 5:  "would n't know"     → condI    subjunctive   (would = subjunctive mood)
Line 11: "could be headed"    → condI    subjunctive   (could = subjunctive mood)
Line 17: "ca n't increase"    → pres     indicative    (can = NO mood change)
Line 25: "could justify"      → condI    subjunctive   (could = subjunctive mood)
Line 35: "will have lived"    → futureII indicative    (will = NO mood change)
```

### 12.3 Implications for Our Extractor

Our current implementation follows TMV-annotator's approach:
- `can/may/must/will/shall` → do NOT change mood (remains INDICATIVE)
- `would/could/should/might` → set mood to SUBJUNCTIVE

**If we want to extend beyond TMV-annotator** (like Doenicke's thesis), we would need to:
1. Add `Mood.MODAL` to the enum
2. Return MODAL for `can/may/must` constructions
3. Distinguish CONDITIONAL from SUBJUNCTIVE for `would/could` based on context

### 12.4 Test Data Alignment Decision

Our test data currently expects `mood_modal` for sentences like "She can write quickly" and "The report must be submitted".

**Options:**
1. **Stay TMV-compatible:** Change test data to expect INDICATIVE for these cases
2. **Extend beyond TMV:** Add MODAL mood detection to the extractor

This is a design decision about whether we want strict TMV-annotator compatibility or richer mood distinctions.

### 12.5 Implementation (2025-12-08): Extended Modal Detection

We chose **Option 2: Extend beyond TMV** to provide richer mood distinctions.

**Changes made to `_determine_mood()` in `tamv_extractor.py`:**

```python
# Extended mood detection logic:
# - would/might in conditional context → CONDITIONAL
# - would/could/might outside conditional → SUBJUNCTIVE (epistemic possibility)
# - can/may/must/should → MODAL (deontic/dynamic modality)
# - will/shall → indicative (future tense marker, not mood)
```

**Results after fix:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Overall accuracy | 70.8% | 76.9% | +6.1% |
| Mood accuracy | 79.0% | 88.7% | +9.7% |
| mood_modal category | 0% | 67% | +67% |
| mood_conditional category | 20% | 40% | +20% |

**Remaining mood issues:**
- `mood_subjunctive` at 0% — mandative subjunctive not detected ("I suggest that he leave")
- `could` in questions treated as SUBJUNCTIVE, test expects MODAL
- Inverted conditionals ("Should you need...") detected as MODAL, test expects CONDITIONAL

---

## 13. Non-Finite Phrase Support (Implemented 2025-12-08)

Our extractor now supports **non-finite verb phrases** in addition to finite phrases, extending beyond TMV-annotator to match Doenicke's thesis approach.

### 13.1 What Was Added

**New Mood values in `tamv_extractor.py`:**
```python
class Mood(Enum):
    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    IMPERATIVE = "imperative"
    CONDITIONAL = "conditional"
    MODAL = "modal"
    INFINITIVAL = "infinitival"    # "to have been told"
    PARTICIPLE = "participle"       # "Having been warned"
```

**New detection methods:**
- `_is_infinitive(main_verb, aux_chain)` — checks for "to" particle or xcomp dependency
- `_is_participle(main_verb, aux_chain)` — checks for VBG/VBN in advcl/ccomp without finite auxiliary
- `_has_finite_auxiliary(aux_chain)` — helper to check if auxiliary chain contains finite verb

**Modified extraction logic in `_extract_verbal_complex()`:**
```python
is_infinitive = self._is_infinitive(main_verb, aux_chain)
is_participle = self._is_participle(main_verb, aux_chain)

if is_infinitive:
    tense = Tense.NONE
    mood = Mood.INFINITIVAL
elif is_participle:
    tense = Tense.NONE
    mood = Mood.PARTICIPLE
else:
    tense = self._determine_tense(main_verb, aux_chain)
    mood = self._determine_mood(main_verb, aux_chain, doc)
```

### 13.2 Examples Handled

| Sentence | Verb | TAMV Output |
|----------|------|-------------|
| "He seems **to have been promoted**" | promoted | NONE-PERFECT-INFINITIVAL-PASSIVE |
| "**Having been warned**, they were careful" | warned | NONE-SIMPLE-PARTICIPLE-PASSIVE |
| "He was supposed **to have been told**" | told | NONE-PERFECT-INFINITIVAL-PASSIVE |

### 13.3 Results After Implementation

**Final validation results (80.0% accuracy):**

| Metric | Value |
|--------|-------|
| Overall accuracy | 80.0% |
| Verb detection | 95.4% |
| Tense accuracy | 93.5% |
| Aspect accuracy | 98.4% |
| Mood accuracy | 88.7% |
| Voice accuracy | 100% |
| Complex category | 85% |

### 13.4 Remaining Limitations

Failures still occur for:

| Issue | Example | Root Cause |
|-------|---------|------------|
| Mandative subjunctive | "I suggest that he leave" | Requires semantic analysis of trigger verbs |
| Negated imperatives | "Don't forget the meeting" | spaCy parses "don't" as main verb |
| `were` in subjunctive | "If it were true" | `were` not detected as main verb |
| `willing` as verb | "I'm willing to..." | spaCy tags as adjective, not verb |
| Aspect in participles | "Having been warned" → SIMPLE | Should be PERFECT (has "been") |

These edge cases would require more sophisticated detection logic beyond what TMV-annotator provides.
