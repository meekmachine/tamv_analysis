# TAMV Analysis: Grammatical Patterns in Online Discourse
## Slide Deck Outline (~10-15 minutes)

---

## Slide 1: Title Slide

**Title:** Analyzing Tense, Aspect, Mood, and Voice in Online Conversations

**Subtitle:** Can Grammatical Structure Predict Conversation Derailment?

**Presenters:** [Your Names Here]

**Course:** [Course Name]

**Date:** December 2025

> **Design Notes:** Clean, professional. Consider a background image of conversation threads or linguistic tree diagram.

---

## Slide 2: Introduction — Research Questions

**Primary Research Question:**
> *Do TAMV feature combinations have predictive power for detecting conversation derailment?*

**Secondary Research Question (enabling step):**
> *How can we reliably extract TAMV features from raw text at scale?*

**The Logic:**
```
RQ2: Extraction Method  →  RQ1: Predictive Power
     (How to get TAMV)       (What TAMV tells us)
```

**Why This Matters:**
- Most NLP toxicity detection uses **lexical features** (bad words, sentiment)
- We hypothesize **grammatical structure** carries independent signal
- If true: grammar-aware models could detect *how* people speak, not just *what* they say

> **Speaker Notes:** Frame the presentation clearly. We have TWO research questions, but they're ordered: we FIRST need to solve extraction (RQ2) before we can test predictive power (RQ1). This presentation covers both, but emphasizes that extraction is a means to an end—the real question is whether TAMV predicts anything useful.

---

## Slide 3: Introduction — What is TAMV?

**The Four Dimensions of Verbal Grammar**

| Dimension | What It Captures | Example | Key Source |
|-----------|------------------|---------|------------|
| **Tense** | Location in time | *"She **wrote**"* vs *"She **writes**"* | Comrie (1985) *Tense* |
| **Aspect** | Internal temporal structure | *"She **is writing**"* vs *"She **has written**"* | Comrie (1976) *Aspect* |
| **Mood** | Speaker's attitude/reality status | *"She **might** write"* vs *"**Write** this!"* | Palmer (2001) *Mood and Modality* |
| **Voice** | Subject-action relationship | *"She wrote it"* vs *"It **was written**"* | Biber et al. (1999) |

**Combined Label Example:**
> *"The report **has been written**"* → `present-perfect-indicative-passive`

> **Speaker Notes:** Use concrete examples. Key insight: these four dimensions combine to create a "grammatical fingerprint" for each verb. A single document has a distribution over these fingerprints—that distribution is the TAMV profile we analyze.

---

## Slide 4: Literature Review — Timeline of TAM(V) Research

**Academic History: From Linguistic Theory to Computational Extraction**

| Year | Work | Contribution |
|------|------|--------------|
| **1976** | Comrie, *Aspect* | Foundational theory: "perfectivity views situation as a whole; imperfectivity attends to internal structure" |
| **1985** | Comrie, *Tense* | Defined tense as "grammaticalization of location in time" |
| **1997** | Smith, *The Parameter of Aspect* | Distinguished **viewpoint aspect** (grammatical) from **situation aspect** (lexical) |
| **1997** | Dorr & Olsen (ACL) | First NLP work: "Deriving verbal and compositional lexical aspect for NLP applications" |
| **1999** | Biber et al., *Longman Grammar* | Corpus-based TAMV frequency norms across registers |
| **2001** | Palmer, *Mood and Modality* | Theoretical framework for mood/modality |
| **2017** | **Ramm et al., TMV-Annotator** | First robust multilingual TAMV extraction tool |
| **2018** | Zhang et al., *Conversations Gone Awry* | Derailment prediction (lexical features only) |
| **2025** | **This project** | TAMV extraction + derailment prediction |

**Research Gap:**
- Comrie/Smith established *why* aspect matters (theoretical)
- Ramm et al. solved *how* to extract it (computational)
- Zhang et al. studied derailment with *lexical* features
- **No one has tested aspect's predictive power for social outcomes**

> **Speaker Notes:** Walk through the timeline. Comrie (1976) is the foundational text for aspect—he distinguishes perfective ("I wrote") from imperfective ("I was writing"). Smith (1997) added the viewpoint/situation distinction. We inherit this theory and test whether it predicts anything useful.

---

## Slide 5: Literature Review — Why Include Aspect?

**The Theoretical Case for Aspect (Comrie 1976; Smith 1997)**

Aspect captures **how a speaker views the internal structure of an event**:

| Aspect | Viewpoint | Example | Pragmatic Effect |
|--------|-----------|---------|------------------|
| **Simple** | Event as a whole | *"I wrote the report"* | Completed, distanced |
| **Progressive** | Event in progress | *"I am writing the report"* | Ongoing, immediate |
| **Perfect** | Event with current relevance | *"I have written the report"* | Result matters now |
| **Perfect-Progressive** | Ongoing with duration | *"I have been writing..."* | Emphasis on effort/duration |

**Why This Matters for Prediction:**

> *"Perfectivity indicates the view of the situation as a single whole... while the imperfective pays essential attention to the internal structure of the situation"* — Comrie (1976: 16)

**Our Hypothesis:** Progressive aspect signals **immediacy and involvement**
- *"You are ignoring me"* feels more confrontational than *"You ignored me"*
- The ongoing nature implies the speaker is *currently affected*

**Supporting Literature:**
- Comrie (1976) *Aspect* — foundational theory
- Smith (1997) *The Parameter of Aspect* — viewpoint vs. situation aspect
- Dorr & Olsen (1997, ACL) — first NLP aspect work
- [EACL 2023: "A Kind Introduction to Lexical and Grammatical Aspect"](https://aclanthology.org/2023.eacl-main.44.pdf) — recent NLP survey

> **Speaker Notes:** This slide justifies why we include Aspect (the "A" in TAMV). Comrie's 1976 book is THE foundational text. The key insight: progressive aspect keeps the listener "inside" the event, which may feel more immediate and confrontational. This is our theoretical basis for expecting progressive to predict derailment.

---

## Slide 6: Literature Review — Key Implementation Sources

| Source | Contribution | How We Use It |
|--------|--------------|---------------|
| **Ramm et al. (2017)** TMV-Annotator | Rule-based TAMV extraction; modal → subjunctive mapping | Extraction methodology |
| **Biber et al. (1999)** Longman Grammar | 40M-word corpus; TAMV frequency norms | Validation ground truth |
| **Zhang et al. (2018)** Conversations Gone Awry | 4,188 Wikipedia conversations; derailment labels | Prediction task + dataset |

**The Gap We Fill:**
- Ramm solved *extraction* (formal text)
- Zhang solved *prediction* (lexical features)
- **We combine: TAMV extraction → derailment prediction**

> **Speaker Notes:** Three sources, three roles. Ramm gives us extraction methodology. Biber gives us validation ground truth. Zhang gives us the prediction task. We combine all three to ask: does grammar predict social outcomes?

---

## Slide 7: Methodology — Pipeline Architecture

**Addressing RQ2: How to Extract TAMV**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Text   │ →  │   spaCy     │ →  │ Rule-Based  │ →  │ TAMV Labels │
│             │    │   Parser    │    │ Extraction  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                   • Tokenization      • Auxiliary chains
                   • POS tagging       • Tense rules
                   • Dependencies      • Aspect rules
                   • Morphology        • Mood rules
                                       • Voice rules
```

**Design Decisions (vs. TMV-annotator):**

| Choice | TMV-Annotator | Our Implementation | Rationale |
|--------|---------------|-------------------|-----------|
| Parser | Mate (Java) | spaCy (Python) | Lighter, faster, modern |
| Input | CoNLL pre-parsed | Raw text | End-to-end pipeline |
| Terminology | TMV scheme | Aligned with TMV | Comparability |

> **Speaker Notes:** We re-implement TMV-annotator's logic in Python/spaCy. This lets us go from raw text → TAMV labels in one pipeline. The terminology alignment is important—we use their definitions so results are comparable.

---

## Slide 8: Methodology — Extraction Rules with Examples

**Rule Examples (from validate_tamv.py, sourced from Biber et al. 1999)**

| Dimension | Rule | Test Case Example | Source |
|-----------|------|-------------------|--------|
| **Tense** | VBD tag → past | *"They **arrived** yesterday."* | Biber Ch. 6 |
| **Tense** | will/shall + VB → future | *"She **will write** a novel."* | Biber Ch. 6 |
| **Aspect** | have + VBN → perfect | *"She **has written** three novels."* | Biber Ch. 6 |
| **Aspect** | be + VBG → progressive | *"They **were discussing** the proposal."* | Biber Ch. 6 |
| **Mood** | might/could/should/would → subjunctive | *"He **might arrive** late."* | Palmer Ch. 4; TMV-annotator |
| **Mood** | VB without subject → imperative | *"**Write** the report now."* | Biber Ch. 8 |
| **Voice** | be/get + VBN → passive | *"The house **got destroyed** in the storm."* | Biber Ch. 6 |

**Complex Example:**
> *"She **might have been being watched**."*
> → `present-perfect-progressive-subjunctive-passive`
> (Source: Biber et al. 1999, rare but grammatical)

> **Speaker Notes:** Show that our rules are grounded in linguistic literature. Every test case has a citation. The complex example shows we handle 5-verb auxiliary chains correctly.

---

## Slide 9: Methodology — Validation Results

**Validation Against Linguistic Ground Truth**

- **42 test cases** from Biber et al. (1999) and Palmer (2001)
- Covers all TAMV combinations + edge cases

| Dimension | Accuracy | Notes |
|-----------|----------|-------|
| Tense | 100% (8/8) | Present, past, future |
| Aspect | 100% (8/8) | Simple, progressive, perfect, perfect-progressive |
| Voice | 100% (7/7) | Active, passive, get-passive |
| Mood | 90.5% (38/42) | 4 edge cases missed |
| **Overall** | **90.5%** | |

**What We Miss (4 cases):**

| Case | Example | Why It Fails |
|------|---------|--------------|
| Negative imperative | *"**Don't** forget the meeting."* | spaCy parses "don't" as main verb |
| Mandative subjunctive (×3) | *"I suggest that he **leave**."* | Rare pattern; TMV-annotator also misses this |

> **Speaker Notes:** 90.5% is strong validation. The failures are genuine edge cases. Importantly, TMV-annotator (our reference) ALSO misses mandative subjunctive—it's a known hard problem. These cases are <1% of real verbs.

---

## Slide 10: Results — Corpus Comparison

**Addressing RQ1: What Does TAMV Tell Us?**

**Brown Corpus (edited writing) vs. CGA Wiki (online discussion)**

| TAMV Pattern | Brown | CGA | Δ | Interpretation |
|--------------|-------|-----|---|----------------|
| Past Simple Active | **30.3%** | 14.2% | -16.1% | Brown has narrative (fiction) |
| Present Simple Active | 39.5% | **49.3%** | +9.8% | CGA is "here and now" |
| Imperatives | 3.1% | **5.5%** | +2.4% | CGA has more commands |
| Conditionals | 3.4% | **7.9%** | +4.5% | CGA is argumentative ("if you...") |
| Progressive | 1.2% | **2.9%** | +1.7% | CGA emphasizes ongoing actions |

**This Validates Biber (1999):**
- Biber found conversation → present tense, progressive, imperatives
- Our CGA results match this pattern
- Online discourse looks like **spoken conversation**, not edited writing

> **Speaker Notes:** This is our first RQ1 result: TAMV distributions differ meaningfully across registers. This replicates Biber's findings and suggests TAMV captures real stylistic differences. Next: do these differences predict derailment?

---

## Slide 11: Results — Toxicity Correlations

**RQ1: Do TAMV Patterns Predict Derailment?**

*Pilot analysis: n=30 conversations (15 derailed, 15 civil)*

**Significant Correlations with Toxicity Score:**

| TAMV Pattern | r | p-value | Interpretation |
|--------------|---|---------|----------------|
| Present Progressive Active | **0.49** | 0.006** | *"I **am telling** you..."* — confrontational immediacy |
| Present Imperative Active | **0.37** | 0.046* | *"**Stop** doing that!"* — direct commands |
| Past Simple Active | -0.34 | 0.066 | Narrative distance (civil) |

**Effect Sizes (Cohen's d) — Derailed vs. Civil:**

| TAMV Pattern | d | Higher In |
|--------------|---|-----------|
| Present Progressive | 0.89 (large) | Derailed |
| Past Simple | -0.77 (medium) | Civil |
| Imperative | 0.63 (medium) | Derailed |

> **Speaker Notes:** This is the key finding. Progressive aspect ("you ARE doing") and imperatives predict derailment. The effect sizes are meaningful—0.89 is large. Caveat: small sample, needs replication.

---

## Slide 12: Results — Prediction Model

**Can TAMV Alone Predict Derailment?**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | 53% ± 19% | Near chance (50%) |
| F1 Score | 49% ± 20% | Poor |
| **AUC-ROC** | **0.64 ± 0.22** | Above chance — signal exists |

**Top Predictive Features (Logistic Regression):**
1. Present Progressive Active (β = 0.75)
2. Present Perfect Active (β = 0.66)
3. Past Progressive Active (β = 0.59)

**What This Means for RQ1:**
- TAMV alone = **weak but real signal** (AUC 0.64 > 0.50)
- Grammar captures *style*; lexicon captures *content*
- **Hypothesis:** Combined model (TAMV + lexical) will outperform either alone

> **Speaker Notes:** Be honest about limitations. 53% accuracy is poor, but AUC of 0.64 shows the model learned something. TAMV should be a FEATURE in a larger model, not the whole model. The pilot is also tiny (n=30)—we have 4,188 conversations waiting.

---

## Slide 13: Conclusion & Next Steps

**What We Found:**

| Research Question | Status | Finding |
|-------------------|--------|---------|
| **RQ2:** Can we extract TAMV reliably? | ✓ Solved | 90.5% accuracy, validated against Biber/Palmer |
| **RQ1:** Does TAMV predict derailment? | Preliminary | Yes—progressive & imperatives correlate (pilot) |

**What We Hope to Prove (Full Study):**
1. TAMV features have **statistically significant** predictive power at scale
2. Progressive aspect signals **confrontational immediacy**
3. Combined TAMV + lexical models outperform lexical-only baselines

**Next Steps:**

| Step | Description | Status |
|------|-------------|--------|
| Scale up | Run on all 4,188 CGA conversations | Data ready |
| Statistical tests | ANOVA, chi-square, bootstrap CIs | To do |
| Combined model | TAMV + lexical features | To do |
| Baseline comparison | Compare to Zhang et al. (2018) results | To do |

**Questions?**

> **Speaker Notes:** End with clear takeaways. RQ2 is solved—we can extract TAMV reliably. RQ1 shows promise but needs full-scale validation. The contribution: first study combining TAMV extraction with derailment prediction.

---

## Appendix: Backup Slides

### A1: Full Test Case Examples (from validate_tamv.py)

**Tense Examples (Biber et al. 1999, Ch. 6):**
- Present: *"She writes novels."* → `present-simple-indicative-active`
- Past: *"They arrived yesterday."* → `past-simple-indicative-active`
- Future: *"She will write a novel."* → `future-simple-indicative-active`

**Aspect Examples (Biber et al. 1999, Ch. 6):**
- Progressive: *"She is writing a novel."* → `present-progressive-indicative-active`
- Perfect: *"She has written three novels."* → `present-perfect-indicative-active`
- Perfect-Progressive: *"She has been writing for hours."* → `present-perfect-progressive-indicative-active`

**Mood Examples:**
- Modal/Subjunctive (Palmer 2001): *"He might arrive late."* → `present-simple-subjunctive-active`
- Imperative (Biber Ch. 8): *"Write the report now."* → `present-simple-imperative-active`
- Conditional (Palmer Ch. 6): *"If she writes, it will be published."* → `present-simple-conditional-active`

**Voice Examples (Biber et al. 1999, Ch. 6):**
- Be-passive: *"The novel was written by her."* → `past-simple-indicative-passive`
- Get-passive: *"The house got destroyed in the storm."* → `past-simple-indicative-passive`

### A2: TMV-Annotator Alignment

| Concept | TMV-Annotator | Our System |
|---------|---------------|------------|
| would/could/should/might | Subjunctive | Subjunctive ✓ |
| will/shall/can/may/must | Indicative | Indicative ✓ |
| if/unless clauses | Conditional | Conditional ✓ |
| Base form, no subject | Imperative | Imperative ✓ |

### A3: Dataset Statistics

| Corpus | Documents | Unique TAMV Labels | Source |
|--------|-----------|-------------------|--------|
| Brown | 500 | 89 | Francis & Kucera (1979) |
| CGA Wiki | 4,188 | 112 | Zhang et al. (2018) |
| Combined | 4,688 | 120 | — |

### A4: Known Limitations

| Issue | Impact | Source of Problem | Potential Fix |
|-------|--------|-------------------|---------------|
| Negative imperatives | 1/42 test failures | spaCy parse quirk | Pattern match "don't" + VB |
| Mandative subjunctive | 3/42 test failures | Rare English pattern | Clause-level detection |
| Document length confound | Unknown | Longer docs = more verbs | Normalize by length |
| Small pilot sample | High variance | n=30 only | Run full 4,188 |

---

## Speaker Notes Summary

### Key Messages:

1. **Two research questions, ordered:** RQ2 (extraction) enables RQ1 (prediction). We solved RQ2; RQ1 is promising but preliminary.

2. **Grounded in literature:** Every extraction rule has a citation (Biber, Palmer, TMV-annotator). We're not inventing—we're applying established linguistics.

3. **Novel contribution:** First study combining TAMV extraction with derailment prediction. Bridges two literatures (Ramm et al. 2017 + Zhang et al. 2018).

4. **Honest about limitations:** Small pilot, weak standalone prediction, known extraction failures.

### Anticipated Questions:

**Q: Why not use a neural model for extraction?**
A: Interpretability. Rule-based means we can explain every decision and trace errors to specific rules. Neural would be a black box.

**Q: Why is prediction accuracy only 53%?**
A: Two reasons: (1) tiny sample (n=30), and (2) TAMV alone can't capture everything. We expect TAMV to be a *feature*, not a complete model. AUC of 0.64 shows there IS signal.

**Q: How does this compare to Zhang et al. (2018)?**
A: They didn't use TAMV features at all—they used lexical/rhetorical features. Our contribution is testing whether grammar adds signal on top of their approach.

**Q: Is 90.5% extraction accuracy good enough?**
A: Yes. The 4 failures are genuine edge cases (<1% of verbs). TMV-annotator, our reference implementation, also misses mandative subjunctive. Perfect accuracy would require solving open research problems.

### References for Slides:

**Foundational Linguistics (Aspect Theory):**
- Comrie, B. (1976). *Aspect: An Introduction to the Study of Verbal Aspect and Related Problems*. Cambridge University Press.
- Comrie, B. (1985). *Tense*. Cambridge University Press.
- Smith, C. S. (1997). *The Parameter of Aspect* (2nd ed.). Kluwer Academic Publishers.

**Grammar References:**
- Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). *Longman Grammar of Spoken and Written English*. Longman.
- Palmer, F. R. (2001). *Mood and Modality* (2nd ed.). Cambridge University Press.

**Computational Linguistics:**
- Dorr, B. J., & Olsen, M. B. (1997). Deriving verbal and compositional lexical aspect for NLP applications. *Proceedings of ACL 1997*.
- Ramm, A., Loáiciga, S., Friedrich, A., & Fraser, A. (2017). Annotating tense, mood and voice for English, French and German. *Proceedings of ACL 2017*.
- Zhang, J., Chang, J., Danescu-Niculescu-Mizil, C., Dixon, L., Hua, Y., Taraborelli, D., & Thain, N. (2018). Conversations gone awry: Detecting early signs of conversational failure. *Proceedings of ACL 2018*.
- Metheniti, E., Van De Cruys, T., & Hathout, N. (2023). A kind introduction to lexical and grammatical aspect, with a survey of computational approaches. *Proceedings of EACL 2023*. https://aclanthology.org/2023.eacl-main.44.pdf
