"""
TAMV Extractor - Rule-based extraction of Tense, Aspect, Mood, and Voice

This module implements a rule-based system for extracting TAMV labels from text,
following the design principles of TMV-annotator (Ramm et al., 2017) and
clause-oriented approaches (Friedrich et al., 2016; Dönicke, 2020).
"""

import spacy
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class Tense(Enum):
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    NONE = "none"  # For non-finite verbs


class Aspect(Enum):
    SIMPLE = "simple"
    PROGRESSIVE = "progressive"
    PERFECT = "perfect"
    PERFECT_PROGRESSIVE = "perfect-progressive"


class Mood(Enum):
    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    IMPERATIVE = "imperative"
    CONDITIONAL = "conditional"
    MODAL = "modal"
    INFINITIVAL = "infinitival"    # Non-finite: "to have been told"
    PARTICIPLE = "participle"       # Non-finite: "Having been warned"


class Voice(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


@dataclass
class TAMVLabel:
    """Represents a complete TAMV annotation for a verbal complex."""
    tense: Tense
    aspect: Aspect
    mood: Mood
    voice: Voice
    verb_text: str
    verb_lemma: str
    auxiliary_chain: List[str]
    span_start: int
    span_end: int

    def to_string(self) -> str:
        """Return canonical TAMV label string."""
        return f"{self.tense.value}-{self.aspect.value}-{self.mood.value}-{self.voice.value}"

    def __repr__(self):
        return f"TAMVLabel({self.to_string()}, verb='{self.verb_text}')"


class TAMVExtractor:
    """
    Rule-based TAMV extractor using spaCy dependency parsing.

    Extracts tense, aspect, mood, and voice from verbal complexes
    by analyzing auxiliary patterns and morphological features.
    """

    # Modal verbs that indicate modal mood
    MODAL_VERBS = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ought'}

    # Auxiliary verbs
    AUX_BE = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being'}
    AUX_HAVE = {'have', 'has', 'had', 'having'}
    AUX_DO = {'do', 'does', 'did'}

    # Conditional markers
    CONDITIONAL_MARKERS = {'if', 'unless', 'whether', 'provided', 'supposing'}

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with a spaCy model."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading spaCy model: {model}")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    def extract_from_text(self, text: str) -> List[TAMVLabel]:
        """
        Extract all TAMV labels from a text.

        Args:
            text: Input text to analyze

        Returns:
            List of TAMVLabel objects for each verbal complex found
        """
        doc = self.nlp(text)
        return self.extract_from_doc(doc)

    def extract_from_doc(self, doc) -> List[TAMVLabel]:
        """
        Extract TAMV labels from a spaCy Doc object.

        Args:
            doc: spaCy Doc object

        Returns:
            List of TAMVLabel objects
        """
        labels = []
        processed_verbs = set()

        for token in doc:
            # Find main verbs (not auxiliaries that are part of another verbal complex)
            if self._is_main_verb(token) and token.i not in processed_verbs:
                label = self._extract_verbal_complex(token, doc, processed_verbs)
                if label:
                    labels.append(label)

        return labels

    def _is_main_verb(self, token) -> bool:
        """Check if a token is a main verb (not an auxiliary)."""
        # Main verbs have POS tag VERB and are not auxiliaries
        if token.pos_ == "VERB":
            return True
        # Also consider AUX that are the root or have no verb head
        if token.pos_ == "AUX" and token.dep_ == "ROOT":
            return True
        return False

    def _extract_verbal_complex(self, main_verb, doc, processed_verbs: set) -> Optional[TAMVLabel]:
        """
        Extract TAMV label for a verbal complex headed by main_verb.

        Args:
            main_verb: The main verb token
            doc: The spaCy Doc
            processed_verbs: Set of already processed verb indices

        Returns:
            TAMVLabel or None if extraction fails
        """
        # Collect the auxiliary chain
        aux_chain = self._collect_auxiliaries(main_verb)

        # Mark all verbs in the complex as processed
        processed_verbs.add(main_verb.i)
        for aux in aux_chain:
            processed_verbs.add(aux.i)

        # Check if this is a non-finite verb phrase
        is_infinitive = self._is_infinitive(main_verb, aux_chain)
        is_participle = self._is_participle(main_verb, aux_chain)

        # Determine TAMV features
        if is_infinitive:
            tense = Tense.NONE
            mood = Mood.INFINITIVAL
        elif is_participle:
            tense = Tense.NONE
            mood = Mood.PARTICIPLE
        else:
            tense = self._determine_tense(main_verb, aux_chain)
            mood = self._determine_mood(main_verb, aux_chain, doc)

        aspect = self._determine_aspect(main_verb, aux_chain)
        voice = self._determine_voice(main_verb, aux_chain)

        # Build auxiliary text chain
        aux_texts = [aux.text for aux in aux_chain]

        return TAMVLabel(
            tense=tense,
            aspect=aspect,
            mood=mood,
            voice=voice,
            verb_text=main_verb.text,
            verb_lemma=main_verb.lemma_,
            auxiliary_chain=aux_texts,
            span_start=min([main_verb.idx] + [aux.idx for aux in aux_chain]),
            span_end=max([main_verb.idx + len(main_verb.text)] +
                        [aux.idx + len(aux.text) for aux in aux_chain])
        )

    def _collect_auxiliaries(self, main_verb) -> List:
        """Collect all auxiliaries that modify the main verb."""
        auxiliaries = []

        # Look for auxiliaries in the children
        for child in main_verb.children:
            if child.dep_ == "aux" or child.dep_ == "auxpass":
                auxiliaries.append(child)

        # Sort by position
        auxiliaries.sort(key=lambda x: x.i)
        return auxiliaries

    def _is_infinitive(self, main_verb, aux_chain: List) -> bool:
        """Check if the verbal complex is an infinitive (non-finite with 'to').

        Examples:
        - "to have been promoted" (xcomp with 'to' aux)
        - "to be told"
        """
        # Check if verb has 'to' as auxiliary
        for aux in aux_chain:
            if aux.pos_ == "PART" and aux.text.lower() == "to":
                return True

        # Also check dep relation - xcomp often indicates infinitive
        if main_verb.dep_ == "xcomp":
            # Verify it's actually an infinitive by checking for 'to' or base form
            for aux in aux_chain:
                if aux.text.lower() == "to":
                    return True
            # Check if main verb is base form without finite aux
            if main_verb.tag_ in {"VB", "VBN"} and not self._has_finite_auxiliary(aux_chain):
                # Look for 'to' before this verb in the sentence
                for tok in main_verb.doc[max(0, main_verb.i - 5):main_verb.i]:
                    if tok.text.lower() == "to" and tok.head == main_verb:
                        return True

        return False

    def _is_participle(self, main_verb, aux_chain: List) -> bool:
        """Check if the verbal complex is a participle phrase (non-finite).

        Examples:
        - "Having been warned" (advcl with VBG start)
        - "Written by hand" (reduced relative)
        """
        # Participle clauses typically have dep = advcl and no finite auxiliary
        if main_verb.dep_ in {"advcl", "ccomp"} and not self._has_finite_auxiliary(aux_chain):
            # Check if it starts with a participle form (VBG or VBN)
            if aux_chain:
                first_aux = aux_chain[0]
                if first_aux.tag_ == "VBG":  # "Having been warned"
                    return True
            # Or main verb itself is participle without finite aux
            if main_verb.tag_ in {"VBG", "VBN"}:
                return True

        return False

    def _has_finite_auxiliary(self, aux_chain: List) -> bool:
        """Check if the auxiliary chain contains a finite verb."""
        finite_tags = {"VBZ", "VBP", "VBD", "MD"}  # Present, past, modal
        for aux in aux_chain:
            if aux.tag_ in finite_tags:
                return True
            # Also check if it's a modal
            if aux.lemma_.lower() in self.MODAL_VERBS:
                return True
        return False

    def _determine_tense(self, main_verb, aux_chain: List) -> Tense:
        """Determine the tense of the verbal complex."""
        # Check for future tense markers first
        for aux in aux_chain:
            if aux.lemma_.lower() in {'will', 'shall'}:
                return Tense.FUTURE
            # "going to" future
            if aux.lemma_.lower() == 'go' and main_verb.tag_ == 'VB':
                return Tense.FUTURE

        # Check morphology of the finite verb (first aux or main verb)
        finite_verb = aux_chain[0] if aux_chain else main_verb

        # Use morphological features
        morph = finite_verb.morph
        tense_feature = morph.get("Tense")

        if tense_feature:
            if "Past" in tense_feature:
                return Tense.PAST
            elif "Pres" in tense_feature:
                return Tense.PRESENT

        # Fallback to tag-based detection
        tag = finite_verb.tag_
        if tag in {'VBD', 'VBN'}:
            # VBN alone is past participle, need context
            if finite_verb == main_verb and not aux_chain:
                return Tense.PAST
            return Tense.PAST
        elif tag in {'VBZ', 'VBP', 'VB'}:
            return Tense.PRESENT
        elif tag == 'VBG':
            return Tense.PRESENT

        return Tense.PRESENT  # Default

    def _determine_aspect(self, main_verb, aux_chain: List) -> Aspect:
        """Determine the aspect of the verbal complex."""
        has_perfect = False
        has_progressive = False

        # Check for perfect aspect (have/has/had + past participle)
        for i, aux in enumerate(aux_chain):
            if aux.lemma_.lower() in {'have', 'has', 'had'}:
                # Check if followed by past participle
                has_perfect = True

        # Check for progressive aspect (be + -ing)
        main_tag = main_verb.tag_
        if main_tag == 'VBG':  # Present participle
            # Check if there's a form of 'be' in auxiliaries
            for aux in aux_chain:
                if aux.lemma_.lower() == 'be':
                    has_progressive = True
                    break
            # Even without explicit 'be', VBG often indicates progressive
            if not has_progressive and any(aux.lemma_.lower() in self.AUX_BE for aux in aux_chain):
                has_progressive = True

        # Also check if any aux is 'being' (for passive progressive)
        for aux in aux_chain:
            if aux.text.lower() == 'being':
                has_progressive = True

        if has_perfect and has_progressive:
            return Aspect.PERFECT_PROGRESSIVE
        elif has_perfect:
            return Aspect.PERFECT
        elif has_progressive:
            return Aspect.PROGRESSIVE
        else:
            return Aspect.SIMPLE

    def _determine_mood(self, main_verb, aux_chain: List, doc) -> Mood:
        """Determine the mood of the verbal complex.

        Extended beyond TMV-annotator (Ramm et al. 2017):
        - would/might in conditional context → CONDITIONAL
        - would/could/might outside conditional → SUBJUNCTIVE (epistemic possibility)
        - can/may/must/should → MODAL (deontic/dynamic modality)
        - will/shall → indicative (future tense marker, not mood)
        """
        # Check for modal auxiliaries
        for aux in aux_chain:
            if aux.lemma_.lower() in self.MODAL_VERBS:
                modal = aux.lemma_.lower()

                # Check if in conditional context first
                if modal in {'would', 'might'} and self._in_conditional_context(main_verb, doc):
                    return Mood.CONDITIONAL

                # would/could/might express epistemic possibility → SUBJUNCTIVE
                if modal in {'would', 'could', 'might'}:
                    return Mood.SUBJUNCTIVE

                # can/may/must/should express deontic/dynamic modality → MODAL
                if modal in {'can', 'may', 'must', 'should'}:
                    return Mood.MODAL

                # will/shall are future tense markers, not mood changers
                break

        # Check for imperative (base form with no subject)
        if main_verb.tag_ == 'VB' and not aux_chain:
            # Look for a subject
            has_subject = False
            for child in main_verb.children:
                if child.dep_ in {'nsubj', 'nsubjpass', 'expl'}:
                    has_subject = True
                    break
            if not has_subject:
                # Could be imperative
                return Mood.IMPERATIVE

        # Check for subjunctive (rare in English)
        # Typically: "I suggest that he go" (base form after certain verbs)
        if main_verb.tag_ == 'VB' and not aux_chain:
            if self._in_subjunctive_context(main_verb, doc):
                return Mood.SUBJUNCTIVE

        # Check for conditional context
        if self._in_conditional_context(main_verb, doc):
            return Mood.CONDITIONAL

        return Mood.INDICATIVE

    def _in_conditional_context(self, verb, doc) -> bool:
        """Check if the verb is in a conditional clause."""
        # Look for conditional markers in the sentence
        sent = verb.sent
        for token in sent:
            if token.text.lower() in self.CONDITIONAL_MARKERS:
                return True
            # Check for "would/could/might have" in past conditional
            if token.lemma_.lower() in {'would', 'could', 'might'}:
                for child in token.children:
                    if child.lemma_.lower() == 'have':
                        return True
        return False

    def _in_subjunctive_context(self, verb, doc) -> bool:
        """Check if the verb is in a subjunctive context."""
        # Subjunctive triggers in English
        subjunctive_triggers = {
            'suggest', 'recommend', 'demand', 'insist', 'require',
            'ask', 'propose', 'urge', 'request', 'important', 'essential',
            'necessary', 'vital', 'crucial'
        }

        # Check if the verb is in a that-clause following a trigger
        sent = verb.sent
        for token in sent:
            if token.lemma_.lower() in subjunctive_triggers:
                # Check if there's a "that" between the trigger and the verb
                if token.i < verb.i:
                    for t in doc[token.i:verb.i]:
                        if t.text.lower() == 'that':
                            return True
        return False

    def _determine_voice(self, main_verb, aux_chain: List) -> Voice:
        """Determine the voice of the verbal complex."""
        # Check for passive: be/get + past participle
        main_tag = main_verb.tag_

        if main_tag == 'VBN':  # Past participle
            for aux in aux_chain:
                # Passive with 'be'
                if aux.lemma_.lower() == 'be':
                    return Voice.PASSIVE
                # Passive with 'get'
                if aux.lemma_.lower() == 'get':
                    return Voice.PASSIVE

        # Check for auxpass dependency
        for aux in aux_chain:
            if aux.dep_ == 'auxpass':
                return Voice.PASSIVE

        # Check if main verb has nsubjpass
        for child in main_verb.children:
            if child.dep_ == 'nsubjpass':
                return Voice.PASSIVE

        return Voice.ACTIVE

    def get_tamv_profile(self, text: str) -> Dict[str, int]:
        """
        Get a frequency profile of TAMV labels for a text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping TAMV label strings to counts
        """
        labels = self.extract_from_text(text)
        profile = {}
        for label in labels:
            key = label.to_string()
            profile[key] = profile.get(key, 0) + 1
        return profile

    def get_normalized_profile(self, text: str) -> Dict[str, float]:
        """
        Get a normalized frequency profile of TAMV labels.

        Args:
            text: Input text

        Returns:
            Dictionary mapping TAMV label strings to proportions
        """
        profile = self.get_tamv_profile(text)
        total = sum(profile.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in profile.items()}


def get_all_tamv_labels() -> List[str]:
    """Generate all possible TAMV label combinations."""
    labels = []
    for tense in Tense:
        if tense == Tense.NONE:
            continue
        for aspect in Aspect:
            for mood in Mood:
                for voice in Voice:
                    labels.append(f"{tense.value}-{aspect.value}-{mood.value}-{voice.value}")
    return labels


if __name__ == "__main__":
    # Example usage
    extractor = TAMVExtractor()

    test_sentences = [
        "The cat sat on the mat.",
        "She is writing a letter.",
        "They have been working all day.",
        "The book was written by the author.",
        "He would have gone if he could.",
        "Go to the store!",
        "I suggest that he leave immediately.",
        "She can speak three languages.",
        "The project will be completed tomorrow.",
    ]

    for sent in test_sentences:
        print(f"\nSentence: {sent}")
        labels = extractor.extract_from_text(sent)
        for label in labels:
            print(f"  {label}")
