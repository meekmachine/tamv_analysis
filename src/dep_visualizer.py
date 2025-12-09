"""
Dependency Parse Visualizer using spaCy and Matplotlib

Creates MATE-parser-style dependency diagrams showing:
- Words with POS tags below
- Morphological features (for verbs)
- Dependency arcs above the sentence
"""

import spacy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


class DependencyVisualizer:
    """
    Visualizes dependency parses in a MATE-parser style diagram.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the visualizer with a spaCy model.

        Args:
            model_name: spaCy model to use (e.g., 'en_core_web_sm', 'en_core_web_trf')
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # Styling parameters
        self.word_font_size = 12
        self.pos_font_size = 9
        self.morph_font_size = 8
        self.dep_font_size = 8
        self.word_spacing = 1.5  # Horizontal spacing between words
        self.arc_height_factor = 0.4  # Controls arc height based on distance
        self.min_arc_height = 0.3

    def parse(self, text: str) -> spacy.tokens.Doc:
        """Parse text with spaCy."""
        return self.nlp(text)

    def _get_morph_string(self, token: spacy.tokens.Token) -> str:
        """Extract relevant morphological features as a string."""
        morph = token.morph
        features = []

        # Get key features for verbs
        if token.pos_ in ['VERB', 'AUX']:
            person = morph.get('Person')
            number = morph.get('Number')
            tense = morph.get('Tense')
            mood = morph.get('Mood')
            voice = morph.get('Voice')
            verbform = morph.get('VerbForm')

            parts = []
            if number:
                parts.append('sg' if number[0] == 'Sing' else 'pl')
            if person:
                parts.append(person[0])
            if tense:
                tense_abbrev = {'Pres': 'pres', 'Past': 'past', 'Fut': 'fut'}
                parts.append(tense_abbrev.get(tense[0], tense[0].lower()))
            if mood:
                mood_abbrev = {'Ind': 'ind', 'Sub': 'subj', 'Imp': 'imp'}
                parts.append(mood_abbrev.get(mood[0], mood[0].lower()))
            if verbform and verbform[0] == 'Part':
                parts.append('part')

            if parts:
                return '|'.join(parts)

        return ''

    def _calculate_arc_height(self, start_idx: int, end_idx: int,
                               occupied_heights: Dict[Tuple[int, int], float],
                               base_height: float = 0.5) -> float:
        """
        Calculate arc height to avoid overlaps.

        Args:
            start_idx: Starting token index
            end_idx: Ending token index
            occupied_heights: Dict of (min_idx, max_idx) -> height for existing arcs
            base_height: Minimum height for arcs

        Returns:
            Height for this arc
        """
        min_idx, max_idx = min(start_idx, end_idx), max(start_idx, end_idx)
        distance = max_idx - min_idx

        # Base height scales with distance
        height = base_height + distance * self.arc_height_factor

        # Check for overlaps with existing arcs
        for (existing_min, existing_max), existing_height in occupied_heights.items():
            # Check if arcs overlap horizontally
            if not (max_idx < existing_min or min_idx > existing_max):
                # Arcs overlap, need to adjust height
                if height <= existing_height + 0.2:
                    height = existing_height + 0.3

        return height

    def visualize(self, text: str,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = None,
                  save_path: Optional[str] = None,
                  show_morph: bool = True,
                  highlight_verbs: bool = True) -> plt.Figure:
        """
        Create a MATE-style dependency diagram.

        Args:
            text: Sentence to parse and visualize
            title: Optional title for the diagram
            figsize: Figure size (auto-calculated if None)
            save_path: Path to save the figure
            show_morph: Whether to show morphological features
            highlight_verbs: Whether to bold verbs

        Returns:
            matplotlib Figure object
        """
        doc = self.parse(text)
        tokens = list(doc)
        n_tokens = len(tokens)

        if n_tokens == 0:
            raise ValueError("No tokens found in text")

        # Auto-calculate figure size
        if figsize is None:
            width = max(12, n_tokens * 1.2)
            height = 6
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate positions
        x_positions = np.arange(n_tokens) * self.word_spacing
        y_base = 0  # Base y position for words

        # Track occupied heights for arcs
        occupied_heights = {}

        # First pass: collect all dependencies and sort by distance (shorter first)
        deps = []
        for token in tokens:
            if token.head != token:  # Skip root
                deps.append((token.i, token.head.i, token.dep_))

        # Sort by distance (shorter arcs drawn first, at lower heights)
        deps.sort(key=lambda x: abs(x[0] - x[1]))

        # Draw dependency arcs
        for child_idx, head_idx, dep_label in deps:
            # Calculate arc height
            height = self._calculate_arc_height(child_idx, head_idx, occupied_heights)

            # Store this arc's height
            min_idx, max_idx = min(child_idx, head_idx), max(child_idx, head_idx)
            occupied_heights[(min_idx, max_idx)] = height

            # Arc coordinates
            x_start = x_positions[head_idx]
            x_end = x_positions[child_idx]

            # Draw arc using a quadratic Bezier curve
            arc_top = y_base + height + 0.8

            # Create arc path
            verts = [
                (x_start, y_base + 0.4),  # Start point (above head word)
                (x_start, arc_top),  # Control point 1
                ((x_start + x_end) / 2, arc_top + 0.1),  # Top of arc
                (x_end, arc_top),  # Control point 2
                (x_end, y_base + 0.4),  # End point (above child word)
            ]

            # Draw the arc
            from matplotlib.path import Path
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor='#2c3e50',
                                        linewidth=1.5, capstyle='round')
            ax.add_patch(patch)

            # Draw arrowhead at child
            arrow_dir = 1 if x_end > x_start else -1
            ax.annotate('', xy=(x_end, y_base + 0.4),
                       xytext=(x_end + arrow_dir * 0.05, y_base + 0.55),
                       arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

            # Draw dependency label in a box
            label_x = (x_start + x_end) / 2
            label_y = arc_top + 0.15

            bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='#2c3e50', linewidth=1)
            ax.text(label_x, label_y, dep_label, ha='center', va='bottom',
                   fontsize=self.dep_font_size, fontweight='bold',
                   bbox=bbox_props)

        # Find ROOT and draw ROOT label
        for token in tokens:
            if token.dep_ == 'ROOT':
                root_x = x_positions[token.i]
                root_height = max(occupied_heights.values(), default=0.5) + 0.5

                # Draw ROOT arc
                ax.plot([root_x, root_x], [y_base + 0.4, y_base + root_height + 0.8],
                       color='#2c3e50', linewidth=1.5)

                # ROOT label
                bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='#2c3e50', linewidth=1)
                ax.text(root_x, y_base + root_height + 0.9, 'ROOT',
                       ha='center', va='bottom',
                       fontsize=self.dep_font_size, fontweight='bold',
                       bbox=bbox_props)
                break

        # Draw words and annotations
        for i, token in enumerate(tokens):
            x = x_positions[i]

            # Determine if this is a verb (for highlighting)
            is_verb = token.pos_ in ['VERB', 'AUX']
            font_weight = 'bold' if (is_verb and highlight_verbs) else 'normal'

            # Word
            ax.text(x, y_base, token.text, ha='center', va='top',
                   fontsize=self.word_font_size, fontweight=font_weight)

            # POS tag
            ax.text(x, y_base - 0.35, token.pos_, ha='center', va='top',
                   fontsize=self.pos_font_size, color='#7f8c8d')

            # Morphological features (for verbs)
            if show_morph:
                morph_str = self._get_morph_string(token)
                if morph_str:
                    ax.text(x, y_base - 0.6, morph_str, ha='center', va='top',
                           fontsize=self.morph_font_size, color='#3498db',
                           style='italic')

        # Set axis limits and styling
        max_height = max(occupied_heights.values(), default=1) + 1.5
        ax.set_xlim(-0.5, x_positions[-1] + 0.5)
        ax.set_ylim(-1.2, max_height + 1)

        # Remove axes
        ax.set_aspect('equal')
        ax.axis('off')

        # Add title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved diagram to {save_path}")

        return fig

    def visualize_with_tmv(self, text: str,
                           tmv_annotations: Optional[Dict] = None,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create dependency diagram with TMV annotations below.

        Similar to the MATE parser output showing:
        (1) Dependency parse
        (2) Verbal complexes extraction
        (3) TMV feature assignment

        Args:
            text: Sentence to parse
            tmv_annotations: Optional dict of {verb_phrase: (tense, mood, voice)}
            title: Optional title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        doc = self.parse(text)
        tokens = list(doc)
        n_tokens = len(tokens)

        # Auto-calculate figure size
        if figsize is None:
            width = max(14, n_tokens * 1.3)
            height = 8
            figsize = (width, height)

        fig, (ax_dep, ax_tmv) = plt.subplots(2, 1, figsize=figsize,
                                              gridspec_kw={'height_ratios': [3, 1]})

        # Draw dependency diagram in top panel
        x_positions = np.arange(n_tokens) * self.word_spacing
        y_base = 0
        occupied_heights = {}

        # Collect and sort dependencies
        deps = []
        for token in tokens:
            if token.head != token:
                deps.append((token.i, token.head.i, token.dep_))
        deps.sort(key=lambda x: abs(x[0] - x[1]))

        # Draw arcs
        for child_idx, head_idx, dep_label in deps:
            height = self._calculate_arc_height(child_idx, head_idx, occupied_heights)
            min_idx, max_idx = min(child_idx, head_idx), max(child_idx, head_idx)
            occupied_heights[(min_idx, max_idx)] = height

            x_start = x_positions[head_idx]
            x_end = x_positions[child_idx]
            arc_top = y_base + height + 0.8

            from matplotlib.path import Path
            verts = [
                (x_start, y_base + 0.4),
                (x_start, arc_top),
                ((x_start + x_end) / 2, arc_top + 0.1),
                (x_end, arc_top),
                (x_end, y_base + 0.4),
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor='#2c3e50',
                                        linewidth=1.5)
            ax_dep.add_patch(patch)

            arrow_dir = 1 if x_end > x_start else -1
            ax_dep.annotate('', xy=(x_end, y_base + 0.4),
                           xytext=(x_end + arrow_dir * 0.05, y_base + 0.55),
                           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

            label_x = (x_start + x_end) / 2
            label_y = arc_top + 0.15
            bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='#2c3e50', linewidth=1)
            ax_dep.text(label_x, label_y, dep_label, ha='center', va='bottom',
                       fontsize=self.dep_font_size, fontweight='bold', bbox=bbox_props)

        # Draw ROOT
        for token in tokens:
            if token.dep_ == 'ROOT':
                root_x = x_positions[token.i]
                root_height = max(occupied_heights.values(), default=0.5) + 0.5
                ax_dep.plot([root_x, root_x], [y_base + 0.4, y_base + root_height + 0.8],
                           color='#2c3e50', linewidth=1.5)
                bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='#2c3e50', linewidth=1)
                ax_dep.text(root_x, y_base + root_height + 0.9, 'ROOT',
                           ha='center', va='bottom',
                           fontsize=self.dep_font_size, fontweight='bold', bbox=bbox_props)
                break

        # Draw words and POS
        for i, token in enumerate(tokens):
            x = x_positions[i]
            is_verb = token.pos_ in ['VERB', 'AUX']
            font_weight = 'bold' if is_verb else 'normal'

            ax_dep.text(x, y_base, token.text, ha='center', va='top',
                       fontsize=self.word_font_size, fontweight=font_weight)
            ax_dep.text(x, y_base - 0.35, token.pos_, ha='center', va='top',
                       fontsize=self.pos_font_size, color='#7f8c8d')

            morph_str = self._get_morph_string(token)
            if morph_str:
                ax_dep.text(x, y_base - 0.6, morph_str, ha='center', va='top',
                           fontsize=self.morph_font_size, color='#3498db', style='italic')

        max_height = max(occupied_heights.values(), default=1) + 1.5
        ax_dep.set_xlim(-0.5, x_positions[-1] + 0.5)
        ax_dep.set_ylim(-1.2, max_height + 1)
        ax_dep.axis('off')

        # Bottom panel: TMV annotations
        ax_tmv.set_xlim(-0.5, x_positions[-1] + 0.5)
        ax_tmv.set_ylim(-0.5, 2)
        ax_tmv.axis('off')

        # Add TMV annotation header
        ax_tmv.text(-0.3, 1.5, "TMV Features:", fontsize=10, fontweight='bold', va='top')

        # Extract verbal complexes and their features
        verbal_complexes = self._extract_verbal_complexes(doc)

        y_offset = 1.0
        for i, (verb_phrase, pos_seq, features) in enumerate(verbal_complexes):
            label = chr(ord('A') + i)

            # Format: A   will be examined    MD[will] VB[be] VBN → futureI indicative passive
            line = f"{label}   {verb_phrase:<25} {pos_seq:<30} → {features}"
            ax_tmv.text(-0.3, y_offset, line, fontsize=9, va='top',
                       fontfamily='monospace')
            y_offset -= 0.4

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved diagram to {save_path}")

        return fig

    def _extract_verbal_complexes(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """
        Extract verbal complexes and their TMV features.

        Returns:
            List of (verb_phrase, pos_sequence, tmv_features) tuples
        """
        complexes = []
        processed = set()

        # Find main verbs (ROOT or with verbal head)
        for token in doc:
            if token.i in processed:
                continue

            # Look for main verbs
            if token.pos_ == 'VERB':
                # Collect auxiliaries that modify this verb
                verb_tokens = []

                # Get all auxiliaries for this verb
                for child in token.children:
                    if child.dep_ in ['aux', 'auxpass']:
                        verb_tokens.append(child)

                # Add the main verb
                verb_tokens.append(token)

                # Sort by position
                verb_tokens.sort(key=lambda t: t.i)

                # Mark as processed
                for t in verb_tokens:
                    processed.add(t.i)

                # Build phrase and POS sequence
                phrase = ' '.join(t.text for t in verb_tokens)
                pos_seq = ' '.join(f"{t.tag_}[{t.text}]" for t in verb_tokens)

                # Determine TMV features
                features = self._determine_tmv_features(verb_tokens)

                complexes.append((phrase, pos_seq, features))

            # Handle standalone auxiliaries (like copulas)
            elif token.pos_ == 'AUX' and token.dep_ == 'ROOT':
                verb_tokens = [token]
                processed.add(token.i)

                phrase = token.text
                pos_seq = f"{token.tag_}[{token.text}]"
                features = self._determine_tmv_features(verb_tokens)

                complexes.append((phrase, pos_seq, features))

        return complexes

    def _get_verbal_complex(self, token: spacy.tokens.Token,
                            doc: spacy.tokens.Doc) -> List[spacy.tokens.Token]:
        """Get all tokens in a verbal complex."""
        complex_tokens = [token]

        # Look for auxiliary chains
        for child in token.children:
            if child.dep_ in ['aux', 'auxpass']:
                complex_tokens.append(child)

        # Look for verb complements
        for child in token.children:
            if child.dep_ == 'xcomp' and child.pos_ == 'VERB':
                complex_tokens.extend(self._get_verbal_complex(child, doc))

        # Sort by position
        complex_tokens.sort(key=lambda t: t.i)

        return complex_tokens

    def _determine_tmv_features(self, verb_tokens: List[spacy.tokens.Token]) -> str:
        """Determine Tense-Mood-Voice features for a verbal complex."""
        tense = 'present'
        mood = 'indicative'
        voice = 'active'

        has_modal = False
        has_will = False
        has_passive_aux = False
        has_past_participle = False
        has_progressive = False
        has_perfect = False

        for token in verb_tokens:
            tag = token.tag_
            text_lower = token.text.lower()

            # Check for modals
            if tag == 'MD':
                has_modal = True
                if text_lower in ['will', "won't", "'ll"]:
                    has_will = True
                elif text_lower in ['would', "wouldn't", "'d"]:
                    mood = 'conditional'
                elif text_lower in ['should', 'could', 'may', 'might', 'must']:
                    mood = 'modal'

            # Check for passive (be + VBN)
            if text_lower in ['be', 'been', 'being', 'am', 'is', 'are', 'was', 'were']:
                has_passive_aux = True

            if tag == 'VBN':
                has_past_participle = True

            # Check for progressive (be + VBG)
            if tag == 'VBG':
                has_progressive = True

            # Check tense from finite verb
            morph = token.morph
            token_tense = morph.get('Tense')
            if token_tense:
                if token_tense[0] == 'Past':
                    tense = 'past'

            # Check for perfect (have + VBN)
            if text_lower in ['have', 'has', 'had', "'ve", "'s"]:
                has_perfect = True

        # Determine tense
        if has_will:
            tense = 'futureI'
        elif has_modal and mood == 'conditional':
            tense = 'conditional'

        # Determine voice
        if has_passive_aux and has_past_participle:
            voice = 'passive'

        # Determine aspect
        aspect = 'simple'
        if has_perfect and has_progressive:
            aspect = 'perfect-progressive'
        elif has_perfect:
            aspect = 'perfect'
        elif has_progressive:
            aspect = 'progressive'

        return f"{tense} {mood} {voice}"


def main():
    """Demo the dependency visualizer."""
    viz = DependencyVisualizer()

    # Example sentence from the MATE parser image
    sentence = "It will, I hope, be examined in a positive light."

    print(f"Parsing: {sentence}")
    print()

    # Create simple dependency diagram
    fig1 = viz.visualize(sentence,
                         title="spaCy Dependency Parse (Universal Dependencies)",
                         save_path="output/dep_parse_simple.png")

    # Create diagram with TMV annotations
    fig2 = viz.visualize_with_tmv(sentence,
                                   title="Dependency Parse with TMV Features",
                                   save_path="output/dep_parse_with_tmv.png")

    # Try another sentence
    sentence2 = "The letters have been answered already."
    fig3 = viz.visualize(sentence2,
                         title="spaCy Dependency Parse",
                         save_path="output/dep_parse_passive.png")

    print("\nDone! Check the output directory for the diagrams.")

    plt.show()


if __name__ == "__main__":
    main()
