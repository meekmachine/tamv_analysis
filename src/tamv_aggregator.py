"""
TAMV Aggregator - Document and corpus-level TAMV profile aggregation

Aggregates clause-level TAMV annotations into document-level distributions
for visualization and analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from .tamv_extractor import TAMVExtractor, TAMVLabel, get_all_tamv_labels
from .corpus_loaders import Document, Conversation


@dataclass
class TAMVProfile:
    """Document-level TAMV profile with counts and proportions."""
    document_id: str
    genre: Optional[str]
    corpus: Optional[str]
    raw_counts: Dict[str, int]
    normalized: Dict[str, float]
    total_clauses: int
    metadata: Optional[Dict] = None

    def to_vector(self, label_order: List[str]) -> np.ndarray:
        """Convert normalized profile to a fixed-order vector."""
        return np.array([self.normalized.get(label, 0.0) for label in label_order])


class TAMVAggregator:
    """
    Aggregates TAMV labels at document and corpus levels.

    Provides methods to:
    - Extract TAMV profiles for individual documents
    - Build corpus-wide TAMV matrices
    - Compute summary statistics across genres/corpora
    """

    def __init__(self, extractor: Optional[TAMVExtractor] = None):
        """
        Initialize the aggregator.

        Args:
            extractor: TAMVExtractor instance. Creates one if not provided.
        """
        self.extractor = extractor or TAMVExtractor()
        self.all_labels = get_all_tamv_labels()
        self.profiles: List[TAMVProfile] = []

    def extract_profile(self, document: Document) -> TAMVProfile:
        """
        Extract TAMV profile for a single document.

        Args:
            document: Document object to analyze

        Returns:
            TAMVProfile with counts and normalized frequencies
        """
        labels = self.extractor.extract_from_text(document.text)

        # Count TAMV labels
        raw_counts = defaultdict(int)
        for label in labels:
            raw_counts[label.to_string()] += 1

        total = sum(raw_counts.values())

        # Normalize
        normalized = {}
        if total > 0:
            for label_str in self.all_labels:
                normalized[label_str] = raw_counts.get(label_str, 0) / total
        else:
            for label_str in self.all_labels:
                normalized[label_str] = 0.0

        return TAMVProfile(
            document_id=document.id,
            genre=document.genre,
            corpus=document.metadata.get('corpus') if document.metadata else None,
            raw_counts=dict(raw_counts),
            normalized=normalized,
            total_clauses=total,
            metadata=document.metadata
        )

    def process_documents(self, documents: List[Document],
                          show_progress: bool = True) -> List[TAMVProfile]:
        """
        Process multiple documents and extract TAMV profiles.

        Args:
            documents: List of Document objects
            show_progress: Whether to show progress bar

        Returns:
            List of TAMVProfile objects
        """
        self.profiles = []
        iterator = tqdm(documents, desc="Extracting TAMV") if show_progress else documents

        for doc in iterator:
            profile = self.extract_profile(doc)
            self.profiles.append(profile)

        return self.profiles

    def to_dataframe(self, profiles: Optional[List[TAMVProfile]] = None) -> pd.DataFrame:
        """
        Convert profiles to a pandas DataFrame.

        Args:
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            DataFrame with document metadata and TAMV frequencies
        """
        profiles = profiles or self.profiles
        if not profiles:
            raise ValueError("No profiles available. Run process_documents first.")

        rows = []
        for p in profiles:
            row = {
                'document_id': p.document_id,
                'genre': p.genre,
                'corpus': p.corpus,
                'total_clauses': p.total_clauses
            }
            # Add metadata fields
            if p.metadata:
                for k, v in p.metadata.items():
                    if k not in row:
                        row[f'meta_{k}'] = v

            # Add TAMV frequencies
            for label in self.all_labels:
                row[label] = p.normalized.get(label, 0.0)

            rows.append(row)

        return pd.DataFrame(rows)

    def to_matrix(self, profiles: Optional[List[TAMVProfile]] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert profiles to a feature matrix.

        Args:
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            Tuple of (matrix, document_ids, label_names)
        """
        profiles = profiles or self.profiles
        if not profiles:
            raise ValueError("No profiles available. Run process_documents first.")

        doc_ids = [p.document_id for p in profiles]
        matrix = np.array([p.to_vector(self.all_labels) for p in profiles])

        return matrix, doc_ids, self.all_labels

    def get_genre_summary(self, profiles: Optional[List[TAMVProfile]] = None) -> pd.DataFrame:
        """
        Compute average TAMV profiles per genre.

        Args:
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            DataFrame with mean TAMV frequencies per genre
        """
        df = self.to_dataframe(profiles)
        tamv_cols = [c for c in df.columns if c in self.all_labels]

        # Group by genre and compute means
        genre_means = df.groupby('genre')[tamv_cols].mean()

        return genre_means

    def get_corpus_summary(self, profiles: Optional[List[TAMVProfile]] = None) -> pd.DataFrame:
        """
        Compute average TAMV profiles per corpus.

        Args:
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            DataFrame with mean TAMV frequencies per corpus
        """
        df = self.to_dataframe(profiles)
        tamv_cols = [c for c in df.columns if c in self.all_labels]

        # Group by corpus and compute means
        corpus_means = df.groupby('corpus')[tamv_cols].mean()

        return corpus_means

    def get_top_tamv_by_genre(self, n: int = 5,
                               profiles: Optional[List[TAMVProfile]] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top N most frequent TAMV labels per genre.

        Args:
            n: Number of top labels to return
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            Dict mapping genre to list of (label, frequency) tuples
        """
        genre_means = self.get_genre_summary(profiles)
        result = {}

        for genre in genre_means.index:
            row = genre_means.loc[genre]
            top_n = row.nlargest(n)
            result[genre] = [(label, freq) for label, freq in top_n.items()]

        return result

    def compute_similarity_matrix(self, profiles: Optional[List[TAMVProfile]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise cosine similarity between documents.

        This can be used for 3D force-directed visualization.

        Args:
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            Tuple of (similarity_matrix, document_ids)
        """
        matrix, doc_ids, _ = self.to_matrix(profiles)

        # Normalize rows for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = matrix / norms

        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)

        return similarity, doc_ids

    def prepare_graph_data(self, similarity_threshold: float = 0.8,
                           profiles: Optional[List[TAMVProfile]] = None) -> Dict:
        """
        Prepare data for 3D force-directed graph visualization.

        Creates nodes and edges based on TAMV similarity.

        Args:
            similarity_threshold: Minimum similarity to create an edge
            profiles: List of profiles (uses stored profiles if None)

        Returns:
            Dict with 'nodes' and 'edges' for graph visualization
        """
        profiles = profiles or self.profiles
        similarity, doc_ids = self.compute_similarity_matrix(profiles)

        # Create nodes
        nodes = []
        for i, profile in enumerate(profiles):
            node = {
                'id': profile.document_id,
                'genre': profile.genre,
                'corpus': profile.corpus,
                'total_clauses': profile.total_clauses
            }
            if profile.metadata:
                if 'max_toxicity' in profile.metadata:
                    node['toxicity'] = profile.metadata['max_toxicity']
                if 'has_derailed' in profile.metadata:
                    node['derailed'] = profile.metadata['has_derailed']
            nodes.append(node)

        # Create edges based on similarity
        edges = []
        n = len(doc_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] >= similarity_threshold:
                    edges.append({
                        'source': doc_ids[i],
                        'target': doc_ids[j],
                        'weight': float(similarity[i, j])
                    })

        return {
            'nodes': nodes,
            'edges': edges,
            'similarity_threshold': similarity_threshold
        }


class ConversationTAMVAnalyzer:
    """
    Specialized analyzer for conversation-level TAMV trajectories.

    Tracks how TAMV profiles evolve across conversation turns.
    """

    def __init__(self, extractor: Optional[TAMVExtractor] = None):
        self.extractor = extractor or TAMVExtractor()
        self.all_labels = get_all_tamv_labels()

    def analyze_trajectory(self, conversation: Conversation) -> Dict:
        """
        Analyze TAMV trajectory across conversation turns.

        Args:
            conversation: Conversation object

        Returns:
            Dict with turn-by-turn TAMV profiles and trajectory metrics
        """
        turn_profiles = []

        for utt in conversation.utterances:
            text = utt.get('text', '')
            if not text:
                continue

            labels = self.extractor.extract_from_text(text)
            counts = defaultdict(int)
            for label in labels:
                counts[label.to_string()] += 1

            total = sum(counts.values())
            normalized = {l: counts.get(l, 0) / total if total > 0 else 0
                         for l in self.all_labels}

            turn_profiles.append({
                'turn_id': utt.get('id'),
                'speaker': utt.get('speaker'),
                'toxicity': utt.get('toxicity'),
                'total_clauses': total,
                'tamv_profile': normalized
            })

        # Compute trajectory metrics
        trajectory = {
            'conversation_id': conversation.id,
            'has_derailed': conversation.has_derailed,
            'num_turns': len(turn_profiles),
            'turns': turn_profiles,
            'tamv_shift': self._compute_tamv_shift(turn_profiles),
            'final_turn_profile': turn_profiles[-1]['tamv_profile'] if turn_profiles else {}
        }

        return trajectory

    def _compute_tamv_shift(self, turn_profiles: List[Dict]) -> float:
        """
        Compute total TAMV shift across conversation.

        Measures how much TAMV profiles change from turn to turn.
        """
        if len(turn_profiles) < 2:
            return 0.0

        total_shift = 0.0
        for i in range(1, len(turn_profiles)):
            prev = turn_profiles[i - 1]['tamv_profile']
            curr = turn_profiles[i]['tamv_profile']

            # Compute L2 distance
            shift = sum((curr.get(l, 0) - prev.get(l, 0)) ** 2
                       for l in self.all_labels) ** 0.5
            total_shift += shift

        return total_shift / (len(turn_profiles) - 1)  # Average shift per turn


if __name__ == "__main__":
    # Test with sample documents
    from .corpus_loaders import BrownCorpusLoader

    print("Testing TAMV Aggregator...")

    # Load a few Brown documents
    loader = BrownCorpusLoader(genres=['news', 'fiction'])
    docs = []
    for i, doc in enumerate(loader.load()):
        docs.append(doc)
        if i >= 4:
            break

    # Process documents
    aggregator = TAMVAggregator()
    profiles = aggregator.process_documents(docs)

    print(f"\nProcessed {len(profiles)} documents")

    # Show sample profile
    print(f"\nSample profile: {profiles[0].document_id}")
    print(f"  Genre: {profiles[0].genre}")
    print(f"  Total clauses: {profiles[0].total_clauses}")
    print(f"  Top 5 TAMV labels:")
    sorted_labels = sorted(profiles[0].normalized.items(),
                          key=lambda x: x[1], reverse=True)[:5]
    for label, freq in sorted_labels:
        print(f"    {label}: {freq:.3f}")

    # Get DataFrame
    df = aggregator.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")

    # Prepare graph data
    graph_data = aggregator.prepare_graph_data(similarity_threshold=0.5)
    print(f"\nGraph data: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
