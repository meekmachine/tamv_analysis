"""
Corpus Loaders for TAMV Analysis

Provides unified interfaces for loading:
- Brown Corpus (from NLTK)
- Conversations Gone Awry datasets (from ConvoKit)
"""

import os
from typing import List, Dict, Generator, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class Document:
    """Represents a document from any corpus."""
    id: str
    text: str
    genre: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Document(id='{self.id}', genre='{self.genre}', text='{text_preview}')"


@dataclass
class Conversation:
    """Represents a conversation from CGA datasets."""
    id: str
    utterances: List[Dict[str, Any]]
    has_derailed: bool
    toxicity_scores: List[float]
    metadata: Optional[Dict[str, Any]] = None

    @property
    def full_text(self) -> str:
        """Get full conversation text."""
        return " ".join(u.get('text', '') for u in self.utterances)

    @property
    def max_toxicity(self) -> float:
        """Get maximum toxicity score in conversation."""
        return max(self.toxicity_scores) if self.toxicity_scores else 0.0

    @property
    def mean_toxicity(self) -> float:
        """Get mean toxicity score."""
        return sum(self.toxicity_scores) / len(self.toxicity_scores) if self.toxicity_scores else 0.0

    def __repr__(self):
        return f"Conversation(id='{self.id}', turns={len(self.utterances)}, derailed={self.has_derailed})"


class CorpusLoader(ABC):
    """Abstract base class for corpus loaders."""

    @abstractmethod
    def load(self) -> Generator:
        """Load and yield documents/conversations."""
        pass

    @abstractmethod
    def get_all(self) -> List:
        """Load all items at once."""
        pass


class BrownCorpusLoader(CorpusLoader):
    """
    Loader for the Brown Corpus via NLTK.

    The Brown Corpus contains 500 samples of written American English
    across 15 genres.
    """

    # Genre codes and their descriptions
    GENRE_MAP = {
        'news': 'Press: Reportage',
        'editorial': 'Press: Editorial',
        'reviews': 'Press: Reviews',
        'religion': 'Religion',
        'hobbies': 'Hobbies',
        'lore': 'Popular Lore',
        'belles_lettres': 'Belles Lettres',
        'government': 'Government Documents',
        'learned': 'Learned/Scientific',
        'fiction': 'General Fiction',
        'mystery': 'Mystery Fiction',
        'science_fiction': 'Science Fiction',
        'adventure': 'Adventure Fiction',
        'romance': 'Romance Fiction',
        'humor': 'Humor'
    }

    def __init__(self, genres: Optional[List[str]] = None):
        """
        Initialize the Brown Corpus loader.

        Args:
            genres: Optional list of genre codes to load. If None, loads all.
        """
        self.genres = genres
        self._ensure_downloaded()

    def _ensure_downloaded(self):
        """Ensure Brown Corpus is downloaded."""
        import nltk
        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            print("Downloading Brown Corpus...")
            nltk.download('brown', quiet=True)

    def load(self) -> Generator[Document, None, None]:
        """
        Yield documents from the Brown Corpus.

        Yields:
            Document objects with text and genre information
        """
        from nltk.corpus import brown

        categories = self.genres if self.genres else brown.categories()

        for category in categories:
            if category not in brown.categories():
                warnings.warn(f"Category '{category}' not found in Brown Corpus")
                continue

            for fileid in brown.fileids(categories=category):
                # Get words and reconstruct text
                words = brown.words(fileid)
                text = self._reconstruct_text(words)

                yield Document(
                    id=fileid,
                    text=text,
                    genre=category,
                    metadata={
                        'genre_description': self.GENRE_MAP.get(category, category),
                        'num_words': len(words)
                    }
                )

    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct readable text from tokenized words."""
        # Simple heuristic to add spaces appropriately
        text_parts = []
        for i, word in enumerate(words):
            if i == 0:
                text_parts.append(word)
            elif word in '.,!?;:)]\'"':
                text_parts.append(word)
            elif words[i-1] in '(["\'' and word not in '.,!?;:':
                text_parts.append(word)
            else:
                text_parts.append(' ' + word)
        return ''.join(text_parts)

    def get_all(self) -> List[Document]:
        """Load all documents at once."""
        return list(self.load())

    def get_genre_summary(self) -> Dict[str, int]:
        """Get count of documents per genre."""
        from nltk.corpus import brown

        summary = {}
        categories = self.genres if self.genres else brown.categories()
        for category in categories:
            if category in brown.categories():
                summary[category] = len(brown.fileids(categories=category))
        return summary


class CGALoader(CorpusLoader):
    """
    Loader for Conversations Gone Awry datasets via ConvoKit.

    Supports both Wikipedia Talk Pages (CGA-WIKI) and Reddit CMV (CGA-CMV).
    """

    DATASETS = {
        'wiki': 'conversations-gone-awry-corpus',
        'cmv': 'conversations-gone-awry-cmv-corpus'
    }

    def __init__(self, dataset: str = 'wiki', data_dir: Optional[str] = None):
        """
        Initialize the CGA loader.

        Args:
            dataset: 'wiki' for Wikipedia or 'cmv' for Reddit ChangeMyView
            data_dir: Optional directory to store downloaded data
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Dataset must be one of: {list(self.DATASETS.keys())}")

        self.dataset = dataset
        self.corpus_name = self.DATASETS[dataset]
        self.data_dir = data_dir
        self._corpus = None

    def _load_corpus(self):
        """Load the ConvoKit corpus."""
        if self._corpus is not None:
            return

        try:
            from convokit import Corpus, download
        except ImportError:
            raise ImportError("Please install convokit: pip install convokit")

        print(f"Loading {self.corpus_name}...")
        if self.data_dir:
            self._corpus = Corpus(
                filename=download(self.corpus_name, data_dir=self.data_dir)
            )
        else:
            self._corpus = Corpus(filename=download(self.corpus_name))

    def load(self) -> Generator[Conversation, None, None]:
        """
        Yield conversations from the CGA dataset.

        Yields:
            Conversation objects with utterances and metadata
        """
        self._load_corpus()

        for convo in self._corpus.iter_conversations():
            utterances = []
            toxicity_scores = []

            for utt in convo.iter_utterances():
                utt_data = {
                    'id': utt.id,
                    'text': utt.text or '',
                    'speaker': utt.speaker.id if utt.speaker else None,
                    'timestamp': utt.timestamp,
                    'reply_to': utt.reply_to
                }

                # Extract toxicity score if available
                if hasattr(utt, 'meta') and utt.meta:
                    toxicity = utt.meta.get('toxicity', utt.meta.get('score', None))
                    if toxicity is not None:
                        toxicity_scores.append(float(toxicity))
                        utt_data['toxicity'] = float(toxicity)

                utterances.append(utt_data)

            # Get conversation-level metadata
            has_derailed = False
            if hasattr(convo, 'meta') and convo.meta:
                # CGA uses 'conversation_has_personal_attack' label
                has_derailed = convo.meta.get('conversation_has_personal_attack', False)
                if not has_derailed:
                    has_derailed = convo.meta.get('has_derailed', False)

            yield Conversation(
                id=convo.id,
                utterances=utterances,
                has_derailed=has_derailed,
                toxicity_scores=toxicity_scores,
                metadata=dict(convo.meta) if hasattr(convo, 'meta') else {}
            )

    def get_all(self) -> List[Conversation]:
        """Load all conversations at once."""
        return list(self.load())

    def get_derailment_summary(self) -> Dict[str, int]:
        """Get count of derailed vs civil conversations."""
        self._load_corpus()

        derailed = 0
        civil = 0

        for convo in self._corpus.iter_conversations():
            if hasattr(convo, 'meta') and convo.meta:
                if convo.meta.get('conversation_has_personal_attack', False):
                    derailed += 1
                else:
                    civil += 1

        return {'derailed': derailed, 'civil': civil}

    def get_toxicity_distribution(self) -> Tuple[List[float], List[float]]:
        """
        Get toxicity score distributions for derailed and civil conversations.

        Returns:
            Tuple of (derailed_toxicities, civil_toxicities)
        """
        derailed_tox = []
        civil_tox = []

        for convo in self.load():
            if convo.toxicity_scores:
                max_tox = convo.max_toxicity
                if convo.has_derailed:
                    derailed_tox.append(max_tox)
                else:
                    civil_tox.append(max_tox)

        return derailed_tox, civil_tox


class CombinedCorpusLoader:
    """
    Load and combine multiple corpora for comparative analysis.
    """

    def __init__(self):
        self.loaders = {}
        self.documents = []

    def add_brown(self, genres: Optional[List[str]] = None):
        """Add Brown Corpus documents."""
        loader = BrownCorpusLoader(genres=genres)
        self.loaders['brown'] = loader

        for doc in loader.load():
            doc.metadata = doc.metadata or {}
            doc.metadata['corpus'] = 'brown'
            self.documents.append(doc)

        return self

    def add_cga(self, dataset: str = 'wiki'):
        """Add CGA conversations as documents."""
        loader = CGALoader(dataset=dataset)
        self.loaders[f'cga_{dataset}'] = loader

        for convo in loader.load():
            doc = Document(
                id=convo.id,
                text=convo.full_text,
                genre=f"cga_{'derailed' if convo.has_derailed else 'civil'}",
                metadata={
                    'corpus': f'cga_{dataset}',
                    'has_derailed': convo.has_derailed,
                    'max_toxicity': convo.max_toxicity,
                    'mean_toxicity': convo.mean_toxicity,
                    'num_turns': len(convo.utterances)
                }
            )
            self.documents.append(doc)

        return self

    def get_all_documents(self) -> List[Document]:
        """Get all loaded documents."""
        return self.documents

    def get_by_corpus(self, corpus: str) -> List[Document]:
        """Get documents from a specific corpus."""
        return [d for d in self.documents
                if d.metadata and d.metadata.get('corpus') == corpus]

    def get_by_genre(self, genre: str) -> List[Document]:
        """Get documents from a specific genre."""
        return [d for d in self.documents if d.genre == genre]


if __name__ == "__main__":
    # Test Brown Corpus loader
    print("Testing Brown Corpus loader...")
    brown_loader = BrownCorpusLoader(genres=['news', 'fiction'])
    print(f"Genre summary: {brown_loader.get_genre_summary()}")

    for i, doc in enumerate(brown_loader.load()):
        print(f"  {doc}")
        if i >= 2:
            break

    # Test CGA loader (only if convokit is available)
    try:
        print("\nTesting CGA loader...")
        cga_loader = CGALoader(dataset='wiki')
        for i, convo in enumerate(cga_loader.load()):
            print(f"  {convo}")
            if i >= 2:
                break
    except ImportError as e:
        print(f"ConvoKit not available: {e}")
