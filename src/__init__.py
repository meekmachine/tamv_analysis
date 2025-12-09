# TAMV Analysis Package
# Tense, Aspect, Mood, Voice analysis for corpus linguistics

from .tamv_extractor import TAMVExtractor, TAMVLabel, Tense, Aspect, Mood, Voice, get_all_tamv_labels
from .corpus_loaders import BrownCorpusLoader, CGALoader, Document, Conversation, CombinedCorpusLoader
from .tamv_aggregator import TAMVAggregator, TAMVProfile, ConversationTAMVAnalyzer
from .visualization import TAMVVisualizer
from .analyze_cga import CGAAnalyzer, CGAVisualizer

# Entry point functions
from .extract_tamv import run_extraction

__version__ = '0.1.0'
__all__ = [
    # Core classes
    'TAMVExtractor',
    'TAMVLabel',
    'Tense',
    'Aspect',
    'Mood',
    'Voice',
    'get_all_tamv_labels',
    # Corpus loaders
    'BrownCorpusLoader',
    'CGALoader',
    'Document',
    'Conversation',
    'CombinedCorpusLoader',
    # Aggregation
    'TAMVAggregator',
    'TAMVProfile',
    'ConversationTAMVAnalyzer',
    # Visualization & analysis
    'TAMVVisualizer',
    'CGAAnalyzer',
    'CGAVisualizer',
    # Entry points
    'run_extraction',
]
