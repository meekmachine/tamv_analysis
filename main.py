#!/usr/bin/env python3
"""
TAMV Analysis - Main Entry Point

Analyzes Tense, Aspect, Mood, and Voice distributions across:
- Brown Corpus (edited written English)
- Conversations Gone Awry (online conversations)

Usage:
    python main.py --corpus brown --genres news fiction
    python main.py --corpus cga --dataset wiki
    python main.py --corpus all --output results
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tamv_extractor import TAMVExtractor
from src.corpus_loaders import BrownCorpusLoader, CGALoader, Document
from src.tamv_aggregator import TAMVAggregator, ConversationTAMVAnalyzer
from src.visualization import TAMVVisualizer
from src.toxicity_analysis import ToxicityAnalyzer


def analyze_brown(genres: list = None, output_dir: str = 'output',
                  max_docs: int = None, verbose: bool = True):
    """
    Analyze TAMV distributions in the Brown Corpus.

    Args:
        genres: List of genre codes (None = all genres)
        output_dir: Directory for output files
        max_docs: Maximum documents to process (for testing)
        verbose: Print progress info
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BROWN CORPUS TAMV ANALYSIS")
        print("=" * 60)

    # Load corpus
    loader = BrownCorpusLoader(genres=genres)
    if verbose:
        print(f"\nGenre summary: {loader.get_genre_summary()}")

    docs = []
    for i, doc in enumerate(loader.load()):
        doc.metadata = doc.metadata or {}
        doc.metadata['corpus'] = 'brown'
        docs.append(doc)
        if max_docs and i + 1 >= max_docs:
            break

    if verbose:
        print(f"Loaded {len(docs)} documents")

    # Extract TAMV profiles
    aggregator = TAMVAggregator()
    profiles = aggregator.process_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Extracted {len(profiles)} TAMV profiles")

    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    viz = TAMVVisualizer(output_dir=output_dir)

    if verbose:
        print("\nCreating visualizations...")

    # PCA plot by genre
    viz.plot_pca_2d(aggregator, color_by='genre',
                   title='Brown Corpus: TAMV Distribution by Genre',
                   save_path=os.path.join(output_dir, 'brown_pca_genre.png'))

    # PCA loadings
    viz.plot_pca_loadings(save_path=os.path.join(output_dir, 'brown_pca_loadings.png'))

    # Genre heatmap
    viz.plot_genre_heatmap(aggregator,
                          save_path=os.path.join(output_dir, 'brown_genre_heatmap.png'))

    # Export data
    df = aggregator.to_dataframe()
    df.to_csv(os.path.join(output_dir, 'brown_tamv_profiles.csv'), index=False)

    # Export graph data for 3D visualization
    viz.export_for_3d_visualization(aggregator,
                                    output_path=os.path.join(output_dir, 'brown_graph_data.json'))

    # Summary statistics
    genre_summary = aggregator.get_genre_summary()
    genre_summary.to_csv(os.path.join(output_dir, 'brown_genre_means.csv'))

    top_tamv = aggregator.get_top_tamv_by_genre(n=5)

    if verbose:
        print("\n" + "-" * 40)
        print("TOP TAMV LABELS BY GENRE")
        print("-" * 40)
        for genre, labels in top_tamv.items():
            print(f"\n{genre}:")
            for label, freq in labels:
                print(f"  {label}: {freq:.3f}")

    return aggregator, viz


def analyze_cga(dataset: str = 'wiki', output_dir: str = 'output',
                max_convos: int = None, verbose: bool = True):
    """
    Analyze TAMV distributions in Conversations Gone Awry.

    Args:
        dataset: 'wiki' or 'cmv'
        output_dir: Directory for output files
        max_convos: Maximum conversations to process (for testing)
        verbose: Print progress info
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"CGA ({dataset.upper()}) TAMV ANALYSIS")
        print("=" * 60)

    # Load corpus
    loader = CGALoader(dataset=dataset)

    docs = []
    conversations = []

    for i, convo in enumerate(loader.load()):
        conversations.append(convo)

        # Convert to document for aggregation
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
        docs.append(doc)

        if max_convos and i + 1 >= max_convos:
            break

    if verbose:
        n_derailed = sum(1 for c in conversations if c.has_derailed)
        print(f"Loaded {len(docs)} conversations")
        print(f"  Derailed: {n_derailed}")
        print(f"  Civil: {len(docs) - n_derailed}")

    # Extract TAMV profiles
    aggregator = TAMVAggregator()
    profiles = aggregator.process_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Extracted {len(profiles)} TAMV profiles")

    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    viz = TAMVVisualizer(output_dir=output_dir)

    if verbose:
        print("\nCreating visualizations...")

    # PCA plot
    viz.plot_pca_2d(aggregator, color_by='genre',
                   title=f'CGA ({dataset}): TAMV Distribution (Derailed vs Civil)',
                   save_path=os.path.join(output_dir, f'cga_{dataset}_pca.png'))

    # PCA loadings
    viz.plot_pca_loadings(save_path=os.path.join(output_dir, f'cga_{dataset}_loadings.png'))

    # Compare derailed vs civil
    viz.plot_tamv_comparison(
        aggregator,
        groups={
            'Civil': ['cga_civil'],
            'Derailed': ['cga_derailed']
        },
        save_path=os.path.join(output_dir, f'cga_{dataset}_comparison.png')
    )

    # Export data
    df = aggregator.to_dataframe()
    df.to_csv(os.path.join(output_dir, f'cga_{dataset}_tamv_profiles.csv'), index=False)

    # Export graph data
    viz.export_for_3d_visualization(aggregator,
                                    output_path=os.path.join(output_dir, f'cga_{dataset}_graph_data.json'))

    # Toxicity analysis
    if verbose:
        print("\nRunning toxicity analysis...")

    tox_analyzer = ToxicityAnalyzer(aggregator)

    # Generate toxicity report
    report = tox_analyzer.generate_report(output_dir=output_dir)
    if verbose:
        print("\n" + report)

    # Toxicity visualizations
    try:
        tox_analyzer.plot_toxicity_distribution(
            save_path=os.path.join(output_dir, f'cga_{dataset}_toxicity_dist.png')
        )
        tox_analyzer.plot_derailed_vs_civil_comparison(
            save_path=os.path.join(output_dir, f'cga_{dataset}_derailed_civil.png')
        )
        tox_analyzer.plot_tamv_by_toxicity_band(
            save_path=os.path.join(output_dir, f'cga_{dataset}_toxicity_bands.png')
        )

        # Toxicity correlations
        corr_df = tox_analyzer.compute_correlations()
        corr_df.to_csv(os.path.join(output_dir, f'cga_{dataset}_tamv_toxicity_corr.csv'), index=False)

        viz.plot_toxicity_tamv_correlation(
            df, aggregator.all_labels,
            save_path=os.path.join(output_dir, f'cga_{dataset}_toxicity_corr.png')
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate some toxicity plots: {e}")

    return aggregator, viz, tox_analyzer


def analyze_combined(output_dir: str = 'output',
                     brown_genres: list = None,
                     cga_dataset: str = 'wiki',
                     max_docs: int = None,
                     verbose: bool = True):
    """
    Combined analysis of Brown Corpus and CGA.

    Args:
        output_dir: Directory for output files
        brown_genres: Brown genres to include (None = all)
        cga_dataset: CGA dataset ('wiki' or 'cmv')
        max_docs: Maximum documents per corpus
        verbose: Print progress
    """
    if verbose:
        print("\n" + "=" * 60)
        print("COMBINED CORPUS TAMV ANALYSIS")
        print("=" * 60)

    all_docs = []

    # Load Brown
    if verbose:
        print("\nLoading Brown Corpus...")
    brown_loader = BrownCorpusLoader(genres=brown_genres)
    for i, doc in enumerate(brown_loader.load()):
        doc.metadata = doc.metadata or {}
        doc.metadata['corpus'] = 'brown'
        all_docs.append(doc)
        if max_docs and i + 1 >= max_docs:
            break

    if verbose:
        print(f"  Loaded {len(all_docs)} Brown documents")

    # Load CGA
    if verbose:
        print(f"\nLoading CGA ({cga_dataset})...")
    cga_loader = CGALoader(dataset=cga_dataset)
    n_cga = 0
    for i, convo in enumerate(cga_loader.load()):
        doc = Document(
            id=convo.id,
            text=convo.full_text,
            genre=f"cga_{'derailed' if convo.has_derailed else 'civil'}",
            metadata={
                'corpus': f'cga_{cga_dataset}',
                'has_derailed': convo.has_derailed,
                'max_toxicity': convo.max_toxicity,
                'mean_toxicity': convo.mean_toxicity,
                'num_turns': len(convo.utterances)
            }
        )
        all_docs.append(doc)
        n_cga += 1
        if max_docs and n_cga >= max_docs:
            break

    if verbose:
        print(f"  Loaded {n_cga} CGA conversations")
        print(f"\nTotal documents: {len(all_docs)}")

    # Extract TAMV profiles
    aggregator = TAMVAggregator()
    profiles = aggregator.process_documents(all_docs, show_progress=verbose)

    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    viz = TAMVVisualizer(output_dir=output_dir)

    if verbose:
        print("\nCreating visualizations...")

    # PCA by corpus
    viz.plot_pca_2d(aggregator, color_by='corpus',
                   title='Combined Corpora: TAMV Distribution by Corpus',
                   save_path=os.path.join(output_dir, 'combined_pca_corpus.png'))

    # PCA by genre
    viz.plot_pca_2d(aggregator, color_by='genre',
                   title='Combined Corpora: TAMV Distribution by Genre',
                   save_path=os.path.join(output_dir, 'combined_pca_genre.png'))

    # Loadings
    viz.plot_pca_loadings(save_path=os.path.join(output_dir, 'combined_pca_loadings.png'))

    # Heatmap
    viz.plot_genre_heatmap(aggregator,
                          save_path=os.path.join(output_dir, 'combined_genre_heatmap.png'))

    # Export data
    df = aggregator.to_dataframe()
    df.to_csv(os.path.join(output_dir, 'combined_tamv_profiles.csv'), index=False)

    # Graph data
    viz.export_for_3d_visualization(aggregator,
                                    output_path=os.path.join(output_dir, 'combined_graph_data.json'))

    # Corpus comparison
    corpus_summary = aggregator.get_corpus_summary()
    corpus_summary.to_csv(os.path.join(output_dir, 'combined_corpus_means.csv'))

    if verbose:
        print("\n" + "-" * 40)
        print("CORPUS SUMMARY")
        print("-" * 40)
        print(corpus_summary.T.to_string())

    return aggregator, viz


def main():
    parser = argparse.ArgumentParser(
        description='TAMV (Tense, Aspect, Mood, Voice) Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --corpus brown --genres news fiction learned
  %(prog)s --corpus cga --dataset wiki
  %(prog)s --corpus all --max-docs 100 --output results
        """
    )

    parser.add_argument('--corpus', choices=['brown', 'cga', 'all'],
                       default='brown',
                       help='Corpus to analyze (default: brown)')

    parser.add_argument('--genres', nargs='+',
                       help='Brown corpus genres to include (default: all)')

    parser.add_argument('--dataset', choices=['wiki', 'cmv'],
                       default='wiki',
                       help='CGA dataset to use (default: wiki)')

    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')

    parser.add_argument('--max-docs', type=int, default=None,
                       help='Maximum documents to process (for testing)')

    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    verbose = not args.quiet

    # Print header
    if verbose:
        print("\n" + "=" * 60)
        print("TAMV ANALYSIS TOOL")
        print("Tense, Aspect, Mood, Voice Distribution Analysis")
        print("=" * 60)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {args.output}")

    try:
        if args.corpus == 'brown':
            analyze_brown(
                genres=args.genres,
                output_dir=args.output,
                max_docs=args.max_docs,
                verbose=verbose
            )

        elif args.corpus == 'cga':
            analyze_cga(
                dataset=args.dataset,
                output_dir=args.output,
                max_convos=args.max_docs,
                verbose=verbose
            )

        elif args.corpus == 'all':
            analyze_combined(
                output_dir=args.output,
                brown_genres=args.genres,
                cga_dataset=args.dataset,
                max_docs=args.max_docs,
                verbose=verbose
            )

        if verbose:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(f"Results saved to: {args.output}/")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == '__main__':
    main()
