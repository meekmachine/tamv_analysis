#!/usr/bin/env python3
"""
TAMV Feature Extraction Script

Extracts TAMV (Tense, Aspect, Mood, Voice) distributions from Brown Corpus
and Conversations Gone Awry, outputting clean data files for analysis.

This script focuses on data extraction without visualization overhead.

Usage:
    python extract_tamv.py                    # Extract from both corpora
    python extract_tamv.py --corpus brown     # Brown corpus only
    python extract_tamv.py --corpus cga       # CGA only
    python extract_tamv.py --max-docs 100     # Limit documents (testing)

Outputs:
    output/tamv_profiles.csv         - Document-level TAMV frequencies
    output/tamv_summary_by_genre.csv - Aggregated by genre
    output/tamv_summary_by_corpus.csv - Aggregated by corpus
    output/extraction_report.txt     - Summary statistics
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

from .tamv_extractor import TAMVExtractor, get_all_tamv_labels
from .corpus_loaders import BrownCorpusLoader, CGALoader, Document
from .tamv_aggregator import TAMVAggregator


def extract_brown(max_docs: int = None, genres: list = None, verbose: bool = True):
    """Extract TAMV from Brown Corpus."""
    if verbose:
        print("\n[Brown Corpus]")
        print("-" * 40)

    loader = BrownCorpusLoader(genres=genres)
    docs = []

    for i, doc in enumerate(loader.load()):
        doc.metadata = doc.metadata or {}
        doc.metadata['corpus'] = 'brown'
        docs.append(doc)
        if max_docs and i + 1 >= max_docs:
            break

    if verbose:
        genre_counts = {}
        for doc in docs:
            genre_counts[doc.genre] = genre_counts.get(doc.genre, 0) + 1
        print(f"Loaded {len(docs)} documents")
        print(f"Genres: {dict(sorted(genre_counts.items()))}")

    return docs


def extract_cga(dataset: str = 'wiki', max_docs: int = None, verbose: bool = True):
    """Extract TAMV from CGA corpus."""
    if verbose:
        print(f"\n[CGA - {dataset.upper()}]")
        print("-" * 40)

    loader = CGALoader(dataset=dataset)
    docs = []
    n_derailed = 0

    for i, convo in enumerate(loader.load()):
        if convo.has_derailed:
            n_derailed += 1

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

        if max_docs and i + 1 >= max_docs:
            break

    if verbose:
        print(f"Loaded {len(docs)} conversations")
        print(f"  Derailed: {n_derailed}")
        print(f"  Civil: {len(docs) - n_derailed}")

    return docs


def run_extraction(
    corpora: list = None,
    cga_dataset: str = 'wiki',
    brown_genres: list = None,
    max_docs: int = None,
    output_dir: str = 'output',
    verbose: bool = True
):
    """
    Run TAMV extraction on specified corpora.

    Args:
        corpora: List of corpora to process ['brown', 'cga'] or None for both
        cga_dataset: CGA dataset ('wiki' or 'cmv')
        brown_genres: Brown genres to include (None = all)
        max_docs: Maximum documents per corpus
        output_dir: Output directory
        verbose: Print progress

    Returns:
        DataFrame with document-level TAMV profiles
    """
    if corpora is None:
        corpora = ['brown', 'cga']

    all_docs = []

    # Load corpora
    if 'brown' in corpora:
        brown_docs = extract_brown(max_docs=max_docs, genres=brown_genres, verbose=verbose)
        all_docs.extend(brown_docs)

    if 'cga' in corpora:
        cga_docs = extract_cga(dataset=cga_dataset, max_docs=max_docs, verbose=verbose)
        all_docs.extend(cga_docs)

    if not all_docs:
        print("No documents to process.")
        return None

    if verbose:
        print(f"\nTotal documents: {len(all_docs)}")
        print("\n[Extracting TAMV Features]")
        print("-" * 40)

    # Extract TAMV profiles
    aggregator = TAMVAggregator()
    profiles = aggregator.process_documents(all_docs, show_progress=verbose)

    if verbose:
        print(f"Extracted {len(profiles)} TAMV profiles")
        print(f"Unique TAMV labels found: {len(aggregator.all_labels)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get DataFrames
    df = aggregator.to_dataframe()
    genre_summary = aggregator.get_genre_summary()
    corpus_summary = aggregator.get_corpus_summary()

    # Save outputs
    profiles_path = os.path.join(output_dir, 'tamv_profiles.csv')
    genre_path = os.path.join(output_dir, 'tamv_summary_by_genre.csv')
    corpus_path = os.path.join(output_dir, 'tamv_summary_by_corpus.csv')
    report_path = os.path.join(output_dir, 'extraction_report.txt')

    df.to_csv(profiles_path, index=False)
    genre_summary.to_csv(genre_path)
    corpus_summary.to_csv(corpus_path)

    if verbose:
        print(f"\nSaved: {profiles_path}")
        print(f"Saved: {genre_path}")
        print(f"Saved: {corpus_path}")

    # Generate report
    report = generate_report(df, genre_summary, corpus_summary, aggregator)
    with open(report_path, 'w') as f:
        f.write(report)

    if verbose:
        print(f"Saved: {report_path}")
        print("\n" + "=" * 60)
        print(report)

    return df, genre_summary, corpus_summary


def generate_report(df, genre_summary, corpus_summary, aggregator):
    """Generate extraction summary report."""
    lines = [
        "=" * 60,
        "TAMV EXTRACTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET SUMMARY",
        "-" * 40,
        f"Total documents: {len(df)}",
        f"Unique TAMV labels: {len(aggregator.all_labels)}",
        "",
    ]

    # Corpus breakdown
    if 'corpus' in df.columns:
        lines.append("By corpus:")
        for corpus, count in df['corpus'].value_counts().items():
            lines.append(f"  {corpus}: {count}")
        lines.append("")

    # Genre breakdown
    if 'genre' in df.columns:
        lines.append("By genre:")
        for genre, count in df['genre'].value_counts().items():
            lines.append(f"  {genre}: {count}")
        lines.append("")

    # Top TAMV labels overall
    tamv_cols = [c for c in df.columns if '-' in c and c not in ['document_id', 'genre', 'corpus']]
    if tamv_cols:
        means = df[tamv_cols].mean().sort_values(ascending=False)
        lines.append("TOP 10 TAMV LABELS (overall mean frequency)")
        lines.append("-" * 40)
        for label, freq in means.head(10).items():
            lines.append(f"  {label}: {freq:.4f}")
        lines.append("")

    # Corpus comparison (if both present)
    if corpus_summary is not None and len(corpus_summary) > 1:
        lines.append("CORPUS COMPARISON (top 5 differentiating labels)")
        lines.append("-" * 40)

        # Find labels with biggest differences between corpora
        if len(corpus_summary.columns) >= 2:
            diff = abs(corpus_summary.iloc[:, 0] - corpus_summary.iloc[:, 1])
            top_diff = diff.sort_values(ascending=False).head(5)

            corpus_names = corpus_summary.columns.tolist()
            for label in top_diff.index:
                vals = [f"{corpus_summary.loc[label, c]:.4f}" for c in corpus_names]
                lines.append(f"  {label}:")
                for c, v in zip(corpus_names, vals):
                    lines.append(f"    {c}: {v}")
        lines.append("")

    # Genre highlights
    lines.append("GENRE HIGHLIGHTS")
    lines.append("-" * 40)

    top_by_genre = aggregator.get_top_tamv_by_genre(n=3)
    for genre, labels in sorted(top_by_genre.items()):
        lines.append(f"\n{genre}:")
        for label, freq in labels:
            lines.append(f"  {label}: {freq:.4f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract TAMV features from corpora',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Both corpora, full extraction
  %(prog)s --corpus brown               # Brown corpus only
  %(prog)s --corpus cga --dataset cmv   # CGA CMV dataset only
  %(prog)s --max-docs 50 --output test  # Quick test run
        """
    )

    parser.add_argument('--corpus', choices=['brown', 'cga', 'both'],
                       default='both',
                       help='Corpus to extract from (default: both)')

    parser.add_argument('--genres', nargs='+',
                       help='Brown corpus genres to include (default: all)')

    parser.add_argument('--dataset', choices=['wiki', 'cmv'],
                       default='wiki',
                       help='CGA dataset (default: wiki)')

    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')

    parser.add_argument('--max-docs', type=int, default=None,
                       help='Max documents per corpus (for testing)')

    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("TAMV FEATURE EXTRACTION")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine corpora
    if args.corpus == 'both':
        corpora = ['brown', 'cga']
    else:
        corpora = [args.corpus]

    # Run extraction
    run_extraction(
        corpora=corpora,
        cga_dataset=args.dataset,
        brown_genres=args.genres,
        max_docs=args.max_docs,
        output_dir=args.output,
        verbose=verbose
    )

    if verbose:
        print(f"\nExtraction complete. Results in: {args.output}/")


if __name__ == '__main__':
    main()
