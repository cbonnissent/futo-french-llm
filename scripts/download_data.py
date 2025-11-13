#!/usr/bin/env python3
"""
Script pour télécharger et préparer les données françaises
"""

import argparse
from pathlib import Path
import gzip
import shutil

from datasets import load_dataset


def download_wikipedia_fr(output_dir: str, max_samples: int = None):
    """Télécharge Wikipedia français"""
    print("Téléchargement de Wikipedia français...")

    dataset = load_dataset(
        "wikipedia",
        "20220301.fr",
        split="train",
        trust_remote_code=True
    )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Dataset chargé: {len(dataset)} articles")

    # Sauvegarder en fichier texte
    output_file = Path(output_dir) / "wikipedia_fr.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            text = item['text'].strip()
            if text:
                f.write(text + '\n')

    print(f"✓ Sauvegardé: {output_file}")
    print(f"  Taille: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return output_file


def download_oscar_fr(output_dir: str, max_samples: int = None):
    """Télécharge OSCAR corpus français"""
    print("Téléchargement de OSCAR français...")

    dataset = load_dataset(
        "oscar-corpus/OSCAR-2201",
        "fr",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    output_file = Path(output_dir) / "oscar_fr.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            text = item['text'].strip()
            if text:
                f.write(text + '\n')
                count += 1

                if max_samples and count >= max_samples:
                    break

    print(f"✓ Sauvegardé: {output_file}")
    print(f"  {count} documents")
    print(f"  Taille: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return output_file


def download_mc4_fr(output_dir: str, max_samples: int = 100000):
    """Télécharge mC4 français (Common Crawl)"""
    print("Téléchargement de mC4 français...")

    dataset = load_dataset(
        "mc4",
        "fr",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    output_file = Path(output_dir) / "mc4_fr.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            text = item['text'].strip()
            if text and len(text) > 100:  # Filtrer les textes trop courts
                f.write(text + '\n')
                count += 1

                if count % 10000 == 0:
                    print(f"  {count} documents...")

                if max_samples and count >= max_samples:
                    break

    print(f"✓ Sauvegardé: {output_file}")
    print(f"  {count} documents")
    print(f"  Taille: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return output_file


def merge_datasets(input_files: list, output_file: str):
    """Fusionne plusieurs datasets"""
    print(f"\nFusion de {len(input_files)} fichiers...")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if not Path(input_file).exists():
                print(f"⚠️  {input_file} n'existe pas, ignoré")
                continue

            print(f"  Ajout de {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_lines += 1

    print(f"\n✓ Dataset fusionné: {output_file}")
    print(f"  {total_lines} lignes")
    print(f"  Taille: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Télécharger et préparer les données françaises'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/raw',
        help='Dossier de sortie'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        choices=['wikipedia', 'oscar', 'mc4', 'all'],
        default=['all'],
        help='Sources de données à télécharger'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Nombre maximum d\'échantillons par source'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Fusionner tous les datasets en un seul'
    )

    args = parser.parse_args()

    # Créer le dossier de sortie
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sources = args.sources
    if 'all' in sources:
        sources = ['wikipedia', 'oscar', 'mc4']

    downloaded_files = []

    # Télécharger les sources
    if 'wikipedia' in sources:
        file_path = download_wikipedia_fr(args.output_dir, args.max_samples)
        downloaded_files.append(file_path)

    if 'oscar' in sources:
        file_path = download_oscar_fr(args.output_dir, args.max_samples)
        downloaded_files.append(file_path)

    if 'mc4' in sources:
        file_path = download_mc4_fr(args.output_dir, args.max_samples)
        downloaded_files.append(file_path)

    # Fusionner si demandé
    if args.merge and len(downloaded_files) > 1:
        merge_datasets(
            downloaded_files,
            Path(args.output_dir) / "french_corpus_merged.txt"
        )

    print("\n✓ Téléchargement terminé!")


if __name__ == '__main__':
    main()
