#!/usr/bin/env python3
"""
Script de finetuning du modèle pour l'autocorrection
Trois phases de finetuning selon la doc FUTO
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict
import json

import torch
from torch.utils.data import Dataset
from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset
import sentencepiece as spm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from synthetic_errors.generate_errors import AzertyErrorGenerator


class FrenchKeyboardTokenizer:
    """Wrapper pour le tokenizer SentencePiece"""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_token_id = self.sp.pad_id()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode_as_ids(text)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        input_ids = [self.encode(t) for t in text]
        max_length = kwargs.get('max_length', None)
        if max_length:
            input_ids = [
                ids[:max_length] + [self.pad_token_id] * (max_length - len(ids))
                if len(ids) < max_length else ids[:max_length]
                for ids in input_ids
            ]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor([
                [1 if id != self.pad_token_id else 0 for id in ids]
                for ids in input_ids
            ])
        }


class CorrectionDataset(Dataset):
    """Dataset pour l'entraînement à la correction"""

    def __init__(
        self,
        examples: List[str],
        tokenizer: FrenchKeyboardTokenizer,
        max_length: int = 128,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(text, max_length=self.max_length)

        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': encoding['input_ids'][0].clone(),
        }


def generate_phase1_data(
    word_list: List[str],
    error_generator: AzertyErrorGenerator,
    num_examples_per_word: int = 5,
) -> List[str]:
    """
    Phase 1: Corrections individuelles sans contexte
    Format: <XBU><CHAR_M><CHAR_E><CHAR_S><CHAR_O><CHAR_N><XBC>maison <XEC>

    Args:
        word_list: Liste de mots français
        error_generator: Générateur de fautes
        num_examples_per_word: Nombre d'exemples par mot

    Returns:
        Liste d'exemples formatés
    """
    print(f"Génération de Phase 1: corrections individuelles...")

    examples = []

    for word in word_list:
        if len(word) < 2:
            continue

        for _ in range(num_examples_per_word):
            misspelled = error_generator.generate_word_error(word)
            if misspelled != word:
                formatted = error_generator.format_for_training(word, misspelled)
                examples.append(formatted)

    print(f"  → {len(examples)} exemples générés")
    return examples


def generate_phase2_data(
    corpus_texts: List[str],
    error_generator: AzertyErrorGenerator,
    error_prob: float = 0.33,
) -> List[str]:
    """
    Phase 2: Corrections en contexte
    ~1/3 des mots avec fautes dans des phrases

    Args:
        corpus_texts: Liste de phrases/textes
        error_generator: Générateur de fautes
        error_prob: Probabilité qu'un mot ait une faute

    Returns:
        Liste de textes avec corrections intégrées
    """
    print(f"Génération de Phase 2: corrections en contexte...")

    examples = []

    for text in corpus_texts:
        # Diviser en phrases si nécessaire
        sentences = text.split('.')
        augmented_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            words = sentence.split()
            result_parts = []

            for word in words:
                # Séparer ponctuation
                punctuation = ''
                clean_word = word
                if word and word[-1] in '.,!?;:':
                    punctuation = word[-1]
                    clean_word = word[:-1]

                # Introduire erreur avec probabilité error_prob
                if (
                    clean_word
                    and len(clean_word) > 2
                    and error_generator.error_rate > 0
                    and torch.rand(1).item() < error_prob
                ):
                    misspelled = error_generator.generate_word_error(clean_word)
                    if misspelled != clean_word:
                        # Format de correction
                        formatted = error_generator.format_for_training(
                            clean_word, misspelled
                        )
                        result_parts.append(formatted)
                    else:
                        result_parts.append(word)
                else:
                    result_parts.append(word)

            if result_parts:
                augmented_sentences.append(' '.join(result_parts))

        if augmented_sentences:
            examples.append('. '.join(augmented_sentences) + '.')

    print(f"  → {len(examples)} exemples générés")
    return examples


def load_word_frequency(corpus_path: str, max_words: int = 50000) -> Dict[str, int]:
    """Charge et compte les fréquences des mots"""
    print(f"Analyse des fréquences de mots dans {corpus_path}...")

    word_freq = {}

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                # Nettoyer
                word = word.strip('.,!?;:').lower()
                if word and len(word) > 1:
                    word_freq[word] = word_freq.get(word, 0) + 1

    # Trier par fréquence
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    print(f"  → {len(sorted_words)} mots uniques trouvés")
    print(f"  → Top 10: {sorted_words[:10]}")

    return dict(sorted_words[:max_words])


def apply_log_sampling(word_freq: Dict[str, int]) -> List[str]:
    """
    Applique le sampling log(N) pour réduire le biais vers mots communs

    Si un mot apparaît N fois, il apparaît log(N) fois dans le dataset
    """
    import math

    sampled_words = []

    for word, freq in word_freq.items():
        # Nombre d'occurrences = log(freq)
        num_occurrences = max(1, int(math.log(freq + 1)))
        sampled_words.extend([word] * num_occurrences)

    print(f"  Sampling log(N): {len(word_freq)} mots → {len(sampled_words)} exemples")

    return sampled_words


def finetune_model(
    model_path: str,
    tokenizer_path: str,
    corpus_path: str,
    output_dir: str,
    phase: str = 'all',
    num_train_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_wandb: bool = False,
):
    """
    Finetuning du modèle en plusieurs phases

    Args:
        model_path: Chemin vers le modèle préentraîné
        tokenizer_path: Chemin vers le tokenizer
        corpus_path: Chemin vers le corpus français
        output_dir: Dossier de sortie
        phase: 'phase1', 'phase2', 'phase3', ou 'all'
    """
    print("=" * 60)
    print(f"Finetuning - Phase: {phase}")
    print("=" * 60)

    # Charger le modèle et tokenizer
    print(f"\nChargement du modèle depuis {model_path}...")
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = FrenchKeyboardTokenizer(tokenizer_path)

    # Générateur de fautes
    error_generator = AzertyErrorGenerator(error_rate=0.15)

    # Générer les données selon la phase
    if phase in ['phase1', 'all']:
        print("\n### PHASE 1: Corrections individuelles ###")

        # Charger le vocabulaire avec fréquences
        word_freq = load_word_frequency(corpus_path, max_words=50000)

        # Appliquer log sampling
        word_list = apply_log_sampling(word_freq)

        # Générer exemples de correction
        phase1_examples = generate_phase1_data(
            word_list=word_list,
            error_generator=error_generator,
            num_examples_per_word=3,
        )

        # Créer dataset
        phase1_dataset = CorrectionDataset(
            examples=phase1_examples,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Entraîner
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/phase1",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            save_steps=2000,
            logging_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            report_to=["wandb"] if use_wandb else [],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=phase1_dataset,
        )

        print("\nEntraînement Phase 1...")
        trainer.train()
        trainer.save_model(f"{output_dir}/phase1/final")
        print("✓ Phase 1 terminée")

    if phase in ['phase2', 'all']:
        print("\n### PHASE 2: Corrections en contexte ###")

        # Si on vient de phase 1, charger le modèle
        if phase == 'phase2':
            model = LlamaForCausalLM.from_pretrained(model_path)

        # Charger le corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_texts = [line.strip() for line in f if len(line.strip()) > 50]

        # Limiter pour l'exemple (vous pouvez augmenter)
        corpus_texts = corpus_texts[:100000]

        # Générer exemples avec corrections en contexte
        phase2_examples = generate_phase2_data(
            corpus_texts=corpus_texts,
            error_generator=error_generator,
            error_prob=0.33,
        )

        phase2_dataset = CorrectionDataset(
            examples=phase2_examples,
            tokenizer=tokenizer,
            max_length=512,
        )

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/phase2",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size // 2,  # Séquences plus longues
            learning_rate=learning_rate * 0.5,  # LR plus faible
            warmup_steps=500,
            save_steps=2000,
            logging_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            report_to=["wandb"] if use_wandb else [],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=phase2_dataset,
        )

        print("\nEntraînement Phase 2...")
        trainer.train()
        trainer.save_model(f"{output_dir}/phase2/final")
        print("✓ Phase 2 terminée")

    if phase in ['phase3', 'all']:
        print("\n### PHASE 3: Langage informel (première personne) ###")

        # Charger modèle de phase 2 si nécessaire
        if phase == 'phase3':
            model = LlamaForCausalLM.from_pretrained(model_path)

        # TODO: Charger un corpus de langage informel français
        # Pour l'instant, on utilise le même corpus avec corrections
        print("⚠️  Phase 3 nécessite un corpus de langage informel")
        print("    (forums, messages, etc.)")
        print("    Ignoré pour l'instant - implémentez avec votre corpus informel")

    print("\n" + "=" * 60)
    print("✓ Finetuning terminé!")
    print(f"Modèle final dans: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Finetuning du modèle pour autocorrection'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Chemin vers le modèle préentraîné'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Chemin vers le tokenizer .model'
    )
    parser.add_argument(
        '--corpus-path',
        type=str,
        required=True,
        help='Chemin vers le corpus français'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/finetune',
        help='Dossier de sortie'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['phase1', 'phase2', 'phase3', 'all'],
        default='all',
        help='Phase de finetuning à exécuter'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Nombre d\'époques'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Utiliser W&B'
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    finetune_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        phase=args.phase,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb,
    )


if __name__ == '__main__':
    main()
