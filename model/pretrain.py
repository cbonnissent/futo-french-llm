#!/usr/bin/env python3
"""
Script de préentraînement du modèle Llama français pour FUTO Keyboard
Optimisé pour GPU B200
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from datasets import load_dataset, load_from_disk
import sentencepiece as spm

import sys
sys.path.append(str(Path(__file__).parent))
from config import get_default_config, get_large_config


class FrenchKeyboardTokenizer:
    """Wrapper pour le tokenizer SentencePiece"""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # IDs spéciaux
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()

        self.vocab_size = self.sp.vocab_size()

    def __call__(self, text, **kwargs):
        """Pour compatibilité avec HuggingFace"""
        if isinstance(text, str):
            text = [text]

        input_ids = [self.sp.encode_as_ids(t) for t in text]

        # Padding si nécessaire
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

    def decode(self, ids, **kwargs):
        """Décodage"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode(ids)

    def batch_decode(self, ids_list, **kwargs):
        """Décodage batch"""
        return [self.decode(ids) for ids in ids_list]


def load_french_corpus(
    data_path: str,
    tokenizer_path: str,
    max_length: int = 512,
    num_proc: int = 8,
) -> tuple:
    """
    Charge et tokenize le corpus français

    Args:
        data_path: Chemin vers les données (dossier ou fichiers)
        tokenizer_path: Chemin vers le tokenizer .model
        max_length: Longueur max des séquences
        num_proc: Nombre de processus pour le traitement

    Returns:
        (train_dataset, eval_dataset)
    """
    print(f"Chargement du corpus depuis {data_path}...")

    # Charger le tokenizer
    tokenizer = FrenchKeyboardTokenizer(tokenizer_path)

    # Charger les données
    if Path(data_path).is_dir():
        # Si c'est un dossier avec dataset preprocessé
        if (Path(data_path) / "dataset_info.json").exists():
            dataset = load_from_disk(data_path)
        else:
            # Charger tous les fichiers texte
            dataset = load_dataset(
                'text',
                data_files={'train': str(Path(data_path) / "*.txt")},
                split='train'
            )
    else:
        # Fichier unique
        dataset = load_dataset('text', data_files=data_path, split='train')

    print(f"Dataset chargé: {len(dataset)} exemples")

    # Tokenization
    def tokenize_function(examples):
        # Tokenize
        result = tokenizer(
            examples['text'],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        # Ajouter les labels (= input_ids pour language modeling)
        result['labels'] = result['input_ids'].clone()
        return result

    print("Tokenization en cours...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Split train/eval
    if 'validation' not in tokenized_dataset:
        split = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']

    print(f"Train: {len(train_dataset)} exemples")
    print(f"Eval: {len(eval_dataset)} exemples")

    return train_dataset, eval_dataset, tokenizer


def pretrain_model(
    model_config,
    train_dataset,
    eval_dataset,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 5e-4,
    warmup_steps: int = 1000,
    save_steps: int = 5000,
    eval_steps: int = 1000,
    logging_steps: int = 100,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    bf16: bool = True,  # B200 supporte bf16
    use_wandb: bool = False,
):
    """
    Préentraîne le modèle

    Args:
        model_config: Configuration du modèle
        train_dataset: Dataset d'entraînement
        eval_dataset: Dataset d'évaluation
        output_dir: Dossier de sortie
        ... autres paramètres d'entraînement
    """
    print("\nInitialisation du modèle...")

    # Créer le modèle
    llama_config = model_config.to_llama_config()
    model = LlamaForCausalLM(llama_config)

    # Afficher la taille
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle initialisé: {num_params:,} paramètres ({num_params/1e6:.1f}M)")

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,

        # Sauvegarde et évaluation
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        evaluation_strategy="steps",

        # Optimisations
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,  # Économiser la mémoire

        # Optimiseur
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Logging
        logging_dir=f"{output_dir}/logs",
        report_to=["wandb"] if use_wandb else [],

        # Divers
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # Entraînement
    print("\n" + "=" * 50)
    print("Début du préentraînement...")
    print("=" * 50 + "\n")

    trainer.train()

    # Sauvegarder le modèle final
    print(f"\nSauvegarde du modèle dans {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")

    # Évaluation finale
    print("\nÉvaluation finale...")
    metrics = trainer.evaluate()
    print(f"Perplexité finale: {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}")

    return model, trainer


def main():
    parser = argparse.ArgumentParser(
        description='Préentraînement du modèle FUTO Keyboard français'
    )

    # Données
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Chemin vers le corpus français'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Chemin vers le tokenizer .model'
    )

    # Modèle
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['default', 'large'],
        default='default',
        help='Taille du modèle (default=36M, large=100M)'
    )

    # Entraînement
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/pretrain',
        help='Dossier de sortie'
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
        help='Batch size par device'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Longueur max des séquences'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Utiliser Weights & Biases pour le logging'
    )

    args = parser.parse_args()

    # Créer le dossier de sortie
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Charger la configuration
    if args.model_size == 'default':
        # Lire la taille du vocab depuis le tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(args.tokenizer_path)
        vocab_size = sp.vocab_size()
        model_config = get_default_config(vocab_size=vocab_size)
    else:
        sp = spm.SentencePieceProcessor()
        sp.load(args.tokenizer_path)
        vocab_size = sp.vocab_size()
        model_config = get_large_config(vocab_size=vocab_size)

    model_config.print_model_size()

    # Charger les données
    train_dataset, eval_dataset, tokenizer = load_french_corpus(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
    )

    # Préentraîner
    model, trainer = pretrain_model(
        model_config=model_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
    )

    print("\n✓ Préentraînement terminé!")
    print(f"Modèle sauvegardé dans: {args.output_dir}/final")


if __name__ == '__main__':
    main()
