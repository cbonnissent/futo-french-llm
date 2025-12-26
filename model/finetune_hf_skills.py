#!/usr/bin/env python3
"""
Finetuning avec HuggingFace TRL (Transformer Reinforcement Learning)
Utilise SFT et DPO comme recommandé par HF Skills

Alternative moderne au finetuning custom pour FUTO Keyboard
"""

import argparse
from pathlib import Path
from typing import List
import torch
from datasets import Dataset, load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import sentencepiece as spm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from synthetic_errors.generate_errors import AzertyErrorGenerator


class FrenchKeyboardDatasetBuilder:
    """Construit des datasets au format HuggingFace pour SFT"""

    def __init__(self, tokenizer_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.error_generator = AzertyErrorGenerator()

    def build_phase1_dataset(
        self,
        word_list: List[str],
        num_examples_per_word: int = 3,
    ) -> Dataset:
        """
        Phase 1: Corrections individuelles en format conversationnel

        Format SFT standard:
        {
            "prompt": "Corriger: bonjoyr",
            "completion": "bonjour"
        }
        """
        data = []

        for word in word_list:
            if len(word) < 2:
                continue

            for _ in range(num_examples_per_word):
                misspelled = self.error_generator.generate_word_error(word)
                if misspelled != word:
                    # Format FUTO intégré dans le prompt
                    formatted = self.error_generator.format_for_training(word, misspelled)

                    data.append({
                        "text": formatted,  # Format SFT unifié
                    })

        return Dataset.from_list(data)

    def build_phase2_dataset(
        self,
        corpus_texts: List[str],
        error_prob: float = 0.33,
    ) -> Dataset:
        """
        Phase 2: Corrections en contexte

        Format: Texte avec corrections intégrées FUTO
        """
        data = []

        for text in corpus_texts:
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue

                # Générer phrase avec corrections
                words = sentence.split()
                result_parts = []

                for word in words:
                    clean_word = word.strip('.,!?;:')

                    if (
                        clean_word
                        and len(clean_word) > 2
                        and torch.rand(1).item() < error_prob
                    ):
                        misspelled = self.error_generator.generate_word_error(clean_word)
                        if misspelled != clean_word:
                            formatted = self.error_generator.format_for_training(
                                clean_word, misspelled
                            )
                            result_parts.append(formatted)
                        else:
                            result_parts.append(word)
                    else:
                        result_parts.append(word)

                if result_parts:
                    data.append({
                        "text": ' '.join(result_parts),
                    })

        return Dataset.from_list(data)


def finetune_with_sft(
    model_path: str,
    tokenizer_path: str,
    dataset: Dataset,
    output_dir: str,
    phase_name: str = "sft",
    num_train_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_seq_length: int = 512,
    use_wandb: bool = False,
):
    """
    Fine-tuning avec SFTTrainer de TRL (recommandé HF Skills)

    Avantages vs script custom:
    - Optimisations automatiques (packing, attention mask)
    - Support Flash Attention 2
    - Meilleure gestion mémoire
    - Logging intégré
    """
    print("=" * 60)
    print(f"Fine-tuning SFT - Phase: {phase_name}")
    print("=" * 60)

    # Charger le modèle
    print(f"\nChargement du modèle depuis {model_path}...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,  # 🚀 Accélération 2-3x
        trust_remote_code=True,
    )

    # Tokenizer (pour SFTTrainer, on peut utiliser AutoTokenizer wrapper)
    # Note: Ici on garde SentencePiece mais on pourrait migrer vers HF tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # Configuration SFT optimisée
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,

        # Optimisations HF Skills recommandées
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # Meilleur que linear
        warmup_ratio=0.1,  # 10% warmup

        # Optimisations mémoire
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,

        # Optimisations performance
        optim="adamw_torch_fused",  # 🚀 Plus rapide qu'adamw_torch
        group_by_length=True,  # 🚀 Regroupe séquences similaires
        dataloader_num_workers=4,

        # Sauvegarde optimisée
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,

        # Logging
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to=["wandb"] if use_wandb else ["tensorboard"],

        # Push to Hub (intégration HF Skills)
        push_to_hub=False,  # Activable avec HF token
    )

    # SFTTrainer avec optimisations
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # Champ contenant le texte

        # Optimisations SFT
        max_seq_length=max_seq_length,
        packing=True,  # 🚀 Pack multiple examples (2-3x plus rapide!)
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    # Entraînement
    print("\n" + "=" * 60)
    print("Début du fine-tuning SFT...")
    print("=" * 60 + "\n")

    trainer.train()

    # Sauvegarder
    print(f"\nSauvegarde du modèle dans {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")

    print("\n✓ Fine-tuning SFT terminé!")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tuning FUTO avec HuggingFace TRL/SFT'
    )

    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--corpus-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output/finetune_sft')
    parser.add_argument('--phase', type=str, choices=['phase1', 'phase2', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--use-wandb', action='store_true')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Builder de datasets
    builder = FrenchKeyboardDatasetBuilder(args.tokenizer_path)

    # Phase 1: Corrections individuelles
    if args.phase in ['phase1', 'all']:
        print("\n### PHASE 1: Corrections individuelles avec SFT ###")

        # Charger vocabulaire
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            # Prendre mots uniques
            unique_words = list(set([w.strip('.,!?;:').lower() for w in words if len(w) > 2]))
            unique_words = unique_words[:50000]  # Limiter

        # Construire dataset
        phase1_dataset = builder.build_phase1_dataset(
            word_list=unique_words,
            num_examples_per_word=3,
        )

        print(f"Dataset Phase 1: {len(phase1_dataset)} exemples")

        # Fine-tuner
        finetune_with_sft(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            dataset=phase1_dataset,
            output_dir=f"{args.output_dir}/phase1",
            phase_name="phase1",
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=128,
            use_wandb=args.use_wandb,
        )

    # Phase 2: Corrections en contexte
    if args.phase in ['phase2', 'all']:
        print("\n### PHASE 2: Corrections en contexte avec SFT ###")

        # Charger corpus
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            corpus_texts = [line.strip() for line in f if len(line.strip()) > 50]
            corpus_texts = corpus_texts[:100000]

        # Construire dataset
        phase2_dataset = builder.build_phase2_dataset(
            corpus_texts=corpus_texts,
            error_prob=0.33,
        )

        print(f"Dataset Phase 2: {len(phase2_dataset)} exemples")

        # Fine-tuner
        model_path_phase2 = (
            f"{args.output_dir}/phase1/final"
            if args.phase == 'all'
            else args.model_path
        )

        finetune_with_sft(
            model_path=model_path_phase2,
            tokenizer_path=args.tokenizer_path,
            dataset=phase2_dataset,
            output_dir=f"{args.output_dir}/phase2",
            phase_name="phase2",
            num_train_epochs=args.epochs,
            batch_size=args.batch_size // 2,
            learning_rate=args.learning_rate * 0.5,
            max_seq_length=512,
            use_wandb=args.use_wandb,
        )

    print("\n" + "=" * 60)
    print("✓ Fine-tuning SFT complet terminé!")
    print(f"Modèle final: {args.output_dir}/phase2/final")
    print("=" * 60)


if __name__ == '__main__':
    main()
