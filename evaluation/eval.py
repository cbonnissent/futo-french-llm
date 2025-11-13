#!/usr/bin/env python3
"""
Script d'évaluation du modèle français
"""

import argparse
from pathlib import Path
from typing import List, Dict
import torch
from transformers import LlamaForCausalLM
import sentencepiece as spm
from tqdm import tqdm
import json


class FrenchKeyboardEvaluator:
    """Évaluateur pour le modèle de clavier français"""

    def __init__(self, model_path: str, tokenizer_path: str):
        print(f"Chargement du modèle depuis {model_path}...")
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✓ Modèle chargé sur GPU")
        else:
            print("⚠️  GPU non disponible, utilisation du CPU")

        print(f"Chargement du tokenizer depuis {tokenizer_path}...")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)

    def compute_perplexity(self, texts: List[str]) -> float:
        """Calcule la perplexité sur un ensemble de textes"""
        print("\nCalcul de la perplexité...")

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in tqdm(texts, desc="Évaluation"):
                # Tokenize
                input_ids = self.tokenizer.encode(text)
                if len(input_ids) < 2:
                    continue

                input_tensor = torch.tensor([input_ids])
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # Forward pass
                outputs = self.model(input_tensor, labels=input_tensor)
                loss = outputs.loss

                total_loss += loss.item() * len(input_ids)
                total_tokens += len(input_ids)

        # Perplexité = exp(moyenne de la loss)
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def test_autocorrect(self, test_cases: List[tuple]) -> Dict:
        """
        Teste la capacité de correction

        Args:
            test_cases: Liste de (mot_correct, mot_erroné)

        Returns:
            Statistiques de correction
        """
        print("\nTest de l'autocorrection...")

        correct = 0
        total = len(test_cases)
        results = []

        for correct_word, misspelled_word in tqdm(test_cases, desc="Tests"):
            # Format FUTO
            char_tokens = ''.join([
                f'<CHAR_{c.upper()}>'
                for c in misspelled_word.upper()
            ])
            prompt = f'<XBU>{char_tokens}<XBC>'

            # Tokenize
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # Générer
            with torch.no_grad():
                output = self.model.generate(
                    input_tensor,
                    max_new_tokens=10,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=self.tokenizer.piece_to_id('<XEC>'),
                )

            # Décoder
            generated_text = self.tokenizer.decode(output[0].tolist())

            # Extraire le mot prédit
            if '<XBC>' in generated_text and '<XEC>' in generated_text:
                predicted = generated_text.split('<XBC>')[1].split('<XEC>')[0].strip()
            else:
                predicted = ""

            # Vérifier si correct
            is_correct = predicted.lower() == correct_word.lower()
            if is_correct:
                correct += 1

            results.append({
                'misspelled': misspelled_word,
                'correct': correct_word,
                'predicted': predicted,
                'success': is_correct,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results,
        }

    def test_next_word_prediction(self, test_sentences: List[str], k: int = 5) -> Dict:
        """
        Teste la prédiction du mot suivant

        Args:
            test_sentences: Phrases de test (dernier mot sera masqué)
            k: Top-k accuracy

        Returns:
            Statistiques de prédiction
        """
        print(f"\nTest de prédiction du mot suivant (top-{k})...")

        top1_correct = 0
        topk_correct = 0
        total = len(test_sentences)
        results = []

        for sentence in tqdm(test_sentences, desc="Tests"):
            words = sentence.split()
            if len(words) < 2:
                continue

            # Contexte = tous les mots sauf le dernier
            context = ' '.join(words[:-1]) + ' '
            target_word = words[-1].lower()

            # Tokenize
            input_ids = self.tokenizer.encode(context)
            input_tensor = torch.tensor([input_ids])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # Prédire
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]  # Logits du dernier token

                # Top-k tokens
                topk_probs, topk_indices = torch.topk(logits, k)
                topk_tokens = [
                    self.tokenizer.decode([idx.item()]).strip().lower()
                    for idx in topk_indices
                ]

            # Vérifier si le mot cible est dans top-k
            is_top1 = topk_tokens[0] == target_word if topk_tokens else False
            is_topk = target_word in topk_tokens

            if is_top1:
                top1_correct += 1
            if is_topk:
                topk_correct += 1

            results.append({
                'context': context,
                'target': target_word,
                'predictions': topk_tokens,
                'top1_match': is_top1,
                f'top{k}_match': is_topk,
            })

        top1_accuracy = top1_correct / total if total > 0 else 0
        topk_accuracy = topk_correct / total if total > 0 else 0

        return {
            'top1_accuracy': top1_accuracy,
            f'top{k}_accuracy': topk_accuracy,
            'total': total,
            'results': results,
        }


def load_test_data(test_file: str) -> Dict:
    """Charge les données de test depuis un fichier JSON"""
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Évaluation du modèle de clavier français'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Chemin vers le modèle'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Chemin vers le tokenizer .model'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='Fichier JSON avec données de test'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation_results.json',
        help='Fichier de sortie pour les résultats'
    )

    args = parser.parse_args()

    # Créer l'évaluateur
    evaluator = FrenchKeyboardEvaluator(args.model_path, args.tokenizer_path)

    results = {}

    # Si fichier de test fourni
    if args.test_file:
        test_data = load_test_data(args.test_file)

        # Perplexité
        if 'perplexity_texts' in test_data:
            perplexity = evaluator.compute_perplexity(test_data['perplexity_texts'])
            results['perplexity'] = perplexity
            print(f"\n✓ Perplexité: {perplexity:.2f}")

        # Autocorrection
        if 'autocorrect_cases' in test_data:
            autocorrect_results = evaluator.test_autocorrect(
                test_data['autocorrect_cases']
            )
            results['autocorrect'] = autocorrect_results
            print(f"\n✓ Accuracy autocorrection: {autocorrect_results['accuracy']*100:.1f}%")

        # Prédiction mot suivant
        if 'next_word_sentences' in test_data:
            next_word_results = evaluator.test_next_word_prediction(
                test_data['next_word_sentences']
            )
            results['next_word'] = next_word_results
            print(f"\n✓ Top-1 accuracy: {next_word_results['top1_accuracy']*100:.1f}%")
            print(f"✓ Top-5 accuracy: {next_word_results['top5_accuracy']*100:.1f}%")

    else:
        # Tests rapides par défaut
        print("\nTests rapides (pas de fichier de test fourni)...")

        # Test autocorrection
        test_cases = [
            ('bonjour', 'bonjoyr'),
            ('merci', 'meric'),
            ('français', 'francais'),
            ('ordinateur', 'ordinteur'),
            ('téléphone', 'telephone'),
        ]
        autocorrect_results = evaluator.test_autocorrect(test_cases)
        results['autocorrect'] = autocorrect_results
        print(f"\n✓ Accuracy autocorrection: {autocorrect_results['accuracy']*100:.1f}%")

        # Test prédiction
        test_sentences = [
            "Comment allez-vous aujourd'hui",
            "Je vais à la maison",
            "Le temps est très beau",
        ]
        next_word_results = evaluator.test_next_word_prediction(test_sentences)
        results['next_word'] = next_word_results
        print(f"\n✓ Top-1 accuracy: {next_word_results['top1_accuracy']*100:.1f}%")

    # Sauvegarder les résultats
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Résultats sauvegardés: {args.output}")


if __name__ == '__main__':
    main()
