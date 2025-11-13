#!/usr/bin/env python3
"""
Script pour entraîner un tokenizer SentencePiece français pour FUTO Keyboard
avec whitespace en suffixe et tokens spéciaux pour l'autocorrection
"""

import sentencepiece as spm
import argparse
from pathlib import Path


def train_french_tokenizer(
    input_file: str,
    model_prefix: str = "french_keyboard",
    vocab_size: int = 15008,
    character_coverage: float = 0.9995,
):
    """
    Entraîne un tokenizer SentencePiece pour le français avec configuration FUTO

    Args:
        input_file: Chemin vers le fichier texte d'entraînement
        model_prefix: Préfixe pour les fichiers de sortie
        vocab_size: Taille du vocabulaire (max ~15k recommandé)
        character_coverage: Couverture des caractères (0.9995 pour inclure accents)
    """

    # Tokens spéciaux requis par FUTO Keyboard
    special_tokens = [
        '<XBU>',  # Beginning of user input
        '<XBC>',  # Beginning of correction
        '<XEC>',  # End of correction
        '<XC0>',  # Swipe typing (optionnel, pour future)
    ]

    # Tokens de caractères individuels A-Z
    char_tokens = [f'<CHAR_{chr(i)}>' for i in range(ord('A'), ord('Z') + 1)]

    # Tokens de caractères français avec accents
    french_chars = ['À', 'Â', 'Ä', 'Æ', 'Ç', 'É', 'È', 'Ê', 'Ë', 'Î', 'Ï',
                    'Ô', 'Ö', 'Ù', 'Û', 'Ü', 'Ÿ', 'Œ']
    french_char_tokens = [f'<CHAR_{char}>' for char in french_chars]

    all_special_tokens = special_tokens + char_tokens + french_char_tokens

    print(f"Entraînement du tokenizer avec {vocab_size} tokens...")
    print(f"{len(all_special_tokens)} tokens spéciaux définis")

    # Configuration SentencePiece avec whitespace en suffixe (CRITIQUE!)
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='unigram',  # ou 'bpe'

        # CRITIQUE: whitespace en suffixe pour FUTO Keyboard
        treat_whitespace_as_suffix=True,

        # Tokens spéciaux
        user_defined_symbols=all_special_tokens,

        # Normalisation pour le français
        normalization_rule_name='nmt_nfkc_cf',
        remove_extra_whitespaces=True,

        # Pas de split sur les chiffres/lettres
        split_digits=False,
        split_by_unicode_script=False,
        split_by_whitespace=True,
        split_by_number=False,

        # Tokens de contrôle
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,

        # Éviter de split les tokens spéciaux
        byte_fallback=False,
    )

    print(f"✓ Tokenizer sauvegardé: {model_prefix}.model et {model_prefix}.vocab")

    # Test du tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    print("\n=== Tests du tokenizer ===")
    test_sentences = [
        "Bonjour, comment allez-vous?",
        "Je n'ai pas compris la leçon d'aujourd'hui.",
        "C'est très intéressant!",
        "L'été à Montréal est magnifique.",
    ]

    for sentence in test_sentences:
        tokens = sp.encode_as_pieces(sentence)
        print(f"\nPhrase: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {sp.encode_as_ids(sentence)}")
        print(f"Décodage: {sp.decode_pieces(tokens)}")

    # Vérifier que les tokens spéciaux sont présents
    print("\n=== Vérification des tokens spéciaux ===")
    for token in special_tokens[:5] + char_tokens[:3] + french_char_tokens[:3]:
        token_id = sp.piece_to_id(token)
        print(f"{token}: ID={token_id}")

    print("\n✓ Tokenizer entraîné avec succès!")
    print(f"  Taille vocabulaire: {sp.vocab_size()}")
    print(f"  Whitespace en suffixe: activé")


def main():
    parser = argparse.ArgumentParser(
        description='Entraîner un tokenizer SentencePiece français pour FUTO Keyboard'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Fichier texte d\'entraînement (corpus français)'
    )
    parser.add_argument(
        '--model-prefix',
        type=str,
        default='french_keyboard',
        help='Préfixe pour les fichiers de sortie (défaut: french_keyboard)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=15008,
        help='Taille du vocabulaire (défaut: 15008, max recommandé: ~15k)'
    )
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='Couverture des caractères (défaut: 0.9995 pour français)'
    )

    args = parser.parse_args()

    # Vérifier que le fichier existe
    if not Path(args.input).exists():
        print(f"Erreur: Le fichier {args.input} n'existe pas")
        return

    train_french_tokenizer(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )


if __name__ == '__main__':
    main()
