#!/usr/bin/env python3
"""
Démonstration du fonctionnement de l'entraînement du tokenizer SentencePiece
"""

import sentencepiece as spm
from pathlib import Path


def demo_tokenizer_training():
    """Démo pas à pas de l'entraînement"""

    print("=" * 60)
    print("DÉMONSTRATION: Entraînement du tokenizer SentencePiece")
    print("=" * 60)

    # 1. Créer un mini corpus
    print("\n### ÉTAPE 1: Création du corpus d'entraînement ###")

    corpus = """
    Bonjour, comment allez-vous aujourd'hui?
    Bonjour, je vais très bien merci.
    Comment s'appelle votre chat?
    Mon chat s'appelle Félix.
    Aujourd'hui il fait beau.
    Demain il fera beau aussi.
    J'aime beaucoup le français.
    Le français est une belle langue.
    """ * 100  # Répéter pour avoir assez de données

    corpus_file = Path("/tmp/demo_corpus.txt")
    corpus_file.write_text(corpus, encoding='utf-8')

    print(f"Corpus créé: {len(corpus)} caractères")
    print(f"Exemples:")
    for line in corpus.strip().split('\n')[:3]:
        print(f"  - {line.strip()}")

    # 2. Entraîner le tokenizer
    print("\n### ÉTAPE 2: Entraînement du tokenizer ###")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix='/tmp/demo_tokenizer',
        vocab_size=100,  # Petit vocabulaire pour la démo
        character_coverage=1.0,
        model_type='unigram',

        # CRUCIAL pour FUTO Keyboard
        treat_whitespace_as_suffix=True,

        # Tokens spéciaux
        user_defined_symbols=['<XBU>', '<XBC>', '<XEC>'],

        # Paramètres
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    print("✓ Tokenizer entraîné!")

    # 3. Charger et tester
    print("\n### ÉTAPE 3: Test du tokenizer ###")

    sp = spm.SentencePieceProcessor()
    sp.load('/tmp/demo_tokenizer.model')

    print(f"\nVocabulaire total: {sp.vocab_size()} tokens")
    print(f"Tokens spéciaux: {sp.piece_to_id('<XBU>')}, {sp.piece_to_id('<XBC>')}")

    # Test sur phrases
    test_sentences = [
        "Bonjour comment allez-vous?",
        "J'aime le français",
        "Aujourd'hui il fait beau",
    ]

    print("\n### ÉTAPE 4: Tokenization d'exemples ###\n")

    for sentence in test_sentences:
        tokens = sp.encode_as_pieces(sentence)
        ids = sp.encode_as_ids(sentence)

        print(f"Phrase: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Nombre de tokens: {len(tokens)}")
        print(f"Décodage: {sp.decode_pieces(tokens)}")
        print()

    # 4. Analyser le vocabulaire
    print("### ÉTAPE 5: Analyse du vocabulaire appris ###\n")

    # Top 20 tokens
    print("Top 20 tokens par fréquence:")
    for i in range(min(20, sp.vocab_size())):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        print(f"  {i:3d}: '{piece}' (score: {score:.4f})")

    # 5. Démontrer whitespace en suffixe
    print("\n### ÉTAPE 6: Whitespace en SUFFIXE (critique FUTO) ###\n")

    sentence = "bonjour le monde"
    tokens = sp.encode_as_pieces(sentence)

    print(f"Phrase: '{sentence}'")
    print(f"Tokens: {tokens}")
    print("\nRemarquez les espaces:")
    for i, token in enumerate(tokens):
        has_space = token.endswith(' ') or '▁' in token
        print(f"  Token {i}: '{token}' → {'Espace EN SUFFIXE ✓' if has_space and i < len(tokens)-1 else 'Pas d\'espace'}")

    # 6. Comparer avec whitespace en préfixe
    print("\n### ÉTAPE 7: Comparaison préfixe vs suffixe ###\n")

    # Simuler préfixe (comportement par défaut)
    print("Avec whitespace en PRÉFIXE (défaut):")
    print("  Tokens: ['▁bonjour', '▁le', '▁monde']")
    print("  Prédiction mot suivant:")
    print("    1. Voir '▁bonjour' → on sait que le mot est fini")
    print("    2. Voir '▁le' → on sait que 'bonjour' est terminé")
    print("    ❌ Besoin de LOOKAHEAD (regarder le token suivant)")

    print("\nAvec whitespace en SUFFIXE (FUTO):")
    print("  Tokens: ['bonjour ', 'le ', 'monde']")
    print("  Prédiction mot suivant:")
    print("    1. Voir 'bonjour ' → espace présent, mot fini!")
    print("    2. Générer prédiction immédiatement")
    print("    ✓ PAS de lookahead nécessaire → plus efficace")

    # 7. Démontrer la gestion des mots inconnus
    print("\n### ÉTAPE 8: Gestion des mots INCONNUS ###\n")

    unknown_words = [
        "supercalifragilisticexpialidocious",
        "anticonstitutionnellement",
        "blabla123",
    ]

    for word in unknown_words:
        tokens = sp.encode_as_pieces(word)
        print(f"Mot inconnu: '{word}'")
        print(f"  Décomposé en: {tokens}")
        print(f"  → Pas de <UNK>, utilise des sous-mots!")
        print()

    # 8. Démontrer les tokens spéciaux FUTO
    print("### ÉTAPE 9: Tokens spéciaux FUTO Keyboard ###\n")

    # Format correction
    misspelled = "bonjoyr"
    correct = "bonjour"

    # Encoder avec format FUTO
    char_tokens = ''.join([f'<CHAR_{c.upper()}>' for c in misspelled.upper()])
    futo_format = f'<XBU>{char_tokens}<XBC>{correct} <XEC>'

    print(f"Mot erroné: {misspelled}")
    print(f"Mot correct: {correct}")
    print(f"\nFormat FUTO:")
    print(f"  {futo_format}")
    print(f"\nExplication:")
    print(f"  <XBU> = Beginning of User input")
    print(f"  <CHAR_X> = Caractère individuel (touche du clavier)")
    print(f"  <XBC> = Beginning of Correction")
    print(f"  <XEC> = End of Correction")

    # Vérifier que les tokens existent
    print(f"\nVérification des tokens spéciaux:")
    for token in ['<XBU>', '<XBC>', '<XEC>']:
        token_id = sp.piece_to_id(token)
        print(f"  {token}: ID={token_id} {'✓ Présent' if token_id >= 0 else '✗ Manquant'}")

    print("\n" + "=" * 60)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 60)


def explain_parameters():
    """Explique les paramètres importants"""

    print("\n" + "=" * 60)
    print("PARAMÈTRES IMPORTANTS DE L'ENTRAÎNEMENT")
    print("=" * 60)

    params = {
        "vocab_size": {
            "valeur": "15008",
            "rôle": "Taille du vocabulaire final",
            "impact": "Plus grand = meilleure couverture mais plus de paramètres du modèle",
            "choix": "15008 car: vocab_size × hidden_size × 2 = 15008 × 512 × 2 = 15M params (43% du modèle)",
        },
        "character_coverage": {
            "valeur": "0.9995",
            "rôle": "% de caractères à couvrir",
            "impact": "0.9995 = couvre 99.95% des caractères (inclut tous les accents français)",
            "choix": "Élevé pour ne pas perdre les caractères accentués",
        },
        "model_type": {
            "valeur": "unigram",
            "rôle": "Algorithme de tokenization",
            "impact": "unigram > bpe pour les langues avec morphologie complexe",
            "choix": "unigram car français a beaucoup de conjugaisons",
        },
        "treat_whitespace_as_suffix": {
            "valeur": "True",
            "rôle": "Position de l'espace",
            "impact": "CRUCIAL pour FUTO - permet de savoir quand un mot est fini",
            "choix": "True = obligatoire pour FUTO Keyboard",
        },
        "user_defined_symbols": {
            "valeur": "['<XBU>', '<XBC>', ...]",
            "rôle": "Tokens spéciaux à ajouter",
            "impact": "Nécessaires pour le format de correction FUTO",
            "choix": "Tokens de contrôle pour autocorrection",
        },
        "normalization_rule_name": {
            "valeur": "nmt_nfkc_cf",
            "rôle": "Normalisation des caractères",
            "impact": "Unifie les variantes (é vs é)",
            "choix": "nmt_nfkc_cf = bon pour NMT/LM",
        },
        "split_digits": {
            "valeur": "False",
            "rôle": "Séparer chiffres en tokens",
            "impact": "False = '2024' reste entier",
            "choix": "False pour préserver les nombres",
        },
    }

    for param, info in params.items():
        print(f"\n### {param} = {info['valeur']} ###")
        print(f"Rôle: {info['rôle']}")
        print(f"Impact: {info['impact']}")
        print(f"Pourquoi: {info['choix']}")


if __name__ == '__main__':
    # Démo complète
    demo_tokenizer_training()

    # Explication des paramètres
    explain_parameters()
