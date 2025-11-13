#!/usr/bin/env python3
"""
Générateur de fautes de frappe synthétiques pour clavier AZERTY français
"""

import random
import string
from typing import List, Tuple


class AzertyErrorGenerator:
    """Génère des fautes de frappe réalistes pour clavier AZERTY"""

    # Layout AZERTY (rangées de touches)
    AZERTY_LAYOUT = {
        'row1': 'azertyuiop',
        'row2': 'qsdfghjklm',
        'row3': 'wxcvbn',
    }

    # Mapping des touches adjacentes sur AZERTY
    ADJACENT_KEYS = {
        'a': ['z', 'q'],
        'z': ['a', 'e', 'q', 's'],
        'e': ['z', 'r', 's', 'd'],
        'r': ['e', 't', 'd', 'f'],
        't': ['r', 'y', 'f', 'g'],
        'y': ['t', 'u', 'g', 'h'],
        'u': ['y', 'i', 'h', 'j'],
        'i': ['u', 'o', 'j', 'k'],
        'o': ['i', 'p', 'k', 'l'],
        'p': ['o', 'l', 'm'],
        'q': ['a', 'z', 's', 'w'],
        's': ['q', 'z', 'e', 'd', 'w', 'x'],
        'd': ['s', 'e', 'r', 'f', 'x', 'c'],
        'f': ['d', 'r', 't', 'g', 'c', 'v'],
        'g': ['f', 't', 'y', 'h', 'v', 'b'],
        'h': ['g', 'y', 'u', 'j', 'b', 'n'],
        'j': ['h', 'u', 'i', 'k', 'n'],
        'k': ['j', 'i', 'o', 'l'],
        'l': ['k', 'o', 'p', 'm'],
        'm': ['l', 'p'],
        'w': ['q', 's', 'x'],
        'x': ['w', 's', 'd', 'c'],
        'c': ['x', 'd', 'f', 'v'],
        'v': ['c', 'f', 'g', 'b'],
        'b': ['v', 'g', 'h', 'n'],
        'n': ['b', 'h', 'j'],
    }

    # Erreurs communes de substitution en français
    COMMON_SUBSTITUTIONS = {
        'é': ['e', 'è', 'ê'],
        'è': ['e', 'é', 'ê'],
        'ê': ['e', 'é', 'è'],
        'à': ['a'],
        'â': ['a'],
        'ù': ['u'],
        'û': ['u'],
        'ô': ['o'],
        'ï': ['i'],
        'î': ['i'],
        'ç': ['c'],
        'œ': ['oe'],
        'æ': ['ae'],
        'ou': ['u'],  # Confusion phonétique
        'au': ['o'],  # Confusion phonétique
        'eau': ['o'], # Confusion phonétique
        'er': ['é'],  # Confusion grammaticale
        'é': ['er'],  # Confusion grammaticale
    }

    def __init__(self, error_rate: float = 0.15):
        """
        Args:
            error_rate: Probabilité d'introduire une erreur par caractère (0.0-1.0)
        """
        self.error_rate = error_rate

    def adjacent_key_error(self, char: str) -> str:
        """Remplace par une touche adjacente sur AZERTY"""
        char_lower = char.lower()
        if char_lower in self.ADJACENT_KEYS:
            adjacent = random.choice(self.ADJACENT_KEYS[char_lower])
            return adjacent.upper() if char.isupper() else adjacent
        return char

    def missing_char_error(self, word: str, pos: int) -> str:
        """Supprime un caractère"""
        return word[:pos] + word[pos + 1:]

    def double_char_error(self, word: str, pos: int) -> str:
        """Double un caractère"""
        return word[:pos] + word[pos] + word[pos:]

    def transpose_error(self, word: str, pos: int) -> str:
        """Transpose deux caractères adjacents"""
        if pos + 1 < len(word):
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        return word

    def accent_error(self, char: str) -> str:
        """Erreur d'accent (fréquent en français)"""
        char_lower = char.lower()
        if char_lower in self.COMMON_SUBSTITUTIONS:
            replacement = random.choice(self.COMMON_SUBSTITUTIONS[char_lower])
            return replacement.upper() if char.isupper() else replacement
        return char

    def generate_word_error(self, word: str) -> str:
        """
        Génère une version avec faute d'un mot

        Args:
            word: Mot original

        Returns:
            Mot avec faute(s) de frappe
        """
        if len(word) < 2:
            return word

        # Choisir le type d'erreur
        error_types = [
            ('adjacent', 0.35),      # Touche adjacente
            ('missing', 0.15),       # Caractère manquant
            ('double', 0.10),        # Caractère doublé
            ('transpose', 0.15),     # Transposition
            ('accent', 0.25),        # Erreur d'accent
        ]

        error_type = random.choices(
            [et[0] for et in error_types],
            weights=[et[1] for et in error_types]
        )[0]

        # Appliquer l'erreur à une position aléatoire
        if error_type == 'missing':
            pos = random.randint(0, len(word) - 1)
            return self.missing_char_error(word, pos)

        elif error_type == 'double':
            pos = random.randint(0, len(word) - 1)
            return self.double_char_error(word, pos)

        elif error_type == 'transpose':
            pos = random.randint(0, len(word) - 2)
            return self.transpose_error(word, pos)

        elif error_type == 'adjacent':
            pos = random.randint(0, len(word) - 1)
            new_word = list(word)
            new_word[pos] = self.adjacent_key_error(word[pos])
            return ''.join(new_word)

        elif error_type == 'accent':
            # Chercher un caractère avec accent
            accent_positions = [
                i for i, c in enumerate(word.lower())
                if c in self.COMMON_SUBSTITUTIONS
            ]
            if accent_positions:
                pos = random.choice(accent_positions)
                new_word = list(word)
                new_word[pos] = self.accent_error(word[pos])
                return ''.join(new_word)

        return word

    def generate_sentence_errors(
        self,
        sentence: str,
        error_prob: float = 0.33
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Génère des erreurs dans une phrase

        Args:
            sentence: Phrase originale
            error_prob: Probabilité qu'un mot ait une erreur (défaut: 33%)

        Returns:
            Tuple de (phrase avec erreurs, liste des (mot_original, mot_erroné))
        """
        words = sentence.split()
        corrupted_words = []
        error_list = []

        for word in words:
            # Séparer la ponctuation
            punctuation = ''
            clean_word = word
            if word and word[-1] in '.,!?;:':
                punctuation = word[-1]
                clean_word = word[:-1]

            # Introduire une erreur avec probabilité error_prob
            if random.random() < error_prob and len(clean_word) > 2:
                corrupted = self.generate_word_error(clean_word)
                corrupted_words.append(corrupted + punctuation)
                error_list.append((clean_word, corrupted))
            else:
                corrupted_words.append(word)

        return ' '.join(corrupted_words), error_list

    def format_for_training(
        self,
        original_word: str,
        misspelled_word: str
    ) -> str:
        """
        Formate au format FUTO pour l'entraînement

        Format: <XBU><CHAR_M><CHAR_E><CHAR_S><CHAR_O><CHAR_N><XBC>maison <XEC>

        Args:
            original_word: Mot correct
            misspelled_word: Mot avec faute

        Returns:
            String formaté pour l'entraînement
        """
        # Convertir le mot erroné en tokens CHAR_X
        char_tokens = ''.join([
            f'<CHAR_{c.upper()}>'
            for c in misspelled_word.upper()
        ])

        # Format final
        return f'<XBU>{char_tokens}<XBC>{original_word.lower()} <XEC>'

    def generate_training_examples(
        self,
        words: List[str],
        num_examples_per_word: int = 3
    ) -> List[str]:
        """
        Génère des exemples d'entraînement au format FUTO

        Args:
            words: Liste de mots corrects
            num_examples_per_word: Nombre d'exemples par mot

        Returns:
            Liste d'exemples formatés
        """
        examples = []

        for word in words:
            if len(word) < 2:
                continue

            for _ in range(num_examples_per_word):
                misspelled = self.generate_word_error(word)
                if misspelled != word:  # Seulement si une erreur a été générée
                    formatted = self.format_for_training(word, misspelled)
                    examples.append(formatted)

        return examples


def main():
    """Exemple d'utilisation"""
    generator = AzertyErrorGenerator()

    # Test sur des mots français
    test_words = [
        'bonjour', 'merci', 'maison', 'français', 'clavier',
        'aujourd\'hui', 'téléphone', 'ordinateur', 'étudiant',
        'médecin', 'hôpital', 'château', 'précis'
    ]

    print("=== Exemples de fautes générées ===\n")

    for word in test_words[:5]:
        for _ in range(3):
            corrupted = generator.generate_word_error(word)
            formatted = generator.format_for_training(word, corrupted)
            print(f"{word:15} → {corrupted:15} → {formatted}")

    print("\n=== Exemple sur phrase ===\n")

    sentence = "Je vais à l'école aujourd'hui pour étudier le français."
    corrupted_sentence, errors = generator.generate_sentence_errors(sentence, error_prob=0.5)

    print(f"Original:  {sentence}")
    print(f"Corrompu:  {corrupted_sentence}")
    print(f"\nErreurs: {errors}")

    print("\n=== Exemples d'entraînement ===\n")

    training_examples = generator.generate_training_examples(test_words[:5], num_examples_per_word=2)
    for example in training_examples[:10]:
        print(example)


if __name__ == '__main__':
    main()
