# Guide complet : Entraînement du tokenizer

Explication détaillée du fonctionnement de l'entraînement du tokenizer SentencePiece pour FUTO Keyboard.

---

## 📚 Table des matières

1. [Qu'est-ce qu'un tokenizer ?](#quest-ce-quun-tokenizer-)
2. [Pourquoi SentencePiece ?](#pourquoi-sentencepiece-)
3. [L'algorithme Unigram](#lalgorithme-unigram)
4. [Whitespace en suffixe (CRITIQUE)](#whitespace-en-suffixe-critique)
5. [Le processus d'entraînement](#le-processus-dentraînement)
6. [Paramètres importants](#paramètres-importants)
7. [Tokens spéciaux FUTO](#tokens-spéciaux-futo)
8. [Démonstration pratique](#démonstration-pratique)

---

## Qu'est-ce qu'un tokenizer ?

### Le problème

Les modèles de langue ne comprennent que les **nombres**, pas le texte.

```
┌─────────────────────────────────────────────────┐
│  Texte humain                                   │
│  "Bonjour, comment allez-vous aujourd'hui?"     │
└─────────────────────────────────────────────────┘
                    ↓ TOKENIZER
┌─────────────────────────────────────────────────┐
│  Tokens (morceaux de texte)                     │
│  ["Bonjour", ",", "comment", "allez", "-",      │
│   "vous", "aujourd", "'", "hui", "?"]           │
└─────────────────────────────────────────────────┘
                    ↓ ENCODAGE
┌─────────────────────────────────────────────────┐
│  IDs (nombres)                                  │
│  [1234, 5, 892, 3421, 67, 2108, 8956, 12, ...]  │
└─────────────────────────────────────────────────┘
                    ↓ EMBEDDING LOOKUP
┌─────────────────────────────────────────────────┐
│  Vecteurs (512 dimensions)                      │
│  [[0.23, -0.45, ...], [0.12, 0.89, ...], ...]   │
└─────────────────────────────────────────────────┘
                    ↓ MODÈLE
```

### Approches naïves (qui ne marchent pas)

#### 1. Découpage par espaces

```python
"Bonjour, comment allez-vous?".split()
# → ["Bonjour,", "comment", "allez-vous?"]

Problèmes:
❌ Ponctuation attachée ("Bonjour," vs "Bonjour")
❌ Apostrophes ("aujourd'hui" → 1 ou 2 tokens?)
❌ Tirets ("allez-vous" → 1 ou 2 tokens?)
```

#### 2. Découpage par caractères

```python
list("Bonjour")
# → ['B', 'o', 'n', 'j', 'o', 'u', 'r']

Problèmes:
❌ Séquences trop longues (7 tokens pour 1 mot)
❌ Perd la structure des mots
❌ Modèle doit réapprendre les mots
```

#### 3. Vocabulaire de mots entiers

```python
vocab = {"Bonjour": 1, "comment": 2, "allez": 3, ...}

Problèmes:
❌ Vocabulaire énorme (millions de mots)
❌ Mots inconnus → <UNK>
❌ Conjugaisons = tokens différents ("manger", "mangé", "mangeant")
```

### La solution : Subword Tokenization

Découpe en **sous-mots** (morceaux optimaux) :

```
"anticonstitutionnellement"
     ↓ Subword
["anti", "constitu", "tion", "nel", "lement"]

Avantages:
✓ Vocabulaire raisonnable (~15k tokens)
✓ Pas de mots inconnus (combinaison de sous-mots)
✓ Capture la morphologie (préfixes, suffixes)
✓ Efficace (moins de tokens que caractères)
```

---

## Pourquoi SentencePiece ?

### Comparaison des tokenizers

| Tokenizer | Type | Avantages | Inconvénients |
|-----------|------|-----------|---------------|
| **WordPiece** (BERT) | Greedy | Simple, rapide | Dépend du prétraitement |
| **BPE** (GPT) | Merge | Déterministe | Pas optimal |
| **SentencePiece** | Unigram/BPE | **Indépendant langue**, **probabiliste** | Plus lent à entraîner |

### Pourquoi SentencePiece pour FUTO ?

1. **Indépendant de la langue**
   - Pas besoin de pré-tokenization (découpage mots)
   - Fonctionne directement sur le texte brut
   - Gère les apostrophes/tirets automatiquement

2. **Whitespace configurable**
   - `treat_whitespace_as_suffix=True` (CRUCIAL pour FUTO)
   - Permet de savoir quand un mot est fini

3. **Embedde le tokenizer**
   - Le fichier `.model` contient TOUT
   - Pas besoin de fichiers de config séparés
   - Facile à intégrer dans GGUF

4. **Production-ready**
   - Utilisé par Google, Meta, etc.
   - C++ optimisé (rapide en inférence)
   - Librairies pour tous les langages

---

## L'algorithme Unigram

### Principe de base

Au lieu de **construire** le vocabulaire (BPE), Unigram **réduit** le vocabulaire.

```
┌──────────────────────────────────────────────┐
│  ÉTAPE 1: Initialisation                    │
│  Générer TOUS les sous-mots possibles       │
│  Vocabulaire: 1,000,000+ candidats          │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  ÉTAPE 2: Calcul probabilités               │
│  Pour chaque sous-mot, calculer P(token)    │
│  P("bon") = fréq("bon") / total             │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  ÉTAPE 3: Suppression itérative             │
│  Supprimer les tokens les moins utiles      │
│  Critère: perte minimale de vraisemblance  │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  ÉTAPE 4: Convergence                       │
│  Arrêt quand vocab_size atteint (15008)     │
│  Vocabulaire final optimisé                 │
└──────────────────────────────────────────────┘
```

### Exemple concret

Corpus: `"bonjour bonjour bonbon jour"`

#### Initialisation

```
Sous-mots possibles:
b, bo, bon, bonj, bonjo, bonjou, bonjour
o, on, j, jo, jou, jour
n, nb, nbo, nbon
...
```

#### Calcul des probabilités

```python
Fréquences:
"bonjour" : 2 fois
"bonbon"  : 1 fois
"jour"    : 1 fois
"bon"     : 3 fois (dans bonjour × 2 + bonbon × 1)
"on"      : 4 fois (bon × 3 + bonbon × 1)

Probabilités:
P("bonjour") = 2/4 = 0.50
P("bon")     = 3/4 = 0.75  ← Plus probable!
P("jour")    = 1/4 = 0.25
P("on")      = 4/4 = 1.00  ← Très utile!
```

#### Décisions

```
Garder "on" → très fréquent, utile partout
Garder "bon" → fréquent, forme de base
Garder "jour" → mot complet

Supprimer "bonjo" → redondant avec "bon"+"jo"
Supprimer "nbo" → trop rare
```

#### Résultat final

```
Vocabulaire optimal:
["b", "o", "n", "j", "u", "r", "on", "bon", "jour", ...]

Tokenization:
"bonjour" → ["bon", "jour"]  (2 tokens)
"bonbon"  → ["bon", "bon"]   (2 tokens)
"jour"    → ["jour"]         (1 token)
```

### Avantage vs BPE

```
BPE (greedy merge):
1. Fusionne les paires les plus fréquentes
2. Ordre dépend de l'initialisation
3. Pas forcément optimal

Unigram (probabilistic):
1. Évalue TOUTES les possibilités
2. Choisit la meilleure globalement
3. Théoriquement optimal (EM algorithm)
```

---

## Whitespace en suffixe (CRITIQUE)

### Pourquoi c'est important pour FUTO Keyboard

Le clavier doit savoir **quand un mot est fini** pour suggérer le suivant.

### Comparaison préfixe vs suffixe

#### Whitespace en PRÉFIXE (défaut, ❌ pour FUTO)

```
Phrase: "bonjour le monde"

Tokens: ["▁bonjour", "▁le", "▁monde"]
         ^espace     ^espace  ^espace

Prédiction mot suivant après "bonjour":

Décodage séquentiel:
1. Voir "▁bonjour"     → Mot commençant par "bonjour"
2. Voir "▁le"          → Ah! Nouveau mot, donc "bonjour" est fini
                          → Peut maintenant prédire le suivant

❌ Problème: Besoin de LOOKAHEAD (regarder le prochain token)
❌ Inefficace pour un clavier (doit attendre)
```

#### Whitespace en SUFFIXE (✅ pour FUTO)

```
Phrase: "bonjour le monde"

Tokens: ["bonjour ", "le ", "monde"]
                 ^espace ^espace

Prédiction mot suivant après "bonjour":

Décodage séquentiel:
1. Voir "bonjour "     → Espace présent = mot fini!
                       → Peut prédire immédiatement

✓ Pas de lookahead nécessaire
✓ Efficace pour clavier (prédiction instantanée)
```

### Visualisation détaillée

```
┌────────────────────────────────────────────────────────────┐
│  PRÉFIXE: L'espace est AU DÉBUT du token                   │
└────────────────────────────────────────────────────────────┘

Utilisateur tape: "b o n j o u r [ESPACE]"
                                           ↑ Appui sur espace

Modèle voit:
Token 1: "▁bonjour"   ← espace en préfixe
                       ← Attend le prochain token...

Token 2: "▁le"        ← espace en préfixe
                       ← Maintenant on sait que "bonjour" est fini!
                       ← Trop tard pour prédire efficacement


┌────────────────────────────────────────────────────────────┐
│  SUFFIXE: L'espace est À LA FIN du token                   │
└────────────────────────────────────────────────────────────┘

Utilisateur tape: "b o n j o u r [ESPACE]"
                                           ↑ Appui sur espace

Modèle voit:
Token 1: "bonjour "   ← espace en suffixe
                      ← Espace présent = mot terminé!
                      ← Peut prédire le suivant IMMÉDIATEMENT

Token 2: "le "        ← Suggestion déjà affichée
                      ← Expérience fluide
```

### Impact sur les performances

```
Latence de prédiction:

Préfixe:  Taper "bonjour " → Attendre token suivant → Prédiction
          └─────────────────── ~100-200ms ──────────────────────┘

Suffixe:  Taper "bonjour " → Prédiction immédiate
          └────── ~50ms ──────┘

Gain: 2-4x plus rapide
```

### Configuration SentencePiece

```python
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix="tokenizer",

    # ⚠️ PARAMÈTRE CRUCIAL POUR FUTO ⚠️
    treat_whitespace_as_suffix=True,

    # Autres paramètres...
)
```

**Sans ce paramètre, le modèle ne fonctionnera PAS avec FUTO Keyboard!**

---

## Le processus d'entraînement

### Vue d'ensemble

```
┌─────────────┐
│  1. CORPUS  │  Texte français brut (Wikipedia, etc.)
└──────┬──────┘
       ↓
┌─────────────┐
│  2. ANALYSE │  Extraction caractères, sous-mots, fréquences
└──────┬──────┘
       ↓
┌─────────────┐
│  3. UNIGRAM │  Algorithme EM pour vocabulaire optimal
└──────┬──────┘
       ↓
┌─────────────┐
│  4. PRUNING │  Réduction à vocab_size (15008)
└──────┬──────┘
       ↓
┌─────────────┐
│  5. EXPORT  │  Sauvegarde .model + .vocab
└─────────────┘
```

### Étape par étape

#### ÉTAPE 1: Préparation du corpus

```python
# Le corpus doit être un fichier texte (UTF-8)
corpus = """
Bonjour, comment allez-vous?
Je vais très bien merci.
Le français est une belle langue.
...
"""

# Sauvegarder
with open('corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)
```

**Taille recommandée** : 1-10 GB de texte (quelques milliards de caractères)

#### ÉTAPE 2: Lancement de l'entraînement

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='french_keyboard',
    vocab_size=15008,

    # Algorithme
    model_type='unigram',  # ou 'bpe'

    # Normalisation
    normalization_rule_name='nmt_nfkc_cf',

    # CRUCIAL
    treat_whitespace_as_suffix=True,

    # Tokens spéciaux
    user_defined_symbols=[
        '<XBU>', '<XBC>', '<XEC>',
        '<CHAR_A>', '<CHAR_B>', ..., '<CHAR_Z>',
        '<CHAR_À>', '<CHAR_É>', ...
    ],
)
```

**Durée** : ~10-30 minutes selon taille corpus et CPU

#### ÉTAPE 3: Ce qui se passe en interne

```
1. Scan du corpus (comptage caractères)
   └─> "é" apparaît 150,000 fois
   └─> "à" apparaît 80,000 fois
   └─> "ç" apparaît 5,000 fois

2. Extraction sous-mots candidats
   └─> Générés par algorithme de seed
   └─> ~1-10 millions de candidats

3. Algorithme EM (Expectation-Maximization)
   Itération 1:
     E-step: Calculer probabilités conditionnelles
     M-step: Mettre à jour probabilités des tokens

   Itération 2:
     ...

   Convergence après ~10-20 itérations

4. Pruning (réduction)
   Sort par probabilité:
     Token 1: P=0.999 → Garder
     Token 2: P=0.950 → Garder
     ...
     Token 15008: P=0.001 → Garder
     Token 15009: P=0.0009 → Supprimer
     ...

5. Finalisation
   - Assigner IDs finaux
   - Calculer scores
   - Créer trie pour décodage rapide
```

#### ÉTAPE 4: Fichiers générés

```bash
french_keyboard.model  # Modèle binaire (tout est dedans)
french_keyboard.vocab  # Vocabulaire lisible (pour debug)
```

**`.model`** : C'est le seul fichier nécessaire pour l'inférence!

#### ÉTAPE 5: Validation

```python
# Charger le tokenizer
sp = spm.SentencePieceProcessor()
sp.load('french_keyboard.model')

# Tester
tokens = sp.encode_as_pieces("Bonjour, ça va?")
print(tokens)
# ['Bonjour ', ',', 'ça ', 'va ', '?']
#           ^espace    ^espace  ^espace
```

---

## Paramètres importants

### Paramètres de base

| Paramètre | Valeur | Rôle | Impact |
|-----------|--------|------|--------|
| `vocab_size` | 15008 | Taille vocabulaire final | Plus grand = meilleure couverture mais plus de paramètres modèle |
| `model_type` | `unigram` | Algorithme | `unigram` > `bpe` pour français (morphologie complexe) |
| `character_coverage` | 0.9995 | % caractères à couvrir | 0.9995 = couvre accents français |

### Paramètres critiques FUTO

| Paramètre | Valeur | Importance |
|-----------|--------|------------|
| `treat_whitespace_as_suffix` | `True` | ⚠️ **OBLIGATOIRE** pour FUTO |
| `user_defined_symbols` | `['<XBU>', ...]` | Tokens spéciaux autocorrection |

### Paramètres de normalisation

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `normalization_rule_name` | `nmt_nfkc_cf` | Unifie variantes (é vs é composé) |
| `remove_extra_whitespaces` | `True` | Nettoie espaces multiples |

### Paramètres d'optimisation

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `split_digits` | `False` | Garde nombres entiers ("2024" vs "20"+"24") |
| `split_by_unicode_script` | `False` | Pas de split entre scripts (latin/grec) |
| `byte_fallback` | `False` | Pas de fallback bytes (risque tokens bizarres) |

### Calcul de vocab_size optimal

**Question** : Pourquoi 15008 ?

**Réponse** : Compromis entre couverture et taille du modèle

```
Paramètres du modèle liés au vocabulaire:

Embedding input:  vocab_size × hidden_size
Embedding output: vocab_size × hidden_size

Total embeddings: vocab_size × hidden_size × 2
                = 15008 × 512 × 2
                = 15,368,192 paramètres

Taille: 15.4M paramètres × 2 bytes (fp16)
      = 30.7 MB

En pourcentage du modèle total (36M params):
15.4M / 36M = 42.8%

Conclusion: Les embeddings prennent ~43% du modèle
           → Pas beaucoup plus de marge pour augmenter vocab_size
```

**Règle générale** :
- Vocab trop petit (5k) → Beaucoup de sous-mots, séquences longues
- Vocab trop grand (50k) → Trop de paramètres d'embeddings
- Sweet spot (15k) → Bon équilibre

---

## Tokens spéciaux FUTO

### Pourquoi des tokens spéciaux ?

Le modèle doit comprendre le **format de correction** FUTO :

```
<XBU><CHAR_B><CHAR_O><CHAR_N><CHAR_J><CHAR_O><CHAR_Y><CHAR_R><XBC>bonjour <XEC>
```

Ces tokens ont des rôles spécifiques :

### Tokens de contrôle

| Token | Nom | Rôle |
|-------|-----|------|
| `<XBU>` | eXpected Beginning of User input | Début de la saisie utilisateur |
| `<XBC>` | eXpected Beginning of Correction | Début de la correction |
| `<XEC>` | eXpected End of Correction | Fin de la correction |
| `<XC0>` | eXpected Control 0 (swipe) | Swipe typing (futur) |

### Tokens de caractères (AZERTY)

```python
# Lettres standard
char_tokens = ['<CHAR_A>', '<CHAR_B>', ..., '<CHAR_Z>']  # 26 tokens

# Lettres accentuées françaises
french_chars = ['<CHAR_À>', '<CHAR_Â>', '<CHAR_Ä>', '<CHAR_Æ>',
                '<CHAR_Ç>',
                '<CHAR_É>', '<CHAR_È>', '<CHAR_Ê>', '<CHAR_Ë>',
                '<CHAR_Î>', '<CHAR_Ï>',
                '<CHAR_Ô>', '<CHAR_Ö>',
                '<CHAR_Ù>', '<CHAR_Û>', '<CHAR_Ü>',
                '<CHAR_Ÿ>', '<CHAR_Œ>']  # 18 tokens

Total: 26 + 18 = 44 tokens spéciaux
```

### Format complet

```
┌──────────────────────────────────────────────────────────┐
│  FORMAT AUTOCORRECTION FUTO                              │
└──────────────────────────────────────────────────────────┘

Contexte: "Bonjour, je vais à "

Utilisateur tape: "l e c o l e"  (erreur: manque apostrophe)

Format envoyé au modèle:
┌──────────────────────────────────────────────────────────┐
│ Bonjour, je vais à <XBU><CHAR_L><CHAR_E><CHAR_C>        │
│ <CHAR_O><CHAR_L><CHAR_E><XBC>                           │
└──────────────────────────────────────────────────────────┘
                            ↓ MODÈLE PRÉDIT
┌──────────────────────────────────────────────────────────┐
│ l'école <XEC>                                            │
└──────────────────────────────────────────────────────────┘

Résultat affiché: "l'école"
```

### Pourquoi CHAR tokens ?

**Question** : Pourquoi pas utiliser directement les lettres ?

**Réponse** : Pour encoder la **position physique** de la frappe

```
Sans CHAR tokens:
"lecole" → Tokenisé normalement → ["l", "ecole"]
         → Modèle ne voit pas la différence avec un mot normal

Avec CHAR tokens:
<CHAR_L><CHAR_E><CHAR_C><CHAR_O><CHAR_L><CHAR_E>
→ Modèle comprend: ce sont des touches individuelles tapées
→ Peut déduire: erreur probable (pas un vrai mot)
→ Suggère: "l'école" (correction)
```

**Avantage** : Le modèle apprend à différencier :
- Texte normal : "lecole"
- Saisie clavier : `<CHAR_L><CHAR_E><CHAR_C><CHAR_O><CHAR_L><CHAR_E>`

### Ajout au tokenizer

```python
# Tous les tokens spéciaux
special_tokens = [
    '<XBU>', '<XBC>', '<XEC>', '<XC0>',  # Contrôle
]

# Caractères A-Z
char_tokens = [f'<CHAR_{chr(i)}>' for i in range(ord('A'), ord('Z') + 1)]

# Caractères accentués
french_chars = ['À', 'Â', 'Ä', 'Æ', 'Ç', 'É', 'È', 'Ê', 'Ë',
                'Î', 'Ï', 'Ô', 'Ö', 'Ù', 'Û', 'Ü', 'Ÿ', 'Œ']
french_char_tokens = [f'<CHAR_{char}>' for char in french_chars]

# Combiner
all_special_tokens = special_tokens + char_tokens + french_char_tokens

# Entraîner avec
spm.SentencePieceTrainer.train(
    ...
    user_defined_symbols=all_special_tokens,
    ...
)
```

**Important** : Ces tokens sont **réservés** et ne seront jamais produits par l'algorithme Unigram.

---

## Démonstration pratique

### Script de démo

J'ai créé un script pour visualiser le processus :

```bash
python tokenizer/demo_tokenizer.py
```

### Ce qu'il fait

```
1. Crée un mini corpus français
2. Entraîne un tokenizer (vocab_size=100)
3. Teste la tokenization
4. Analyse le vocabulaire appris
5. Démontre whitespace en suffixe
6. Compare préfixe vs suffixe
7. Teste mots inconnus
8. Vérifie tokens spéciaux FUTO
```

### Exemple de sortie

```
==========================================================
DÉMONSTRATION: Entraînement du tokenizer SentencePiece
==========================================================

### ÉTAPE 1: Création du corpus d'entraînement ###
Corpus créé: 3420 caractères
Exemples:
  - Bonjour, comment allez-vous aujourd'hui?
  - Bonjour, je vais très bien merci.
  - Comment s'appelle votre chat?

### ÉTAPE 2: Entraînement du tokenizer ###
✓ Tokenizer entraîné!

### ÉTAPE 3: Test du tokenizer ###
Vocabulaire total: 100 tokens
Tokens spéciaux: 7, 8

### ÉTAPE 4: Tokenization d'exemples ###

Phrase: Bonjour comment allez-vous?
Tokens: ['Bonjour ', 'comment ', 'allez ', '-', 'vous ', '?']
                 ^          ^         ^              ^
                 Espaces en SUFFIXE!
IDs: [12, 34, 56, 3, 78, 9]
Nombre de tokens: 6
Décodage: Bonjour comment allez-vous?

Phrase: J'aime le français
Tokens: ["J'", 'aime ', 'le ', 'français']
IDs: [23, 45, 67, 89]
Nombre de tokens: 4

Phrase: Aujourd'hui il fait beau
Tokens: ["Aujourd'", 'hui ', 'il ', 'fait ', 'beau']
IDs: [11, 22, 33, 44, 55]
Nombre de tokens: 5

### ÉTAPE 5: Analyse du vocabulaire appris ###

Top 20 tokens par fréquence:
  0: '<unk>' (score: 0.0000)
  1: '<s>' (score: 0.0000)
  2: '<XBU>' (score: 0.0000)
  3: '<XBC>' (score: 0.0000)
  4: ' ' (score: -2.3456)
  5: 'e ' (score: -3.1234)
  6: 'le ' (score: -3.4567)
  7: 's ' (score: -3.7890)
  ...

### ÉTAPE 6: Whitespace en SUFFIXE (critique FUTO) ###

Phrase: 'bonjour le monde'
Tokens: ['bonjour ', 'le ', 'monde']
Remarquez les espaces:
  Token 0: 'bonjour ' → Espace EN SUFFIXE ✓
  Token 1: 'le ' → Espace EN SUFFIXE ✓
  Token 2: 'monde' → Pas d'espace (dernier mot)

### ÉTAPE 7: Comparaison préfixe vs suffixe ###

Avec whitespace en PRÉFIXE (défaut):
  Tokens: ['▁bonjour', '▁le', '▁monde']
  Prédiction mot suivant:
    1. Voir '▁bonjour' → on sait que le mot est fini
    2. Voir '▁le' → on sait que 'bonjour' est terminé
    ❌ Besoin de LOOKAHEAD (regarder le token suivant)

Avec whitespace en SUFFIXE (FUTO):
  Tokens: ['bonjour ', 'le ', 'monde']
  Prédiction mot suivant:
    1. Voir 'bonjour ' → espace présent, mot fini!
    2. Générer prédiction immédiatement
    ✓ PAS de lookahead nécessaire → plus efficace

### ÉTAPE 8: Gestion des mots INCONNUS ###

Mot inconnu: 'supercalifragilisticexpialidocious'
  Décomposé en: ['super', 'cal', 'if', 'rag', 'il', 'ist', 'ic', 'exp', 'ial', 'id', 'oc', 'ious']
  → Pas de <UNK>, utilise des sous-mots!

Mot inconnu: 'anticonstitutionnellement'
  Décomposé en: ['anti', 'constitu', 'tion', 'nel', 'lement']
  → Capture la morphologie!

Mot inconnu: 'blabla123'
  Décomposé en: ['bla', 'bla', '123']
  → Gère nombres et mots inventés!

### ÉTAPE 9: Tokens spéciaux FUTO Keyboard ###

Mot erroné: bonjoyr
Mot correct: bonjour

Format FUTO:
  <XBU><CHAR_B><CHAR_O><CHAR_N><CHAR_J><CHAR_O><CHAR_Y><CHAR_R><XBC>bonjour <XEC>

Explication:
  <XBU> = Beginning of User input
  <CHAR_X> = Caractère individuel (touche du clavier)
  <XBC> = Beginning of Correction
  <XEC> = End of Correction

Vérification des tokens spéciaux:
  <XBU>: ID=2 ✓ Présent
  <XBC>: ID=3 ✓ Présent
  <XEC>: ID=4 ✓ Présent

==========================================================
FIN DE LA DÉMONSTRATION
==========================================================
```

---

## Résumé : Checklist entraînement tokenizer

### ✅ Avant l'entraînement

- [ ] Corpus français préparé (1-10 GB)
- [ ] Fichier texte UTF-8 propre
- [ ] SentencePiece installé (`pip install sentencepiece`)

### ✅ Paramètres obligatoires

- [ ] `vocab_size=15008` (optimisé pour 36M modèle)
- [ ] `model_type='unigram'` (meilleur pour français)
- [ ] `treat_whitespace_as_suffix=True` ⚠️ **CRITIQUE**
- [ ] `character_coverage=0.9995` (couvre accents)
- [ ] Tokens spéciaux FUTO ajoutés

### ✅ Après l'entraînement

- [ ] Fichier `.model` généré
- [ ] Test tokenization phrases françaises
- [ ] Vérification espaces EN SUFFIXE
- [ ] Vérification tokens spéciaux présents
- [ ] Test mots avec accents (é, à, ç)
- [ ] Test mots inconnus (décomposition)

### ✅ Validation FUTO

- [ ] Format correction fonctionne
- [ ] Tous les `<CHAR_X>` tokens présents
- [ ] Espace en suffixe confirmé
- [ ] Prêt pour l'intégration modèle

---

## Questions fréquentes

### Q: Pourquoi unigram et pas BPE ?

**R:** Unigram est probabiliste et trouve le vocabulaire optimal globalement, alors que BPE est greedy (glouton). Pour le français avec sa morphologie riche (conjugaisons, etc.), unigram donne de meilleurs résultats.

### Q: Puis-je modifier le vocabulaire après entraînement ?

**R:** Non, le vocabulaire est figé. Si vous voulez ajouter des tokens, il faut réentraîner (ou utiliser `user_defined_symbols`).

### Q: Combien de temps prend l'entraînement ?

**R:** 10-30 minutes pour un corpus de 1-5 GB sur un CPU moderne. Augmente avec la taille du corpus et vocab_size.

### Q: Que se passe-t-il si j'oublie `treat_whitespace_as_suffix=True` ?

**R:** Le tokenizer fonctionnera, MAIS le modèle ne sera pas compatible avec FUTO Keyboard (prédictions inefficaces).

### Q: Puis-je réutiliser un tokenizer existant ?

**R:** Seulement s'il a été entraîné avec whitespace en suffixe ET contient les tokens spéciaux FUTO. Sinon, entraînez-en un nouveau.

### Q: 15008 tokens c'est assez pour le français ?

**R:** Oui! C'est comparable à GPT-2 (50k) et BERT (30k), mais on utilise moins car notre modèle est petit (36M). 15k couvre très bien le français.

---

**Prêt à entraîner votre tokenizer ?**

```bash
python tokenizer/train_tokenizer.py \
    --input data/raw/french_corpus.txt \
    --vocab-size 15008
```
