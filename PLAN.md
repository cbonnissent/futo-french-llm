# Plan détaillé - Modèle de langue français FUTO Keyboard

Ce document résume le plan complet pour créer un modèle de langue français pour FUTO Keyboard.

## 📊 Résumé exécutif

**Objectif**: Créer un modèle de langue Llama de 36M paramètres pour prédiction de mots sur clavier AZERTY français

**Matériel**: GPU B200 (location cloud)

**Durée totale estimée**: 15-25 heures d'entraînement

**Résultat final**: Fichier `.gguf` importable dans FUTO Keyboard Android

---

## 🎯 Spécifications techniques

### Architecture du modèle

```
Llama Configuration (36M paramètres):
├── Vocabulaire: 15,008 tokens
├── Hidden size: 512
├── Intermediate size: 1,024
├── Couches: 8
├── Attention heads: 8
└── Max sequence length: 2048
```

### Tokens spéciaux FUTO

```
Correction: <XBU>, <XBC>, <XEC>
Caractères: <CHAR_A>...<CHAR_Z>
Accents FR: <CHAR_À>, <CHAR_É>, <CHAR_È>, etc.
Swipe (futur): <XC0>
```

### Métadonnées GGUF requises

```yaml
keyboardlm.languages: "fr"
keyboardlm.features: "base_v1 inverted_space xbu_char_autocorrect_v1"
keyboardlm.finetuning_count: 0
keyboardlm.ext_tokenizer_type: "SentencePiece"
```

---

## 📋 Pipeline d'entraînement (6 étapes)

### Étape 1: Préparation des données (1-2h)

**Sources de données françaises**:
- Wikipedia français (~2GB)
- OSCAR corpus français (~10GB subset)
- mC4 français (~5GB subset)

**Total**: ~5-10 milliards de tokens

**Script**: `scripts/download_data.py`

```bash
python scripts/download_data.py \
    --sources all \
    --merge \
    --output-dir ./data/raw
```

**Output**: `data/raw/french_corpus_merged.txt`

---

### Étape 2: Tokenizer SentencePiece (30 min)

**Particularités**:
- ✅ `treat_whitespace_as_suffix=True` (CRITIQUE pour FUTO)
- ✅ Vocabulaire ~15k tokens max
- ✅ Coverage 0.9995 (pour accents français)

**Script**: `tokenizer/train_tokenizer.py`

```bash
python tokenizer/train_tokenizer.py \
    --input data/raw/french_corpus_merged.txt \
    --vocab-size 15008
```

**Output**:
- `tokenizer/french_keyboard.model`
- `tokenizer/french_keyboard.vocab`

**Validation**:
```python
sp.encode_as_pieces("bonjour le monde")
# Attendu: ['bonjour ', 'le ', 'monde']  (espaces EN SUFFIXE!)
```

---

### Étape 3: Préentraînement (6-12h sur B200)

**Objectif**: Apprendre la structure générale du français

**Hyperparamètres**:
- Epochs: 3
- Batch size: 64 (avec gradient accumulation × 2 = effective 128)
- Learning rate: 5e-4
- Optimizer: AdamW
- Precision: BF16 (B200 natif)
- Gradient checkpointing: Oui

**Script**: `model/pretrain.py`

```bash
python model/pretrain.py \
    --data-path data/raw/french_corpus_merged.txt \
    --tokenizer-path tokenizer/french_keyboard.model \
    --epochs 3 \
    --batch-size 64
```

**Output**: `output/pretrain/final/`

**Métriques attendues**:
- Perplexité finale: 15-25 (selon taille corpus)
- Loss finale: ~2.7-3.2

---

### Étape 4: Finetuning Phase 1 - Corrections individuelles (2-4h)

**Objectif**: Apprendre à corriger des mots isolés

**Format d'entraînement**:
```
<XBU><CHAR_B><CHAR_O><CHAR_N><CHAR_J><CHAR_O><CHAR_Y><CHAR_R><XBC>bonjour <XEC>
```

**Données**:
1. Extraire 50k mots les plus fréquents du corpus
2. Appliquer log(N) sampling pour réduire biais
3. Générer 3 fautes par mot avec `AzertyErrorGenerator`
4. Total: ~150k exemples

**Types de fautes AZERTY**:
- Touche adjacente: 35% (e→r, a→z)
- Caractère manquant: 15%
- Caractère doublé: 10%
- Transposition: 15%
- Erreur d'accent: 25% (é→e, à→a)

**Script**: `model/finetune.py --phase phase1`

**Hyperparamètres**:
- Epochs: 3
- Batch size: 32
- Learning rate: 1e-4 (10× plus faible que pretrain)
- Max length: 128

**Output**: `output/finetune/phase1/final/`

---

### Étape 5: Finetuning Phase 2 - Corrections en contexte (4-6h)

**Objectif**: Corriger en tenant compte du contexte environnant

**Format d'entraînement**:
```
Bonjour, je <XBU><CHAR_V><CHAR_A><CHAR_I><CHAR_S><XBC>vais <XEC> très bien.
```

**Données**:
1. Prendre 100k phrases du corpus
2. Pour chaque phrase, introduire fautes sur ~33% des mots
3. Total: ~100k phrases augmentées

**Script**: `model/finetune.py --phase phase2`

**Hyperparamètres**:
- Epochs: 3
- Batch size: 16 (séquences plus longues)
- Learning rate: 5e-5 (encore plus faible)
- Max length: 512

**Output**: `output/finetune/phase2/final/`

---

### Étape 6 (Optionnelle): Finetuning Phase 3 - Langage informel (2-4h)

**Objectif**: Comprendre "J'ai", "On va", argot, SMS, etc.

**Données nécessaires**:
- Corpus de forums français
- Tweets/messages français
- Conversations informelles

⚠️ **Note**: Cette phase nécessite un corpus spécifique non inclus par défaut

**Script**: `model/finetune.py --phase phase3`

---

## 🔄 Export et déploiement

### Conversion GGUF

**Méthode 1**: Via llama.cpp (recommandé)
```bash
# 1. Convertir HF → GGUF
python llama.cpp/convert.py \
    output/finetune/phase2/final \
    --outfile french_keyboard.gguf \
    --outtype f16

# 2. Ajouter métadonnées FUTO (modification manuelle requise)
```

**Méthode 2**: Via Python gguf lib
```bash
python export/to_gguf.py \
    --model-path output/finetune/phase2/final \
    --tokenizer-path tokenizer/french_keyboard.model \
    --output french_keyboard.gguf \
    --use-python-gguf
```

### Quantization (optionnel)

Pour réduire la taille:
```bash
# Q8_0: ~40MB (précision élevée)
# Q4_0: ~20MB (précision moyenne)

llama.cpp/quantize french_keyboard.gguf french_keyboard_q8.gguf q8_0
```

---

## 📊 Métriques d'évaluation

### Tests automatiques

```bash
python evaluation/eval.py \
    --model-path output/finetune/phase2/final \
    --tokenizer-path tokenizer/french_keyboard.model \
    --test-file evaluation/test_data_example.json
```

### Métriques cibles

| Métrique | Objectif | Excellent |
|----------|----------|-----------|
| Perplexité | < 20 | < 15 |
| Autocorrect accuracy | > 75% | > 85% |
| Top-1 next word | > 30% | > 40% |
| Top-5 next word | > 60% | > 75% |

---

## 💾 Ressources requises

### Stockage

```
Corpus brut:           ~15 GB
Corpus tokenizé:       ~20 GB
Modèle checkpoints:    ~30 GB (plusieurs versions)
Modèle final:          ~140 MB (fp32), ~70 MB (fp16)
GGUF quantized:        ~20-40 MB
-------------------------------------------
Total:                 ~65-70 GB
```

### GPU (B200)

```
VRAM utilisée:         ~40-60 GB (selon batch size)
Temps préentraînement: 6-12 heures
Temps finetuning:      6-10 heures
-------------------------------------------
Total:                 15-25 heures
```

### Coût estimé (location B200)

```
Tarif moyen B200:      ~$3-5/heure
Durée totale:          ~20 heures
-------------------------------------------
Coût total estimé:     $60-100
```

---

## ✅ Checklist de validation

Avant de considérer le modèle terminé:

- [ ] Tokenizer encode avec espaces en SUFFIXE
- [ ] Vocabulaire contient tous les tokens spéciaux FUTO
- [ ] Perplexité < 20 sur corpus de validation français
- [ ] Autocorrect accuracy > 75% sur test set
- [ ] Top-5 next word accuracy > 60%
- [ ] Fichier GGUF contient toutes les métadonnées FUTO
- [ ] Tokenizer SentencePiece embedé dans le GGUF
- [ ] Test d'import réussi dans FUTO Keyboard Android
- [ ] Latence < 100ms pour prédiction sur smartphone moyen

---

## 🚨 Points critiques à ne pas oublier

### 1. Whitespace en suffixe (CRUCIAL!)

❌ **Faux** (whitespace préfixé):
```python
["_bonjour", "_le", "_monde"]
```

✅ **Correct** (whitespace suffixé):
```python
["bonjour ", "le ", "monde"]
```

Configuration SentencePiece:
```python
treat_whitespace_as_suffix=True  # OBLIGATOIRE!
```

### 2. Tokens spéciaux obligatoires

Le tokenizer DOIT contenir:
- `<XBU>`, `<XBC>`, `<XEC>`
- `<CHAR_A>` à `<CHAR_Z>` (26 tokens)
- `<CHAR_À>`, `<CHAR_É>`, etc. (accents français)

### 3. Log sampling en Phase 1

Pour éviter le biais vers mots fréquents:
```python
if word appears N times → generate log(N) training examples
```

### 4. Métadonnées GGUF

Sans ces métadonnées, FUTO Keyboard **rejettera** le modèle:
- `keyboardlm.languages`
- `keyboardlm.features`
- `keyboardlm.ext_tokenizer_type`
- `keyboardlm.ext_tokenizer_data`

---

## 🔍 Debugging courant

### Problème 1: "Model too slow on device"

**Solutions**:
- Quantize en Q4_0 ou Q8_0
- Réduire vocab_size à 10k
- Réduire num_layers à 6

### Problème 2: "Poor autocorrect accuracy"

**Solutions**:
- Augmenter epochs en Phase 1
- Vérifier que fautes sont réalistes (test manuel)
- Augmenter diversité des fautes (modifier error_generator)

### Problème 3: "Suggestions not contextual"

**Solutions**:
- Augmenter epochs en Phase 2
- Augmenter max_length à 1024
- Vérifier que Phase 2 utilise bien le contexte

---

## 📚 Références

- [FUTO Keyboard Docs](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Llama Architecture](https://arxiv.org/abs/2302.13971)
- [SentencePiece](https://github.com/google/sentencepiece)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Date de création**: 2025-11-13
**Auteur**: Claude (Assistant IA)
**Statut**: ✅ Prêt pour l'exécution
