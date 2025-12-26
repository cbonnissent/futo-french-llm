# FUTO French Keyboard LLM

Modèle de langue français pour le clavier FUTO, optimisé pour la **prédiction du mot suivant** et l'autocorrection sur clavier **AZERTY**.

## 📋 Vue d'ensemble

Ce projet implémente un modèle de langue Llama de ~36M paramètres pour le clavier FUTO, basé sur la [documentation officielle](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs).

### Caractéristiques

- ✅ Architecture Llama 36M paramètres
- ✅ Tokenizer SentencePiece avec whitespace en suffixe ([TOKENIZER_EXPLAINED.md](TOKENIZER_EXPLAINED.md))
- ✅ Support AZERTY français
- ✅ Génération de fautes synthétiques adaptées au français
- ✅ **Pipeline optimisé HuggingFace Skills** (2x plus rapide, 30% moins de VRAM)
- ✅ Entraînement en 3 phases avec SFT/DPO
- ✅ Export GGUF avec métadonnées FUTO
- ✅ Optimisé pour GPU B200/A100/RTX 4090

### 🚀 Nouveautés (Optimisations HF Skills)

- ⚡ **SFTTrainer** avec packing → 2-3x plus rapide
- ⚡ **Flash Attention 2** → +40-50% de vitesse
- ⚡ **Fused AdamW** → +10-15% de vitesse
- 💾 **30% moins de VRAM** nécessaire
- 📊 **Monitoring W&B** intégré
- 🔧 **Validation automatique** des datasets

### 📖 Documentation complète disponible

- 🏃 [QUICKSTART.md](QUICKSTART.md) - Démarrage rapide en 5 minutes
- 🔤 [TOKENIZER_EXPLAINED.md](TOKENIZER_EXPLAINED.md) - Comprendre le tokenizer
- ⚡ [IMPROVEMENTS_HF_SKILLS.md](IMPROVEMENTS_HF_SKILLS.md) - Détails des optimisations
- 💰 [COST_ESTIMATION.md](COST_ESTIMATION.md) - Budget et choix de GPU

---

## ⚡ Quick Start (TL;DR)

Pour entraîner rapidement votre modèle :

```bash
# 1. Installation
pip install -r requirements-optimized.txt

# 2. Télécharger données + Tokenizer + Pipeline complet
./scripts/train_full_pipeline.sh

# OU étape par étape avec méthode optimisée:

# Données
python scripts/download_data.py --sources wikipedia --merge

# Tokenizer
python tokenizer/train_tokenizer.py \
    --input data/raw/french_corpus_merged.txt

# Entraînement complet (optimisé)
python model/pretrain.py ...
python model/finetune_hf_skills.py ...  # ⚡ 2x plus rapide
```

**Temps total** : ~15-20h sur A100 (~30€)
**Alternative rapide** : Voir [QUICKSTART.md](QUICKSTART.md)

---

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd futo-french-llm

# Installer les dépendances (version standard)
pip install -r requirements.txt

# OU version optimisée (recommandé - avec TRL, Flash Attention)
pip install -r requirements-optimized.txt
```

**💡 Recommandation**: Utilisez `requirements-optimized.txt` pour les meilleures performances (2x plus rapide)

### 2. Télécharger les données

```bash
# Télécharger Wikipedia + OSCAR + mC4 français
python scripts/download_data.py \
    --output-dir ./data/raw \
    --sources all \
    --merge

# Cela créera: ./data/raw/french_corpus_merged.txt
```

Ou fournissez votre propre corpus français (fichier texte).

### 3. Entraîner le tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --input ./data/raw/french_corpus_merged.txt \
    --model-prefix ./tokenizer/french_keyboard \
    --vocab-size 15008

# Cela créera:
# - ./tokenizer/french_keyboard.model
# - ./tokenizer/french_keyboard.vocab
```

### 4. Préentraînement

Sur votre B200 :

```bash
python model/pretrain.py \
    --data-path ./data/raw/french_corpus_merged.txt \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --output-dir ./output/pretrain \
    --model-size default \
    --epochs 3 \
    --batch-size 64 \
    --learning-rate 5e-4 \
    --gradient-accumulation-steps 2 \
    --use-wandb  # Optionnel: tracking avec W&B

# Temps estimé sur B200: ~6-12h selon la taille du corpus
```

### 5. Finetuning (2 méthodes disponibles)

#### ⚡ Méthode OPTIMISÉE (recommandée - 2x plus rapide)

Utilise SFTTrainer + Flash Attention + Packing

```bash
python model/finetune_hf_skills.py \
    --model-path ./output/pretrain/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --corpus-path ./data/raw/french_corpus_merged.txt \
    --output-dir ./output/finetune_sft \
    --phase all \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --use-wandb  # Monitoring temps réel (optionnel)

# Temps estimé: ~7-10h (vs 15-20h avec méthode standard)
# VRAM: 14-18 GB (vs 20-24 GB)
```

#### 🔧 Méthode STANDARD (compatible tous GPU)

```bash
python model/finetune.py \
    --model-path ./output/pretrain/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --corpus-path ./data/raw/french_corpus_merged.txt \
    --output-dir ./output/finetune \
    --phase all \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 1e-4

# Temps estimé: ~15-20h
```

**Phases exécutées (les deux méthodes)** :
1. Corrections individuelles (log(N) sampling)
2. Corrections en contexte (33% de mots avec fautes)
3. Langage informel (optionnel, nécessite corpus spécifique)

**Quelle méthode choisir ?**
- ✅ **Optimisée** : Si vous avez installé `requirements-optimized.txt` (Flash Attention compatible)
- ✅ **Standard** : Si problèmes de compatibilité ou GPU ancien

### 6. Export GGUF

```bash
python export/to_gguf.py \
    --model-path ./output/finetune/phase2/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --output ./output/french_keyboard.gguf \
    --quantization f16 \
    --language fr

# Note: Cela créera un script pour llama.cpp
# Suivez les instructions affichées pour finaliser la conversion
```

---

## 📁 Structure du projet

```
futo-french-llm/
├── 📚 Documentation
│   ├── README.md                    # Ce fichier
│   ├── QUICKSTART.md                # Guide démarrage rapide
│   ├── PLAN.md                      # Plan détaillé du projet
│   ├── TOKENIZER_EXPLAINED.md       # Explication tokenizer
│   ├── IMPROVEMENTS_HF_SKILLS.md    # Optimisations HF Skills
│   └── COST_ESTIMATION.md           # Estimation coûts
│
├── 📦 Dependencies
│   ├── requirements.txt             # Dépendances standard
│   ├── requirements-optimized.txt   # ⭐ Optimisées (recommandé)
│   └── config.example.yaml          # Configuration d'exemple
│
├── 💾 Data
│   ├── data/raw/                    # Corpus bruts
│   ├── data/processed/              # Données prétraitées
│   └── data/synthetic/              # Fautes synthétiques
│
├── 🔤 Tokenizer
│   ├── tokenizer/train_tokenizer.py # Entraînement tokenizer
│   └── tokenizer/demo_tokenizer.py  # Démo interactive
│
├── 🧠 Modèle
│   ├── model/config.py              # Configuration Llama
│   ├── model/pretrain.py            # Préentraînement
│   ├── model/finetune.py            # Finetuning standard
│   └── model/finetune_hf_skills.py  # ⭐ Finetuning optimisé (SFT)
│
├── ⌨️ Synthetic Errors
│   └── synthetic_errors/generate_errors.py  # Générateur fautes AZERTY
│
├── 📊 Evaluation
│   ├── evaluation/eval.py           # Évaluation du modèle
│   └── evaluation/test_data_example.json
│
├── 📤 Export
│   └── export/to_gguf.py            # Conversion GGUF
│
└── 🚀 Scripts
    ├── scripts/download_data.py     # Téléchargement données
    └── scripts/train_full_pipeline.sh  # Pipeline complet
```

**⭐ = Recommandé pour meilleures performances**

---

## 🔧 Configuration du modèle

### Configuration par défaut (36M paramètres)

```python
vocab_size = 15008
hidden_size = 512
intermediate_size = 1024
num_hidden_layers = 8
num_attention_heads = 8
```

**Taille du modèle**: ~70 MB (float16), ~140 MB (float32)

### Configuration large (100M paramètres)

```python
vocab_size = 15008
hidden_size = 768
intermediate_size = 2048
num_hidden_layers = 12
num_attention_heads = 12
```

Pour utiliser la config large:
```bash
python model/pretrain.py --model-size large ...
```

---

## 🎯 Générateur de fautes AZERTY

Le générateur crée des fautes réalistes pour clavier AZERTY français:

### Types de fautes

1. **Touches adjacentes** (35%): e → r, a → z, etc.
2. **Caractère manquant** (15%): bonjour → bonjur
3. **Caractère doublé** (10%): bonjour → bonnjour
4. **Transposition** (15%): bonjour → bonojur
5. **Erreurs d'accents** (25%): français → francais, été → ete

### Exemple d'utilisation

```python
from synthetic_errors.generate_errors import AzertyErrorGenerator

generator = AzertyErrorGenerator()

# Générer une faute
word = "bonjour"
misspelled = generator.generate_word_error(word)
# → "bonjoyr", "bonjur", "bonnjour", etc.

# Format pour l'entraînement
formatted = generator.format_for_training(word, misspelled)
# → "<XBU><CHAR_B><CHAR_O><CHAR_N><CHAR_J><CHAR_O><CHAR_Y><CHAR_R><XBC>bonjour <XEC>"
```

---

## 📊 Évaluation

Évaluez le modèle avec:

```bash
python evaluation/eval.py \
    --model-path ./output/finetune/phase2/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --output ./evaluation_results.json
```

### Métriques

- **Perplexité**: Qualité globale du modèle de langue
- **Accuracy autocorrection**: % de corrections réussies
- **Top-1/Top-5 accuracy**: Prédiction du mot suivant

---

## 🔄 Format FUTO

### Tokens spéciaux

- `<XBU>`: Beginning of user input
- `<CHAR_A>` à `<CHAR_Z>`: Caractères A-Z
- `<CHAR_À>`, `<CHAR_É>`, etc.: Caractères accentués français
- `<XBC>`: Beginning of correction
- `<XEC>`: End of correction

### Exemple de correction

```
Entrée utilisateur: "bonjoyr"
Format: <XBU><CHAR_B><CHAR_O><CHAR_N><CHAR_J><CHAR_O><CHAR_Y><CHAR_R><XBC>
Prédiction modèle: bonjour <XEC>
```

### Métadonnées GGUF

```json
{
  "keyboardlm.languages": "fr",
  "keyboardlm.finetuning_count": 0,
  "keyboardlm.features": "base_v1 inverted_space xbu_char_autocorrect_v1",
  "keyboardlm.ext_tokenizer_type": "SentencePiece",
  "keyboardlm.ext_tokenizer_data": [binary data]
}
```

---

## 📝 Plan d'entraînement détaillé

### Phase 1: Préentraînement
- **Objectif**: Apprendre la structure du français
- **Données**: 1-5 milliards de tokens (Wikipedia, OSCAR, mC4)
- **Durée**: 3 époques (~6-12h sur B200)
- **Loss**: Cross-entropy standard

### Phase 2: Finetuning corrections individuelles
- **Objectif**: Apprendre à corriger des mots isolés
- **Données**: Mots avec log(N) sampling + fautes synthétiques
- **Format**: `<XBU>...<XBC>mot_correct <XEC>`
- **Durée**: 3 époques (~2-4h)

### Phase 3: Finetuning corrections en contexte
- **Objectif**: Corriger en tenant compte du contexte
- **Données**: Phrases avec ~33% de mots avec fautes
- **Durée**: 3 époques (~4-6h)

### Phase 4 (optionnelle): Langage informel
- **Objectif**: Comprendre "J'ai", "On va", argot
- **Données**: Forums, tweets, messages
- **Durée**: 2-3 époques

### ⏱️ Durées totales estimées

| GPU | Méthode | Préentraînement | Finetuning | Total | Coût* |
|-----|---------|-----------------|------------|-------|-------|
| **B200** | Standard | 6-12h | 10-14h | **20-30h** | ~110€ |
| **B200** | Optimisée | 4-6h | 5-7h | **10-15h** | ~60€ |
| **A100 80GB** | Standard | 10-15h | 15-20h | **30-40h** | ~60€ |
| **A100 80GB** | Optimisée | 6-8h | 8-12h | **15-20h** | ~30€ |
| **RTX 4090** | Standard | 18-24h | 20-28h | **40-50h** | ~30€** |
| **RTX 4090** | Optimisée | 10-14h | 12-16h | **25-30h** | ~18€** |

*Prix location cloud
**Vast.ai (0.44€/h)

**💡 Recommandation** : Utilisez **A100 80GB avec méthode optimisée** (~30€ total) pour le meilleur rapport qualité/prix.

Voir [COST_ESTIMATION.md](COST_ESTIMATION.md) pour analyse détaillée des coûts.

---

## 💡 Conseils d'optimisation

### ⚡ Utiliser le pipeline optimisé (recommandé)

**Méthode 1 : Utiliser `finetune_hf_skills.py`**

Les optimisations HuggingFace Skills sont déjà intégrées :

```bash
# Installer les dépendances optimisées
pip install -r requirements-optimized.txt

# Utiliser le script optimisé
python model/finetune_hf_skills.py ...
```

**Gains automatiques** :
- ✅ **Packing** : 2-3x plus rapide (combine exemples courts)
- ✅ **Flash Attention 2** : +40-50% vitesse (attention optimisée)
- ✅ **Fused AdamW** : +10-15% vitesse (optimiseur optimisé)
- ✅ **Group by length** : Réduit le padding
- ✅ **VRAM** : -30% d'utilisation mémoire

**Résultat total** : **2x plus rapide** + **30% moins de VRAM**

### 🔧 Optimisations manuelles (pour scripts custom)

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Configuration optimisée
training_args = TrainingArguments(
    # Performance
    bf16=True,                      # Précision mixte (B200/A100)
    gradient_checkpointing=True,     # Économie mémoire
    optim="adamw_torch_fused",      # Optimiseur rapide
    group_by_length=True,           # Réduit padding

    # Batch settings
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,

    # Monitoring
    report_to=["wandb"],            # Temps réel
)

# Utiliser SFTTrainer au lieu de Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    max_seq_length=512,
    packing=True,  # ⚡ 2-3x plus rapide!
)
```

### 📊 Comparaison des performances

| Configuration | Temps | VRAM | Coût (A100) |
|---------------|-------|------|-------------|
| **Standard** | 20h | 24 GB | ~40€ |
| **Optimisée** | 10h | 16 GB | ~20€ |
| **Gain** | **-50%** | **-33%** | **-50%** |

Voir [IMPROVEMENTS_HF_SKILLS.md](IMPROVEMENTS_HF_SKILLS.md) pour plus de détails.

### 💾 Réduire l'utilisation mémoire

Si problèmes de VRAM :

1. Activer `gradient_checkpointing=True`
2. Réduire `max_length` (512 → 256)
3. Réduire `batch_size` (64 → 32)
4. Utiliser `bf16=True` au lieu de `fp32`
5. Utiliser packing (`packing=True` dans SFTTrainer)

### 🚀 Accélérer encore plus

1. Utiliser plusieurs GPUs :
```bash
accelerate launch --multi_gpu model/finetune_hf_skills.py ...
```

2. Précomputer les tokens (cache) :
```bash
# Tokenize une fois, réutilise ensuite
python scripts/precompute_tokens.py
```

3. Utiliser spot instances (cloud) :
- AWS/GCP/Azure : -50% du prix
- Risque interruption mais checkpoints auto

---

## 🐛 Débogage

### Problème: Tokenizer n'encode pas correctement

**Solution**: Vérifiez que `treat_whitespace_as_suffix=True`

```python
sp = spm.SentencePieceProcessor()
sp.load('tokenizer/french_keyboard.model')

# Test
tokens = sp.encode_as_pieces("bonjour le monde")
print(tokens)
# Attendu: ['bonjour ', 'le ', 'monde'] (espaces en suffixe)
```

### Problème: Le modèle ne corrige pas bien

**Solutions**:
1. Augmenter les époques de finetuning Phase 1
2. Vérifier que les fautes synthétiques sont réalistes
3. Utiliser plus de données pour le préentraînement
4. Réduire le `learning_rate` en finetuning

### Problème: Out of memory

**Solutions**:
1. Réduire `per_device_train_batch_size`
2. Activer `gradient_checkpointing=True`
3. Réduire `max_length`
4. Utiliser une config plus petite

---

## 📚 Ressources

### Documentation du projet

- 📖 [QUICKSTART.md](QUICKSTART.md) - Démarrage rapide (5 min)
- 📋 [PLAN.md](PLAN.md) - Plan détaillé avec timeline et métriques
- 🔤 [TOKENIZER_EXPLAINED.md](TOKENIZER_EXPLAINED.md) - Guide complet du tokenizer
- ⚡ [IMPROVEMENTS_HF_SKILLS.md](IMPROVEMENTS_HF_SKILLS.md) - Optimisations HF Skills
- 💰 [COST_ESTIMATION.md](COST_ESTIMATION.md) - Estimation des coûts détaillée

### Ressources externes

- [FUTO Keyboard Wiki](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs) - Documentation officielle
- [HuggingFace TRL](https://huggingface.co/docs/trl) - SFT, DPO, GRPO
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Framework principal
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenizer
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inférence C++
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Attention optimisée

---

## 📄 Licence

Ce projet suit les guidelines FUTO Keyboard. Le modèle est destiné à un usage personnel sur le clavier FUTO.

**⚠️ Important**: Ne partagez pas les modèles fintuned avec vos données personnelles (voir avertissement de confidentialité FUTO).

---

## 🤝 Contribution

Les contributions sont bienvenues! Domaines d'amélioration:

- [ ] Améliorer le générateur de fautes (plus réaliste)
- [ ] Ajouter support pour swipe typing
- [ ] Optimiser la taille du modèle (quantization)
- [ ] Créer un corpus de langage informel français
- [ ] Implémenter `char_embed_mixing_v1` (interpolation des embeddings)

---

## ❓ FAQ

### Quelle méthode de finetuning choisir ?

**Méthode optimisée (`finetune_hf_skills.py`)** :
- ✅ 2x plus rapide
- ✅ 30% moins de VRAM
- ✅ Monitoring W&B intégré
- ⚠️ Nécessite Flash Attention 2 (GPU compatible)
- **Recommandée si vous avez un GPU moderne** (A100, RTX 4090, etc.)

**Méthode standard (`finetune.py`)** :
- ✅ Compatible tous GPUs
- ✅ Pas de dépendances spécifiques
- ⚠️ Plus lent
- **Recommandée si problèmes de compatibilité**

### Quel GPU choisir ?

Pour **UN modèle** (test/POC) :
- **A100 80GB cloud** (~30€) - Meilleur rapport qualité/prix
- RTX 4090 cloud (~15€) - Budget serré

Pour **10+ modèles** (itérations) :
- **RTX 4090 local** (2810€) - S'amortit après ~43 entraînements
- A100 cloud - Si usage occasionnel

Voir [COST_ESTIMATION.md](COST_ESTIMATION.md) pour analyse complète.

### Pourquoi 36M paramètres?

C'est la taille recommandée par FUTO pour un bon compromis performance/latence sur mobile.

### Puis-je utiliser un modèle plus grand?

Oui, utilisez `--model-size large` (100M params), mais l'inférence sera plus lente.

### Combien de données faut-il?

Minimum: 1GB de texte français (~200M tokens)
Recommandé: 5-10GB (~1-2B tokens)

### Le modèle fonctionne sur quel matériel?

Le modèle final (GGUF) est conçu pour smartphone Android via llama.cpp.

### Comment tester le modèle avant export?

Utilisez `evaluation/eval.py` sur le modèle HuggingFace.

### Flash Attention 2 ne s'installe pas, que faire ?

Utilisez la méthode standard (`finetune.py`) ou installez sans Flash Attention :
```bash
pip install -r requirements.txt  # Sans Flash Attention
```

### Combien coûte l'entraînement d'un modèle ?

- **A100 optimisé** : ~30€ (recommandé)
- **B200 optimisé** : ~60€ (si vitesse critique)
- **RTX 4090 cloud** : ~18€ (budget serré)

Budget conseillé pour démarrer (3-5 itérations) : **100-150€**

### Le tokenizer est-il réutilisable pour d'autres langues ?

Non, il faut réentraîner pour chaque langue. Voir [TOKENIZER_EXPLAINED.md](TOKENIZER_EXPLAINED.md).

---

**Bon entraînement! 🚀**
