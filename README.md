# FUTO French Keyboard LLM

Modèle de langue français pour le clavier FUTO, optimisé pour la **prédiction du mot suivant** et l'autocorrection sur clavier **AZERTY**.

## 📋 Vue d'ensemble

Ce projet implémente un modèle de langue Llama de ~36M paramètres pour le clavier FUTO, basé sur la [documentation officielle](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs).

### Caractéristiques

- ✅ Architecture Llama 36M paramètres
- ✅ Tokenizer SentencePiece avec whitespace en suffixe
- ✅ Support AZERTY français
- ✅ Génération de fautes synthétiques adaptées au français
- ✅ Entraînement en 3 phases (corrections individuelles → contexte → langage informel)
- ✅ Export GGUF avec métadonnées FUTO
- ✅ Optimisé pour GPU B200

---

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd futo-french-llm

# Installer les dépendances
pip install -r requirements.txt
```

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

### 5. Finetuning (3 phases)

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

# Phases exécutées:
# 1. Corrections individuelles (log(N) sampling)
# 2. Corrections en contexte (33% de mots avec fautes)
# 3. Langage informel (optionnel, nécessite corpus spécifique)
```

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
├── data/
│   ├── raw/              # Corpus bruts
│   ├── processed/        # Données prétraitées
│   └── synthetic/        # Fautes synthétiques
├── tokenizer/
│   └── train_tokenizer.py
├── model/
│   ├── config.py         # Configuration Llama
│   ├── pretrain.py       # Préentraînement
│   └── finetune.py       # Finetuning (3 phases)
├── synthetic_errors/
│   └── generate_errors.py  # Générateur de fautes AZERTY
├── evaluation/
│   └── eval.py           # Évaluation du modèle
├── export/
│   └── to_gguf.py        # Conversion GGUF
├── scripts/
│   └── download_data.py  # Téléchargement de données
├── requirements.txt
└── README.md
```

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

**Total estimé**: ~15-25 heures sur B200

---

## 💡 Conseils d'optimisation

### Pour GPU B200

```python
# Dans pretrain.py / finetune.py
training_args = TrainingArguments(
    bf16=True,  # B200 supporte bf16 nativement
    gradient_checkpointing=True,
    per_device_train_batch_size=64,  # Ajuster selon VRAM
    gradient_accumulation_steps=2,
    dataloader_num_workers=8,  # Paralléliser le chargement
    optim="adamw_torch",  # Ou adamw_torch_fused pour plus de vitesse
)
```

### Réduire l'utilisation mémoire

- Utiliser `gradient_checkpointing=True`
- Réduire `max_length` (512 → 256)
- Réduire `batch_size`
- Utiliser `bf16=True` au lieu de `fp32`

### Accélérer l'entraînement

- Augmenter `batch_size` × `gradient_accumulation_steps`
- Utiliser plusieurs GPUs avec `accelerate`
- Pré-tokenizer les données (voir scripts/preprocess.py)

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

- [FUTO Keyboard Wiki](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SentencePiece](https://github.com/google/sentencepiece)

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

---

**Bon entraînement! 🚀**
