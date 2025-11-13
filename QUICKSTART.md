# Démarrage rapide - 5 minutes ⚡

Guide ultra-rapide pour commencer l'entraînement du modèle français FUTO.

## Prérequis

- Python 3.8+
- GPU avec CUDA (recommandé: B200)
- 50GB d'espace disque libre

## Installation express

```bash
# 1. Cloner et installer
git clone <votre-repo>
cd futo-french-llm
pip install -r requirements.txt

# 2. Rendre le script exécutable
chmod +x scripts/train_full_pipeline.sh

# 3. Lancer le pipeline complet
./scripts/train_full_pipeline.sh
```

C'est tout! Le script va:
1. ✅ Télécharger les données françaises
2. ✅ Entraîner le tokenizer
3. ✅ Préentraîner le modèle
4. ✅ Finetuner (2 phases)
5. ✅ Évaluer le modèle

**Temps estimé sur B200**: 15-25 heures

---

## Étapes manuelles (si vous préférez le contrôle)

### 1. Données (5 min)

```bash
python scripts/download_data.py \
    --output-dir ./data/raw \
    --sources wikipedia \
    --max-samples 50000 \
    --merge
```

### 2. Tokenizer (10 min)

```bash
python tokenizer/train_tokenizer.py \
    --input ./data/raw/french_corpus_merged.txt \
    --model-prefix ./tokenizer/french_keyboard
```

### 3. Préentraînement (6-12h sur B200)

```bash
python model/pretrain.py \
    --data-path ./data/raw/french_corpus_merged.txt \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --output-dir ./output/pretrain \
    --batch-size 64 \
    --epochs 3
```

### 4. Finetuning (4-8h sur B200)

```bash
python model/finetune.py \
    --model-path ./output/pretrain/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --corpus-path ./data/raw/french_corpus_merged.txt \
    --output-dir ./output/finetune \
    --phase all \
    --epochs 3
```

### 5. Export GGUF (5 min)

```bash
python export/to_gguf.py \
    --model-path ./output/finetune/phase2/final \
    --tokenizer-path ./tokenizer/french_keyboard.model \
    --output ./output/french_keyboard.gguf
```

---

## Test rapide du générateur de fautes

```bash
python synthetic_errors/generate_errors.py
```

Vous verrez des exemples de fautes AZERTY:
```
bonjour → bonjoyr → <XBU><CHAR_B>...<XBC>bonjour <XEC>
merci → meric → <XBU><CHAR_M>...<XBC>merci <XEC>
```

---

## Vérifier que ça marche

Après l'entraînement:

```bash
python evaluation/eval.py \
    --model-path ./output/finetune/phase2/final \
    --tokenizer-path ./tokenizer/french_keyboard.model
```

Vous devriez voir:
```
✓ Accuracy autocorrection: 75-90%
✓ Top-1 accuracy: 30-50%
✓ Top-5 accuracy: 60-80%
```

---

## Problèmes courants

### "CUDA out of memory"
```bash
# Réduire le batch size
python model/pretrain.py --batch-size 32  # au lieu de 64
```

### "Pas de données"
```bash
# Vérifier que les données sont téléchargées
ls -lh data/raw/
# Devrait afficher french_corpus_merged.txt
```

### "Tokenizer introuvable"
```bash
# Vérifier que le tokenizer est entraîné
ls -lh tokenizer/
# Devrait afficher french_keyboard.model
```

---

## Support

- Voir le [README.md](README.md) complet pour plus de détails
- Consulter la [doc FUTO](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)

Bon entraînement! 🚀
