#!/bin/bash
# Script complet d'entraînement du modèle FUTO français
# Usage: ./scripts/train_full_pipeline.sh

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "Pipeline d'entraînement FUTO French LLM"
echo "=========================================="

# Configuration
DATA_DIR="./data/raw"
TOKENIZER_DIR="./tokenizer"
OUTPUT_DIR="./output"
VOCAB_SIZE=15008
MODEL_SIZE="default"  # ou "large"
BATCH_SIZE=64
EPOCHS_PRETRAIN=3
EPOCHS_FINETUNE=3

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Étape 1: Téléchargement des données
echo -e "\n${BLUE}[1/6] Téléchargement des données françaises...${NC}"
if [ ! -f "$DATA_DIR/french_corpus_merged.txt" ]; then
    python scripts/download_data.py \
        --output-dir "$DATA_DIR" \
        --sources wikipedia oscar \
        --max-samples 100000 \
        --merge
    echo -e "${GREEN}✓ Données téléchargées${NC}"
else
    echo -e "${YELLOW}⚠ Données déjà présentes, skip${NC}"
fi

# Étape 2: Entraînement du tokenizer
echo -e "\n${BLUE}[2/6] Entraînement du tokenizer...${NC}"
if [ ! -f "$TOKENIZER_DIR/french_keyboard.model" ]; then
    python tokenizer/train_tokenizer.py \
        --input "$DATA_DIR/french_corpus_merged.txt" \
        --model-prefix "$TOKENIZER_DIR/french_keyboard" \
        --vocab-size $VOCAB_SIZE
    echo -e "${GREEN}✓ Tokenizer entraîné${NC}"
else
    echo -e "${YELLOW}⚠ Tokenizer déjà présent, skip${NC}"
fi

# Étape 3: Préentraînement
echo -e "\n${BLUE}[3/6] Préentraînement du modèle...${NC}"
if [ ! -d "$OUTPUT_DIR/pretrain/final" ]; then
    python model/pretrain.py \
        --data-path "$DATA_DIR/french_corpus_merged.txt" \
        --tokenizer-path "$TOKENIZER_DIR/french_keyboard.model" \
        --output-dir "$OUTPUT_DIR/pretrain" \
        --model-size $MODEL_SIZE \
        --epochs $EPOCHS_PRETRAIN \
        --batch-size $BATCH_SIZE \
        --learning-rate 5e-4 \
        --gradient-accumulation-steps 2
    echo -e "${GREEN}✓ Préentraînement terminé${NC}"
else
    echo -e "${YELLOW}⚠ Modèle préentraîné déjà présent, skip${NC}"
fi

# Étape 4: Finetuning Phase 1 (corrections individuelles)
echo -e "\n${BLUE}[4/6] Finetuning Phase 1: corrections individuelles...${NC}"
python model/finetune.py \
    --model-path "$OUTPUT_DIR/pretrain/final" \
    --tokenizer-path "$TOKENIZER_DIR/french_keyboard.model" \
    --corpus-path "$DATA_DIR/french_corpus_merged.txt" \
    --output-dir "$OUTPUT_DIR/finetune" \
    --phase phase1 \
    --epochs $EPOCHS_FINETUNE \
    --batch-size 32 \
    --learning-rate 1e-4
echo -e "${GREEN}✓ Phase 1 terminée${NC}"

# Étape 5: Finetuning Phase 2 (corrections en contexte)
echo -e "\n${BLUE}[5/6] Finetuning Phase 2: corrections en contexte...${NC}"
python model/finetune.py \
    --model-path "$OUTPUT_DIR/finetune/phase1/final" \
    --tokenizer-path "$TOKENIZER_DIR/french_keyboard.model" \
    --corpus-path "$DATA_DIR/french_corpus_merged.txt" \
    --output-dir "$OUTPUT_DIR/finetune" \
    --phase phase2 \
    --epochs $EPOCHS_FINETUNE \
    --batch-size 16 \
    --learning-rate 5e-5
echo -e "${GREEN}✓ Phase 2 terminée${NC}"

# Étape 6: Évaluation
echo -e "\n${BLUE}[6/6] Évaluation du modèle...${NC}"
python evaluation/eval.py \
    --model-path "$OUTPUT_DIR/finetune/phase2/final" \
    --tokenizer-path "$TOKENIZER_DIR/french_keyboard.model" \
    --test-file evaluation/test_data_example.json \
    --output "$OUTPUT_DIR/evaluation_results.json"
echo -e "${GREEN}✓ Évaluation terminée${NC}"

# Résumé
echo -e "\n=========================================="
echo -e "${GREEN}✓ Pipeline complet terminé!${NC}"
echo "=========================================="
echo ""
echo "Modèle final: $OUTPUT_DIR/finetune/phase2/final"
echo "Tokenizer: $TOKENIZER_DIR/french_keyboard.model"
echo "Résultats évaluation: $OUTPUT_DIR/evaluation_results.json"
echo ""
echo "Prochaines étapes:"
echo "1. Vérifier les résultats d'évaluation"
echo "2. Convertir en GGUF:"
echo "   python export/to_gguf.py \\"
echo "       --model-path $OUTPUT_DIR/finetune/phase2/final \\"
echo "       --tokenizer-path $TOKENIZER_DIR/french_keyboard.model \\"
echo "       --output $OUTPUT_DIR/french_keyboard.gguf"
echo ""
