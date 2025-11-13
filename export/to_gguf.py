#!/usr/bin/env python3
"""
Script de conversion du modèle HuggingFace vers GGUF pour FUTO Keyboard
avec métadonnées appropriées
"""

import argparse
import struct
from pathlib import Path
from datetime import datetime
from typing import List

import torch
import numpy as np
from transformers import LlamaForCausalLM
import sentencepiece as spm


def write_gguf_metadata(
    gguf_writer,
    language: str = "fr",
    features: List[str] = None,
    finetuning_count: int = 0,
    history: str = None,
    tokenizer_path: str = None,
):
    """
    Ajoute les métadonnées FUTO au fichier GGUF

    Métadonnées requises selon la doc:
    - keyboardlm.languages: ISO language code(s)
    - keyboardlm.finetuning_count: Integer starting from 0
    - keyboardlm.history: Human-readable history
    - keyboardlm.features: Space-separated feature list
    - keyboardlm.ext_tokenizer_type: "SentencePiece"
    - keyboardlm.ext_tokenizer_data: Binary tokenizer data
    """
    if features is None:
        features = [
            "base_v1",
            "inverted_space",
            "xbu_char_autocorrect_v1"
        ]

    if history is None:
        history = f"{datetime.now().strftime('%Y-%m-%d')} Initial French keyboard model for FUTO"

    # Métadonnées FUTO
    gguf_writer.add_string("keyboardlm.languages", language)
    gguf_writer.add_uint32("keyboardlm.finetuning_count", finetuning_count)
    gguf_writer.add_string("keyboardlm.history", history)
    gguf_writer.add_string("keyboardlm.features", " ".join(features))
    gguf_writer.add_string("keyboardlm.ext_tokenizer_type", "SentencePiece")

    # Embedder le tokenizer sentencepiece
    if tokenizer_path:
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = f.read()
        gguf_writer.add_array("keyboardlm.ext_tokenizer_data", list(tokenizer_data))

    print("✓ Métadonnées FUTO ajoutées:")
    print(f"  - Langue: {language}")
    print(f"  - Features: {', '.join(features)}")
    print(f"  - Finetuning count: {finetuning_count}")
    print(f"  - Tokenizer: {'embedded' if tokenizer_path else 'not included'}")


def convert_to_gguf(
    model_path: str,
    tokenizer_path: str,
    output_path: str,
    quantization: str = "f16",
    language: str = "fr",
    features: List[str] = None,
    finetuning_count: int = 0,
    history: str = None,
):
    """
    Convertit un modèle HuggingFace Llama vers GGUF avec métadonnées FUTO

    Args:
        model_path: Chemin vers le modèle HuggingFace
        tokenizer_path: Chemin vers le tokenizer .model SentencePiece
        output_path: Chemin de sortie pour le fichier GGUF
        quantization: Type de quantization (f32, f16, q8_0, q4_0, etc.)
        language: Code ISO de langue
        features: Liste des features FUTO
        finetuning_count: Compteur de finetuning
        history: Historique du modèle
    """
    print("=" * 60)
    print("Conversion vers GGUF pour FUTO Keyboard")
    print("=" * 60)

    # Note: Cette implémentation utilise la conversion standard
    # Pour une conversion complète, utilisez convert.py de llama.cpp

    print(f"\nChargement du modèle depuis {model_path}...")
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    print(f"Chargement du tokenizer depuis {tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # Créer le dossier de sortie
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("\nPour convertir vers GGUF, utilisez le script convert.py de llama.cpp:")
    print("\n" + "-" * 60)
    print("# 1. Sauvegarder le modèle au format compatible")
    model.save_pretrained(f"{model_path}_for_conversion")

    # Créer un script de conversion
    conversion_script = f"""#!/bin/bash
# Script de conversion vers GGUF pour FUTO Keyboard

# 1. Convertir le modèle HuggingFace vers GGUF
python /path/to/llama.cpp/convert.py \\
    {model_path}_for_conversion \\
    --outfile {output_path} \\
    --outtype {quantization}

# 2. Ajouter les métadonnées FUTO (requiert modification de convert.py)
# Les métadonnées suivantes doivent être ajoutées:
#   keyboardlm.languages = "{language}"
#   keyboardlm.finetuning_count = {finetuning_count}
#   keyboardlm.history = "{history or 'Initial model'}"
#   keyboardlm.features = "{' '.join(features or ['base_v1', 'inverted_space', 'xbu_char_autocorrect_v1'])}"
#   keyboardlm.ext_tokenizer_type = "SentencePiece"
#   keyboardlm.ext_tokenizer_data = [binary data from {tokenizer_path}]

echo "✓ Conversion terminée: {output_path}"
"""

    script_path = Path(output_path).parent / "convert_to_gguf.sh"
    with open(script_path, 'w') as f:
        f.write(conversion_script)

    print(f"\n✓ Script de conversion créé: {script_path}")
    print("\nPour utiliser llama.cpp avec métadonnées FUTO:")
    print("1. Cloner llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print("2. Modifier convert.py pour ajouter les métadonnées FUTO")
    print("3. Exécuter le script de conversion")
    print("-" * 60)

    # Créer aussi un fichier de métadonnées JSON pour référence
    metadata = {
        "keyboardlm.languages": language,
        "keyboardlm.finetuning_count": finetuning_count,
        "keyboardlm.history": history or f"{datetime.now().strftime('%Y-%m-%d')} Initial French model",
        "keyboardlm.features": " ".join(features or ["base_v1", "inverted_space", "xbu_char_autocorrect_v1"]),
        "keyboardlm.ext_tokenizer_type": "SentencePiece",
        "keyboardlm.ext_tokenizer_path": tokenizer_path,
    }

    import json
    metadata_path = Path(output_path).parent / "futo_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Métadonnées sauvegardées: {metadata_path}")

    # Sauvegarder les informations du modèle
    print(f"\n✓ Modèle sauvegardé pour conversion: {model_path}_for_conversion")
    print("\nInformations du modèle:")
    print(f"  - Vocabulaire: {sp.vocab_size()}")
    print(f"  - Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Architecture: Llama")

    return metadata_path, script_path


def create_gguf_with_python_gguf(
    model_path: str,
    tokenizer_path: str,
    output_path: str,
    quantization: str = "f16",
):
    """
    Alternative: Utiliser la librairie Python gguf directement
    """
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        print("⚠️  La librairie 'gguf' n'est pas installée")
        print("   Installez avec: pip install gguf")
        return None

    print("\nUtilisation de la librairie Python gguf...")

    # Charger le modèle
    model = LlamaForCausalLM.from_pretrained(model_path)

    # Charger le tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # Créer le writer GGUF
    gguf_writer = GGUFWriter(output_path, arch="llama")

    # Ajouter les métadonnées du modèle
    config = model.config

    # Architecture
    gguf_writer.add_name("French FUTO Keyboard Model")
    gguf_writer.add_description("French language model for FUTO Keyboard")
    gguf_writer.add_block_count(config.num_hidden_layers)
    gguf_writer.add_context_length(config.max_position_embeddings)
    gguf_writer.add_embedding_length(config.hidden_size)
    gguf_writer.add_feed_forward_length(config.intermediate_size)
    gguf_writer.add_head_count(config.num_attention_heads)
    gguf_writer.add_head_count_kv(config.num_key_value_heads or config.num_attention_heads)
    gguf_writer.add_layer_norm_rms_eps(config.rms_norm_eps)

    # Tokenizer
    tokens = []
    scores = []
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        tokens.append(piece)
        scores.append(sp.get_score(i))

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_bos_token_id(sp.bos_id())
    gguf_writer.add_eos_token_id(sp.eos_id())
    gguf_writer.add_pad_token_id(sp.pad_id())

    # Métadonnées FUTO
    write_gguf_metadata(
        gguf_writer,
        language="fr",
        tokenizer_path=tokenizer_path,
    )

    # Convertir et écrire les tenseurs
    print("\nConversion des tenseurs...")
    for name, tensor in model.state_dict().items():
        # Convertir en numpy
        data = tensor.cpu().numpy()

        # Quantization si nécessaire
        if quantization == "f16":
            data = data.astype(np.float16)
        elif quantization == "f32":
            data = data.astype(np.float32)

        gguf_writer.add_tensor(name, data)

    # Écrire le fichier
    print(f"\nÉcriture du fichier GGUF: {output_path}")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"\n✓ Fichier GGUF créé: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convertir le modèle vers GGUF pour FUTO Keyboard'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Chemin vers le modèle HuggingFace'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Chemin vers le tokenizer .model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output/french_keyboard.gguf',
        help='Chemin de sortie'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['f32', 'f16', 'q8_0', 'q4_0'],
        default='f16',
        help='Type de quantization'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='fr',
        help='Code ISO de langue'
    )
    parser.add_argument(
        '--finetuning-count',
        type=int,
        default=0,
        help='Compteur de finetuning'
    )
    parser.add_argument(
        '--history',
        type=str,
        help='Historique du modèle'
    )
    parser.add_argument(
        '--use-python-gguf',
        action='store_true',
        help='Utiliser la librairie Python gguf'
    )

    args = parser.parse_args()

    if args.use_python_gguf:
        # Méthode avec librairie Python
        create_gguf_with_python_gguf(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            output_path=args.output,
            quantization=args.quantization,
        )
    else:
        # Méthode avec llama.cpp
        convert_to_gguf(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            output_path=args.output,
            quantization=args.quantization,
            language=args.language,
            finetuning_count=args.finetuning_count,
            history=args.history,
        )


if __name__ == '__main__':
    main()
