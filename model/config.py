#!/usr/bin/env python3
"""
Configuration du modèle Llama pour FUTO Keyboard français
"""

from transformers import LlamaConfig
from dataclasses import dataclass
from typing import Optional


@dataclass
class FutoKeyboardConfig:
    """Configuration pour le modèle de clavier FUTO"""

    # Architecture Llama (36M paramètres)
    vocab_size: int = 15008
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: Optional[int] = None  # Si None, = num_attention_heads
    max_position_embeddings: int = 2048  # Contexte max

    # Paramètres additionnels Llama
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    rope_theta: float = 10000.0

    # Paramètres d'entraînement
    dropout: float = 0.1
    attention_dropout: float = 0.0

    # Métadonnées FUTO
    language: str = "fr"
    features: list = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.features is None:
            self.features = [
                "base_v1",
                "inverted_space",
                "xbu_char_autocorrect_v1"
            ]

    def to_llama_config(self) -> LlamaConfig:
        """Convertit en LlamaConfig de HuggingFace"""
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout,
        )

    def print_model_size(self):
        """Affiche la taille estimée du modèle"""
        # Embedding: vocab_size * hidden_size * 2 (in + out)
        embedding_params = self.vocab_size * self.hidden_size * 2

        # Attention: 4 * hidden_size^2 par couche (Q, K, V, O)
        attention_params_per_layer = 4 * (self.hidden_size ** 2)

        # FFN: 2 * hidden_size * intermediate_size par couche
        ffn_params_per_layer = 2 * self.hidden_size * self.intermediate_size

        # LayerNorm: 2 * hidden_size par couche
        norm_params_per_layer = 2 * self.hidden_size

        # Total par couche
        params_per_layer = (
            attention_params_per_layer +
            ffn_params_per_layer +
            norm_params_per_layer
        )

        # Total
        total_params = embedding_params + (params_per_layer * self.num_hidden_layers)

        # Taille en MB (float16)
        size_mb = (total_params * 2) / (1024 ** 2)

        print("=" * 50)
        print("Configuration du modèle:")
        print("=" * 50)
        print(f"Vocabulaire: {self.vocab_size:,}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Intermediate size: {self.intermediate_size}")
        print(f"Couches: {self.num_hidden_layers}")
        print(f"Attention heads: {self.num_attention_heads}")
        print("-" * 50)
        print(f"Embedding params: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
        print(f"Params par couche: {params_per_layer:,}")
        print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Taille (float16): {size_mb:.1f} MB")
        print(f"Taille (float32): {size_mb*2:.1f} MB")
        print("=" * 50)


def get_default_config(vocab_size: int = 15008) -> FutoKeyboardConfig:
    """Retourne la configuration par défaut"""
    return FutoKeyboardConfig(vocab_size=vocab_size)


def get_large_config(vocab_size: int = 15008) -> FutoKeyboardConfig:
    """
    Configuration plus grande pour B200 (si vous voulez un modèle plus performant)
    ~100M paramètres
    """
    return FutoKeyboardConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
    )


def main():
    """Test de la configuration"""
    print("\n### Configuration par défaut (36M params) ###\n")
    config = get_default_config()
    config.print_model_size()

    print("\n### Configuration large (100M params) ###\n")
    config_large = get_large_config()
    config_large.print_model_size()


if __name__ == '__main__':
    main()
