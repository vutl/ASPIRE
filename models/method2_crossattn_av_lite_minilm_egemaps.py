
"""ASPIRE-Lite with eGeMAPS + MiniLM-L3 text + tiny mel audio encoder."""
from models.method2_crossattn_av_lite_base import Method2CrossAttentionAVLiteBase


class Method2CrossAttentionAVLiteMiniLMEGeMAPS(Method2CrossAttentionAVLiteBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('roberta_model_name', 'sentence-transformers/paraphrase-MiniLM-L3-v2')
        kwargs.setdefault('wavlm_model_name', 'tiny-mel-audio')
        super().__init__(*args, **kwargs)
