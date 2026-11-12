
"""ASPIRE-Lite with eGeMAPS + ALBERT-small text + tiny mel audio encoder."""
from models.method2_crossattn_av_lite_base import Method2CrossAttentionAVLiteBase


class Method2CrossAttentionAVLiteAlbertEGeMAPS(Method2CrossAttentionAVLiteBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('roberta_model_name', 'sentence-transformers/paraphrase-albert-small-v2')
        kwargs.setdefault('wavlm_model_name', 'tiny-mel-audio')
        super().__init__(*args, **kwargs)
