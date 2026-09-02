import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel



class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size, # size of embedding vectors 
        intermediate_size, # size of mlp layers
        num_hidden_layers, # how many attention layers
        num_attention_heads, # no. of query attention heads 
        num_key_value_heads, # no. of key-value attention heads (since we use grouped query attention)
        head_dim=256, # dimension of vector each attention head works with
        max_position_embeddings=8192, # ctx window size
        rms_norm_eps=1e-6,
        rope_theta=10000.0, 
        attention_bias=False, # if we want to have bias in attention matrixes 
        attention_dropout=0.0, # not used in paligemma
        pad_token_id=None, # No padding used
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig:

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100, # needed during training, not needed here as its only inference code
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id = None,
        **kwargs
    ):
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder=False #needed by huggingface

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2
        self.vision_config.projection_dim = projection_dim

class PaliGemmaMultiModalProjection(nn.Module):
    pass

class GemmaForCausalLm(nn.Module):
    pass

class KVCache:
    pass


class PaliGemmaForConditionalGeneration(nn.Module):
    """
        Called conditional generation as we are conditioning the text on the image provided
        Also called so because of how we set the attention mask w.r.t causality, so its like a condition 
    """

    def __init__(self, config:PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multimodal_projector = PaliGemmaMultiModalProjection(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLm(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def tie_weights(self):
        """
            Shares the weights of the embedding layer with the final linear layer (the one that converts the contextualized embeddings produced after the
            transformer layers back to token_id probabilities).
            It's possible to share their weights as they are essentially inverse actions of each other
            Weight tieing is not done by all LLMs, but it is present in Gemma
            Useful for reduction of parameters in the model
        """
        return self.language_model.tie_weights()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded" # We haven't implemented padding related codes in this version, so attention mask must be all 1s

        # 1. Extract the input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # (Batch_size, Seq_Len, Hidden_size)

        # 2. Merge images and text
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
            # [Batch_size, Channels, Height, Width]  -> [Batch_size, Num_patches, Embed_dim]
        
        image_features = self.multimodal_projector(selected_image_feature)
            # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Hidden_size]
                # Embed_dim is image embedding size, Hidden_size is text embedding size

        # Merge the embeddings of the text and the image tokens (replaces the placeholder `<image>` tokens' embeddings in input_embeds with the 
        # actual embedding representing the image patches)
        input_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)


        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs

