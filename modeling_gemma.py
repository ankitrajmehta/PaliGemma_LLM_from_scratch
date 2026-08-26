import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaConfig:
    pass

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
        super().__init()
        self.config = config
        self.vision_tower = SiglipVisionModel
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

        # Merge the embeddings of the text and the image tokens (replaces the `<image>` tokens' embeddings in input_embeds with the actual embedding representing the image patches)
        input_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)


        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs

