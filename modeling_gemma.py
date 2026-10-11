"""
    Implementation of the Gemma Text LLM and then connection with text to make the PaliGemma model.
"""

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

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class GemmaForCausalLm(nn.Module):

    """
        CausalLm in most huggingface libs usually means transfer model + language modeling head(the linear layer that projects each embeddings to logits ) 
    """

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weight(self):
        self.lm_head.weight = self.model.embed_token.weight


    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data

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
        self.multimodal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLm(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def tie_weights(self):
        """
            Weight tying means using the same set of parameters (weights) for two different parts of the model
            This function shares the weights of the embedding layer with the final linear layer (the one that converts the contextualized embeddings produced after the
            transformer layers back to token_id probabilities).
            It's possible to share their weights as they are essentially inverse actions of each other
            Weight tieing is not done by all LLMs, but it is present in Gemma
            Useful for reduction of parameters in the model
        """
        return self.language_model.tie_weights()

    def merge_input_ids_with_image_features(
            self, 
            image_features: torch.Tensor, 
            inputs_embeds: torch.Tensor, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor, 
            kv_cache: Optional[KVCache] = None
        ):

        """
        Merge the embeddings of the text and the image tokens 
        (replaces the placeholder `<image>` tokens' embeddings in input_embeds with the actual embedding representing the image patches)
        Also, creates the causal mask and position_ids (to be used for RoPE later)
        """


        _,_,embed_dim = image_features.shape #image_feature and input_embeds have same sized vector embeddings
        batch_size, sequence_length = input_ids.shape # input_ids has image tokens (placeholder), text, bos token, etc
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size ** .5) # div by d^1/2 to maintain scale even if the input image was different dimensions
            # [Batch_size, Num_patches, Hidden_Size]
            # Num_patches -> no. of image tokens per image


        # combine the embeddings of the image tokens and text tokens with use of masks:  

        final_embedding: torch.Tensor = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device) # will fill values  of actual embeddings into this

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
            # shape: [Batch_size, Seq_Len]
            # Seq_Len or sequence_length-> context window length, including image tokens, text tokens, and any other tokens
        image_mask = input_ids == self.config.image_token_index
            # shape: [Batch_size, Seq_Len]
        pad_mask = input_ids == self.pad_token_id
            # shape: [Batch_size, Seq_Len]

        # we need to expand the masks to the embedding dimension as well to use in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
            # text_mask.unsqueeze(-1) adds a dimension of size 1 at the very end of the tensor's shape.
                #so, [Batch_size, Seq_Len] -> [Batch_size, Seq_Len, 1]
            # then .expand(-1,-1,embed_dim) extends a tensor's shape by broadcasting, where -1 means to keep the existing size of that dimension
                # so, [Batch_size, Seq_Len, 1] -> [Batch_size, Seq_Len, embed_dim]

        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
            #  [Batch_size, Seq_Len, embed_dim]

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
            # where ever the text_mask_expanded is 1, put value from corresponding position of inputs_embeds, else put value from final_embedding

        # Add image embeddings. Can't use torch.where as scaled_image_features.shape != final_embedding.shape because Num_patches != Seq_len
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Zero out padding tokens. pad_mask_expanded is all 0s anyway (i.e all false) as we dont implemnt padding, so all values of final_embedding are gonna be the same
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Creating the causal attention mask:

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items()==0:
            #i.e prefill stage
            """
                In causal attention, each token can only attend to itself and the tokens that came before it in the sequence. 
                Normally, for images we want to allow each image token to attend to all other image tokens, while text tokens can only attend to themselves and the tokens that came before them (even of prompt (i.e prefill stage)).
                But in Paligemma, during prefill stage, Paligemma allows all tokens to attend to all other tokens, including text tokens.
                For most other LLMs, the attention mask is a lower triangular matrix, but in Paligemma, the attention mask is a full matrix(i.e all tokens can attend to all other tokens)
                (This is for inference) 
            """
            # This func only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
            # for most other models, we would keep -inf to upper triangle, making a lower triangular martix of 0s. This would be causal mask. As when -inf is added to the Q*k^T, it masks out the upper triangle. None of this is done in paligemma, it just uses causal_mask of all 0s only.
            # note: most LLMs use causal mask with 1 = unmasked (used), 0 = masked (unused/ un-attended). This mask is multipled over all elements. 
            # But, in Paligemma, we add the mask (see further up in code), and so we use 0 = unmasked (used) and -inf = masked (unused/un-attended)
        else:
          
            assert q_len ==1   # for generating tokens, only latest token is used as query. rest of data we get from the kv cache
            kv_len = kv_cache.num_items() + q_len #kv size is being increased by one more

            """while generating tokens, we dont mask anything as the latest token should attend to all previous tokens"""

            # This func only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the dimension for head:
        causal_mask = causal_mask.unsqueeze(1)
            # [Batch_sieze, q_len, kv_len] -> [Batch_size, 1, q_len, kv_len]
            # This is 1 dim is for Num_heads_Q dimension (by broadcasting later)


        # Getting the position_ids now, which will be use in RoPE (Rotary Position Embedding):

        if kv_cache is not None and kv_cache.num_items() > 0: # the generation phase
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1] 
                # the attention_mask comes from the tokenizer is just all ones as this implementation doesn't use padding.
                # attention_mask.cumsum(-1) gives cumulative sum going from 1 to len(attention_mask[1]).
                # we do [:, -1] as we want the position_id of the final token only from each of the batch. For generation, we are going to encode only this
            
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0) # to ensure it has a batch_dim at start always, even if only 1 batch exists

        else: # the prefill stage
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)    
                # Create a position_ids based on the size of the attention_mask
                # For masked tokens, use the number 1 as position. (gives position id of 1 to all the padding positions (where attention_mask ==0))
            
        return final_embedding, causal_mask, position_ids


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

        # 2. Get image embeddings
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
            # [Batch_size, Channels, Height, Width]  -> [Batch_size, Num_patches, Embed_dim]
        
        image_features = self.multimodal_projector(selected_image_feature)
            # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Hidden_size]
                # Embed_dim is image embedding size, Hidden_size is text embedding size

        # 3. Merge the embeddings of the text and the image tokens (replaces the placeholder `<image>` tokens' embeddings in input_embeds with the 
            # actual embedding representing the image patches)
        input_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        # 4. Put through final LLM model
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs

if __name__ == "__main__":
    config = PaliGemmaConfig()
    pg = PaliGemmaForConditionalGeneration(config=config)
