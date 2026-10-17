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
        rope_theta=10000.0, # Based freq of RoPE
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


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim)) # learnable parameter per feature (like a gamma parameter)

    def _norm(self,x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x: Optional[torch.Tensor] = None):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float()) # multiply with learnable parameter
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # equal to head_dim
        self.max_position_embeddings = max_position_embeddings # the context size, will be set as 8192 by GemmaConfig
        self.base = base # =10000, as described in RoPE paper

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        # Calculate the theta as theta_i = base^(-2i/dim) where i = 0 to dim // 2
        
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        self.inv_freq.to(x.device)
            # Unlike query_states, key_states, value_states, cos, sin, etc., inv_freq isn't derived from an input tensor that's already on the right device, 
            # so it has no "source" to inherit device from. Hence we explicitly move it to device.

        # Copy the inv_freq tensor for each element of batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # this temporarily disables automatic mixed-precision casting inside the with block, so operations use their normal dtype rather than autocast-selected dtypes.
            # used to prevent precision loss in numerically sensitive calculations, such as RoPE frequency calculations, where lower-precision types like FP16 may cause rounding errors or overflow.

            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Head_Dim // 2, Seq_Len] .transpose(1,2) --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1) 
            # as desribed in the paper, position * theta_i is repeated twice accross the embedding for all the i. 
            # But in paper it is desribed in a different way, going in sequence: position * theta_1, position * theta_1, position * theta_2, position * theta_2, and so on
            # here we are doing in sequence: position * theta_1, position * theta_2, position * theta_3, ......, position * theta_1, position * theta_2, position * theta_3, .....
            # Our implementation is how huggingface does it


            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    pass

class KVCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, we create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # else we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        #  then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout # will not be used
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0      

        # making attention layers as per grouped query attention required sizes
        """
        Grouped query attention uses fewer key and value embedding divisions (the embedding size for them is smaller and hence divided into fewer pieces, but the size of each inidivual piece is still head_dim (same as query's division).) 
        Each of the multiple attention heads uses a different part from the division of the query embedding (q_proj) (Each Head_dim sized array goes to one attention head)
        but groups of attention heads share the same division from the key and value embeddings (k_proj & v_proj) (Each Head_dim sized array goes to one group of attention heads)
        This makes the calculations much faster as in the GPU, fewer values need to be copied/exchanged between HBM and local memory and cuda cores.
        """     

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # [Batch_Size, Seq_Len, Hidden_Size]] -> [Batch_Size, Seq_Len, num_heads * head_dim]
        # where, num_heads * head_dim = Hidden_Size. so it is [Batch_Size, Seq_Len, Hidden_Size]] -> [Batch_Size, Seq_Len, Hidden_Size]] 
        # It bascially divides the representation space amongst all the heads equally without reducing the total representation space

        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # [Batch_Size, Seq_Len, Hidden_Size]] -> [Batch_Size, Seq_Len, num_key_value_heads * head_dim]
        # if head_dim = 128 and hidden_size = 1024, 
        #   then [Batch_Size, Seq_Len, 1024]] -> [Batch_Size, Seq_Len, num_key_value_heads * 128]
        # This basically reduces representation size of each token for the key and value

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size() # [Batch_Size, Seq_Len, Hidden_Size]

        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)

        #note: during the  prefil stage, query_states, key_states, and value_states are all full vectors containing all the sequence len number of elements' embeddings
        # but during generation phase, they contain only vectors containing embedding of the last token. 
        # for query_states this is enough, for key and value states, the past embeddings are retained from the kv cache


        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim] -> [Batch_Size, Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        
        # Repeat the key and values to match the number of heads of the query
        # done so num_key_value_heads no. of key (and value) divisions can be mapped to Num_Heads_Q number of query divisions (which is also the number of attention heads)
        # if we used flash attention, we would actually use the lower number of key and values in claculations too to make computation faster
        # but since we dont have a custom cuda kernel for that, we are reversing the grouped query attention (as if it was never done) by repeating the keys and values
        # hence the only benifit we got from grouped query attention is a smaller KV cache, no computational benifits
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout (none in paligemma)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # Multiply by the values. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous() # contiguous -> see in modeling_siglip.py 

        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)

        # [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states
        

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
