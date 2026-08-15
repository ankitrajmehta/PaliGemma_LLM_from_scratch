from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
        def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_lyers=12,
            num_channels=3,
            image_size=224,
            patch_size=16, #i.e, take 16*16 sized patches to perform convolution on, and produce one embedding for each 
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
            ):
                super().__init__()

                self.hidden_size=hidden_size,
                self.intermediate_size=intermediate_size,
                self.num_hidden_lyers=hidden_size,
                self.num_channels=num_channels,
                self.image_size=image_size,
                self.patch_size=patch_size,
                self.layer_norm_eps=layer_norm_eps,
                self.attention_dropout=attention_dropout,
                self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.image_size = config.image_size
		self.patch_size = config.patch_size

		self.patch_embedding = nn.Conv2d(
			in_channels=config.num_channels,
			out_channels=self.embed_dim,
			kernel_size=self.patch_size,
			stride=self.patch_size,
			padding="valid", #Indicates no padding is added
		)

		self.num_patches = (self.image_size//self.patch_size)**2 #here: (224/16)**2 = 14**2 = 196 patches
		self.num_positions = self.num_patches
		self.positional_embeddings = nn.Embedding(self.num_positions, self.embed_dim) #learned positonal embeddings
		self.register_buffer(
			"position_ids",
			torch.arange(self.num_positions).expand((-1,1)), #.expand((1,-1)) resizes its shape from [self.num_positions] to [1,self.num_positions]
			persistent=False # no  need to store this in state_dict as it easily calculateable 
		)

	def forward(self, pixel_values: torch.FloatTensor)->torch.Tensor:
		pass

class SiglipVisionTransformer(nn.Module):
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		embed_dim = config.hidden_size

		self.embeddings = SiglipVisionEmbeddings(config)
		self.encoder = SiglipEncoder(config)
		self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		#putting it through the transformer
		#pixel_values: [batfch_size, Channels, height, Width] -> [Batch_size, Num_patches, Embed_dim]

		hidden_state = self.embeddings(pixel_values)

		last_hidden_state = self.post_layernorm(hidden_state)

		return last_hidden_state


class SiglipVisionModel(nn.Module):

	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.vision_model = SiglipVisionTransformer(config)

	def forward(self, pixel_values) -> Tuple:
		# [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_patches, Embed_dim]
		return self.vision_model(pixel_values=pixel_values)