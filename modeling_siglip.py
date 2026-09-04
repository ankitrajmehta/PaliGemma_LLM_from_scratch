from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
		def __init__(
			self,
			hidden_size=768,
			intermediate_size=3072, # size of intermediate layer in MLP
			num_hidden_layers=12,
			num_attention_heads = 12,
			num_channels=3,
			image_size=224,
			patch_size=16, #i.e, take 16*16 sized patches to perform convolution on, and produce one embedding for each 
			layer_norm_eps=1e-6,
			attention_dropout=0.0,
			num_image_tokens: int = None,
			**kwargs
			):
				super().__init__()

				self.hidden_size=hidden_size
				self.intermediate_size=intermediate_size
				self.num_hidden_layers=num_hidden_layers
				self.num_attention_heads = num_attention_heads
				self.num_channels=num_channels
				self.image_size=image_size
				self.patch_size=patch_size
				self.layer_norm_eps=layer_norm_eps
				self.attention_dropout=attention_dropout
				self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):

	def __init__(self, config: SiglipVisionConfig):
		"""
			This class is responsible for converting image to embeddings which can be passed to the vision transformer. 
			Each single image will be converted to a list of tokens, each single token being a list of embeddings of 'config.hidden_size' size.
			No. of tokens per image = num_patches = image_size//patch_size
		"""
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.image_size = config.image_size
		self.patch_size = config.patch_size

		self.patch_embedding = nn.Conv2d(
			in_channels=config.num_channels,
			out_channels=self.embed_dim,
			kernel_size=self.patch_size,
			stride=self.patch_size, # no overlap of the kernels as stried = patch_size
			padding="valid", #Indicates no padding is added
		)

		self.num_patches = (self.image_size//self.patch_size)**2 #here: (224/16)**2 = 14**2 = 196 patches
		self.num_positions = self.num_patches
		self.positional_embeddings = nn.Embedding(self.num_positions, self.embed_dim) #learned positonal embeddings
		self.register_buffer(
			"position_ids",
			torch.arange(self.num_positions).expand((1,-1)), #.expand((1,-1)) resizes its shape from [self.num_positions] to [1,self.num_positions]
			persistent=False # no  need to store this in state_dict as it easily calculateable 
		)

	def forward(self, pixel_values: torch.FloatTensor)->torch.Tensor:
		b, c, height, width = pixel_values.shape #[Batch_Size, Channels, Height, Width]

		patch_embeds: torch.Tensor = self.patch_embedding(pixel_values) # Convolve 'patch_size' patches(kernels) over the entire image with no overlap of the kernels
			#converts shape [Batch_Size, Channels, Height, Width] -> [Batch_Size, embed_dim, Num_patches_H, Num_patches_W]
			# where Num_patches_H = no. of rows of patches (i.e height wise) = self.image_size//self.patch_size
			# similarly Num_patches_W = no. of coumns of patches = self.image_size//self.patch_size

		embeddings = patch_embeds.flatten(2) # It preserves dimensions 0 and 1, and flattens all dimensions from index 2 onward.
			#converts [Batch_Size, embed_dim, Num_patches_H, Num_patches_W] -> [Batch_Size, embed_dim, Num_patches]
			# where Num_patches = Num_patches_H * Num_patches_W

		embeddings = embeddings.transpose(1,2) # transpose dimension 1 and 2
			#converts [Batch_Size, embed_dim, Num_patches] ->  [Batch_Size, Num_patches, embed_dim]

		embeddings = embeddings + self.positional_embeddings(self.position_ids)
			#[Batch_Size, Num_patches, embed_dim] + [Num_patches, embed_dim] = [Batch_Size, Num_patches, embed_dim] (broadcasts across all batch images)

		return embeddings
				

class SiglipAttention(nn.Module):
	"Multi-headed attention"

	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim//self.num_heads
		self.scale = self.head_dim**-0.5 # 1/sq.root(d)
		self.dropout = config.attention_dropout

		self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
		self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
		  
		self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


	def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"hidden_states: [Batch_size, Num_patches, Embed_dim]"
		batch_size, seq_len, _ = hidden_states.size()

		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)
			# all three states shape: [Batch_size, Num_patches, Embed_dim]

		query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
			# query_states: [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Num_heads, Head_Dim]  (done to breakdown the q array (product of hidden_states * W^q) into Num_heads number of arrays, each of size Head_dim. Each Head_dim sized array goes to one attention head)
		query_states = query_states.transpose(1,2)
			# query_states: [Batch_size, Num_patches, Num_heads, Head_Dim]  -> [Batch_size, Num_heads, Num_patches, Head_Dim]  (done so its easier to pass the Head_dim sized arrays from each of all the patches to the respective attention head)
		key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
		value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
			# same as query_states operations done above
			# [Batch_size, Num_heads, Num_patches, Head_Dim]

		attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))* self.scale)
			# calculate the attention using the formula Q * K^T / sqrt(d_k). 
			# (Q:[Batch_size, Num_heads, Num_patches, Head_Dim] * K^T:[Batch_size, Num_heads, Head_Dim, Num_patches]) / sqrt(d_k) ->
			# attn_weights: [Batch_size, Num_Heads, Num_patches, Num_patches]

		if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
			raise ValueError(
				f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is of size {attn_weights.size()}"
			)

		attn_weights = nn.functional.softmax(attn_weights, dim =-1, dtype=torch.float32).to(query_states.dtype)
			# Apply the softmax, row-wise. attn_weights: [Batch_size, Num_Heads, Num_patches, Num_patches]
		
		attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
			# Apply dropout only during training. For paligemma, we keep dropout = 0

		attn_output = torch.matmul(attn_weights, value_states)
			# Multiply the attention weights by the value states 
			# does {Q * K^T / sqrt(d_k)} * V
			# attn_output shape: [Batch_size, Num_Heads, Num_patches, Head_Dim]

		if attn_output.size() != (batch_size, self.num_heads, seq_len, seq_len):
			raise ValueError(
				f"Attention output should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is of size {attn_output.size()}"
			)

		attn_output = attn_output.transpose(1,2).contiguous()
			# [Batch_size, Num_heads, Num_patches, Head_Dim] -> [Batch_size, Num_patches, Num_heads, Head_Dim] 
			# done to represent the data in a per patch (per token) manner instead of per head manner. 
			# contiguous() makes all the data contiguous in the memory. This makes future reshape functions faster to run as only indexing has to change instead of actually moving the data around

		attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
			# [Batch_size, Num_patches, Num_heads, Head_Dim] -> [Batch_size, Num_patches, Embed_dim]
			# concatinates the outputs from all the individual attention heads into a single embedding, since there are Num_heads heads each with Head_dim number of output. Num_heads * Head_dim = Embed_dim

		attn_output = self.out_proj(attn_output)
			# [Batch_size, Num_patches, Embed_dim]
			# basically does attn_output * W^o. 
			# W^o is a Embed_dim by Embed_dim matrix. (out_proj is a linear layer with Embed_dim inputs and Embed_dim neurons, i.e, Embed_dim outputs)
			# This mixes the outputs of all the individual attention heads into a single represention of full size for each of the tokens (patches).
		
		return attn_output, attn_weights
		
		

class SiglipMLP(nn.Module):
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
		self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

		hidden_states = self.fc1(hidden_states)
			#[Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Intermediate_size]
		hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
			# [Batch_size, Num_patches, Intermediate_size]
		hidden_states = self.fc2(hidden_states)
			# [Batch_size, Num_patches, Intermediate_size] ->] [Batch_size, Num_patches, Embed_dim]
		return hidden_states


class SiglipEncoderLayer(nn.Module):
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.embed_dim = config.hidden_size
		self.self_attn = SiglipAttention(config)
		self.layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
		self.mlp = SiglipMLP(config)
		self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"forward pass through the encoder layer"

		residual = hidden_states
			# residual: [Batch_size, Num_patches, Embed_dim]

		hidden_states = self.layer_norm(hidden_states)
			#[Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]

		hidden_states, _ = self.self_attn(hidden_states)
			#[Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
		
		hidden_states = residual + hidden_states
		residual = hidden_states

		hidden_states = self.layer_norm2(hidden_states)
			#[Batch_size, Num_patches, Embed_dim]
		
		hidden_states = self.mlp(hidden_states)
			# [Batch_size, Num_patches, Embed_dim] => [Batch_size, Num_patches, Embed_dim]

		hidden_states = residual + hidden_states

		return hidden_states
		

class SiglipEncoder(nn.Module):
	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList(
			[SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
		)

	def forward(self, inputs_embeds: torch.Tensor) -> torch.tensor:
		hidden_states = inputs_embeds

		for encoder_layer in self.layers:
			hidden_states = encoder_layer(hidden_states)
				# [Batch_size, Num_patches, Embed_dim]
		
		return hidden_states

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
		#pixel_values: [batch_size, Channels, height, Width] -> [Batch_size, Num_patches, Embed_dim]

		hidden_state = self.embeddings(pixel_values)

		last_hidden_state = self.encoder(hidden_state)

		last_hidden_state = self.post_layernorm(last_hidden_state)

		return last_hidden_state


class SiglipVisionModel(nn.Module):

	def __init__(self, config: SiglipVisionConfig):
		super().__init__()
		self.config = config
		self.vision_model = SiglipVisionTransformer(config)

	def forward(self, pixel_values) -> Tuple:
		# [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_patches, Embed_dim]
		return self.vision_model(pixel_values=pixel_values)

if __name__ == "__main__":
	config = SiglipVisionConfig()
	model = SiglipVisionModel(config=config)
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Vision encoder size: ",  trainable_params)

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(f"{name}: {param.numel():,}")
