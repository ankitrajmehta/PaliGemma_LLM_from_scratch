from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [.5,.5,.5]
IMAGENET_STANDARD_STD = [.5,.5,.5] #mean and std of an image's pixel values based on imageNET dataest

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
        Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
        The input text is tokenized normally.
        A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
        This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
        The tokenized text is also prefixed with a fixed number of <image> tokens. How many? 
        It depends on the input image resolution and the patch size used by the SigLIP model. 
        PaliGemma models are pre-trained on one of three square sizes (224x224, 448x448, or 896x896), 
        and always use a patch size of 14. Therefore, the number of <image> tokens to prepend is 256 for the 224 models (224/14 * 224/14),
        1024 for the 448 models, and 4096 for the 896 models.
        NOTE: \n acts as the seperator token ([SEP]), which divides input prompt from output tokens during autoregressive generation.
            from the paligemma paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
            ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

    

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]]
) -> np.ndarray:
    """Normalize the pixel values to be in a Gaussian of mean 0 and variance of 1"""
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean)/std
    return image

def resize(
    image: Image,
    size: Dict[int, int]= None,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """change recevied image to size supported by our vision encoder"""
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    "change the pixel values from 0-255 to 0-1"
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def process_image(
    images: List[Image.Image],
    size: Dict[str, int]= None, # TODO : BOTH SHOULD BE int 
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None, # used to ensure images' pixel values are normalized in each of the 3 channels. 
    image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]:

    """made as per: https://huggingface.co/blog/paligemma#detailed-inference-process"""
    
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height,width),resample=resample) for image in images
    ] # resizes the images

    images = [np.array(image) for image in images]
        #convert each image to an numpy array

    images = [rescale(image, scale=rescale_factor) for image in images]
        # rescale the pixel values to be in the range of [0,1]

    images = [normalize(image, mean=image_mean, std = image_std) for image in images]
        # Normalize the images pixel values to have mean 0 std 1
    
    images = [image.transpose(2,0,1) for image in images]
        # (height, width, channels) -> (channels, height, width)
        # move the channel dimension to the first dimension. The model expects [channels, height, width]
    return images
        
class PaliGemmaProcessor:

    IMAGE_TOKEN ="<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int = 224):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Toeknizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS =[
            f"<loc{i:04d}>" for i in range(1024)
        ] # Tokens userd for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # tokens Used for object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False # we will be adding the BOS and EOS tokens manualy when needed
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images and {len(text)} prompts"

        pixel_values = process_image(
            images, 
            size = (self.image_size,self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        pixel_values = np.stack(pixel_values, axis=0)
            # convert the list of numpy arrays to a single numpy array with shape [batch_size, channel, height, width]

        pixel_values = torch.tensor(pixel_values)
            # convert from np array to pytorch tensor

        input_strings =[
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN
            ) 
            for prompt in text
        ]   #prepend a self.image_seq_length number of image tokens to the prompt 
            #(these image tokens will be replaced by the actual image patch embeddings later)
       
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding = padding,
            trunction=truncation,
        )   # Returns the input_ids and attention_mask as PyTorch Tensors
            # input_ids are id numbers from the vocab for text tokens
            #The attention mask is a tensor that tells the model:
                # 1 = this token is real and should be attended to
                # 0 = this token is padding and should be ignored
            # padding="longest" makes all prompts in a batch the same length by padding the shorter ones to match the longest prompt.

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
