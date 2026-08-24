from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [.5,.5,.5]
IMAGENET_STANDARD_STD = [.5,.5,.5] #mean and std of an image's pixel values based on imageNET dataest

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]]
) -> np.ndarray:
    "Normalize the pixel values to be in a Gaussian of mean 0 and variance of 1"
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean)/std
    return image

def resize(
    image: Image,
    size: Dict[int, int]= None,
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    "change recevied image to size supported by our vision encoder"
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

       