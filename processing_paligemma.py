from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [.5,.5,.5]
IMAGENET_STANDARD_STD = [.5,.5,.5]


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
