# implementation of a VLM (Vision Language Model) - PaliGemma From scratch

My implementation of the PaliGemma model's architecture from Google's Gemma family, from scratch in pytorch, building both the SigLIP vision encoder and the Gemma-2B language model. PaliGemma combines these two components via a linear projection layer to produce a vision-language model, adding multimodal capabilities to the originally text-only Gemma architecture of early days.

More about PaliGemma: https://huggingface.co/blog/paligemma and https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md



[View the PaliGemma's ViT architecture diagram](diagrams/siglip_arch.pdf)


[View base Gemma's architecture](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)


---
# To Run it:
(Commands below are for Linux. Use appropirate commands for your OS)
1.  First download the model weights from huggingface
    ```
    hf auth login
    ```
    ```
    hf download google/paligemma-3b-pt-224 \
    --local-dir ~/palligemma/model_repo_from_hg/paligemma-3b-pt-224
    ```

2. Edit inference.sh with required prompt, image path and parameters

3. Run using:
    ```
    ./inference.sh
    ```