class StableDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and unet.(mainly for LDM such as stable diffusion)
    """

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    