import numpy as np
import torch
def image_to_patches(images, res_patch, H, W, C):
    
    N = images.shape[0]
    patch_width = res_patch
    n_rows = H // patch_width
    n_cols = W // patch_width
    print(images.shape)

    cropped_img = images[:,:n_rows * patch_width, :n_cols * patch_width, :]

    #
    # Into patches
    # [n_rows, n_cols, patch_width, patch_width, C]
    #
    patches = torch.empty(N, n_rows, n_cols, patch_width, patch_width, C).to(int)
    for chan in range(C):
        patches[..., chan] = (
            cropped_img[..., chan]
            .reshape(N, n_rows, patch_width, n_cols, patch_width)
            .permute(0, 1, 3, 2, 4)
        )
       
    return patches.view(N, -1, patch_width, patch_width, C)
images = torch.rand((1,6,6,3))
print(images)
a=image_to_patches(images,2,6,6,3)
print(a.shape)