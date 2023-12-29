import torch.nn as nn

def _conv_filter(state_dict, patch_size=16):
    """
    Convert patch embedding weight from manual patchify + linear proj to conv.
    
    Args:
        state_dict (dict): Model state dictionary.
        patch_size (int): Patch size used in the original model.

    Returns:
        dict: Updated state dictionary.
    """
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            # Reshape the linear projection to convolutional weights
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            # Convert to 2D convolutional weights
            conv_weight = nn.Conv2d(3, v.shape[0], kernel_size=patch_size, stride=patch_size, bias=False)
            conv_weight.weight.data = v
            # Update the state dictionary
            out_dict[k.replace('patch_embed.proj.weight', 'patch_embed.proj.weight_conv')] = conv_weight.weight.data
        else:
            out_dict[k] = v
    return out_dict
