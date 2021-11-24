import ml_collections
'''
********************************************************
                   ViT-V-Net
********************************************************
'''
def get_3DReg_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8, 8)})
    config.patches.grid = (8, 8, 8)
    config.hidden_size = 252
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 8

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 32, 16)
    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 3
    config.n_skip = 5
    return config