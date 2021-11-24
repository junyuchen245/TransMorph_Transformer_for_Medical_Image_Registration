import ml_collections

def get_CycleMorph_config():
    config = ml_collections.ConfigDict()
    config.batchSize = 1
    config.inputSize = (160, 192, 224)
    config.fineSize = (160, 192, 224)
    config.input_nc = 2
    config.encoder_nc = (16, 32, 32, 32, 32)
    config.decoder_nc = (32, 32, 32, 8, 8, 3)
    config.beta1 = 0.5
    config.lr = 0.0002
    config.lambda_A = 0.1
    config.lambda_B = 0.5
    config.lambda_R = 0.02
    config.lr_policy = 'lambda'
    config.lr_decay_iters = 50
    config.isTrain = True
    config.gpu_ids = [0]
    config.which_model_net = 'registUnet'
    config.init_type = 'normal'
    config.continue_train = False
    config.epoch_count = 0
    config.niter = 100
    config.niter_decay = 100
    return config