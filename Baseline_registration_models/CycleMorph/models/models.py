
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycleMorph':
        from .cycleMorph_model import cycleMorph
        model = cycleMorph()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
