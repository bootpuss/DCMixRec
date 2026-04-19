from models import du_recurrent_model


def create_model(opts):
    model = du_recurrent_model.RecurrentModel(opts)
    return model
