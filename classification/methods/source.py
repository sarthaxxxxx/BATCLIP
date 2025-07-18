from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class Source(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        logits, _, _, img_pre_features, text_pre_features = self.model(imgs_test, return_features=True)
        return logits.detach()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
