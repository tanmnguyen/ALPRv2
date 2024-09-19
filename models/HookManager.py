import re


# Define a hook manager class
class FeaturesExtractorHookManager:
    def __init__(self, model, exclude_layers=["None", "model.0", "model.23"]):
        self.model = model
        self.features_output = dict()
        self.layer_names = []
        self.exclude_layers = exclude_layers
        self._register_hook()

    def _register_hook(self):
        # Get the layer from the model
        layers = dict([*self.model.named_modules()])
        layer_names = [name for name in layers]

        # Get the output for each layer stage
        layer_output = dict()
        for name in layer_names:
            stage_name = next(
                (match.group() for match in re.finditer(r"model\.\d+", name)), None
            )
            layer_output[stage_name] = name

        # Exclude first and last layers
        for exc_layer in self.exclude_layers:
            if exc_layer in layer_output:
                del layer_output[exc_layer]

        # Register the hook
        for name in layer_output:
            layer = layers[layer_output[name]]
            self.layer_names.append(name)
            layer.register_forward_hook(self.create_hook_fn(name))

    def create_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            self.features_output[layer_name] = output

        return hook_fn

    def get_layer_output(self):
        try:
            return [*self.features_output.values()]
        finally:
            self.features_output = dict()
