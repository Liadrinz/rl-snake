from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

class CustomModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())
    
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)
    
    def value_function(self):
        return self.model.value_function()
