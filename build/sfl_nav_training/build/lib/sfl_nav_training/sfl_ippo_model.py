from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class SFLIPPOGRUModel(TorchModelV2, nn.Module):
    """Sampling-for-learnability style shared Actor-Critic GRU model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        self.num_outputs = int(num_outputs)

        self.fc_dim = int(cfg.get("fc_dim", 512))
        self.hidden_size = int(cfg.get("hidden_size", 512))
        self.use_layer_norm = bool(cfg.get("use_layer_norm", False))

        self.obs_proj = nn.Linear(self.obs_dim, self.fc_dim)
        self.obs_norm = nn.LayerNorm(self.fc_dim) if self.use_layer_norm else nn.Identity()

        self.gru_cell = nn.GRUCell(self.fc_dim, self.hidden_size)

        self.actor_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.actor_head = nn.Linear(self.hidden_size, self.action_dim)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.critic_fc = nn.Linear(self.hidden_size, self.fc_dim)
        self.critic_head = nn.Linear(self.fc_dim, 1)

        self._cur_value = None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.obs_proj.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.obs_proj.bias, 0.0)

        nn.init.orthogonal_(self.actor_fc.weight, gain=2.0)
        nn.init.constant_(self.actor_fc.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

        nn.init.orthogonal_(self.critic_fc.weight, gain=2.0)
        nn.init.constant_(self.critic_fc.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        w = next(self.parameters())
        return [w.new_zeros(self.hidden_size)]

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs_flat"].float()

        batch = int(seq_lens.shape[0])
        time_steps = int(obs.shape[0] // max(batch, 1))
        obs_3d = obs.view(batch, time_steps, self.obs_dim)

        h_t = state[0]
        actor_out = []
        values = []

        for t in range(time_steps):
            emb = torch.relu(self.obs_norm(self.obs_proj(obs_3d[:, t, :])))
            h_t = self.gru_cell(emb, h_t)

            actor_hidden = torch.relu(self.actor_fc(h_t))
            mean = self.actor_head(actor_hidden)

            critic_hidden = torch.relu(self.critic_fc(h_t))
            v = self.critic_head(critic_hidden).squeeze(-1)

            actor_out.append(mean)
            values.append(v)

        mean_out = torch.stack(actor_out, dim=1).reshape(-1, self.action_dim)
        self._cur_value = torch.stack(values, dim=1).reshape(-1)

        if self.num_outputs == self.action_dim * 2:
            log_std = self.log_std.unsqueeze(0).expand(mean_out.shape[0], -1)
            logits = torch.cat([mean_out, log_std], dim=-1)
        else:
            logits = mean_out

        return logits, [h_t]

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "forward() must run before value_function()"
        return self._cur_value


MODEL_NAME_SFL_IPPO = "sfl_ippo_gru"
