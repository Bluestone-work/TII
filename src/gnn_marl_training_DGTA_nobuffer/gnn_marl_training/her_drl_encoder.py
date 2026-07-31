"""HeR-DRL heterogeneous relation encoder adapted to the local actor graph.

Zhou et al. (IEEE RA-L 2025, DOI: 10.1109/LRA.2025.3553050) construct
three node types (ego robot, other robots, and crowd entities) and apply a
two-layer GraphConv independently to every directed relation type.  PyG's
``to_hetero(..., aggr='sum')`` then sums incoming relation outputs.  This
module implements that released architecture directly with dense tensors so
the ROS/RLlib stack does not gain a torch-geometric binary dependency.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


RELATIONS = (
    ("ego", "obstacle"),
    ("obstacle", "ego"),
    ("neighbor", "ego"),
    ("ego", "neighbor"),
    ("neighbor", "obstacle"),
    ("obstacle", "neighbor"),
    ("obstacle", "obstacle"),
    ("neighbor", "neighbor"),
)


def _type_mlp(input_dim: int, output_dim: int = 32) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
        nn.ReLU(),
    )


class RelationGraphConv(nn.Module):
    """Dense equivalent of one PyG ``GraphConv`` relation module."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.message = nn.Linear(input_dim, output_dim, bias=True)
        self.root = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.bmm(adjacency, source)
        return self.message(aggregate) + self.root(target)


class HeterogeneousRelationLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.relation_convs = nn.ModuleDict(
            {
                f"{source}_to_{target}": RelationGraphConv(input_dim, output_dim)
                for source, target in RELATIONS
            }
        )

    @staticmethod
    def _adjacency(
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        same_type: bool,
    ) -> torch.Tensor:
        adjacency = target_mask.unsqueeze(2) * source_mask.unsqueeze(1)
        if same_type and adjacency.shape[1] == adjacency.shape[2]:
            identity = torch.eye(
                adjacency.shape[1],
                device=adjacency.device,
                dtype=adjacency.dtype,
            ).unsqueeze(0)
            adjacency = adjacency * (1.0 - identity)
        return adjacency

    def forward(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outputs = {
            node_type: torch.zeros(
                feature.shape[0],
                feature.shape[1],
                next(iter(self.relation_convs.values())).message.out_features,
                device=feature.device,
                dtype=feature.dtype,
            )
            for node_type, feature in features.items()
        }
        for source_type, target_type in RELATIONS:
            adjacency = self._adjacency(
                masks[source_type],
                masks[target_type],
                same_type=source_type == target_type,
            )
            key = f"{source_type}_to_{target_type}"
            relation_output = self.relation_convs[key](
                features[source_type],
                features[target_type],
                adjacency,
            )
            outputs[target_type] = outputs[target_type] + relation_output
        return {
            node_type: F.relu(output) * masks[node_type].unsqueeze(-1)
            for node_type, output in outputs.items()
        }


class HeRDRLEncoder(nn.Module):
    """Released HeR-DRL relation topology with its original 32-D width."""

    output_dim = 32

    def __init__(
        self,
        ego_input_dim: int,
        neighbor_input_dim: int,
        obstacle_input_dim: int,
    ):
        super().__init__()
        self.type_encoders = nn.ModuleDict(
            OrderedDict(
                (
                    ("ego", _type_mlp(ego_input_dim, self.output_dim)),
                    ("obstacle", _type_mlp(obstacle_input_dim, self.output_dim)),
                    ("neighbor", _type_mlp(neighbor_input_dim, self.output_dim)),
                )
            )
        )
        self.conv1 = HeterogeneousRelationLayer(self.output_dim, self.output_dim)
        self.conv2 = HeterogeneousRelationLayer(self.output_dim, self.output_dim)

    def forward(
        self,
        ego: torch.Tensor,
        neighbors: torch.Tensor,
        obstacles: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ego.ndim != 2 or neighbors.ndim != 3 or obstacles.ndim != 3:
            raise ValueError("HeR-DRL inputs must be [B,D], [B,N,D], [B,M,D]")
        batch_size = ego.shape[0]
        if neighbor_mask is None:
            neighbor_mask = (neighbors.abs().sum(dim=-1) > 1.0e-6).to(ego.dtype)
        if obstacle_mask is None:
            obstacle_mask = (obstacles.abs().sum(dim=-1) > 1.0e-6).to(ego.dtype)
        masks = {
            "ego": ego.new_ones((batch_size, 1)),
            "obstacle": obstacle_mask.to(dtype=ego.dtype),
            "neighbor": neighbor_mask.to(dtype=ego.dtype),
        }
        features = {
            "ego": self.type_encoders["ego"](ego).unsqueeze(1),
            "obstacle": self.type_encoders["obstacle"](obstacles),
            "neighbor": self.type_encoders["neighbor"](neighbors),
        }
        features = self.conv1(features, masks)
        features = self.conv2(features, masks)
        return features["ego"][:, 0, :]

    def parameter_manifest(self) -> dict[str, int]:
        return {
            "type_encoders": sum(p.numel() for p in self.type_encoders.parameters()),
            "relation_conv1": sum(p.numel() for p in self.conv1.parameters()),
            "relation_conv2": sum(p.numel() for p in self.conv2.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
