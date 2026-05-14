from dataclasses import dataclass, field

import haven


@dataclass
class ModelConfig:
    num_layers: int = 5
    embed_dim: int = 512


@dataclass
class TrainConfig:
    workers: int = 5
    steps: list[int] = field(default_factory=lambda: [50, 100, 150])
    model: ModelConfig = field(default_factory=ModelConfig)


cfg = haven.load(
    TrainConfig,
    """
steps: [1, 2, 3]
model:
  num_layers: 16
""",
)
assert cfg.steps == [1, 2, 3]
assert cfg.model.num_layers == 16
assert cfg.model.embed_dim == 512

updated = haven.update_from_dotlist(cfg, ["workers=3", "model.num_layers=2"])
assert updated.workers == 3
assert updated.model.num_layers == 2
assert cfg.workers == 5
assert cfg.model.num_layers == 16

roundtrip = haven.load(TrainConfig, haven.dump(updated))
assert roundtrip == updated
