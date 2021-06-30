# TransMorph: Transformer for Medical Image Registration


## Flops counting:
Model           | Input Resolution | Params(M) | Flops(G)|
---             |---               |---        |---      |
VoxelMoprh-1    |1x2x160x192x224   | 274.39    |305.07   |
VoxelMoprh-2    |1x2x160x192x224   | 301.41    |400.06   |
VoxelMoprhx2    |1x2x160x192x224   | 1.2       |1554.61  |
VoxelMorph-diff |1x2x160x192x224   | 0.307     |91.71    |
VoxelMoprh-huge |1x2x160x192x224   | 63.25     |3656.19  |
SwinNet-v0      |1x2x160x192x224   | 63.63     |686.75   |
