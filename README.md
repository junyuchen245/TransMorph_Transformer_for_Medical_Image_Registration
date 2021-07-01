# TransMorph: Transformer for Medical Image Registration


## Flops counting:
Model           | Input Resolution | Params(M) | Flops/MACs(G)|
---             |---               |---        |---           |
VoxelMoprh-1    |1x2x160x192x224   | 0.27      |305.07        |
VoxelMoprh-2    |1x2x160x192x224   | 0.30      |400.06        |
VoxelMoprhx2    |1x2x160x192x224   | 1.2       |1554.61       |
VoxelMorph-diff |1x2x160x192x224   | 0.30      |91.71         |
VoxelMoprh-huge |1x2x160x192x224   | 63.25     |3656.19       |
ICNet           |1x2x160x192x224   | 1.11      |215.12        |
PVTNet          |1x2x160x192x224   | 7.24      |195.98        |
CoTr            |1x2x160x192x224   | 38.74     |2275.16       |
ViT-V-Net       |1x2x160x192x224   | 110.62    |406.44        |
SwinNet-v0      |1x2x160x192x224   | 63.63     |686.75        |
