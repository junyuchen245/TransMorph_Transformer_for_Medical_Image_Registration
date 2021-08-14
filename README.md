# TransMorph: Transformer for Medical Image Registration


## Flops counting for JHU MRI:
Model           | Input Resolution | Params(M) | Flops/MACs(G)| DSC          |
---             |---               |---        |---           |---           |
VoxelMoprh-1    |1x2x160x192x224   | 0.27      |305.07        |0.706 +- 0.137|
VoxelMoprh-2    |1x2x160x192x224   | 0.30      |400.06        |0.711 +- 0.135|
VoxelMoprhx2    |1x2x160x192x224   | 1.2       |1554.61       |0.726 +- 0.130|
VoxelMorph-diff |1x2x160x192x224   | 0.30      |91.71         |0.701 +- 0.139|
VoxelMoprh-huge |1x2x160x192x224   | 63.25     |3656.19       |0.732 +- 0.127|
CycleMorph      |1x2x160x192x224   | 0.72      |966.9         |0.694 +- 0.138|
BSplineNet      |1x2x160x192x224   | 0.27      |47.38         |0.700 +- 0.135|
ICNet           |1x2x160x192x224   | 1.11      |215.12        |0.648 +- 0.149|
PVTNet-V0       |1x2x160x192x224   | 7.24      |195.98        |0.712 +- 0.134|
PVTNet-V1       |1x2x160x192x224   | 61.84     |246.67        |0.718 +- 0.133|
CoTr            |1x2x160x192x224   | 38.74     |2275.16       |
ViT-V-Net       |1x2x160x192x224   | 110.62    |406.44        |0.725 +- 0.130|
SwinNet-v0      |1x2x160x192x224   | 63.63     |686.75        |0.733 +- 0.128|
SwinNet-v0-diff |1x2x160x192x224   | 63.42     |280.98        |0.718 +- 0.131|
SwinNet-v0-SYM  |1x2x160x192x224   | 63.63     |686.75        |
SwinNet-v0-BSpl |1x2x160x192x224   | 63.67     |454.41        |0.730 +- 0.127|
