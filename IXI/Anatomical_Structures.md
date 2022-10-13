Here is a small typo in the paper, where the number of anatomical structures for evaluation should be **30** (instead of 29). 
The evaluation of the Dice scores can be found in [analysis.py](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/analysis.py) and [analysis_trans.py](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/analysis_trans.py), which should provide results that are compatible with those in our publication.\
The anatomical structures are:
|Number | Structures  | Substructures        |
| ----------- | ----------- | -----------          |
|1| Brain-Stem  | Brain-Stem           |
|2| Thalamus    | Right-Thalamus-Proper|
|3|             | Left-Thalamus-Proper |
|4|Cerebellum-Cortex |Left-Cerebellum-Cortex |
|5| |Right-Cerebellum-Cortex |
|6| Cerebral-White-Matter|Left-Cerebral-White-Matter |
|7| |Right-Cerebral-White-Matter |
|8|Cerebellum-White-Matter |Left-Cerebellum-White-Matter |
|9| |Right-Cerebellum-White-Matter |
|10|Putamen |Left-Putamen |
|11| |Right-Putamen |
|12| VentralDC|Left-VentralDC |
|13| |Right-VentralDC |
|14|Pallidum |Left-Pallidum |
|15| |Right-Pallidum |
|16| Caudate|Left-Caudate |
|17| |Right-Caudate |
|18| Lateral-Ventricle|Left-Lateral-Ventricle |
|19| |Right-Lateral-Ventricle |
|20|Hippocampus |Left-Hippocampus |
|21| |Right-Hippocampus |
|22|3rd-Ventricle |3rd-Ventricle |
|23|4th-Ventricle |4th-Ventricle |
|24|Amygdala |Left-Amygdala |
|25| |Right-Amygdala |
|26|Cerebral-Cortex |Left-Cerebral-Cortex |
|27| |Right-Cerebral-Cortex |
|28| CSF|CSF |
|29| choroid-plexus|Left-choroid-plexus |
|30| |Right-choroid-plexus |
