# Nomad Outcrops
 
Team Nomad Outcrops, for the Frank Arnott – Next Generation Explorers Award. The work is on Geoscience BC QUEST, workshop Block A, for PDAC 2027 in Toronto.
 
Two models look at the airborne magnetic picture. One paints the published fault ribbon. One names a tile intrusive or other. The scores below are the held-out tests. The paint does not follow the published lines, so it is not a new fault map. The intrusive calls are not a bedrock map.
 
Questions: yerkebulan.tazabek@nu.edu.kz
 
## Weights in this repository
 
| File | Model | What it answers |
| --- | --- | --- |
| `semantic_fault_weights.pt` | DeepLabv3+ | Where is the published fault ribbon on this tile? |
| `intrusive_vit_weights.pt` | ViT-base, patch 16 | Is this tile mapped intrusive rock, or other? |
 
The magnetic tiles themselves are not in this repository. They are 256 by 256 pixels from the QUEST reduced-to-pole image, 100 m per pixel, so 25.6 km on a side. The next tile starts 128 pixels over, and neighbours share half their pixels. Because of that overlap, both splits follow northing. A random cut would show the model ground it had already seen.
 
The fault weights load in Ultralytics:
 
```python
from ultralytics import YOLO
model = YOLO("semantic_fault_weights.pt")
```
 
The class is `fault`. Pixel 0 is background. Pixel 1 is the ribbon.
 
## Fault paint
 
DeepLabv3+. The tile stays 256 pixels. The ribbon the model is asked to paint is the published fault file, 500 m either side of each line.
 
- Train: 391 tiles south of 6,000,000 N.
- Check: 121 tiles from 6,000,000 N to 6,050,000 N.
- Test: 401 tiles on the Block A side, including the Mount Milligan tile.
 
Fault IoU on that test is 4.7%. The same weights, run again and stitched, score 4.5% (261,518 / 5,755,645). A painter that colours about 7% of the pixels at random scores about the same. On most of the belt the ribbon is no sharper than the ground beside it. The published lines stay.
 
Chen, Zhu, Papandreou, Schroff, and Adam, 2018, ECCV, pp. 801–818.
 
## Intrusive or other
 
ViT-base, patch 16. The 256-pixel tile is resized to 224. The label comes from the regional bedrock map. A tile counts as intrusive when at least 20% of its pixels sit in an intrusive polygon.
 
- Train: 41 intrusive and 82 other, between 6,000,000 N and 6,164,400 N.
- Check: 11 intrusive and 16 other, in the same belt.
- Test: 17 intrusive and 83 other, north of 6,190,000 N.
- One full tile is left empty between the test and the tiles used to fit the model, so they share no pixels.
- Mount Milligan is in no folder.
- South of 6,000,000 N the intrusive tiles are not darker than the other tiles, so that ground is left out.
 
On the 100-tile test, at a 50% threshold, intrusive precision is 78, recall is 82, and F1 is 80. The model finds 14 of the 17 intrusive tiles and calls 4 ordinary tiles intrusive. A rule that simply picks the dark tiles, fit off this same test, reaches F1 73. North of 6,000,000 N the intrusive tiles are darker, about 125 against 152. The network is lower on precision and higher on recall.
 
The headline average of both classes is higher. The number to quote is the intrusive F1, 80.
 
A later run of the same kind of model, on 1,092 tiles, calls 454 of them intrusive. 53 of those meet Block A. That run is not the 100-tile score.
 
Dosovitskiy and others, 2021, ICLR. An image is worth 16×16 words.
 
## A separate fault test
 
YOLO26-small was scored on 90 tiles. Fault IoU on that test is 20.4%. Those are not the DeepLab weights, and that score is not the 401-tile result.
 
## Licence
 
MIT. See `LICENSE`.
