# 02_onlyedge_classificationupgrade

This branch contains two main updates:

1. **Image task panels**
   - `global_response` / `pdk_response` now visualize the **edge map of each processed image itself**.
   - No more raw-relative edge-gain for the top-row `Global` / `PDK` panels.

2. **Classification upgrade**
   - Tiny ImageNet uses **ShuffleNetV2 x0.5 ImageNet-pretrained weights**.
   - Tiny ImageNet input is resized/cropped to **224x224**.
   - **ImageNet normalization is applied after Raw / Global / PDK preprocessing**, so spatial filtering still operates on `[0,1]` images.
   - Epoch logs include **TrainCE / TrainAcc / TestAcc / Best / LR**.

## Running

```bash
python main.py --task synthetic
python main.py --task lowlight
python main.py --task natural
python main.py --task classification
python main.py --task all
```
