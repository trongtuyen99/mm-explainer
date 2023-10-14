# MM-Explainer
- Explainable multi-modality neural network using

# Implemented method:
- PID (Partial Information Decomposition in information theory)

# How to compute PID for new dataset using mult
1. Prepare dataset: see [Multibench](https://github.com/pliang279/MultiBench/tree/main) for more detail 
2. Training model and extract features: 
```python
!python3 examples/affect/train_affect_mult.py --data_path /content/mosi_data.pkl --save_path /content/ --train_modal 0 1 --num_epochs 20 --extract_features
```
3. Compute PID with use output from step 2
```python
from utils.compute_pid import compute_pid
dataset_name = 'mosi'
measure = compute_pid('/content/vision_audio', dataset_name=dataset_name)
print(measure)
```


*See Example: examples/Training - Extract feaure - Compute PID.ipynb*