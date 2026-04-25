# Bash Log

## Commands Run During Debugging

### 1. Run patch script
`../01_data_augmentation/venv/bin/python /tmp/patch_h5_models.py`
```
[OK]   mobilenetv2_leaf_classifier.h5 → mobilenetv2_leaf_classifier_patched.h5
[OK]   model_inceptionv2_withaug500andsegmented.h5 → ...patched.h5
```

### 2. Verify model loads
`../01_data_augmentation/venv/bin/python -c "from prediction import predict_leaf; ..."`
```
Testing with: static/uploads/image.jpg
Raw prediction: Coriender (99.05%)
PREDICTION RESULT: Coriender
```

### Resolution
Used native Keras 3 `load_model` with `from_config` overrides on:
- `DepthwiseConv2D` — strip `groups` kwarg
- `RandomFlip` — strip `data_format` kwarg
- `InputLayer` — strip `optional` kwarg

Segmented model bypassed — unfixable dual-input architecture incompatibility.
