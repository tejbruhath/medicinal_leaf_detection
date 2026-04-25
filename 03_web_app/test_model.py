import sys
from prediction import predict_leaf

files = ["test_images/tulsi_leaf_1777137366444.png", "test_images/mint_leaf_1777137382122.png", "test_images/neem_leaf_1777137396585.png"]
try:
    for f in files:
        cls, conf = predict_leaf(f)
        print(f"File: {f} -> {cls} ({conf:.4f})")
except Exception as e:
    print(f"Error: {e}")
