# Bash Command Log — Codebase Refactoring

## 2026-04-18T20:42 — Directory Renames
```bash
mv "Integration of Traditional Knowledge M1" "01_data_augmentation"
mv "Integration of Traditional Knowledge M2" "02_model_training"
mv "Integration of Traditional Knowledge M3" "03_web_app"
# Result: SUCCESS — All 3 top-level directories renamed
```

## 2026-04-18T20:42 — File Renames (M1)
```bash
mv "Agumentation.ipynb" "augmentation.ipynb"
mv "Agumentation-Copy1.ipynb" "augmentation_backup.ipynb"
mv "Agumentation_Fixed.ipynb" "augmentation_pillow.ipynb"
# Result: SUCCESS
```

## 2026-04-18T20:42 — Permission Fix (M2/M3)
```bash
chmod -R u+w "02_model_training" "03_web_app"
# Result: SUCCESS — read-only permission removed
```

## 2026-04-18T20:42 — File Renames (M2/M3)
```bash
mv "Model_Creation.ipynb" "model_training.ipynb"
mv "web_app.py" "app.py"
mv "model" "models"
mv "static/image" "static/images"
mkdir -p static/uploads static/js
# Result: SUCCESS
```

## 2026-04-18T20:42 — Docs Consolidation
```bash
mkdir -p docs
cp 01_data_augmentation/01_Documentation/*.md docs/
cp 01_data_augmentation/02_Architecture_Diagrams/*.md docs/
cp 01_data_augmentation/03_Bugs_and_Errors/*.md docs/
cp 01_data_augmentation/04_Improvements/*.md docs/
# Result: SUCCESS — 10 documentation files consolidated
```

## 2026-04-18T20:42 — Notebook Path Patching
```bash
python3 /tmp/patch_paths.py
# Result: SUCCESS — 8 path references updated in model_training.ipynb
```
