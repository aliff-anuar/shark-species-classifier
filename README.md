<<<<<<< HEAD
# Shark Species Classification — ISB46703 AI Project

Made by:
1. Muhammad Aliff bin Anuar - B03
2. Ahmad Fahim bin Zulkifli - B03

Group: 24

## Domain: Animal Subspecies (Shark)
## Classes: 8 shark species
## Models: ResNet50, DenseNet121, MobileNetV3

---

## How to Run (in order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Crawl images (~10,000 total)
python step1_crawl.py

# 3. Remove corrupt images + split dataset (70/15/15)
python step2_clean_split.py

# 4. Standardize all images to 224x224 RGB JPEG
python step3_standardize.py

# 5. Train all 3 CNN models (50 epochs each)
python step4_train_models.py

# 6. Generate all plots and conclusion
python step5_visualize.py
```

---

## Folder Structure After Running

```
shark_dataset/
├── raw/                  ← Downloaded images (step 1)
└── split/
    ├── train/            ← 70% of data (step 2)
    ├── val/              ← 15% of data (step 2)
    └── test/             ← 15% of data (step 2)

shark_results/
├── ResNet50_best.h5
├── ResNet50_final.h5
├── ResNet50_results.json
├── ResNet50_log.csv
├── DenseNet121_*.h5 / .json / .csv
├── MobileNetV3_*.h5 / .json / .csv
└── all_results_summary.json

shark_plots/
├── 1_dataset_distribution.png
├── 2_accuracy_curves.png
├── 3_loss_curves.png
├── 4_confusion_matrices.png
├── 5_model_comparison.png
├── 6_per_class_map.png
└── conclusion.txt
```

---

## Team Roles

| Role | Responsibilities | Files |
|---|---|---|
| Data Engineer | Crawl, clean, standardize | step1, step2, step3 |
| Data Scientist | Model training | step4 |
| Data Analyst | Visualization, conclusion | step5 |

---

## Shark Classes
1. Great White Shark
2. Hammerhead Shark
3. Tiger Shark
4. Bull Shark
5. Whale Shark
6. Nurse Shark
7. Mako Shark
8. Blacktip Reef Shark
=======
# shark-species-classifier
ISB46703 AI Project - Shark Species Classification
>>>>>>> 9492417e93613c8dd313c17dc623185ca515f794
