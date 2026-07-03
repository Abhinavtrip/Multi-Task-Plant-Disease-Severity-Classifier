# Multi-Task-Plant-Disease-Severity-Classifier

A deep learning system that looks at a single photo of a plant leaf and simultaneously identifies **which disease** is present and **how severe** the infection is — powered by a multi-task CNN built on a fine-tuned EfficientNetB0 backbone, and served through a live Streamlit web app.


##  Objective

To build a multi-task convolutional neural network that, from a single leaf image, jointly predicts both the plant disease category and its severity level — going beyond simple classification to give growers an actionable read on how urgently a plant needs treatment.

##  Highlights

- **97.85%** disease classification accuracy and **98.19%** severity classification accuracy, achieved via two-phase transfer learning on a pretrained **EfficientNetB0**
- Trained on **70,295** training images and validated on **17,572** images spanning **38** crop/disease categories
- Architected a **dual-output network** using the Keras Functional API — a shared EfficientNetB0 backbone branching into two task-specific heads (disease + severity)
- Jointly optimized both heads with the Adam optimizer, combining categorical cross-entropy losses at a **1.0 : 0.5** weighting (disease : severity)
- Deployed as a live **Streamlit** web app on **Hugging Face Spaces**, containerized with **Docker** for real-time inference

##  How It Works

### 1. Dataset
- **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)** (augmented version of PlantVillage), sourced via the Kaggle API
- 38 classes covering multiple crops (apple, corn, grape, potato, tomato, etc.), each labeled as either healthy or a specific disease
- Images resized to 224×224 and batched at 64 using `tf.keras.utils.image_dataset_from_directory`

### 2. Severity Labeling
Since the raw dataset only labels *disease*, a second label — **severity** — is derived by mapping each of the 38 disease classes into one of three buckets:

| Severity | Meaning |
|---|---|
| `0` |  Healthy |
| `1` |  Early stage (e.g. bacterial spot, early blight, powdery mildew) |
| `2` |  Critical (e.g. late blight, black rot, viral infections) |

This mapping is applied on-the-fly to every batch via a `tf.data` pipeline, so each image ends up with two one-hot labels: `disease_output` and `severity_output`.

### 3. Model Architecture
- **Backbone:** EfficientNetB0 (ImageNet weights), initially frozen
- **Shared layers:** Global Average Pooling → BatchNormalization → Dense(256, ReLU) → Dropout(0.4)
- **Head 1 — `disease_output`:** Dense(38, softmax) — predicts the specific disease
- **Head 2 — `severity_output`:** Dense(3, softmax) — predicts severity level
- Built with the Keras **Functional API**, since a single shared trunk feeds two independent output heads

### 4. Two-Phase Training
1. **Phase 1 — Feature extraction:** Backbone frozen, only the new top layers are trained (Adam, lr=0.001, up to 10 epochs)
2. **Phase 2 — Fine-tuning:** Last 20 layers of EfficientNetB0 unfrozen and fine-tuned at a lower learning rate (Adam, lr=0.0001, up to 10 epochs)

Both phases use `EarlyStopping` (restores best weights) and `ReduceLROnPlateau` to stabilize training.

### 5. Loss & Optimization
```python
loss_weights = {"disease_output": 1.0, "severity_output": 0.5}
```
The disease head is treated as the primary task and weighted twice as heavily as the severity head, while still training both jointly in a single forward/backward pass.

##  The App

The Streamlit app (`app.py`) lets a user upload a leaf photo and get an instant read:

- **Disease card:** identified plant + condition, with a healthy/diseased badge and confidence score
- **Severity card:** Healthy / Early Stage / Critical, with a plain-language recommendation (e.g. *"Severe infection. Immediate treatment required."*)
- **Top 3 predictions:** the model's next-most-likely disease guesses, for cases where the top prediction is uncertain

The model and class labels are cached with `@st.cache_resource` so they load once per session rather than on every prediction.

##  Project Structure

```
plant-disease-classifier/
├── app.py                       # Streamlit inference app
├── plantDisease.ipynb           # Data pipeline, model build & training notebook
├── model/
│   ├── plant_multitask_model.keras   # Trained multi-task model
│   ├── class_names.json              # 38 disease class labels
│   └── severity_map.json             # Disease → severity mapping
├── requirements.txt
├── Dockerfile
└── README.md
```

##  Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras, EfficientNetB0 (transfer learning)
- **Data pipeline:** `tf.data`, `image_dataset_from_directory`
- **App/UI:** Streamlit
- **Deployment:** Docker, Hugging Face Spaces
- **Data source:** Kaggle API (New Plant Diseases Dataset)

##  Getting Started

### Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Place the three model artifacts (`plant_multitask_model.keras`, `class_names.json`, `severity_map.json`) inside a `model/` folder next to `app.py`, then:

```bash
streamlit run app.py
```

### Run with Docker

```bash
docker build -t plant-disease-classifier .
docker run -p 8501:8501 plant-disease-classifier
```

### Try it live

 Hugging Face Spaces: `< https://huggingface.co/spaces/Abhitrip/plant-disease-classifier>`

##  Results

| Task | Accuracy |
|---|---|
| Disease classification | **97.85%** |
| Severity classification | **98.19%** |

Evaluated on the held-out validation set (17,572 images) after Phase 2 fine-tuning.

## Future Improvements

- Add Grad-CAM visualizations so users can see *which part* of the leaf drove the prediction
- Expand severity labeling with continuous scoring instead of 3 discrete buckets
- Add batch/multi-image upload support
- Track experiments with MLflow or Weights & Biases


##  Acknowledgments

- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (augmented PlantVillage) on Kaggle
