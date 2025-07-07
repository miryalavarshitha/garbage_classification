# Garbage Classification using EfficientNetV2B2 with Integrated AI Chatbot

🚀 **Live Demo:** [Garbage Classification App on Hugging Face Spaces](https://huggingface.co/spaces/miryalavarshitha/garbageclassification_app)

This project aims to classify images of garbage into six categories using a Convolutional Neural Network (CNN) built on EfficientNetV2B2. The workflow covers everything from dataset extraction to model deployment.

---

## WEEK-1

### 1. Dataset Extraction

- The dataset (`garbage_dataset.zip`) is unzipped into the `/garbage_dataset` directory using Python's `zipfile` module.

### 2. Image Dataset Loading

- The dataset is loaded from `/garbage_dataset/archive/TrashType_Image_Dataset` using TensorFlow’s `image_dataset_from_directory`:
  - 600 images are found, divided into 6 classes.
  - 80% (480 images) used for training.
  - 20% (120 images) initially used as validation.

### 3. Validation Split for Testing

- The original validation set is split into:
  - **Test Set:** First half (60 images)
  - **Final Validation Set:** Second half (60 images)
- The test set is optimized using caching and prefetching for efficient evaluation.

### 4. Class Names Check

- Class names are printed and confirmed to match:
  ```
  ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
  ```

### 5. Visualizing Sample Images

- A batch of 12 images from the training dataset is displayed in a 4×3 grid using `matplotlib`.
- Each image is labeled with its class name to verify correct labeling and loading.

---

## WEEK-2

### 5. Visualizing Sample Images (Continued)

- Displayed a batch of **12 sample images from the training dataset**.
- Plotted using **matplotlib (4×3 grid)**.
- Each image labeled with its respective class name to verify correct labeling.

### 6. Data Preprocessing and Normalization

- **Pixel Scaling:**  
  All image pixel values normalized from **[0, 255] to [0, 1]**.
- **Data Pipeline Enhancements (for training & validation sets):**
  - Caching
  - Shuffling
  - Batching
  - Prefetching

### 7. Model Building – CNN using EfficientNetV2B2

- Utilized **EfficientNetV2B2** from TensorFlow Keras Applications.
- **Base Model Settings:**
  - `include_top=False` (removed original classification head)
  - Using **pre-trained weights** for feature extraction.
- **Custom Layers Added:**
  - `GlobalAveragePooling2D`
  - `Dropout(0.2)`
  - `Dense(6, activation='softmax')` for 6-class garbage classification.
- **Model Summary:**
  - Total Trainable Parameters: Approximately **7 million**

### 8. Model Compilation

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

  ```python
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )
  ```

### 9. Model Training

- Model is trained on the training set.
- Validation set is used for monitoring overfitting and model selection.

### 10. Model Evaluation

- The model achieves an accuracy of **93%** on the test set.
- Model performance is evaluated on the test set using accuracy and loss metrics.
- Optionally, confusion matrix and classification report are visualized for detailed analysis.

### 11. AI Chatbot Integration – Jarvis 🤖
To enhance user experience and provide real-time assistance, an AI Chatbot named Jarvis is integrated into the web application. Jarvis helps users by: -Explaining the project workflow and purpose. -Answering frequently asked questions related to the model, accuracy, and applications. -Jarvis is designed to provide an interactive, WhatsApp-style chat interface, making the app both informative and engaging.

### 12. Model Deployment

- **The trained model is deployed using Gradio as an interactive web app.**
- **Live App:** [Garbage Classification App](https://huggingface.co/spaces/miryalavarshitha/garbageclassification_app)

---
