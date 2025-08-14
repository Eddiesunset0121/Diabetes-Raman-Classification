# Diabetes Classification using Raman Spectroscopy and Neural Networks
This project uses Raman spectroscopy data from earlobes to build a neural network that classifies patients as diabetic or non-diabetic. The model demonstrates the potential for a non-invasive, rapid screening method for Type 2 Diabetes Mellitus (DM2).

📋 Project Overview
Inspired by the research paper "In-vitro and rapid identification of patients with type 2 diabetes mellitus by Raman spectroscopy," this project replicates and expands upon the use of machine learning for diabetes detection. After rigorous preprocessing of the Raman spectra, a neural network with regularization techniques was trained to distinguish between diabetic and non-diabetic individuals. The final model achieves high accuracy and excellent clinical relevance.

✨ Key Features
Data Preprocessing: Implements area normalization to correct for variations in sample measurements and standardization to prevent feature dominance.

Baseline Modeling: Establishes a strong baseline with a Logistic Regression model, achieving 87.5% accuracy.

Neural Network Development: Builds and iteratively improves a neural network by tuning the learning rate and incorporating dropout layers to mitigate overfitting.

Performance: The final model achieves a validation accuracy of 93.75%.

Clinical Relevance: The model demonstrates a perfect recall of 1.00 for the 'Diabetic' class, making it a promising tool for medical screening as it correctly identifies all diabetic patients in the test set.

📊 Dataset
The dataset used in this project is publicly available from the publisher's website and is based on Raman spectra collected from the earlobes of patients. It consists of Raman intensity readings at different wavenumbers, with a binary label indicating the patient's diabetes status (1 for diabetic, 0 for non-diabetic).

🔬 Methodology
Exploratory Data Analysis (EDA): The project begins with an analysis of the class distribution, revealing a slight imbalance between diabetic and non-diabetic samples. The average Raman spectra for both classes are plotted to identify potential distinguishing features.

Preprocessing:

Area Normalization: Each spectrum is normalized by its total area to correct for variations in laser intensity and sample concentration between measurements.

Standardization: The StandardScaler from scikit-learn is used to standardize each feature (wavenumber) across all samples, ensuring that all features are given equal importance by the model.

Modeling:

A Logistic Regression model is first trained to establish a baseline performance.

A Neural Network is then built using TensorFlow/Keras. The model is improved through the following steps:

Learning Rate Tuning: A learning rate scheduler is used to find the optimal learning rate.

Dropout Regularization: Dropout layers are added to prevent the model from overfitting the training data.

Early Stopping: An early stopping callback is implemented to halt training when the validation loss stops improving, ensuring the model generalizes well to unseen data.

🏆 Results
The final model, a neural network with two dense layers and dropout, achieved the following performance on the test set:

Validation Accuracy: 93.75%

Recall (Diabetic Class): 1.00

Precision (Diabetic Class): 0.90

The model's perfect recall for the diabetic class is particularly noteworthy, as it indicates that the model is highly effective at identifying all positive cases, a critical requirement for a medical screening tool. The ROC curve and a high AUC score further validate the model's excellent discriminative ability.
