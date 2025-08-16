# Project: Diabetes Classification with Raman Spectroscopy

<img width="1380" height="1100" alt="Flowchart" src="https://github.com/user-attachments/assets/73e96eda-0e0f-4ae1-9578-185b64a39af8" />


🔷 Project Objective:
- Inspired by the paper "In-vitro and rapid identification of patients with type 2 diabetes mellitus by Raman spectroscopy".
https://opg.optica.org/boe/fulltext.cfm?uri=boe-9-10-4998&id=398623

- This project analyzes Raman spectroscopy data to develop a predictive model for the rapid and non-invasive identification of patients with Type 2 Diabetes Mellitus (DM2). The primary goal is to build an accurate and reliable classification model that can distinguish between diabetic and non-diabetic individuals based on their Raman spectra.

🌟 Key Skills & Tools
- Tensorflow for deep learning

🌿 Analysis & Key Findings
- Comprehensive EDA, the establishment of a strong baseline model, and the iterative development of a neural network to achieve high predictive accuracy and clinical relevance.

🌿 Foundational Analysis & Preprocessing:
- Key challenge: inherent variability in spectroscopy measurements. This was addressed by implementing area normalization to standardize the spectra and correct for sample-to-sample variations.

- EDA revealed subtle but potential differences in the average Raman spectra between diabetic and non-diabetic individuals, particularly around the 1350 cm-1 and 1380 cm-1 Raman shifts.

🌿 Model Development & Selection:
- A Logistic Regression model was first developed to establish a baseline, achieving a strong accuracy of 87.5%.

- A neural network was then built and iteratively improved. The introduction of dropout layers was the most critical enhancement, mitigating overfitting and boosting the model's performance.

🌿 Final Model Performance:
- The final, tuned neural network demonstrates excellent predictive power, achieving an overall accuracy of 93.75% on unseen test data.

- The model's standout feature is its perfect recall of 1.00 for the 'Diabetic' class, as shown in the confusion matrix. This means it correctly identified all diabetic patients in the test set, making it a highly reliable screening tool.

💡 Clinical Application
- Non-Invasive Screening: Offering a rapid and non-invasive method for diabetes screening using a simple Raman scan of the earlobe, replacing the need for blood tests in initial screenings.

- Early Detection: Providing a highly sensitive tool for early detection, as the model is exceptionally good at identifying diabetic individuals (zero false negatives in the test set).

- Point-of-Care Diagnostics: With the use of portable Raman spectrometers, this model could be deployed in clinics for on-the-spot risk assessment, allowing for immediate patient consultation and follow-up.
