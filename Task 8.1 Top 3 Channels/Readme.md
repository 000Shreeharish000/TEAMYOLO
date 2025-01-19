# Task 8: Analyzing and Documenting How Masking Crucial Channels Impacts Classification Performance

## Overview

In this task, we explore how masking crucial EEG channels impacts classification performance across various classes. The analysis is conducted using LightGBM as the machine learning model. The steps involve preprocessing EEG data, selecting important channels through SHAP (SHapley Additive exPlanations), and evaluating the performance of the model. The evaluation includes measuring the ROC-AUC score, Balanced Accuracy Score, and the impact of masking specific EEG channels on model performance.

## Steps Involved

1. **Data Preprocessing**:
   - Multiple EEG datasets corresponding to four different classes (Normal, Complex Partial Seizures, Electrographic Seizures, and Video Detected Seizures with no Visual Change) were combined and preprocessed.
   - Missing values were imputed using the mean strategy.
   - Data was scaled using `StandardScaler` to normalize the features.
   - The dataset was split into training and validation sets (80% train, 20% validation).

2. **Model Training**:
   - A LightGBM classifier was trained using the preprocessed data.
   - The model was optimized using Optuna, a hyperparameter optimization library, to find the best performing set of hyperparameters.
   - The performance of the optimized model was evaluated on the validation set using ROC-AUC and Balanced Accuracy Score.

3. **Channel Importance Using SHAP**:
   - SHAP values were computed to interpret the model's predictions.
   - For each class, the top three most important EEG channels were identified based on their average SHAP values.

4. **Impact of Masking Crucial Channels**:
   - The crucial EEG channels identified by SHAP were masked one by one, and the classification performance was evaluated again to measure the effect of masking these features.

## Key Findings

- The LightGBM model achieved perfect performance with:
  - **ROC-AUC Score**: 1.0
  - **Balanced Accuracy Score**: 1.0

- The top three most important EEG channels for each class, based on SHAP values, are:

  | Class                                                   | Top 1 EEG Channel      | Top 2 EEG Channel      | Top 3 EEG Channel      |
  |---------------------------------------------------------|------------------------|------------------------|------------------------|
  | **Normal**                                              | Mean                   | Variance               | Zero Crossing Rate     |
  | **Complex Partial Seizures**                            | Unnamed: 0             | Zero Crossing Rate     | Mean                   |
  | **Electrographic Seizures**                             | Variance               | Mean                   | Zero Crossing Rate     |
  | **Video Detected Seizures with No Visual Change Over EEG** | Variance             | Mean                   | Zero Crossing Rate     |

### Optimized Model Evaluation

| Model                | ROC-AUC Score | Balanced Accuracy Score | Trainable Parameters |
|----------------------|---------------|-------------------------|----------------------|
| LightGBM Optimized   | 1.0           | 1.0                     | 4                    |

- The optimized LightGBM model achieved a perfect ROC-AUC score and balanced accuracy score, demonstrating the effectiveness of the model and the importance of feature engineering and hyperparameter tuning.

## Visualizations

- **SHAP Summary Plot**:
 A global view of the model's feature importance. It visualizes the SHAP values for each feature across all instances.

- **Class-Specific SHAP Values**: 
SHAP values were computed for each class to identify the most crucial EEG channels that contribute to the model's decision-making.

  - **Normal**: Top channels are `Mean`, `Variance`, and `Zero Crossing Rate`.
  - **Complex Partial Seizures**: Top channels are `Unnamed: 0`, `Zero Crossing Rate`, and `Mean`.
  - **Electrographic Seizures**: Top channels are `Variance`, `Mean`, and `Zero Crossing Rate`.
  - **Video Detected Seizures with No Visual Change Over EEG**: Top channels are `Variance`, `Mean`, and `Zero Crossing Rate`.

## Future Steps

1. **Masking Impact**:
Future work will involve systematically masking the crucial channels identified and measuring the impact on classification performance to evaluate the significance of these channels.

2. **Model Refinement**
 Further optimizations and testing on additional features can be performed to improve the model's robustness and handle potential overfitting.
## Conclusion
This task provided valuable insights into how the classification performance of a machine learning model is influenced by the features (EEG channels) used for training. Identifying and masking crucial features gives us an understanding of their importance in achieving optimal performance, which could be pivotal for further enhancing EEG-based classification systems.
