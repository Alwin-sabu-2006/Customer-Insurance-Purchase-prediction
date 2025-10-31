# Customer Purchase Prediction & Model Comparison

This project is a Jupyter/Google Colab notebook that demonstrates a complete machine learning workflow for a binary classification problem. The goal is to predict whether a customer will purchase a product based on their **Age** and **Estimated Salary**.

The notebook's primary feature is a "bake-off," where five different classification algorithms are trained and evaluated to determine the most effective model for this dataset.

## 📊 Data & Visualization

The project uses the `insurance_data.csv` dataset, which includes the following columns:
* **Age**: The customer's age.
* **EstimatedSalary**: The customer's estimated salary.
* **Purchased**: The target variable (0 = No, 1 = Yes).

An initial scatter plot reveals a clear pattern: customers who are older and have a higher salary are more likely to make a purchase. This indicates that the data is well-suited for a classification model.



## ✨ Key Features

* **Exploratory Data Analysis (EDA):** Visualizes the class separation using a `seaborn` scatter plot.
* **Data Preprocessing:** Implements crucial preprocessing steps, including:
    * **Feature Scaling:** Uses `StandardScaler` to normalize `Age` and `EstimatedSalary`, which is essential for models like Logistic Regression, KNN, and SVM.
    * **Stratified Splitting:** Uses `stratify=y` during the `train_test_split` to ensure that both the training and test sets have the same proportion of "Purchased" and "Not Purchased" samples.
* **Model "Bake-Off":** Systematically trains, evaluates, and compares five different machine learning models:
    1.  Logistic Regression
    2.  K-Nearest Neighbors (KNN)
    3.  Support Vector Machine (SVM)
    4.  Decision Tree
    5.  Random Forest
* **Comprehensive Evaluation:** Measures each model's performance using four key metrics: **Accuracy, Precision, Recall, and F1-Score**.
* **Inference Pipeline:** Demonstrates how to use the final, best-performing model (`RandomForestClassifier`) and the saved `StandardScaler` to make predictions on new, hypothetical data.

## ⚙️ Functioning (How it Works)

The notebook follows a clear, step-by-step process:

1.  **Load Data:** The `insurance_data.csv` file is loaded into a `pandas` DataFrame.
2.  **Visualize:** `seaborn.scatterplot` is used to visualize the `Age` vs. `EstimatedSalary` relationship, with `Purchased` as the hue.
3.  **Define Features (X) and Target (y):**
    * `X` is set to the `['Age', 'EstimatedSalary']` columns.
    * `y` is set to the `['Purchased']` column.
4.  **Scale Features:**
    * An instance of `StandardScaler` is created.
    * `scaler.fit_transform(X)` is called to learn the mean and standard deviation of the features and then transform them into a normalized format (mean of 0, std of 1).
5.  **Split Data:** The scaled data (`X_scaled` and `y`) is split into a 20% test set and an 80% training set using `train_test_split`.
6.  **Train & Compare Models:**
    * A dictionary (`models`) is created to hold the five classifiers.
    * The code iterates through this dictionary. In each loop, it:
        * Fits the model to the training data (`X_train`, `y_train`).
        * Makes predictions on the test data (`X_test`).
        * Calculates and prints the Accuracy, Precision, Recall, and F1-Score for that model.
7.  **Predict on New Data:**
    * A final `RandomForestClassifier` is trained (as it was one of the top performers).
    * A list of new, hypothetical `test_cases` (e.g., `[30, 87000]`) is defined.
    * The **original `scaler`** is used to `transform` this new data, ensuring it is in the same format the model was trained on.
    * The model's `.predict()` method is called on the *scaled* test cases to get the final predictions.

## 📈 Results

### Model Comparison

The model "bake-off" produced the following results on the test set. **KNN**, **SVM**, and **Random Forest** were the clear top performers, all achieving 90% accuracy.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8375 | 0.8636 | 0.6552 | 0.7451 |
| **KNN** | **0.9000** | **0.8621** | **0.8621** | **0.8621** |
| **SVM** | **0.9000** | 0.8387 | **0.8966** | **0.8667** |
| Decision Tree | 0.8625 | 0.7647 | 0.8966 | 0.8254 |
| **Random Forest** | **0.9000** | 0.8387 | **0.8966** | **0.8667** |

### Final Predictions

Using the trained `RandomForestClassifier`, the notebook predicted the following outcomes for new customers:
