# README

## Project Title: Linear Regression Model Example

### Overview
`pro1.py` is a Python script that demonstrates a simple implementation of linear regression using the `scikit-learn` library. It includes data preparation, model training, evaluation, and prediction. The script is a great starting point for understanding how to use `scikit-learn` to perform basic regression tasks.

### Requirements
To run this script, ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `scikit-learn`

You can install them using:
```bash
pip install numpy pandas scikit-learn
```

### Script Details
The script performs the following tasks:

1. **Data Preparation**
   - Creates a sample dataset with `Feature` and `Target` columns.
   - Splits the dataset into training and testing sets.

2. **Model Training**
   - Utilizes a linear regression model from `scikit-learn`.
   - Trains the model on the training set.

3. **Model Evaluation**
   - Calculates and displays the Mean Squared Error (MSE) and R-squared value of the predictions on the test set.

4. **Prediction**
   - Predicts the target value for a new feature input.

### How to Use
1. Clone or download the repository containing the `pro1.py` script.
2. Open a terminal or command prompt and navigate to the directory containing `pro1.py`.
3. Run the script using Python:
   ```bash
   python pro1.py
   ```
4. View the output, which includes the model’s coefficients, intercept, evaluation metrics, and a prediction example.

### Example Output
Sample output of the script:
```
Coefficients: [2.03]
Intercept: 0.15
Mean Squared Error: 0.02
R-squared: 0.99
Predicted target for 11: 22.38
```

### Notes
- The dataset used in this example is hardcoded. For real-world applications, you can load your dataset by modifying the data loading section of the script.
- This implementation assumes a linear relationship between the feature and target. Ensure that your dataset satisfies this assumption for accurate predictions.

### License
This project is released under the MIT License. Feel free to use and modify the script as needed.

