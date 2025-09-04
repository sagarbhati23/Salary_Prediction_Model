# Salary Prediction Model

A machine learning project that predicts employee's expected salaries (CTC) based on features such as education, experience, and job role. It ensures fair compensation and reduces human judgment and bias. The project includes data exploration, visualization, preprocessing, and model building using a Linear Regression pipeline.

## Project Structure

```
├── data/
│   └── expected_ctc.csv   # dataset (already available)
├── notebooks/
│   └── EDA_and_Model.ipynb   # Jupyter notebook for analysis and model training
├── scripts/
│   ├── train.py      # script to train model and save as joblib
│   ├── predict.py    # script to load trained model and predict salary
│   └── evaluate.py   # script to evaluate model performance
├── models/
│   └── model.joblib  # trained model file (generated after training)
├── requirements.txt  # dependencies
└── README.md         # project overview and usage guide
```

## Features

- Data preprocessing and feature engineering
- Multiple regression models
- Model evaluation and comparison
- Prediction on new data

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sagarbhati23/salary_prediction_model.git
    cd salary_prediction_model
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the `data/` directory.
2. Run the main script or use the provided notebooks for training and prediction.
    ```bash
    python scripts/train.py
    ```
3. Evaluate model performance and generate predictions.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.