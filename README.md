# Customer Segmentation Project

Customer clustering using Gaussian Mixture Model.

## Project Structure

```
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── data/
│   ├── raw/                       # Original CSV data
│   └── preprocessed/              # Processed data with clusters
├── models/
│   ├── best_model.pkl            # Trained model pipeline
│   └── feature_columns.pkl       # Feature names
├── src/
│   ├── preprocessing.py          # Preprocessing pipeline
│   ├── utils.py                  # Feature engineering
│   ├── train.py                  # Training script
│   └── inference.py              # Prediction function
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train Model

```bash
cd src
python train.py
```

### Run Dashboard

```bash
cd app
streamlit run streamlit_app.py
```

### Make Prediction

```python
from inference import make_prediction
import pandas as pd

df = pd.read_csv('customer_transactions.csv')
result = make_prediction(df)
print(result)
```

## Customer Segments

| Cluster | Name | Description |
|---------|------|-------------|
| 0 | At-Risk | Haven't purchased recently |
| 1 | One-Timers | Single purchase customers |
| 2 | Rising Stars | Active high-value customers |
| 3 | Explorers | Frequent diverse shoppers |
| 4 | VIPs | Top spending customers |

## Required Columns

- `User_Id`
- `Customer_Age`
- `Trx_Vlu`
- `Trx_Age`
- `Trx_Rank`
- `Points`
- `Category In English`
- `Mer_Id`