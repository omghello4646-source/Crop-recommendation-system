# 🌾 Crop Recommendation System

A machine learning web app that recommends the most suitable crop for a field, based on soil nutrients and weather conditions.

## Problem

Farmers often choose crops by tradition or guesswork, without checking whether the soil and weather suit the crop. A wrong choice can reduce yield and income. This project predicts the best crop from seven inputs and explains the result.

## Features

- Type your values in simple number boxes: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH and rainfall
- Best crop with a confidence score
- Top 3 recommended crops
- Feature importance chart showing which inputs matter most
- What-if analysis: see how the recommendation changes when rainfall changes
- Built-in assistant that explains the result, shows what each crop needs, and gives soil and fertilizer hints from the dataset

## Model results

Dataset: 2,200 records, 22 crops, 7 input features, no missing values. Split: 80% training, 20% testing.

| Model | CV accuracy | Test accuracy |
|---|---|---|
| Decision Tree | 0.977 | 0.964 |
| Random Forest | 0.994 | 0.995 |
| KNN | 0.966 | 0.980 |

Random Forest performed best and is used in the app.

## Tech stack

Python, pandas, scikit-learn, Streamlit, joblib

## How to run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser, enter your values and click **Recommend crop**.

## Dataset

Crop Recommendation Dataset from Kaggle. Please check the dataset page for its license and credit the original author.

## Limitations

- The dataset is clean and the crops are well separated, so the high accuracy may not carry over to messy real-world fields.
- The app is a decision aid, not a replacement for expert agricultural advice or a soil test.

## Future scope

- Live weather data
- Soil test report upload
- Fertilizer quantity suggestions
- Regional language support
- AI-model-powered chatbot