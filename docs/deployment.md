# Deployment Guide

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Click **New app**.
4. Choose your GitHub repository.
5. Set the main file path to:

```text
app/streamlit_app.py
```

6. Click **Deploy**.

Streamlit will install packages from `requirements.txt` automatically.

## Local run command

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app/streamlit_app.py
```
