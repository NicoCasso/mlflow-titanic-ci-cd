# MLflow Titanic

A quick demo to show how MLflow works using the Titanic dataset.

![demo](https://user-images.githubusercontent.com/17039389/65383212-cc848280-dd4c-11e9-9f4a-16c8577e6622.gif)

## Getting Started  (with venv)

```
# create virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate  # for Linux/macOS
# .venv\Scripts\activate    # for Windows

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# preprocess data
python src/preprocess.py

# train model
python src/train.py

# predict with test data

```

## Export Environment

```
pip freeze > requirements.txt
