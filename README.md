# ML bootcamp: from idea to MVP

This is a project for ML bootcamp by HSE: exploratory data analysis, modeling and building a service.

Selected topic: flight satisfaction. Dataset: [here](https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/clients.csv)

## Files:
* `Flight_satisfaction.ipynb`: Colab notebook with exploratory data analysis, modelling and interpretations
* `app.py`: Streamlit app file
* `model.py`: script that transforms data, trains and runs the model 
* `requirements.txt`: package requirements files
* `/data` folder has:
  * `clients.csv`: copy of dataset
  * `importances.csv`: features sorted by their importance
  * `model_weights.mw`: pretrained model
  * and png visualizations for EDA

## Service:
Streamlit service is available at [rateyourflight.streamlit.app](https://rateyourflight.streamlit.app/) via Streamlit Cloud

To run locally, clone the repo and do the following:
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
```
The app will be available at `http://localhost:8501`
