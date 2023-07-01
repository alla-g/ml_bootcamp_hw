import numpy as np
import pandas as pd

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

pd.options.mode.chained_assignment = None


def open_data(path="data/clients.csv"):
    """ reads df from given path """
    df = pd.read_csv(path)

    return df


def remove_outliers(df: pd.DataFrame):
    """ removes outliers from feature columns """

    # колонки с оценками в шкалах
    # убираем оценки больше 5
    score_cols = list(df.select_dtypes(include=['float64']))[4:]
    for col in score_cols:
        df[col] = np.where(df[col] > 5, np.NaN, df[col])

    # возраст
    # убираем старше 100 лет
    df['Age'] = np.where(df['Age'] > 100, np.NaN, df['Age'])

    # расстояние
    # убираем больше 95 перцентиля
    flight_95 = np.nanpercentile(df['Flight Distance'], 95)
    df['Flight Distance'] = np.where(df['Flight Distance'] > flight_95, np.NaN, df['Flight Distance'])

    # задержка рейса
    # убираем больше 99 перцентиля
    departure_99 = np.nanpercentile(df['Departure Delay in Minutes'], 99)
    df['Departure Delay in Minutes'] = np.where(df['Departure Delay in Minutes'] > departure_99, np.NaN,
                                                df['Departure Delay in Minutes'])

    arrival_99 = np.nanpercentile(df['Arrival Delay in Minutes'], 99)
    df['Arrival Delay in Minutes'] = np.where(df['Arrival Delay in Minutes'] > arrival_99, np.NaN,
                                              df['Arrival Delay in Minutes'])

    return df


def remove_missing(df: pd.DataFrame):
    """ processes missing data from feature columns """

    # для числовых колонок
    nulls = pd.DataFrame(df.isna().sum(), columns=['NaN count'])
    has_nulls = nulls[nulls['NaN count'] > 0].index

    for col in has_nulls:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            df[col].fillna(df[col].mean(), inplace=True)

    # для не числовых колонок
    object_cols = list(df.select_dtypes(include=['object']))
    df.dropna(subset=object_cols, inplace=True)
    df = df[df['satisfaction'] != '-']

    return df


def encode_categories(df: pd.DataFrame):
    """ encodes categorical columns """

    # бинарные признаки
    df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1)
    df['Loyalty'] = df['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
    df['Business_travel'] = df['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)
    df['Satisfied'] = df['satisfaction'].apply(lambda x: 0 if x == 'neutral or dissatisfied' else 1)

    # one-hot для признака с 3 значениями
    enc_class = pd.get_dummies(df['Class'], drop_first=True)
    df = pd.concat([df, enc_class], axis=1)

    df.drop(['Type of Travel', 'Customer Type', 'satisfaction', 'Class'], axis=1, inplace=True)

    return df


def preprocess_data(df: pd.DataFrame):
    """ runs preprocessing on dataset """

    df_no_outliers = remove_outliers(df)
    df_no_missing = remove_missing(df_no_outliers)
    df_encoded_cats = encode_categories(df_no_missing)

    # разделение данных
    X, y = df_encoded_cats.drop(['id', 'Satisfied'], axis=1), df_encoded_cats['Satisfied']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # масштабирование
    ss = MinMaxScaler()
    ss.fit(X_train)

    X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test, ss


def fit_and_save_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                       path="data/model_weights.mw",
                       test_model=True,
                       metric='accuracy'):
    """ fits logistic regression model """
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    if test_model:
        preds = model.predict(X_test)
        if metric == 'accuracy':
            score = accuracy_score(y_test, preds)
        elif metric == 'recall':
            score = recall_score(y_test, preds)
        elif metric == 'precision':
            score = precision_score(y_test, preds)
        print(f'{metric.title()}: {round(score, 3)}')

    dump_model(model)
    save_importances(model, X_train.columns)


def dump_model(model, path="data/model_weights.mw"):
    """ saves model as pickle file """

    with open(path, "wb") as file:
        dump(model, file)

    print(f'Model was saved to {path}')


def save_importances(model, feature_names, path='data/importances.csv'):
    """ saves sorted feature weights as df """

    importances = pd.DataFrame({'Признак': feature_names, 'Вес': model.coef_[0]})
    importances.sort_values(by='Вес', key=abs, ascending=False, inplace=True)

    importances.to_csv(path, index=False)
    print(f'Importances were saved to {path}')


def load_model(path="data/model_weights.mw"):
    """ load model from saved weights """

    with open(path, "rb") as file:
        model = load(file)

    return model


def get_importances(top_n=5, importance='most', path='data/importances.csv'):
    """ returns top n most important or least important weights """

    importances = pd.read_csv(path, encoding='utf-8')
    if importance == 'most':
        return importances.head(top_n)
    else:
        return importances.tail(top_n).iloc[::-1]


def predict_on_input(df: pd.DataFrame):
    """ loads model and returns prediction and probability """

    model = load_model()
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)

    return pred, proba


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)
