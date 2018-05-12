import csv
import numpy as np
import pandas as pd

DATA_LABELS = {
    "age": 0,
    "gender": 1,
    "tb": 2,
    "db": 3,
    "alkphos": 4,
    "sgpt": 5,
    "sgot": 6,
    "tp": 7,
    "alp": 8,
    "ag": 9,
    "class": 10
}


def load_data():
    with open("data.csv", newline='') as data_csv:
        data_read = csv.reader(data_csv, delimiter=',', quotechar='|')
        a = np.array(list(list(data_read)))
        a_pd = pd.read_csv("data.csv", sep=",")
        a_pd = parser(a_pd)
        return a_pd.T


def parser(data):
    return pd.DataFrame([parse_gender(col) if it == DATA_LABELS["gender"] else parse_class(col) if it == DATA_LABELS[
        "class"] else normalize(col) for it, col in enumerate(np.array(data.T))])


def normalize(row):
    max_v = max(row)
    return list(map(lambda el, max_v=max_v: el / max_v, row))


def parse_gender(row):
    return list(map(lambda el: 0 if el == "Female" else 1, row))


def parse_class(row):
    return list(map(lambda el: abs(el - 2), row))  # '1' - zywy, '0' - martwy
