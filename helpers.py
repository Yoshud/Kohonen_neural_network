import numpy as np

# funcja zwracająca indeksy danych w losowej kolejności by móc później "karmić" sieć danymi w losowej kolejności
def random_indexes(data):
    one = list(range(0, len(data)))
    second = list(np.random.rand(len(data)))
    x = np.concatenate((np.array([one]), np.array([second])), axis=0)
    lista = []
    for el in x.transpose():
        lista.append((el[0], el[1]))
    lista.sort(key=lambda x: x[1])
    return list(map(lambda el: int(el[0]), lista))