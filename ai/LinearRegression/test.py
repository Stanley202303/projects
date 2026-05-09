# IMPORTS
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import pandas

# LOAD DATA
# data = [(d, t) for d in list(range(0, 500, 5))
#                for t in list(range(200, 1200, 10))]
# X = [k for k in data]
# y = [k[1] for k in data]

csv = pandas.read_csv("ai/test.csv")
X = list(zip(csv.x1, csv.x2, csv.x3))
y = [k for k in csv.y]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y)

# MODEL
model = LinearRegression()

# TRAIN
model.fit(X_train, y_train)

# PREDICT
new_data = [
    (5, 3, 3),
    (10, 5, 11),
    (23, 15, 9)
]
y_pred = model.predict(new_data)

# EVALUATE
score = model.score(X_test, y_test)
print(score, y_pred)