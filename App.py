from flask import Flask, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)


def iris_check(iris_value):
    model = load_model('iris-weight.h5')
    test = np.expand_dims(iris_value, axis=0)
    result = model.predict(test)
    ans = np.argmax(result)
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return class_names[ans]


# API
@app.route('/output', methods=['POST', 'GET'])
def output():
    if request.method == 'POST':
        iris_value = request.form['iris-data']

        L = []
        print(iris_value)
        for i in iris_value.split(','):
            L.append(float(i))
        iris_array = np.array(L)

        ans = iris_check(iris_array)

        return ans


app.run()
