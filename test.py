from keras.models import load_model
import numpy as np

model = load_model('iris-weight.h5')

test = np.array([6.0,2.2,5.0,1.5])

test = np.expand_dims(test,axis=0)

result = model.predict(test)

print(result)

ans = np.argmax(result)

class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
print("Predicated value ",class_names[ans])