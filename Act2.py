from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

'''
1.

La linea que encontramos para realizar backpropagation en python esta dada por TensorFlow
y para la generalización y entrenamiento de una red feedforward tenemos la biblioteca que estamos utilizando la 
cual es sklearn.

Backpropagation con Tensorflow: Calcula los gradientes de la pérdida con respecto a los parámetros del modelo.
grads = tape.gradient(loss_value, model.trainable_variables)
https://www.geeksforgeeks.org/back-propagation-with-tensorflow/

Para el entremaiento de una red feedforward tenemos la funcion MLP que crea un perceptron multicapa:
MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', max_iter=2000, random_state=1)
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.fit
'''

'''
2. 3.

Algunos de los parametros que se pueden modificar para mejorar el rendimiento de la red
o probar distintos tipos de modelos o arquitecturas son los siguientes:

1. Numero de capas en la red: El número de capas ocultas en la red neuronal puede afectar su capacidad para 
aprender y representar relaciones complejas en los datos.

2. Numero de neuronas por capa: Un mayor número de neuronas puede permitir que el modelo capture relaciones más 
complejas en los datos, pero también puede aumentar la complejidad computacional y el riesgo de sobreajuste.

3. Función de activacion del modelo (identity  f(x) = x, logistic f(x) = 1 / (1 + exp(-x)), tanh f(x) = tanh(x), 
relu f(x) = max(0, x)): Las funciones de activación determinan cómo se calculan las salidas de cada neurona en la red.

4. Tasa de aprendizaje (constant, invscaling, adaptive)
La tasa de aprendizaje controla la magnitud de los ajustes de los pesos durante el proceso de optimización. 

5. Algorimo de optimizacion (lbfgs, sgd, adam)
El algoritmo de optimización determina cómo se actualizan los pesos del modelo durante el entrenamiento. 

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.fit
'''

'''
4. 5. 6.
'''
# Cargar la base de datos de iris
iris = load_iris()
X = iris.data
Y = iris.target

# Estandarizar Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (60% - 40%) respectivamente
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4, random_state=42)

# Se probarán diferentes configuraciones para generar distintos aprendizajes cambiando los números de capas y
# diferentes números de neuronas por capa

configurations = [
    (),                     # Una capa oculta sin neuronas (modelo lineal)
    (3,),                   # Una capa oculta con 3 neuronas
    (6,),                   # Una capa oculta con 6 neuronas
    (5, 5),                 # Dos capas ocultas con 5 neuronas cada una
    (8, 3),                 # Dos capas ocultas con 8 y 3 neuronas respectivamente
    (10, 3),                # Dos capas ocultas con 10 y 3 neuronas respectivamente
    (20, 15),               # Dos capas ocultas con 20 y 15 neuronas respectivamente
    (10, 10),               # Dos capas ocultas con 10 neuronas cada una
    (15, 15),               # Dos capas ocultas con 15 neuronas cada una
    (2, 2),                 # Dos capas ocultas con 2 neuronas cada una
]

better_model = None
better_accuarcy = 0

# Ciclo para probar todas las configuraciones anteriormente mencionadas
for config in configurations:
    # Crear el modelo MLPClassifier con la configuración actual
    model = MLPClassifier(hidden_layer_sizes=config, activation='relu', solver='sgd', max_iter=3000, random_state=1)

    # Entrenar el modelo
    model.fit(X_train, Y_train)

    # Evaluar el modelo en el conjunto de prueba X_test
    Y_pred = model.predict(X_test)

    # Precisión del modelo
    accuracy = accuracy_score(Y_test, Y_pred)

    # Imprimir la precisión del modelo para cada configuracion
    print(f"Configuración de red: {config}, Precisión: {accuracy}")

    # Actualizar el mejor modelo y precisión
    if accuracy > better_accuarcy:
        better_accuarcy = accuracy
        better_model = model

# Mejor modelo y precisión
print("\nMejor configuración de red:", better_model)

print("\nPrecisión en el conjunto de prueba:", better_accuarcy)
