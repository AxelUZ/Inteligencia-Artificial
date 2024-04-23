from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    {'hidden_layer_sizes': (), 'activation': 'logistic', 'solver': 'lbfgs', 'learning_rate': 'constant'},
    {'hidden_layer_sizes': (5,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate': 'adaptive'},
    {'hidden_layer_sizes': (3,), 'activation': 'tanh', 'solver': 'adam', 'learning_rate': 'invscaling'},
    {'hidden_layer_sizes': (5, 3), 'activation': 'relu', 'solver': 'lbfgs', 'learning_rate': 'constant'},
    {'hidden_layer_sizes': (2,), 'activation': 'tanh', 'solver': 'sgd', 'learning_rate': 'adaptive'},
    {'hidden_layer_sizes': (8,), 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'adaptive'},
    {'hidden_layer_sizes': (3, 2), 'activation': 'logistic', 'solver': 'lbfgs', 'learning_rate': 'constant'},
    {'hidden_layer_sizes': (5, 2), 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'constant'},
    {'hidden_layer_sizes': (4,), 'activation': 'tanh', 'solver': 'sgd', 'learning_rate': 'invscaling'},
    {'hidden_layer_sizes': (7,), 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'constant'},
]

better_model = None
better_accuarcy = 0

# Ciclo para probar todas las configuraciones anteriormente mencionadas
for config in configurations:
    # Crear el modelo MLPClassifier con la configuración actual
    model = MLPClassifier(max_iter=3000, random_state=1, **config)

    # Entrenar el modelo
    model.fit(X_train, Y_train)

    # Evaluar el modelo en el conjunto de prueba X_test y tambien para X_train y asi calcular errores y eficiencia en
    # entrenamiento
    Y_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)

    # Precisión del modelo en conjunto de prueba
    accuracy_test = accuracy_score(Y_test, Y_pred)

    # Error de eficiencia para aprendizaje y generalización
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    learning_error = mean_squared_error(Y_train, Y_train_pred)
    generalization_error = mean_squared_error(Y_test, Y_pred)

    # Imprimir la precisión del modelo para cada configuracion y sus errores
    print(
        "Configuración de red: {}\nPrecisión de pruebas: "
        "{:.2f}\nError de aprendizaje: {:.2f}\nError de generalización: {:.2f}\n---"
        .format(config, accuracy_test, learning_error, generalization_error)
    )

    # Actualizar el mejor modelo y precisión
    if accuracy_test > better_accuarcy:
        better_accuarcy = accuracy_test
        better_model = model

# Mejor modelo y precisión
print("\nMejor configuración de red:", better_model)
print("\nPrecisión en el conjunto de prueba:", better_accuarcy)
