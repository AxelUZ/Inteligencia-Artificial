from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

'''
Basicmanete lo que sucede aqui es que Entrenamos al modelo dandole "X" que representan las entradas de las medidad de las flores
e identifique que clase de flor es osea "Y", entonces esto provocara en abse a un conjunto "X" que yo le de el modelo predecira que "Y" le pertenece
o lo que e slo mismo en abse a las medidas de la flor que clase de flor es.
'''

'''
Cargar el conjunto de datos Iris
Iris es una DB incluida en Scikit para el entrenamiento de modelos
Tiene dos parametros los cuales recibe data(que son las medidas de cada flor los datos sobre sus medidas) y target(Que es como la clase que es es flor( 
0 Setosa.
1 Versicolor.
2 Virginica.)

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris

'''
iris = load_iris()
X = iris.data
Y = iris.target

'''
Escalar los datos para mejorar el rendimiento del modelo eliminando la media y escalando a la varianza unitaria

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
'''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
Dividir los datos en conjuntos de entrenamiento y prueba (60% - 40%)
train test split corresponde  a una llamada de model_selection, test_size indica la cantidad en porcentaje del conjunto total que sera destinada a pruebas
Se establece el conjunto para pruebas como 0.4 y se asignan a X_test y Y_test y automaticmanete el otro 0.6 se asigna a entrenamiento para X_train y Y_train.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
'''
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4, random_state=42)

'''
Crea el modelo de perceptrón multicapa existe una alternativa la cual es MLPRegressor
EL parametro hidden_layer_sizes establece la estructura que tendra el modleo en este caso cuenta con 5 neuronas me parece que por default son 100
El parametro activation hay 4 diferentes pero en este caso utilizamos la función unitaria lineal rectificada que devuelve f(x) = max(0, x)
El parametro solver que es el encargado de la optimizacion de los pesos de la red en este caso se utiliza Descenso de gradiente estocástico(hay 3 distintos ver link de abajo).
El parametro max_iter establece el numero de iteraciones que se realizaran para el caso especifico de sdg se establece que "determina la cantidad de épocas (cuántas veces se usará cada punto de datos), no la cantidad de pasos de gradiente."
El parametro random_state es una seed

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
'''

model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', max_iter=2000, random_state=1)

'''
Fit es un metodo de MLPClassifier el cual basicmanete se encarga de entrenar el modelo recibiendo como parametro 
En este caso le pasamos X_train, Y_train porque son los que guardan o 60% de los datos que son para entrenamiento como se nos pidio
Naturalemnete debemos pasarle tanto los datos medidas de las plantas que es X como las clases de plantas que son Y

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
'''

model.fit(X_train, Y_train)


'''
Predict es un metodo de MLPClassifier, Ahora bien aqui realizamos las predicciones para esto le pasaremos X_test que guarda el 40% retsante d elos datos para test
Esta vez unicamnete le pasamos X por queremos que en base a los datos de X de medidad de las plantas me diga a que Clase d eplanata correpsonde
o bien a que Y corresponde por ejemplo si en mi X_test viene:

Longitud del sépalo: 3.5
Ancho del sépalo: 2
Longitud del pétalo: 3
Ancho del pétalo: 1

Entonces quiero que mi modelo me diga a que clase de flor corresponde esas medidas osea mi Y 
0 Setosa.
1 Versicolor.
2 Virginica.

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
'''
Y_pred = model.predict(X_test)

'''
Calcula la precision del modelo comparando el Y_test que son los valores correctos de Y y nuestro Y_pred que acabamos de calular
mientras mas se parezcan los datos de nuestro Y_pred a nuestro Y_test quiere decir que tenemos mas precisión

accuracy = accuracy_score(Y_test, Y_pred, normalize=False)

Si la linea fuera con un normalize=False comos e muestra arriba nos devolveria l cantidad de muestars que clasifico correctamente
Entonces ya que nuestro conjunto de pruebas es el 40% de 150 muestras tenemos 60 muestras para pruebas, en este caso nos arroja
que se clasifican correctamente 59 pruebas lo que correponde con el 0.98333 de precision del modelo

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
'''
accuracy = accuracy_score(Y_test, Y_pred)
Correctly_data = accuracy_score(Y_test, Y_pred, normalize=False)

count_Y_test = len(Y_test)

print("Precisión del modelo:", accuracy)
print("Datos totales del conjunto para prueba:",count_Y_test,"Datos totales predecidos correctamente:",Correctly_data)