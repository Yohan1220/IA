import numpy as np
import matplotlib.pyplot as plt

# ===============================
# FUNCIONES DE ACTIVACIÓN
# ===============================

def linear_fn(x): return x
def dlinear_fn(x): return np.ones_like(x)

def relu_fn(x): return np.maximum(0, x)
def drelu_fn(x): return (x > 0).astype(float)

def sigmoid_fn(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid_fn(x):
    s = sigmoid_fn(x)
    return s * (1 - s)

# ===============================
# FUNCIONES DE PÉRDIDA
# ===============================

def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def dmse(y_true, y_pred): return 2 * (y_pred - y_true) / y_true.size

def bce(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def dbce(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# ===============================
# CLASE LAYER (CAPA)
# ===============================

class Layer:
    def __init__(self, input_size, neuronas, activacion="linear"):
        # Inicialización de Xavier para evitar gradientes muertos
        self.wk = np.random.randn(neuronas, input_size) * np.sqrt(1. / input_size)
        self.w0 = np.zeros((neuronas, 1))

        self.activation, self.dactivation = self.setActivation(activacion)
        self.z_k = None 
        self.a_k = None 
        self.delta_k = None

    def __call__(self, x):
        # x: (input_size, 1)
        self.z_k = np.dot(self.wk, x) + self.w0
        self.a_k = self.activation(self.z_k)
        return self.a_k

    def setActivation(self, activation):
        posibles = {"linear": linear_fn, "relu": relu_fn, "sigmoid": sigmoid_fn}
        derivadas = {"linear": dlinear_fn, "relu": drelu_fn, "sigmoid": dsigmoid_fn}
        return posibles[activation], derivadas[activation]


# ===============================================================
# CLASE LAYERSTACK (EL CONTENEDOR DE LA RED)
# ===============================================================
class LayerStack:
    def __init__(self, input_shape, neurons, activations, alpha=0.1, loss="mse"):
        """
        Esta clase organiza las capas y coordina el aprendizaje.
        - input_shape: número de variables de entrada (ej. 2 para XOR).
        - neurons: lista con la cantidad de neuronas por capa (ej. [4, 1]).
        - activations: lista con las funciones para cada capa (ej. ["relu", "sigmoid"]).
        """
        self.layers = []
        self.alpha = alpha  # Tasa de aprendizaje (learning rate)
        
        # --- CONSTRUCCIÓN DE LA PILA (STAGING) ---
        # Conectamos cada capa con la anterior.
        prev_size = input_shape
        for n, act in zip(neurons, activations):
            # Cada capa 'sabe' cuántas entradas recibe (prev_size) 
            # y cuántas salidas genera (n).
            self.layers.append(Layer(prev_size, n, act))
            prev_size = n  # La salida de esta capa será la entrada de la siguiente
            
        # Selección de la función de coste
        self.loss_fn = bce if loss == "bce" else mse
        self.dloss_fn = dbce if loss == "bce" else dmse

    def forward(self, x):
        """
        PROPAGACIÓN HACIA ADELANTE:
        Los datos entran por la primera capa y el resultado 'salta' 
        a la siguiente hasta llegar al final.
        """
        for layer in self.layers:
            x = layer(x) # Llama al método __call__ de la clase Layer
        return x

    def backpropagate(self, x_input, y_true, y_pred):
        """
        RETROPROPAGACIÓN (EL CORAZÓN DEL APRENDIZAJE):
        Aquí calculamos 'quién tuvo la culpa' del error, desde el final hacia el inicio.
        """
        
        # 1. ERROR EN LA ÚLTIMA CAPA (Salida)
        # Delta = (derivada_pérdida) * (derivada_activación_de_la_capa_final)
        delta = self.dloss_fn(y_true, y_pred) * \
                self.layers[-1].dactivation(self.layers[-1].z_k)
        self.layers[-1].delta_k = delta

        # 2. PROPAGACIÓN DEL ERROR (Capas ocultas)
        # Vamos en orden inverso: de la penúltima capa hasta la primera.
        for i in reversed(range(len(self.layers) - 1)):
            # El error de una capa depende de los pesos de la capa siguiente y su delta.
            # Delta_actual = (Pesos_siguiente^T * Delta_siguiente) * derivada_activación_actual
            delta = np.dot(self.layers[i+1].wk.T, delta) * \
                    self.layers[i].dactivation(self.layers[i].z_k)
            self.layers[i].delta_k = delta

        # 3. ACTUALIZACIÓN DE PARÁMETROS (Descenso del Gradiente)
        # Ahora que cada capa tiene su 'delta_k', ajustamos los pesos.
        input_to_layer = x_input  # La primera capa recibió la entrada original (X)
        for layer in self.layers:
            # Peso = Peso - alpha * (Error * Entrada_que_recibió_la_capa)
            layer.wk -= self.alpha * np.dot(layer.delta_k, input_to_layer.T)
            layer.w0 -= self.alpha * layer.delta_k
            
            # Para la siguiente capa, la 'entrada' es la activación (a_k) de la actual.
            input_to_layer = layer.a_k

    def fit(self, X, Y, epochs=1000, verbose=True):
        """
        ENTRENAMIENTO:
        Repite el ciclo de Forward y Backprop muchas veces.
        """
        history = []
        for epoch in range(epochs):
            error_acumulado = 0
            for x_i, y_i in zip(X, Y):
                # Aseguramos que los datos sean vectores columna (n, 1)
                x_i = x_i.reshape(-1, 1)
                y_i = y_i.reshape(-1, 1)

                # Paso 1: Ver qué predice la red actualmente
                pred = self.forward(x_i)
                error_acumulado += self.loss_fn(y_i, pred)

                # Paso 2 y 3: Calcular error y corregir pesos
                self.backpropagate(x_i, y_i, pred)

            # Guardamos el promedio del error en esta época
            loss_epoch = error_acumulado / len(X)
            history.append(loss_epoch)
            
            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}: Error = {loss_epoch:.4f}")
        return history

# ===============================
# EJEMPLO XOR
# ===============================

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

mlp = LayerStack(
    input_shape=2,
    neurons=[4, 1],
    activations=["relu", "sigmoid"],
    alpha=0.1,
    loss="bce"
)

losses = mlp.fit(X, Y, epochs=1000, verbose=True)

# Visualización de resultados
plt.plot(losses)
plt.title("Evolución del Error (XOR)")
plt.show()

# Prueba final
print("\nPredicciones finales:")
for x in X:
    print(f"Entrada: {x} -> Predicción: {mlp.forward(x.reshape(-1,1)).round(4)}")