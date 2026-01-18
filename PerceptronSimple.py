#implementar el algoritmo del perceptron simple usando los siguientes casos
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PerceptronSimple:
    def __init__(self,input_shape, alpha):
        #inicializamos el sesgo(bias) a 1 y los pesos aleatoriamente
        self.w0=1
        self.wk= np.random.normal(size=input_shape)
        #guardamos el valor de alpha para la inicializacion de los pesos
        self.alpha=alpha
        
    #x---> son las entradas de los datos que le vamos a pasar al perceptron
    def __call__(self, x):
        #este metodo es el que se ejecutara cuando hagamos PerceptronSimple(x)
        #tiene que implementar la sumatoria del producto de los pesos por las entradas mas el sesgo
        suma=np.sum(self.wk*x,axis=-1)+self.w0 #axis es para trabajar con lña ultima dimension
        #luego aplicar la funcion signo
        activacion = 1 if suma >= 0 else -1
        return activacion
    def update(self,error,x):

        self.w0=self.w0+self.alpha*error
        self.wk=self.wk+self.alpha*error*x

def train(model,X,Y,repeticiones=20,verbose=True):
    #creamos una lista para almacenar los errores de cada epoca
    #de esta forma luego podemos representarlos
    trainError=[]
    for i in range(repeticiones):
        errorLocal=[]
        for entradaXi, salidaYi in zip(X,Y):
            prediccion=model(entradaXi)
            error=salidaYi-prediccion
            errorLocal.append(error**2)
            model.update(error,entradaXi)
        trainError.append(np.sum(errorLocal))
        if verbose:
            print(f"epoca{i+1}--->error: {trainError[i]}")
    return trainError

entradasPosibles=np.array([[0,0],[0,1],[1,0],[1,1]])
entradaNot= np.array([[0],[1]])

salidaAnd=np.array([-1,-1,-1,1])
salidaOr=np.array([-1,1,1,1])
salidaXor=np.array([-1,1,1,-1])
salidaNot=np.array([1,-1])


alpha = 0.05
erroresAND = train(PerceptronSimple(2,alpha), entradasPosibles, salidaAnd)
erroresOR  = train(PerceptronSimple(2,alpha), entradasPosibles, salidaOr)
erroresXOR = train(PerceptronSimple(2,alpha), entradasPosibles, salidaXor)
erroresNOT = train(PerceptronSimple(1,alpha), entradaNot, salidaNot)

plt.figure()
plt.plot(erroresAND)
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Error vs Época - AND")
plt.show()

plt.figure()
plt.plot(erroresOR)
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Error vs Época - OR")
plt.show()

plt.figure()
plt.plot(erroresXOR)
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Error vs Época - XOR")
plt.show()

plt.figure()
plt.plot(erroresNOT)
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Error vs Época - NOT")
plt.show()