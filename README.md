# Aprendizaje federado con MNIST

Flujo de trabajo de aprendizaje federado simulado: la base de datos MNIST se divide entre los cuatro integrantes del equipo, cada integrante entrena localmente el mismo modelo con su partición, y un servidor combina los modelos locales en un modelo global.

- Raúl Correa Ocañas, A01722401
- Ricardo Kaleb Flores Alfonso, A01198716
- Sebastián Miramontes Soto, A01285296
- Alberto Rodríguez Reyes, A01383805

## Estructura del repositorio

```
├── CaffeLeNet.py            # Modelo global (arquitectura LeNet del ejemplo de MNIST de Caffe)
├── local_training.ipynb     # Entrenamiento local: training loop + evaluación
├── global_training.ipynb    # Cómputo del modelo global (FedAvg, matched averaging, FedAdam) y comparación
├── models/                  # Modelos locales (lmodel_k, lmodel_indep_k; n_k en .json) y globales (global_*)
├── reports/                 # Classification reports (.txt) y resumen de resultados (resultados.csv)
├── viz/                     # Curvas de aprendizaje de cada modelo local
├── requirements.txt
└── README.md
```

La división de los datos no forma parte del repositorio (ver [División de los datos](#1-división-de-los-datos-privada)).

## Modelo: LeNet (ejemplo de MNIST de Caffe)

`CaffeLeNet.py` reimplementa en TensorFlow/Keras la red LeNet definida en el ejemplo de MNIST de Caffe (Jia et al., 2014; `examples/mnist/lenet_train_test.prototxt`), basada en LeCun et al. (1998). Es distinta del modelo visto en clase: usa kernels de 5×5, 20 y 50 filtros, capas convolucionales sin activación (la no linealidad la aporta el max-pooling) y una sola capa oculta densa de 500 unidades.

| Capa (Caffe) | Keras | Salida | Parámetros |
|---|---|---|---|
| conv1 | `Conv2D(20, 5×5, stride 1, valid)`, sin activación | 24×24×20 | 520 |
| pool1 | `MaxPooling2D(2×2, stride 2)` | 12×12×20 | 0 |
| conv2 | `Conv2D(50, 5×5, stride 1, valid)`, sin activación | 8×8×50 | 25,050 |
| pool2 | `MaxPooling2D(2×2, stride 2)` | 4×4×50 | 0 |
| — | `Flatten` | 800 | 0 |
| ip1 + relu1 | `Dense(500, ReLU)` | 500 | 400,500 |
| ip2 | `Dense(10)` (logits) | 10 | 5,010 |
| **Total** | | | **431,080** |

- **Inicialización:** "xavier" de Caffe (uniforme, escala por *fan-in*), equivalente a `VarianceScaling(1.0, "fan_in", "uniform")`; sesgos en 0.
- **Pérdida:** entropía cruzada categórica sobre logits (`SparseCategoricalCrossentropy(from_logits=True)`), equivalente a `SoftmaxWithLoss` de Caffe.
- **Métrica:** exactitud (*accuracy*).
- **Optimizador:** SGD con tasa de aprendizaje 0.01, momentum 0.9 y lotes de 64, como en el *solver* de Caffe.

**Diferencias respecto a Caffe:** los píxeles se escalan entre 255 (Caffe usa 1/256); la tasa de aprendizaje es constante (Caffe usa la política `inv`); no se usa *weight decay* (Caffe usa 0.0005) ni se duplica la tasa de aprendizaje de los sesgos (`lr_mult: 2`).

## Flujo de trabajo

### 1. División de los datos (privada)

El conjunto de entrenamiento de MNIST (60,000 imágenes) se divide en cuatro particiones de 15,000 imágenes, una por integrante, con un script privado que no se versiona (`private/` está en `.gitignore`). La división es estratificada por etiqueta (`StratifiedKFold` con barajado y semilla fija), de modo que cada partición conserva la proporción de cada dígito.

Evidencia de que las particiones son estadísticamente equivalentes:

- **Etiquetas:** prueba χ² de homogeneidad sobre la tabla clientes × dígitos: χ² = 0.0057, gl = 27, p = 1.00. Los conteos por dígito difieren a lo más en 1 ejemplo entre clientes.
- **Imágenes:** prueba de Kruskal-Wallis sobre la intensidad media de cada imagen: H = 1.14, p = 0.77. Intensidad media por cliente: 0.1305, 0.1307, 0.1305 y 0.1309 (escala [0, 1]).

En ambas pruebas no hay evidencia de diferencias entre clientes.

### 2. Entrenamiento local (`local_training.ipynb`)

Cada cliente entrena el modelo durante 5 épocas con su partición, reservando el 10% de sus datos (estratificado) como validación local para las curvas de aprendizaje. Al terminar se evalúa sobre el conjunto de prueba de MNIST (10,000 imágenes) y se guardan el modelo, su número de ejemplos de entrenamiento $n_k$, el *classification report* y las curvas de aprendizaje.

Se entrenan dos conjuntos de modelos locales:
- `lmodel_k`: todos los clientes parten del mismo modelo inicial $w_0$, como si el servidor lo distribuyera antes del entrenamiento, que es la condición de la que parte FedAvg (McMahan et al., 2017).
- `lmodel_indep_k`: cada cliente parte de una inicialización aleatoria propia, sin coordinación con el servidor.

### 3. Modelo global (`global_training.ipynb`)

El servidor carga los modelos locales, verifica que tengan la arquitectura del modelo global y compara tres formas de computar el modelo global, cada una en las condiciones para las que fue diseñada y con el mismo presupuesto de 5 épocas de entrenamiento local por cliente:

| Método | Idea | Escenario de comparación |
|---|---|---|
| **FedAvg** (McMahan et al., 2017) | Promedio de los pesos ponderado por el número de ejemplos: $w = \sum_k \frac{n_k}{n} w_k$. | Línea base en todos los escenarios. |
| **Matched averaging** (Wang et al., 2020; Ainsworth et al., 2023) | Antes de promediar, permuta las neuronas ocultas de cada modelo local para alinearlas con un modelo de referencia (algoritmo húngaro, capa por capa). La permutación no cambia la función de cada modelo. | Una ronda, modelos con inicialización independiente; control con $w_0$ compartido. |
| **FedAdam** (Reddi et al., 2021) | En cada ronda, el servidor trata el cambio promedio de los pesos como un pseudo-gradiente y le aplica Adam (momentum y paso adaptativo por parámetro). | 5 rondas de una época local, $w_0$ compartido, contra FedAvg con las mismas rondas; 3 semillas. La tasa de aprendizaje del servidor se elige con validación federada, sin usar el conjunto de prueba. |

Los modelos globales se comparan también con los modelos locales y con un modelo centralizado (el mismo modelo entrenado con los 60,000 ejemplos en un solo dispositivo), que sirve como cota superior de referencia.

## Cómo reproducir

1. Crear un entorno con Python 3.12 e instalar las dependencias: `pip install -r requirements.txt`.
2. Colocar la partición propia en `private/data/client_{k}.npz` (arreglos `x`: imágenes uint8 de 28×28, `y`: etiquetas).
3. Ejecutar `local_training.ipynb` (*Restart & Run All*). Cada integrante usa `CLIENT_IDS = [k]`; para simular el flujo completo, `CLIENT_IDS = [0, 1, 2, 3]`.
4. Ejecutar `global_training.ipynb` una vez que `models/` contenga los dos conjuntos de modelos locales. La sección de FedAdam simula las rondas de comunicación y necesita las particiones de `private/data/`.

## Resultados

<!-- TODO: tabla con reports/resultados.csv después de ejecutar ambos cuadernos -->

## Referencias

- Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging models modulo permutation symmetries. *International Conference on Learning Representations (ICLR 2023)*.
- Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S., & Darrell, T. (2014). Caffe: Convolutional architecture for fast feature embedding. *Proceedings of the 22nd ACM International Conference on Multimedia*.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of AISTATS 2017*.
- Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2021). Adaptive federated optimization. *International Conference on Learning Representations (ICLR 2021)*.
- Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D., & Khazaeni, Y. (2020). Federated learning with matched averaging. *International Conference on Learning Representations (ICLR 2020)*.
