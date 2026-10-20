"""
Modelo global: LeNet tal como se define en el ejemplo de MNIST de Caffe.

Fuentes:
- Jia, Y. et al. (2014). Caffe: Convolutional Architecture for Fast Feature
  Embedding. Tutorial "Training LeNet on MNIST with Caffe" y
  examples/mnist/lenet_train_test.prototxt (repositorio BVLC/caffe).
- LeCun, Y. et al. (1998). Gradient-based learning applied to document
  recognition. Proceedings of the IEEE, 86(11).

Diferencias respecto al ejemplo de Caffe:
- Los píxeles se escalan dividiendo entre 255 (Caffe usa 1/256).
- Tasa de aprendizaje constante de 0.01 (Caffe usa la política "inv").
- Sin weight decay (Caffe usa 0.0005) y sin duplicar la tasa de aprendizaje de
  los sesgos (lr_mult: 2 en Caffe).
"""
from tensorflow import keras


def _xavier():
    """Equivalente al filler "xavier" de Caffe (escala por fan-in, uniforme).

    Se crea una instancia nueva por capa: reutilizar la misma instancia de un
    inicializador sin semilla produce valores idénticos en cada llamada.
    """
    return keras.initializers.VarianceScaling(
        scale = 1.0, mode = "fan_in", distribution = "uniform"
    )


class build:
    @staticmethod
    def build_it(name = "lenet_caffe"):
        """Construye y compila el modelo (431,080 parámetros entrenables).

        Los nombres de las capas siguen los del prototxt de Caffe. ip2 devuelve
        logits; la función softmax está incluida en la pérdida, como en la capa
        SoftmaxWithLoss de Caffe.
        """
        model = keras.Sequential([
            keras.Input(shape = (28, 28, 1)),
            keras.layers.Conv2D(
                filters = 20,
                kernel_size = 5,
                strides = 1,
                padding = "valid",
                activation = None,
                kernel_initializer = _xavier(),
                name = "conv1",
            ),
            keras.layers.MaxPooling2D(pool_size = 2, strides = 2, name = "pool1"),
            keras.layers.Conv2D(
                filters = 50,
                kernel_size = 5,
                strides = 1,
                padding = "valid",
                activation = None,
                kernel_initializer = _xavier(),
                name = "conv2",
            ),
            keras.layers.MaxPooling2D(pool_size = 2, strides = 2, name = "pool2"),
            keras.layers.Flatten(name = "flatten"),
            keras.layers.Dense(
                500, activation = "relu", kernel_initializer = _xavier(), name = "ip1"
            ),
            keras.layers.Dense(
                10, activation = None, kernel_initializer = _xavier(), name = "ip2"
            ),
        ], name = name)

        model.compile(
            optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ["accuracy"],
        )
        return model
