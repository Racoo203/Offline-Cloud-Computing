from tensorflow import keras

class build:
    @staticmethod
    def build_it():
        model = keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(
                filters = 20, 
                kernel_size = 5, 
                strides = 1, 
                padding = "valid", 
                activation = None,
                kernel_initializer = keras.initializers.VarianceScaling(
                    scale = 1.0, mode = "fan_in", distribution = "uniform"
                )
            ),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(
                filters = 50, 
                kernel_size = 5, 
                strides = 1, 
                padding = "valid", 
                activation = None,
                kernel_initializer = keras.initializers.VarianceScaling(
                    scale = 1.0, mode = "fan_in", distribution = "uniform"
                )
            ),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(
                500, activation = 'relu',
                kernel_initializer = keras.initializers.VarianceScaling(
                    scale = 1.0, mode = "fan_in", distribution = "uniform"
                )
            ),
            keras.layers.Dense(
                10, activation = None,
                kernel_initializer = keras.initializers.VarianceScaling(
                    scale = 1.0, mode = "fan_in", distribution = "uniform"
                )
            )
        ])

        model.compile(
            optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ["accuracy"],
        )
        return model