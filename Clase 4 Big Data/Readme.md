## 1- Hallazgos

El modelo no aumenta el accuracy principalmente porque la función de perdida se esta estancando, es decir,  los cambios existentes entre una epoca y otra son demasiado pequeños. 

Al aumentar la cantidad de decimales en el código html 
```
logs.loss.toFixed(10)
```

se observa que, si bien la función paciera que no cambia, si esta teniendo cambios pero son muy pequeños, esto se puede observar en la siguiente resumen. 

| Época | Pérdida       | Accuracy |
|-------|---------------|----------|
| 1     | 0.3968225121  | 0.8997   |
| 2     | 0.3266133666  | 0.9000   |
| 3     | 0.3265218735  | 0.9000   |
| 4     | 0.3263985217  | 0.9000   |
| 5     | 0.3263185024  | 0.9000   |



 ## 2- Pruebas

 ### 2.1- Pruebas con el número de épocas

 Lo primero que se intenta es tratar de aumentar la cantidad de épocas para ver si el modelo puede aprender a medida que se tienen más épocas y así mejorar las métricas.


| # épocas | Loss | Accuracy |
|-----------|-----------|-----------|
| 5    | 0.32  | 0.9000   |
| 10   | 0.3  | 0.9000   |
| 15    | 0.3   | 0.9000   |



### 2.2- Pruebas cambiando el tamaño de los filtros

Se prueba cambiando el tamaño de los filtros asignados a las capas convolucionales, de esta manera:

```


      model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 5,
        activation: 'tanh',
        padding: 'same'
      }));

      // Second convolution layer
      model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 5,
        activation: 'tanh'
      }));

```

el modelo se estanca en el aprendizaje de igual manera.

### 2.3- Pruebas ajustando la tasa de aprendizaje


Se intenta cambiar el optimizador para ajustar la tasa de aprendizaje del modelo.

```

model.compile({
    optimizer: tf.train.adam(1),
    loss: 'categoricalCrossentropy',
    metrics:['accuracy']
})

```

se realizan varias pruebas, y se esquematizan los resultados


| # épocas | Loss | Accuracy |
|-----------|-----------|-----------|
| 1   | 0.32  | 0.9000   |
| 0.01   | 0.32  | 0.9000   |
| 0.005    | 0.9211   | 0.9000   |


Con la tasa de aprendizaje 0.005, se evidencia que la perdida es mucho mayor, pero el accuracy se mantiene estable, no se modifica

### 2.4- Pruebas ajustando las funciones de activación

Se realizan pruebas cambiando las funciones de activación de las capas convolucionales y las capas densas,  se le asigna la función Relu a todas 

```
 // First convolution layer
      model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 6,
        kernelSize: 5,
        activation: 'relu',
        padding: 'same'
      }));
      model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));

      // Second convolution layer
      model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 5,
        activation: 'tanh'
      }));
      model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));

      // Flatten layer
      model.add(tf.layers.flatten());

      // Fully connected layers
      model.add(tf.layers.dense({ units: 120, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 84, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

```


Con la tasa de aprendizaje anterior (0.005) se genera un resultado similar, `0.9211` de perdida y `0.90` de accuracy.