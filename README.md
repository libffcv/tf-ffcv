# tf-ffcv
This library provide (experimental) support for Tensorflow and FFCV.

## Usage example

To demonstrate how to use it we will use the cifar example of the main FFCV repository.

First make sure that you either import tensorflow or at least `tf_ffcv` before `ffcv`

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import torch as ch
from tf_ffcv import FFCVKerasSequence, ToTFImage
```

Now we reuse the the function `make_dataloaders` from our CIFAR example. Our only change is to add `ToTFImage()` in the pipeline.

```python
....
image_pipeline.extend([
    ToTensor(),
    ToDevice('cuda:0', non_blocking=True),
    ToTorchImage(),
    Convert(ch.float16),
    torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ToTFImage()  #  <===================
])
....
```
We create our Keras model as usual:
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
And wrap `FFCV` loaders using `FFCVKerasSequence`

```python
loaders = make_dataloaders()
dl = FFCVKerasSequence(loaders['train'])
dl_test = FFCVKerasSequence(loaders['test'])
```

These two dataset can be use to train/validate like any Keras one!
```python
history = model.fit_generator(dl, epochs=10)
model.evaluate(dl_test)
```



