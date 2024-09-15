from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import MyModel


mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

# imgs =x_test[0:3]
# labs=y_test[0:3]
# print(labs)
# plot_imgs=np.hstack(imgs)
# plt.imshow(plot_imgs,cmap='gray')
# plt.show()

x_train=x_train[...,tf.newaxis]
x_test=x_test[...,tf.newaxis]

#数据生成器
train_ds=tf.data.Dataset.from_tensor_slices(
    (x_train,y_train)
).shuffle(10000).batch(32)
test_ds=tf.data.Dataset.from_tensor_slices(
    (x_test,y_test)
).batch(32)

model=MyModel()
loss_obj=tf.keras.losses.SparseCategoricalCrossentropy()

optimizer=tf.keras.optimizers.Adam()


train_loss=tf.keras.metrics.Mean(name='train_loss')
train_acc=tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

test_loss=tf.keras.metrics.Mean(name='test_loss')
test_acc=tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

@tf.function
def train_step(images,lables):
    with tf.GradientTape() as tape:
        predictions=model(images)
        loss=loss_obj(lables,predictions)
    graients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(graients,model.trainable_variables))
    train_loss(loss)
    train_acc(lables,predictions)

@tf.function
def test_step(images,lables):
    with tf.GradientTape() as tape:
        predictions=model(images)
        t_loss=loss_obj(lables,predictions)
    
    test_loss(t_loss)
    test_acc(lables,predictions)

EPOCHS=5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for images,lables in train_ds:
        train_step(images,lables)

    for test_images,test_lables in test_ds:
        test_step(test_images,test_lables)
    template='Epoch {},Loss: {},Accuracy: {},Test Loss: {}, Test Accuracy: {}'
    print(template.format(
        epoch+1,
        train_loss.result(),
        train_acc.result()*100,
        test_loss.result(),
        test_acc.result()*100
                          ))


