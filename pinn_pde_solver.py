import numpy as np
import tensorflow as tf

# Generate synthetic data
N = 1000
t = np.linspace(0, 1, N)
x = np.linspace(0, 1, N)
t, x = np.meshgrid(t, x)

# Define the neural network model
inputs = tf.keras.Input(shape=(2,))
layer = tf.keras.layers.Dense(20, activation='relu')(inputs)
layer = tf.keras.layers.Dense(20, activation='relu')(layer)
outputs = tf.keras.layers.Dense(1)(layer)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Define the loss function to include the residual of the PDE
def loss(model, t, x):
    with tf.GradientTape() as tape:
        tape.watch(t)
        tape.watch(x)
        u = model(tf.stack([t, x], axis=-1))
        u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    u_t = tape.gradient(u, t)
    return tf.reduce_mean((u_t - u_xx)**2)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Generate training data
t_train = tf.convert_to_tensor(t.flatten(), dtype=tf.float32)
x_train = tf.convert_to_tensor(x.flatten(), dtype=tf.float32)
u_train = tf.convert_to_tensor(np.exp(-np.pi**2*t)*np.sin(np.pi*x).flatten(), dtype=tf.float32)

# Training loop
for epoch in range(1000):
    with tf.GradientTape() as tape:
        current_loss = loss(model, t_train, x_train)
    grads = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {current_loss.numpy()}')

# Generate test data
t_test = tf.convert_to_tensor(np.random.uniform(0, 1, 100), dtype=tf.float32)
x_test = tf.convert_to_tensor(np.random.uniform(0, 1, 100), dtype=tf.float32)

# Test the model
u_test = model(tf.stack([t_test, x_test], axis=-1))
print('Test results:', u_test)
