import tensorflow as tf
import os

class Linear_QNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def save(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        self.save_weights(file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.bool)
        
        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            done = tf.expand_dims(done, 0)
        
        with tf.GradientTape() as tape:
            pred = self.model(state)
            target = pred.numpy()
            for idx in range(len(done)):
                Q_new = reward[idx].numpy()
                if not done[idx].numpy():
                    Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(tf.expand_dims(next_state[idx], 0))).numpy()
                target[idx][action[idx].numpy()] = Q_new
            target = tf.convert_to_tensor(target)
            loss = self.loss_fn(target, pred)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
