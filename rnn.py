import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.Wx = np.random.randn(hidden_dim, input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim)
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def forward(self, x):
        h_prev = np.zeros((self.hidden_dim, 1))
        self.h_states = []
        for t in range(x.shape[0]):
            x_t = x[t].reshape(-1, 1)  # 转换为列向量 (input_dim, 1)
            h_next = self.tanh(np.dot(self.Wx, x_t) + np.dot(self.Wh, h_prev) + self.bh)
            self.h_states.append(h_next)
            h_prev = h_next
        y = self.sigmoid(np.dot(self.Wy, h_prev) + self.by)
        return y

    def backward(self, x, y_true, y_pred):
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        dy = y_pred - y_true
        dWy += np.dot(dy, self.h_states[-1].T)
        dby += dy

        dh_next = np.dot(self.Wy.T, dy)
        for t in reversed(range(x.shape[0])):
            dh = dh_next * self.tanh_derivative(self.h_states[t])
            dbh += dh
            dWx += np.dot(dh, x[t].reshape(1, -1))
            if t > 0:
                dWh += np.dot(dh, self.h_states[t-1].T)
                dh_next = np.dot(self.Wh.T, dh)
            else:
                dWh += np.dot(dh, np.zeros_like(self.h_states[t]).T)

        return dWx, dWh, dWy, dbh, dby

    def update_parameters(self, dWx, dWh, dWy, dbh, dby):
        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.Wy -= self.learning_rate * dWy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                for x, y_true in zip(X_batch, y_batch):
                    y_pred = self.forward(x)
                    dWx, dWh, dWy, dbh, dby = self.backward(x, y_true, y_pred)
                    self.update_parameters(dWx, dWh, dWy, dbh, dby)
            print(f"Epoch {epoch+1}/{epochs} completed")