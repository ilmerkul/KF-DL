import numpy as np


class KalmanFilter(object):
    def __init__(self, F: np.ndarray = None, H: np.ndarray = None,
                 G: np.ndarray = None, Q: np.ndarray = None,
                 R: np.ndarray = None, P: np.ndarray = None,
                 x0: np.ndarray = None, bts: bool = False):
        if (F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.G = 0 if G is None else G
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

        self.bts = bts
        self.x_s = self.x
        self.P_s = self.P

    def predict(self, u: np.ndarray = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x, self.P

    def update(self, z: np.ndarray, u: np.ndarray = 0):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        I_K = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_K, self.P), I_K.T) + \
                 np.dot(np.dot(K, self.R), K.T)

        if self.bts:
            self.bts_update(u)

    def bts_update(self, u: np.ndarray = 0):
        z, P = self.predict(u)
        G = np.dot(self.P, np.dot(self.F.T, np.linalg.inv(P)))
        self.x_s = self.x + np.dot(G, np.dot(self.x_s - z))
        self.P_s = self.P + np.dot(G, np.dot(self.P_s - P, G.T))


class EMAlgorithm:
    def __init__(self, F: np.ndarray = None, H: np.ndarray = None,
                 G: np.ndarray = None, Q: np.ndarray = None,
                 R: np.ndarray = None, P: np.ndarray = None,
                 x0: np.ndarray = None):
        self.kf = KalmanFilter(F, H, G, Q, R, P, x0, bts=True)

    def E_step(self, z: np.ndarray, u: np.ndarray = 0):
        self.kf.update(z, u)

    def M_step(self):
        pass


def example():
    dt = 1.0 / 60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x ** 2 + 2 * x - 2) + np.random.normal(0, 2, 100)

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, bts=False)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H, kf.predict())[0])
        kf.update(z)

    import matplotlib.pyplot as plt
    plt.plot(range(len(measurements)), measurements, label='Measurements')
    plt.plot(range(len(predictions)), np.array(predictions),
             label='Kalman Filter Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
