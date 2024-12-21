import numpy as np

np.random.seed(42)


class KalmanFilter(object):
    def __init__(self, T1: np.ndarray, T2: np.ndarray,
                 D: np.ndarray, Q: np.ndarray = None,
                 R: np.ndarray = None, P: np.ndarray = None,
                 z0: np.ndarray = None, bts: bool = False):
        assert T1.shape[0] == T1.shape[1]
        assert T1.shape[0] == T2.shape[0]
        assert T1.shape[0] == D.shape[1]

        self.z_size = T1.shape[1]
        self.u_size = T2.shape[1]
        self.x_size = D.shape[0]

        self.T1 = T1
        self.T2 = T2
        self.D = D
        self.Q = np.eye(self.z_size) if Q is None else Q
        self.R = np.eye(self.x_size) if R is None else R
        self.P = np.eye(self.z_size) if P is None else P
        z0 = np.ones(self.z_size) if z0 is None else z0
        self.z = np.random.multivariate_normal(z0, self.P, size=1).squeeze(0)

        self.bts = bts
        self.z_s = self.z
        self.P_s = self.P
        self.G = np.dot(self.P, np.dot(self.T1.T, np.linalg.inv(self.P)))

    def predict(self, u: np.ndarray):
        # P (z_size, z_size)
        # T1 (z_size, z_size)
        # T2 (z_size, u_size)
        # Q (z_size, z_size)
        z = np.dot(self.T1, self.z) + np.dot(self.T2, u)
        P = np.dot(np.dot(self.T1, self.P), self.T1.T) + self.Q
        return z, P

    def update(self, x: np.ndarray, u: np.ndarray):
        self.z, self.P = self.predict(u)

        # D (x_size, z_size)
        # P (z_size, z_size)
        y = x - np.dot(self.D, self.z)
        S = self.R + np.dot(self.D, np.dot(self.P, self.D.T))
        K = np.dot(np.dot(self.P, self.D.T), np.linalg.inv(S))
        self.z = self.z + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, S), K.T)

        if self.bts:
            self.bts_update(u)

    def bts_update(self, u: np.ndarray):
        z, P = self.predict(u)
        self.G = np.dot(self.P, np.dot(self.T1.T, np.linalg.inv(P)))
        self.z_s = self.z + np.dot(self.G, self.z_s - z)
        self.P_s = self.P + np.dot(self.G, np.dot(self.P_s - P, self.G.T))


class Model(KalmanFilter):
    def __init__(self, bts: bool = True, K: int = 10, **kwargs):
        super(Model, self).__init__(**kwargs, bts=bts)

        self.K = K
        self.Sigma = np.zeros_like(self.P)
        self.Phi = np.zeros_like(self.P)
        self.B = np.zeros_like(self.P)
        self.C = np.zeros_like(self.P)
        self.A = np.zeros_like(self.P)
        self.F = np.zeros_like(self.P)
        self.I = np.zeros_like(self.P)
        self.Delta = np.zeros_like(self.P)

    def E_step(self, x: np.ndarray, u: np.ndarray = None):
        if u is None:
            u = np.zeros((self.K, self.z_size, 1))

        Z_K_s = [self.z_s]
        P_K_s = [self.P_s]
        G_k = [self.G]

        for k in range(self.K):
            self.update(x[k], u[k])

            Z_K_s.append(self.z_s)
            P_K_s.append(self.P_s)
            G_k.append(self.G)

        Z_K_s = np.array(Z_K_s)
        P_K_s = np.array(P_K_s)
        G_k = np.array(G_k)

        self.Sigma = (np.sum(P_K_s[1:], axis=0) + np.einsum("ki,kj->ij", Z_K_s[1:],
                                                       Z_K_s[1:])) / self.K
        self.Phi = (np.sum(P_K_s[:-1], axis=0) + np.einsum("ki,kj->ij", Z_K_s[:-1],
                                                      Z_K_s[:-1])) / self.K
        self.B = np.einsum("kx,kz->xz", x, Z_K_s[1:]) / self.K
        self.C = (np.einsum("kiz,kzj->ij", P_K_s[1:], G_k[:-1]) + np.einsum(
            "ki,kj->ij", Z_K_s[1:], Z_K_s[:-1])) / self.K
        self.A = np.einsum("kz,ku->zu", Z_K_s[1:], u) / self.K
        self.F = np.einsum("kz,ku->zu", Z_K_s[:-1], u) / self.K
        self.I = np.einsum("ki,kj->ij", u, u) / self.K
        self.Delta = np.einsum("ki,kj->ij", x, x) / self.K

    def M_step(self):
        Q = 0.5 * self.K * np.trace(np.dot(np.linalg.inv(self.Q), self.Sigma - np.dot(self.T1.T, self.C) - np.dot(self.A, self.T2.T) - np.dot(self.T1, self.C.T) + np.dot(np.dot(self.T1, self.Phi), self.T1.T) + np.dot(np.dot(self.T1, self.F), self.T2.T) - np.dot(self.T2, self.A.T) + np.dot(np.dot(self.T2, self.F.T), self.T1.T) + np.dot(np.dot(self.T2, self.I), self.T2.T))) + 0.5 * self.K * np.trace(np.dot(np.linalg.inv(self.R), self.Delta) - np.dot(self.B, self.D.T) - np.dot(self.D, self.B.T) + np.dot(np.dot(self.D, self.Sigma), self.D.T))
        # TODO minimization Q
        return Q

if __name__ == "__main__":
    model = Model(T1=np.random.random((10, 10)), T2=np.random.random((10, 12)), D=np.random.random((13, 10)), K=20)

    model.E_step(np.zeros((20, 13)), np.zeros((20, 12)))
    print(model.M_step())