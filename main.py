import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)


class KalmanFilter(object):
    def __init__(
        self, T1: np.ndarray, T2: np.ndarray, D: np.ndarray, bts: bool = False
    ):
        assert T1.shape[0] == T1.shape[1]
        assert T1.shape[0] == T2.shape[0]
        assert T1.shape[0] == D.shape[1]

        self.z_size = T1.shape[1]
        self.u_size = T2.shape[1]
        self.x_size = D.shape[0]

        # T1 (z_size, z_size)
        self.T1 = None
        # T2 (z_size, u_size)
        self.T2 = None
        # D (x_size, z_size)
        self.D = None
        # Q (z_size, z_size)
        self.Q = np.eye(self.z_size)
        # R (x_size, x_size)
        self.R = np.eye(self.x_size)
        # P (z_size, z_size)
        self.P = np.eye(self.z_size)
        # z0 (z_size,)
        z0 = np.ones(self.z_size)
        # z (z_size,)
        self.z = np.random.multivariate_normal(z0, self.P, size=1).squeeze(0)

        self.set_matrix(T1, T2, D)

        self.bts = bts
        # z_s (z_size,)
        self.z_s = self.z
        # P_s (z_size, z_size)
        self.P_s = self.P
        # G (z_size, z_size)
        self.G = np.dot(self.P, np.dot(self.T1.T, np.linalg.inv(self.P)))

    def get_size(self) -> (int, int, int):
        return self.z_size, self.u_size, self.x_size

    def set_matrix(self, T1: np.ndarray, T2: np.ndarray, D: np.ndarray):
        self.T1 = T1
        self.T2 = T2
        self.D = D

    def predict(self, u: np.ndarray):
        z = np.dot(self.T1, self.z) + np.dot(self.T2, u)
        P = np.dot(np.dot(self.T1, self.P), self.T1.T) + self.Q
        return z, P

    def update(self, x: np.ndarray, u: np.ndarray):
        self.z, self.P = self.predict(u)

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


class Model:
    def __init__(
        self,
        T10: np.ndarray,
        T11: np.ndarray,
        T12: np.ndarray,
        T20: np.ndarray,
        T21: np.ndarray,
        T22: np.ndarray,
        D0: np.ndarray,
        D1: np.ndarray,
        D2: np.ndarray,
        bts: bool = True,
        K: int = 10,
    ):

        self.T10 = T10
        self.T11 = T11
        self.T12 = T12
        self.T20 = T20
        self.T21 = T21
        self.T22 = T22
        self.D0 = D0
        self.D1 = D1
        self.D2 = D2

        T1 = self.T10 @ self.T11 @ self.T12
        T2 = self.T20 @ self.T21 @ self.T22
        D = self.D0 @ self.D1 @ self.D2

        self.kf = KalmanFilter(T1, T2, D, bts=bts)

        self.K = K

        shape_P = self.kf.P.shape
        self.Sigma = np.zeros(shape_P)
        self.Phi = np.zeros(shape_P)
        self.B = np.zeros(shape_P)
        self.C = np.zeros(shape_P)
        self.A = np.zeros(shape_P)
        self.F = np.zeros(shape_P)
        self.I = np.zeros(shape_P)
        self.Delta = np.zeros(shape_P)

        self.z_size, self.u_size, self.x_size = self.kf.get_size()

    def E_step(self, x: np.ndarray, u: np.ndarray = None):
        if u is None:
            u = np.zeros((self.K, self.z_size))

        Z_K_s = [self.kf.z_s]
        P_K_s = [self.kf.P_s]
        G_k = [self.kf.G]

        for k in range(self.K):
            self.kf.update(x[k], u[k])

            Z_K_s.append(self.kf.z_s)
            P_K_s.append(self.kf.P_s)
            G_k.append(self.kf.G)

        Z_K_s = np.array(Z_K_s)
        P_K_s = np.array(P_K_s)
        G_k = np.array(G_k)

        self.Sigma = (
            np.sum(P_K_s[1:], axis=0) + np.einsum("ki,kj->ij", Z_K_s[1:], Z_K_s[1:])
        ) / self.K
        self.Phi = (
            np.sum(P_K_s[:-1], axis=0) + np.einsum("ki,kj->ij", Z_K_s[:-1], Z_K_s[:-1])
        ) / self.K
        self.B = np.einsum("kx,kz->xz", x, Z_K_s[1:]) / self.K
        self.C = (
            np.einsum("kiz,kzj->ij", P_K_s[1:], G_k[:-1])
            + np.einsum("ki,kj->ij", Z_K_s[1:], Z_K_s[:-1])
        ) / self.K
        self.A = np.einsum("kz,ku->zu", Z_K_s[1:], u) / self.K
        self.F = np.einsum("kz,ku->zu", Z_K_s[:-1], u) / self.K
        self.I = np.einsum("ki,kj->ij", u, u) / self.K
        self.Delta = np.einsum("ki,kj->ij", x, x) / self.K

    def M_step(self):
        self.T10 = (
            self.C @ self.T12.T @ self.T11.T
            - self.T20
            @ self.T21
            @ self.T22
            @ self.F.T
            @ self.T12.T
            @ self.T11.T
            @ np.linalg.pinv(self.T11 @ self.T12 @ self.Phi @ self.T12.T @ self.T11.T)
        )
        self.T10 = np.where(self.T10 > 0, self.T10, 0)

        self.T11 = (
            self.T10.T
            @ np.linalg.inv(self.kf.Q)
            @ np.linalg.inv(self.T10)
            @ (
                self.T10.T @ np.linalg.inv(self.kf.Q) @ self.C @ self.T12
                - self.T10.T
                @ np.linalg.inv(self.kf.Q)
                @ self.T20
                @ self.T21
                @ self.T22
                @ self.F.T
                @ self.T12
                @ np.linalg.pinv(self.T12 @ self.Phi @ self.T12.T)
            )
        )
        self.T11 = np.where(self.T11 > 0, self.T11, 0)

        self.T12 = np.linalg.pinv(
            self.T11.T @ self.T10.T @ np.linalg.inv(self.kf.Q) @ self.T10 @ self.T11
        ) @ self.T11.T @ self.T10.T @ self.C @ np.linalg.inv(
            self.kf.Q
        ) - self.T11.T @ self.T10.T @ np.linalg.inv(
            self.kf.Q
        ) @ self.T20 @ self.T21 @ self.T22 @ self.F.T @ np.linalg.inv(
            self.Phi
        )
        self.T12 = np.where(self.T12 > 0, self.T12, 0)
        print(self.T22.shape, self.T21.shape, self.F.shape)
        self.T20 = (
            self.T22.T @ self.T21.T
            - self.T10 @ self.T11 @ self.T12 @ self.F @ self.T22.T @ self.T21.T
        ) @ np.linalg.pinv((self.T21 @ self.T22) @ self.I @ (self.T21 @ self.T22).T)
        self.T20 = np.where(self.T20 > 0, self.T20, 0)

        self.T21 = np.linalg.pinv(self.T20 @ np.linalg.inv(self.kf.Q) @ self.T20.T) @ (
            self.T20 @ self.A @ np.linalg.inv(self.kf.Q) @ self.T22.T
            - self.T20.T
            @ np.linalg.inv(self.kf.Q)
            @ self.T10
            @ self.T11
            @ self.T12
            @ self.F
            @ self.T22.T
            @ np.linalg.pinv(self.T22 @ self.I @ self.T22.T)
        )
        self.T21 = np.where(self.T21 > 0, self.T21, 0)

        self.T22 = (
            self.T21.T
            @ self.T20.T
            @ np.linalg.inv(self.kf.Q)
            @ self.T20
            @ self.T21
            @ (
                self.T21.T @ self.T20.T @ np.linalg.inv(self.kf.Q) @ self.A
                - self.T21.T
                @ self.T20.T
                @ np.linalg.inv(self.kf.Q)
                @ self.T10.T
                @ self.T11.T
                @ self.T12.T
                @ self.F
            )
            @ np.linalg.inv(self.I)
        )
        self.T22 = np.where(self.T22 > 0, self.T22, 0)

        self.D0 = (
            self.B
            @ self.D2.T
            @ self.D1.T
            @ np.linalg.pinv(self.D1 @ self.D2 @ self.Sigma @ self.D2.T @ self.D1.T)
        )
        self.D0 = np.where(self.D0 > 0, self.D0, 0)

        self.D1 = (
            np.linalg.pinv(self.D0.T @ np.linalg.inv(self.kf.R) @ self.D0)
            @ self.D0.T
            @ np.linalg.inv(self.kf.R)
            @ self.B
            @ self.D2.T
            @ np.linalg.pinv(self.D2 @ self.Sigma @ self.D2.T)
        )
        self.D1 = np.where(self.D1 > 0, self.D1, 0)

        self.D2 = (
            np.linalg.pinv(
                self.D1.T @ self.D0.T @ np.linalg.inv(self.kf.R) @ self.D0 @ self.D1
            )
            @ self.D1.T
            @ self.D0.T
            @ np.linalg.inv(self.kf.R)
            @ self.B
            @ np.linalg.inv(self.Sigma)
        )
        self.D2 = np.where(self.D2 > 0, self.D2, 0)

        T1 = self.T10 @ self.T11 @ self.T12
        T2 = self.T20 @ self.T21 @ self.T22
        D = self.D0 @ self.D1 @ self.D2

        self.kf.set_matrix(T1, T2, D)


class MStepOptim(nn.Module):
    def __init__(
        self,
        K: int,
        Q: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        D: torch.Tensor,
        R: torch.Tensor,
        Sigma: torch.Tensor,
        Phi: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A: torch.Tensor,
        F: torch.Tensor,
        I: torch.Tensor,
        Delta: torch.Tensor,
    ):
        super(MStepOptim, self).__init__()

        self.K = K
        self.Q = Q
        self.T1 = T1
        self.T2 = T2
        self.D = D
        self.R = R
        self.Sigma = nn.Parameter(Sigma)
        self.Phi = nn.Parameter(Phi)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.A = nn.Parameter(A)
        self.F = nn.Parameter(F)
        self.I = nn.Parameter(I)
        self.Delta = nn.Parameter(Delta)

    def forward(self):
        return -1 * (
            0.5
            * self.K
            * torch.trace(
                torch.matmul(
                    torch.inverse(self.Q),
                    self.Sigma
                    - self.T1.T @ self.C
                    - self.A @ self.T2.T
                    - self.T1 @ self.C.T
                    + (self.T1 @ self.Phi) @ self.T1.T
                    + (self.T1 @ self.F) @ self.T2.T
                    - self.T2 @ self.A.T
                    + (self.T2 @ self.F.T) @ self.T1.T
                    + (self.T2 @ self.I) @ self.T2.T,
                )
            )
            + 0.5
            * self.K
            * torch.trace(
                torch.inverse(self.R) @ self.Delta
                - self.B @ self.D.T
                - self.D @ self.B.T
                + (self.D @ self.Sigma) @ self.D.T
            )
        )


if __name__ == "__main__":
    model = Model(
        T10=np.eye(10),
        T11=np.eye(10),
        T12=np.eye(10),
        T20=np.eye(10),
        T21=np.eye(10),
        T22=np.random.random((10, 12)),
        D0=np.random.random((13, 10)),
        D1=np.eye(10),
        D2=np.eye(10),
        K=20,
    )
    print(model.kf.get_size())

    for i in range(100):
        model.E_step(
            np.random.standard_normal((20, 13)), np.random.standard_normal((20, 12))
        )

        model.M_step()
        print("m step")
