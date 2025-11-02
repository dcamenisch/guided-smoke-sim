import numpy as np
import numpy.typing as npt


class MACGrid3:
    def __init__(self, res_x: int, res_y: int, res_z: int, dx: float) -> None:
        self.dx = dx
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        self.x_data = np.zeros((res_z, res_y, res_x + 1))
        self.y_data = np.zeros((res_z, res_y + 1, res_x))
        self.z_data = np.zeros((res_z + 1, res_y, res_x))

    def x(self) -> npt.NDArray[np.float64]:
        return self.x_data

    def y(self) -> npt.NDArray[np.float64]:
        return self.y_data

    def z(self) -> npt.NDArray[np.float64]:
        return self.z_data

    def reset(self) -> None:
        self.x_data.fill(0.0)
        self.y_data.fill(0.0)
        self.z_data.fill(0.0)

    def getResolutionX(self) -> int:
        return self.res_x

    def getResolutionY(self) -> int:
        return self.res_y

    def getResolutionZ(self) -> int:
        return self.res_z

    def getGridSpacing(self) -> float:
        return self.dx
