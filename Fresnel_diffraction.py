import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

class DiffractionSimulator:
    def __init__(self):
        self.wavelength = 632.8e-9  # 波长（m）
        self.k = 2 * np.pi / self.wavelength  # 波数
        self.aperture_size = 0.001  # 孔径大小（m）
        self.z1 = 1.0  # 光源距离（m）
        self.z2 = 1.0  # 观察屏距离（m）
        self.aperture_type = 'circular'
        self.N = 512  # 网格分辨率
        self.setup_plot()

    def setup_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.35)

        # 添加滑块
        axsize = plt.axes([0.2, 0.2, 0.3, 0.03])
        axz2 = plt.axes([0.2, 0.15, 0.3, 0.03])
        self.size_slider = Slider(axsize, '孔径大小 (mm)', 0.1, 5.0, valinit=1.0)
        self.z2_slider = Slider(axz2, '观察屏距离 (m)', 0.1, 5.0, valinit=1.0)

        # 添加形状和类型选择按钮
        rax = plt.axes([0.7, 0.1, 0.15, 0.15])
        self.radio_shape = RadioButtons(rax, ('圆形', '方形', '三角形'))
        rax2 = plt.axes([0.85, 0.1, 0.15, 0.15])
        self.radio_type = RadioButtons(rax2, ('夫琅禾费', '菲涅尔'))

        # 绑定回调
        self.size_slider.on_changed(self.update)
        self.z2_slider.on_changed(self.update)
        self.radio_shape.on_clicked(self.update_shape)
        self.radio_type.on_clicked(self.update_type)
        self.update(None)

    def create_aperture(self, X, Y):
        if self.aperture_type == 'circular':
            return (X**2 + Y**2) <= (self.aperture_size / 2)**2
        elif self.aperture_type == 'square':
            return (np.abs(X) <= self.aperture_size / 2) & (np.abs(Y) <= self.aperture_size / 2)
        elif self.aperture_type == 'triangle':
            return (Y >= -X - self.aperture_size / 2) & (Y >= X - self.aperture_size / 2) & (Y <= self.aperture_size / 2)

    def fresnel_diffraction(self, aperture):
        fx = np.fft.fftfreq(self.N, d=1/self.N)
        fy = np.fft.fftfreq(self.N, d=1/self.N)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(-1j * np.pi * self.wavelength * self.z2 * (FX**2 + FY**2))
        U = np.fft.fftshift(np.fft.fft2(aperture))
        U = U * H
        return np.abs(np.fft.ifft2(U))**2

    def fraunhofer_diffraction(self, aperture):
        U = np.fft.fftshift(np.fft.fft2(aperture))
        return np.abs(U)**2

    def update(self, val):
        self.aperture_size = self.size_slider.val / 1000
        self.z2 = self.z2_slider.val
        x = np.linspace(-1, 1, self.N)
        y = np.linspace(-1, 1, self.N)
        X, Y = np.meshgrid(x, y)
        aperture = self.create_aperture(X, Y)

        if self.radio_type.value_selected == '夫琅禾费':
            intensity = self.fraunhofer_diffraction(aperture)
        else:
            intensity = self.fresnel_diffraction(aperture)

        self.ax1.clear()
        self.ax1.imshow(aperture, cmap='gray', extent=(-1, 1, -1, 1))
        self.ax1.set_title("透光孔形状")
        self.ax2.clear()
        self.ax2.imshow(intensity, cmap='inferno', extent=(-1, 1, -1, 1))
        self.ax2.set_title("衍射图样")
        plt.draw()

    def update_shape(self, label):
        self.aperture_type = {'圆形': 'circular', '方形': 'square', '三角形': 'triangle'}[label]
        self.update(None)

    def update_type(self, label):
        self.update(None)

    def show(self):
        plt.ion()  # 打开交互模式
        plt.show()


if __name__ == '__main__':
    sim = DiffractionSimulator()
    sim.show()
