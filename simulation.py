import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.special import j1
import matplotlib as mpl

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class DiffractionSimulator:
    def __init__(self):
        self.wavelength = 632.8e-9  # 波长（米）
        self.k = 2 * np.pi / self.wavelength  # 波数
        self.aperture_size = 0.0002  # 通光孔尺寸（米）
        self.distance_to_screen = 1.0  # 通光孔到衍射屏的距离（米）
        self.distance_to_source = 0.1  # 光源到通光孔的距离（米）
        self.screen_size = 0.01  # 衍射屏尺寸（米）
        self.N = 1000  # 采样点数
        self.aperture_type = 'circular'  # 默认为圆形通光孔

        self.setup_plot()

    def calculate_diffraction(self):
        x = np.linspace(-self.screen_size / 2, self.screen_size / 2, self.N)
        y = np.linspace(-self.screen_size / 2, self.screen_size / 2, self.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)

        # 计算等效距离（考虑光源距离的影响）
        z_effective = (self.distance_to_source * self.distance_to_screen) / (
                    self.distance_to_source + self.distance_to_screen)

        if self.aperture_type == 'circular':
            # 圆形通光孔的菲涅尔衍射
            r = self.aperture_size / 2
            phase = self.k * (R ** 2) / (2 * z_effective)
            u = r * R * self.k / z_effective
            intensity = (np.pi * r ** 2) ** 2 * (2 * j1(u) / (u + 1e-10)) ** 2

        elif self.aperture_type == 'square':
            # 方形通光孔的菲涅尔衍射
            a = self.aperture_size / 2

            # 创建方形孔径函数
            aperture = np.zeros_like(X, dtype=complex)
            mask = (np.abs(X) <= a) & (np.abs(Y) <= a)
            aperture[mask] = 1

            # 计算衍射场
            phase = np.exp(1j * self.k * (X ** 2 + Y ** 2) / (2 * z_effective))
            field = aperture * phase

            # 使用FFT计算衍射图样
            field_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            intensity = np.abs(field_fft) ** 2

        elif self.aperture_type == 'triangle':
            # 三角形通光孔的菲涅尔衍射
            a = self.aperture_size

            # 创建三角形孔径函数
            aperture = np.zeros_like(X, dtype=complex)
            height = a * np.sqrt(3) / 2

            # 定义三角形区域
            y_top = height / 2
            y_bottom = -height / 2
            for i in range(len(X)):
                for j in range(len(Y)):
                    x, y = X[i, j], Y[i, j]
                    if (y <= y_top and y >= y_bottom):
                        x_bound = (height / 2 - np.abs(y)) / np.sqrt(3)
                        if np.abs(x) <= x_bound:
                            aperture[i, j] = 1

            # 计算衍射场
            phase = np.exp(1j * self.k * (X ** 2 + Y ** 2) / (2 * z_effective))
            field = aperture * phase

            # 使用FFT计算衍射图样
            field_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            intensity = np.abs(field_fft) ** 2

        # 归一化强度
        intensity = intensity / np.max(intensity)
        return intensity, x

    def setup_plot(self):
        self.fig = plt.figure(figsize=(15, 7))

        # 创建子图
        self.ax_2d = self.fig.add_subplot(121)  # 2D衍射图
        self.ax_1d = self.fig.add_subplot(122)  # X轴强度分布

        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.95)

        # 初始衍射图样
        intensity, x = self.calculate_diffraction()

        # 绘制2D衍射图
        extent_mm = [-self.screen_size / 2 * 1000, self.screen_size / 2 * 1000,
                     -self.screen_size / 2 * 1000, self.screen_size / 2 * 1000]
        self.im = self.ax_2d.imshow(intensity, cmap='viridis',
                                    extent=extent_mm,
                                    interpolation='bilinear')
        self.ax_2d.set_xlabel('x (mm)')
        self.ax_2d.set_ylabel('y (mm)')
        self.ax_2d.set_title('菲涅尔衍射图样')

        # 绘制X轴强度分布
        center_index = intensity.shape[0] // 2
        x_intensity = intensity[center_index, :]
        self.line, = self.ax_1d.plot(x * 1000, x_intensity, 'b-', linewidth=2)
        self.ax_1d.set_xlabel('x (mm)')
        self.ax_1d.set_ylabel('相对光强')
        self.ax_1d.set_title('X轴强度分布')
        self.ax_1d.grid(True)
        self.ax_1d.set_ylim(0, 1.1)

        # 添加颜色条
        cbar = plt.colorbar(self.im, ax=self.ax_2d, label='相对光强')
        cbar.ax.tick_params(labelsize=10)

        # 添加滑动条
        ax_distance_screen = plt.axes([0.1, 0.15, 0.65, 0.03])
        ax_distance_source = plt.axes([0.1, 0.1, 0.65, 0.03])
        ax_size = plt.axes([0.1, 0.2, 0.65, 0.03])

        self.slider_distance_screen = Slider(ax_distance_screen, '通光孔到屏幕距离 (m)', 0.1, 10,
                                             valinit=self.distance_to_screen)
        self.slider_distance_source = Slider(ax_distance_source, '光源到通光孔距离 (m)', 0.01, 0.5,
                                             valinit=self.distance_to_source)
        self.slider_size = Slider(ax_size, '通光孔尺寸 (mm)', 0.05, 2.0,
                                  valinit=self.aperture_size * 1000)

        # 添加通光孔形状选择按钮
        ax_radio = plt.axes([0.8, 0.05, 0.15, 0.15])
        self.radio = RadioButtons(ax_radio, ('圆形', '方形', '三角形'))
        self.shape_map = {'圆形': 'circular', '方形': 'square', '三角形': 'triangle'}

        # 注册回调函数
        self.slider_distance_screen.on_changed(self.update)
        self.slider_distance_source.on_changed(self.update)
        self.slider_size.on_changed(self.update)
        self.radio.on_clicked(self.set_aperture_type)

    def update(self, val):
        self.distance_to_screen = self.slider_distance_screen.val
        self.distance_to_source = self.slider_distance_source.val
        self.aperture_size = self.slider_size.val / 1000
        intensity, x = self.calculate_diffraction()

        # 更新2D图
        self.im.set_data(intensity)

        # 更新1D图
        center_index = intensity.shape[0] // 2
        x_intensity = intensity[center_index, :]
        self.line.set_ydata(x_intensity)
        self.ax_1d.relim()
        self.ax_1d.autoscale_view()

        self.fig.canvas.draw_idle()

    def set_aperture_type(self, label):
        self.aperture_type = self.shape_map[label]
        intensity, x = self.calculate_diffraction()

        # 更新2D图
        self.im.set_data(intensity)

        # 更新1D图
        center_index = intensity.shape[0] // 2
        x_intensity = intensity[center_index, :]
        self.line.set_ydata(x_intensity)
        self.ax_1d.relim()
        self.ax_1d.autoscale_view()

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == '__main__':
    simulator = DiffractionSimulator()
    simulator.show()