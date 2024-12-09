import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def single_slit_diffraction(a, wavelength, z, x_range, y_range):
    """
    计算单缝夫琅禾费衍射光强分布
    参数:
    - a: 单缝宽度 (m)
    - wavelength: 光的波长 (m)
    - z: 传播距离 (m)
    - x_range, y_range: x 和 y 坐标范围 (m)
    返回:
    - X, Y: 网格坐标
    - I: 光强分布
    """
    x = np.linspace(-x_range, x_range, 500)  # x 方向的坐标
    y = np.linspace(-y_range, y_range, 500)  # y 方向的坐标
    X, Y = np.meshgrid(x, y)  # 构建网格

    # 计算光强分布，仅与 x 方向相关
    beta = (np.pi * a * X) / (wavelength * z)
    I = (np.sin(beta) / beta) ** 2
    I[np.isnan(I)] = 1  # 处理 beta = 0 的情况，使光强最大
    return X, Y, I

def update(val):
    """
    滑块更新函数，动态更新图像
    """
    a = slider_a.val * 1e-3  # 单缝宽度 (m)
    wavelength = slider_wavelength.val * 1e-9  # 光波长 (m)
    z = slider_z.val  # 传播距离 (m)

    # 重新计算光强分布
    X, Y, I = single_slit_diffraction(a, wavelength, z, x_range, y_range)
    center_line_intensity = I[int(I.shape[0] / 2), :]

    # 更新二维图像
    im.set_data(I)
    im.set_clim(0, I.max())

    # 更新相对强度分布图
    line.set_ydata(center_line_intensity / np.max(center_line_intensity))

    fig.canvas.draw_idle()

# 初始参数
initial_a = 0.1e-3  # 单缝宽度 0.1 mm
initial_wavelength = 500e-9  # 光波长 500 nm
initial_z = 0.6  # 传播距离 0.6 m
x_range, y_range = 0.01, 0.01  # 观察屏的坐标范围 (10 mm)

# 计算初始光强分布
X, Y, I = single_slit_diffraction(initial_a, initial_wavelength, initial_z, x_range, y_range)

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

# 左图：二维光强分布
im = ax[0].imshow(I, extent=[-x_range * 1e3, x_range * 1e3, -y_range * 1e3, y_range * 1e3],
                  cmap='jet', origin='lower')
ax[0].set_title(f"单缝衍射光强分布 (z = {initial_z*1e3:.0f} mm)")
ax[0].set_xlabel("x (mm)")
ax[0].set_ylabel("y (mm)")
fig.colorbar(im, ax=ax[0], label="光强")

# 右图：相对强度分布 (y = 0 截面)
center_line_intensity = I[int(I.shape[0] / 2), :]
x = np.linspace(-x_range, x_range, 500) * 1e3  # 转换为 mm
line, = ax[1].plot(x, center_line_intensity / np.max(center_line_intensity), color='red')
ax[1].set_title("x 轴上的相对强度分布 (y = 0)")
ax[1].set_xlabel("x (mm)")
ax[1].set_ylabel("相对强度 I/I_max")
ax[1].grid()

# 添加滑块
ax_a = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_wavelength = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_z = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_a = Slider(ax_a, '单缝宽度 (mm)', 0.05, 1.0, valinit=initial_a*1e3, valstep=0.01)
slider_wavelength = Slider(ax_wavelength, '波长 (nm)', 400, 700, valinit=initial_wavelength*1e9, valstep=10)
slider_z = Slider(ax_z, '传播距离 (m)', 0.1, 2.0, valinit=initial_z, valstep=0.1)

# 连接滑块与更新函数
slider_a.on_changed(update)
slider_wavelength.on_changed(update)
slider_z.on_changed(update)

plt.show()
