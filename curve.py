"""
curve.py - 彩色曲线类

模拟论文中的"三维场景表面"。
在2D世界中，这是一条带颜色的贝塞尔曲线，可以被拖动变形。
默认采用有自遮挡的S形曲线。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


class Curve:
    """
    彩色曲线类

    用三次贝塞尔曲线来表示场景中的"表面"。
    曲线上的每个点都有颜色，颜色沿曲线渐变（红→橙→黄→绿→蓝→紫）。

    Attributes:
        control_points: 控制点列表，形状为 (4, 2)
        num_samples: 曲线上的采样点数量
        sample_points: 采样点坐标，形状为 (num_samples, 2)
        sample_colors: 采样点颜色，形状为 (num_samples, 3)，范围0~1
    """

    def __init__(self, control_points=None, num_samples=200):
        """
        初始化曲线

        Args:
            control_points: 控制点坐标，默认为有自遮挡的S形曲线
            num_samples: 采样点数量
        """
        self.num_samples = num_samples

        # 默认控制点：有自遮挡的S形曲线
        # 曲线向前上方凸起，再向下凹陷，最后回到中间
        # 这样某些部分从特定角度是看不到的（遮挡）
        if control_points is None:
            self.control_points = np.array([
                [0.10, 0.50],   # P0: 起点，左中
                [0.35, 0.15],   # P1: 向上拉（形成前凸）
                [0.65, 0.85],   # P2: 向下拉（形成凹陷）
                [0.90, 0.50],   # P3: 终点，右中
            ])
        else:
            self.control_points = np.array(control_points)

        # 初始化采样点和颜色
        self.sample_points = np.zeros((num_samples, 2))
        self.sample_colors = np.zeros((num_samples, 3))

        # 计算采样点和颜色
        self._resample()

    # ==================== 贝塞尔曲线计算 ====================

    def _bezier(self, t):
        """
        计算三次贝塞尔曲线上参数t处的点

        公式：B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3

        Args:
            t: 参数，范围 [0, 1]，可以是单个值或数组

        Returns:
            曲线上对应点的坐标，形状为 (len(t), 2)
        """
        t = np.asarray(t)
        P0, P1, P2, P3 = self.control_points

        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3

        x = b0 * P0[0] + b1 * P1[0] + b2 * P2[0] + b3 * P3[0]
        y = b0 * P0[1] + b1 * P1[1] + b2 * P2[1] + b3 * P3[1]

        return np.column_stack([x, y])

    # ==================== 颜色映射 ====================

    def _colormap(self, t):
        """
        根据参数t生成彩虹色渐变

        Args:
            t: 参数，范围 [0, 1]

        Returns:
            RGB颜色，形状为 (len(t), 3)，范围0~1
        """
        t = np.asarray(t)
        hue = 0.8 * t  # 红(0) → 紫(0.8)

        hsv = np.zeros((len(t), 3))
        hsv[:, 0] = hue
        hsv[:, 1] = 1.0
        hsv[:, 2] = 1.0

        return hsv_to_rgb(hsv)
    # ==================== 采样 ====================

    def _resample(self):
        """对曲线进行均匀采样，计算每个采样点的坐标和颜色"""
        t = np.linspace(0, 1, self.num_samples)
        self.sample_points = self._bezier(t)
        self.sample_colors = self._colormap(t)

    # ==================== 公共接口 ====================

    def update_control_point(self, index, new_position):
        """更新指定控制点的位置并重新采样"""
        self.control_points[index] = new_position
        self._resample()

    def get_point_at(self, t):
        """获取曲线上参数t处的坐标"""
        return self._bezier(np.array([t]))[0]

    def get_color_at(self, t):
        """获取曲线上参数t处的颜色"""
        return self._colormap(np.array([t]))[0]

    def find_nearest_control_point(self, pos, threshold=0.05):
        """查找距离pos最近的控制点"""
        distances = np.linalg.norm(self.control_points - pos, axis=1)
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] < threshold:
            return nearest_idx
        return None

    @classmethod
    def random_curve(cls, num_samples=200):
        """
        随机生成一条曲线，控制点坐标在 [0.05, 0.95] 范围内

        Args:
            num_samples: 采样点数量

        Returns:
            Curve 对象
        """
        cp = np.random.uniform(0.05, 0.95, (4, 2))
        return cls(control_points=cp, num_samples=num_samples)

    def compute_average_visibility(self, cameras):
        """
        计算曲线的相机平均可见度，作为几何复杂度的量化指标

        对曲线上每个采样点，统计有多少个相机能成功投影到该点，
        取所有采样点的平均值。值越低说明自遮挡越严重、几何越复杂。

        Args:
            cameras: Camera 对象列表

        Returns:
            float: 平均可见相机数（范围 0 ~ 相机总数）
        """
        total_visible = 0

        for point in self.sample_points:
            visible_count = 0
            for cam in cameras:
                _, hit = cam.project(point)
                if hit:
                    visible_count += 1
            total_visible += visible_count

        return total_visible / self.num_samples

    @classmethod
    def with_hue_range(cls, hue_range, control_points=None, num_samples=200):
        """
        创建一条指定色相变化范围的曲线

        Args:
            hue_range: 色相变化范围，0.01=几乎全红，0.80=红到紫彩虹
            control_points: 控制点坐标，默认使用S形曲线
            num_samples: 采样点数量

        Returns:
            Curve 对象，其 _colormap 方法使用指定的 hue_range
        """
        curve = cls(control_points=control_points, num_samples=num_samples)
        curve.hue_range = hue_range
        # 重新采样，使用新的色相范围
        curve._resample_with_hue_range()
        return curve

    def _resample_with_hue_range(self):
        """使用 self.hue_range 重新采样颜色"""
        t = np.linspace(0, 1, self.num_samples)
        self.sample_points = self._bezier(t)
        # 使用 self.hue_range 而不是默认的 0.8
        hue = self.hue_range * t
        hsv = np.zeros((len(t), 3))
        hsv[:, 0] = hue
        hsv[:, 1] = 1.0
        hsv[:, 2] = 1.0
        self.sample_colors = hsv_to_rgb(hsv)