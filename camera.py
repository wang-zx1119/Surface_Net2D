"""
camera.py - 相机类

模拟论文中的多视角相机。
在2D世界中，每个相机是一个有方向的线段：
  - 线段位置 = 相机光心
  - 线段方向 = 相机朝向
  - 线段上的点 = 成像平面上的像素
"""

import numpy as np


class Camera:
    """
    相机类

    模拟一个2D世界中的"线阵相机"。
    相机由光心位置、朝向角度和视场大小定义。
    它可以将空间中的点投影到成像平面上，也可以对曲线"成像"。

    Attributes:
        center: 光心坐标 (cx, cy)，范围在 [0, 1]
        angle: 朝向角度（弧度），0 = 向右，π/2 = 向上
        fov: 视场角（弧度），决定成像平面的长度
        length: 成像平面的半长度（在归一化坐标系中）
        resolution: 成像分辨率（像素/采样点数量）
        image: 成像结果，形状为 (resolution, 3) 的 RGB 数组
    """

    def __init__(self, center, angle, fov=np.pi / 3, resolution=100):
        """
        初始化相机

        Args:
            center: 光心坐标 (cx, cy)
            angle: 朝向角度（弧度），相机朝向场景的方向
            fov: 视场角，默认 60°
            resolution: 成像分辨率（像素数量）
        """
        self.center = np.array(center, dtype=float)
        self.angle = angle
        self.fov = fov
        self.resolution = resolution

        # 成像平面的半长度（物理尺寸）
        # tan(fov/2) = length / 1.0（假设焦距为1）
        self.length = np.tan(fov / 2)

        # 初始化成像结果为黑色
        self.image = np.zeros((resolution, 3))

        # 计算成像平面的端点和像素位置（世界坐标系）
        self._compute_image_plane()

    # ==================== 成像平面计算 ====================

    def _compute_image_plane(self):
        """
        计算成像平面上所有像素点的世界坐标

        成像平面垂直于相机朝向，中心在 center + direction_vector
        （假设焦距=1，即成像平面在光心前方距离1处）
        """
        # 相机朝向的单位向量
        self.direction = np.array([
            np.cos(self.angle),
            np.sin(self.angle)
        ])

        # 垂直于朝向的向量（成像平面的方向）
        self.perp = np.array([
            -np.sin(self.angle),
            np.cos(self.angle)
        ])

        # 成像平面的中心点（光心前方距离1处）
        focal_distance = 1.0
        self.image_center = self.center + focal_distance * self.direction

        # 成像平面的两个端点
        self.endpoint1 = self.image_center - self.length * self.perp
        self.endpoint2 = self.image_center + self.length * self.perp

        # 所有像素点的世界坐标
        t = np.linspace(-self.length, self.length, self.resolution)
        self.pixel_positions = np.array([
            self.image_center + ti * self.perp
            for ti in t
        ])

    # ==================== 投影 ====================

    def project(self, point):
        """
        将空间中的一个点投影到成像平面上

        投影过程：
        1. 计算从光心到目标点的射线
        2. 求这条射线与成像平面的交点
        3. 返回交点在成像平面上的像素坐标

        Args:
            point: 空间中的点 (x, y)

        Returns:
            pixel_index: 像素索引 (0 ~ resolution-1)
            hit: 是否成功投影（点是否在成像平面范围内）

        如果点在相机后方或超出成像平面范围，返回 (None, False)
        """
        point = np.array(point, dtype=float)

        # 1. 计算从光心到目标点的向量
        ray = point - self.center
        ray_len = np.linalg.norm(ray)

        if ray_len < 1e-8:
            return None, False  # 点与光心重合

        ray_dir = ray / ray_len

        # 2. 检查点是否在相机前方（射线与朝向的点积 > 0）
        if np.dot(ray_dir, self.direction) <= 0:
            return None, False  # 点在相机后方

        # 3. 求射线与成像平面的交点
        #    射线: P = center + t * ray_dir
        #    成像平面: (P - image_center) · direction = 0  (法线是direction)
        #    代入: (center + t*ray_dir - image_center) · direction = 0
        #    t * (ray_dir · direction) = (image_center - center) · direction
        #    t = focal_distance / (ray_dir · direction)

        t = 1.0 / np.dot(ray_dir, self.direction)  # 焦距=1

        intersection = self.center + t * ray_dir

        # 4. 计算交点在成像平面上的位置（相对于中心）
        local_pos = np.dot(intersection - self.image_center, self.perp)

        # 5. 转换为像素坐标
        #    local_pos 范围是 [-length, +length]
        #    映射到 [0, resolution-1]
        pixel_f = (local_pos + self.length) / (2 * self.length) * (self.resolution - 1)
        pixel_index = int(round(pixel_f))

        # 6. 检查是否在有效范围内
        if 0 <= pixel_index < self.resolution:
            return pixel_index, True
        else:
            return None, False

    # ==================== 成像 ====================

    def capture(self, curve):
        """
        对一条彩色曲线进行"拍照"

        将曲线上的每个采样点投影到成像平面上，
        如果该位置还没有颜色，就记录它的颜色。

        这是一个简化模型：假设遇到遮挡时，离相机近的点覆盖远的点。

        Args:
            curve: Curve对象，需要有 sample_points 和 sample_colors 属性
        """
        # 重置成像为黑色
        self.image = np.zeros((self.resolution, 3))

        # 记录每个像素位置最近的交点距离（用于处理遮挡）
        depth_buffer = np.full(self.resolution, np.inf)

        # 对曲线上的每个采样点
        for i in range(curve.num_samples):
            point = curve.sample_points[i]
            color = curve.sample_colors[i]

            # 投影到成像平面
            pixel_index, hit = self.project(point)

            if not hit:
                continue  # 点不在成像平面范围内

            # 计算这个点到相机的距离
            distance = np.linalg.norm(point - self.center)

            # 遮挡处理：只记录最近的点
            if distance < depth_buffer[pixel_index]:
                depth_buffer[pixel_index] = distance
                self.image[pixel_index] = color

    # ==================== 交互接口 ====================

    def update_position(self, new_center):
        """
        更新相机光心位置

        Args:
            new_center: 新的光心坐标 (cx, cy)
        """
        self.center = np.array(new_center, dtype=float)
        self._compute_image_plane()

    def update_angle(self, new_angle):
        """
        更新相机朝向角度

        Args:
            new_angle: 新的朝向角度（弧度）
        """
        self.angle = new_angle
        self._compute_image_plane()

    def contains_point(self, point, threshold=0.03):
        """
        判断一个点是否在相机线段上（用于鼠标拖动）

        Args:
            point: 鼠标位置 (x, y)
            threshold: 距离阈值

        Returns:
            True 如果点到相机线段的距离小于阈值
        """
        # 相机线段的两个端点
        p1 = self.endpoint1
        p2 = self.endpoint2

        # 计算点到线段的距离
        v = p2 - p1
        w = np.array(point) - p1

        # 投影参数
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)

        if c1 <= 0:
            distance = np.linalg.norm(np.array(point) - p1)
        elif c2 <= c1:
            distance = np.linalg.norm(np.array(point) - p2)
        else:
            b = c1 / c2
            pb = p1 + b * v
            distance = np.linalg.norm(np.array(point) - pb)

        return distance < threshold

    def find_near_center(self, point, threshold=0.03):
        """
        判断鼠标是否在光心附近（用于拖动整个相机）

        Args:
            point: 鼠标位置 (x, y)
            threshold: 距离阈值

        Returns:
            True 如果点到光心的距离小于阈值
        """
        dist = np.linalg.norm(np.array(point) - self.center)
        return dist < threshold