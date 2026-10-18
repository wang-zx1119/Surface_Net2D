"""
reconstruction.py - 重建逻辑

模拟 SurfaceNet 论文的核心思想：
  1. 创建一个 2D 网格（对应论文中的 3D 体素空间）
  2. 对每个网格点，投影到各个相机，取对应像素的颜色
  3. 比较各个相机看到的颜色是否一致（photo-consistency）
  4. 颜色越一致，该点越可能在真实表面上
  5. 提取高置信度的点作为重建结果
"""

import numpy as np


class Reconstructor:
    """
    多视角重建器

    在 2D 网格上，利用多个相机的成像结果，
    通过颜色一致性来判断每个网格点是否在表面上。

    Attributes:
        resolution: 网格分辨率（resolution × resolution 个网格点）
        grid_points: 网格点坐标，形状为 (resolution*resolution, 2)
        confidence: 每个网格点的置信度，形状为 (resolution, resolution)
        surface_mask: 二值表面掩码，形状为 (resolution, resolution)
        surface_points: 提取出的表面点坐标列表
    """

    def __init__(self, resolution=50, consistency_threshold=0.1):
        """
        初始化重建器

        Args:
            resolution: 网格分辨率，数字越大越精细但计算越慢
            consistency_threshold: 颜色一致性阈值，低于此值认为"颜色一致"
        """
        self.resolution = resolution
        self.consistency_threshold = consistency_threshold

        # 创建均匀网格（坐标范围 [0, 1]）
        self._create_grid()

        # 重建结果
        self.confidence = np.zeros((resolution, resolution))
        self.surface_mask = np.zeros((resolution, resolution), dtype=bool)
        self.surface_points = []

    # ==================== 网格创建 ====================

    def _create_grid(self):
        """
        创建均匀的 2D 网格

        生成 resolution × resolution 个网格点，
        均匀分布在 [0, 1] × [0, 1] 的坐标空间中。
        """
        xs = np.linspace(0, 1, self.resolution)
        ys = np.linspace(0, 1, self.resolution)
        xx, yy = np.meshgrid(xs, ys)
        self.grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        self.xs = xs
        self.ys = ys
        self.xx = xx
        self.yy = yy

    # ==================== 颜色查询 ====================

    def _get_color_from_camera(self, point, camera):
        """
        从一个相机获取空间中某个点的颜色

        Args:
            point: 空间中的点 (x, y)
            camera: Camera 对象

        Returns:
            RGB 颜色 (R, G, B)，如果投影失败则返回 None
        """
        pixel_index, hit = camera.project(point)
        if not hit:
            return None
        return camera.image[pixel_index]

    # ==================== 颜色一致性计算 ====================

    def _compute_color_consistency(self, point, cameras):
        """
        计算多个相机在同一空间点看到的颜色是否一致

        方法：
        1. 对每个相机，投影该点并获取对应像素颜色
        2. 排除纯黑色背景（相机没拍到任何东西的情况）
        3. 对成功获取的有效颜色，计算两两之间的差异
        4. 取平均差异作为"不一致度"（越低越好）

        Args:
            point: 空间中的点 (x, y)
            cameras: Camera 对象列表

        Returns:
            inconsistency: 颜色不一致度（0 = 完全一致，越大越不一致）
            num_valid: 成功投影且拍到有效颜色的相机数量
            valid_colors: 有效颜色列表
        """
        valid_colors = []

        for cam in cameras:
            color = self._get_color_from_camera(point, cam)
            # 过滤条件：颜色存在 且 不是纯黑色背景（RGB之和 > 0.01）
            if color is not None and np.sum(color) > 0.01:
                valid_colors.append(color)

        num_valid = len(valid_colors)

        # 至少需要 2 个相机拍到有效颜色才能判断一致性
        if num_valid < 2:
            return 1.0, num_valid, valid_colors

        # 计算所有有效颜色两两之间的差异
        total_diff = 0.0
        count = 0

        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                diff = np.linalg.norm(valid_colors[i] - valid_colors[j])
                total_diff += diff
                count += 1

        avg_inconsistency = total_diff / count if count > 0 else 1.0

        return avg_inconsistency, num_valid, valid_colors

    # ==================== 表面重建 ====================

    def reconstruct(self, cameras, min_valid_cameras=2):
        """
        执行表面重建

        对网格中的每个点：
        1. 计算各相机看到的颜色一致性
        2. 将不一致度转换为置信度（一致度越高，置信度越高）
        3. 提取高置信度的点作为表面

        Args:
            cameras: Camera 对象列表
            min_valid_cameras: 最少需要几个相机拍到有效颜色

        Returns:
            confidence: 置信度图，形状为 (resolution, resolution)
        """
        self.confidence = np.zeros((self.resolution, self.resolution))

        for idx, point in enumerate(self.grid_points):
            inconsistency, num_valid, _ = self._compute_color_consistency(point, cameras)

            if num_valid < min_valid_cameras:
                confidence_val = 0.0
            else:
                sigma = self.consistency_threshold
                confidence_val = np.exp(-inconsistency ** 2 / (2 * sigma ** 2))

            row = idx // self.resolution
            col = idx % self.resolution
            self.confidence[row, col] = confidence_val

        self._extract_surface()
        return self.confidence

    # ==================== 表面提取 ====================

    def _extract_surface(self, threshold=0.5):
        """
        从置信度图中提取表面

        步骤（对应论文中的 binarization）：
        1. 阈值化：置信度 > threshold 的网格点标记为表面
        2. 收集表面点的坐标

        Args:
            threshold: 置信度阈值
        """
        self.surface_mask = self.confidence > threshold

        self.surface_points = []
        for row in range(self.resolution):
            for col in range(self.resolution):
                if self.surface_mask[row, col]:
                    x = self.xs[col]
                    y = self.ys[row]
                    c = self.confidence[row, col]
                    self.surface_points.append((x, y, c))

    # ==================== 非极大值抑制（对应论文的 thinning） ====================

    def thin_surface(self, cameras):
        """
        对重建表面进行细化

        对应论文中的 "ray pooling" 操作：
        沿每个相机的投影射线，只保留置信度最高的那个点。

        Args:
            cameras: Camera 对象列表
        """
        if len(self.surface_points) == 0:
            return

        new_mask = np.zeros_like(self.surface_mask)

        for cam in cameras:
            for pixel_idx in range(cam.resolution):
                candidates = []

                for x, y, conf in self.surface_points:
                    point = np.array([x, y])
                    proj_idx, hit = cam.project(point)

                    if hit and proj_idx == pixel_idx:
                        candidates.append((x, y, conf))

                if len(candidates) > 0:
                    best = max(candidates, key=lambda c: c[2])
                    col = np.argmin(np.abs(self.xs - best[0]))
                    row = np.argmin(np.abs(self.ys - best[1]))
                    new_mask[row, col] = True

        self.surface_mask = new_mask
        self.surface_points = []
        for row in range(self.resolution):
            for col in range(self.resolution):
                if self.surface_mask[row, col]:
                    x = self.xs[col]
                    y = self.ys[row]
                    c = self.confidence[row, col]
                    self.surface_points.append((x, y, c))

    # ==================== 参数调整接口 ====================

    def set_consistency_threshold(self, threshold):
        """设置颜色一致性阈值"""
        self.consistency_threshold = threshold

    def set_resolution(self, resolution):
        """设置网格分辨率并重建网格"""
        self.resolution = resolution
        self.confidence = np.zeros((resolution, resolution))
        self.surface_mask = np.zeros((resolution, resolution), dtype=bool)
        self.surface_points = []
        self._create_grid()

    def get_confidence_heatmap(self):
        """获取置信度热力图数据，用于可视化"""
        return self.xx, self.yy, self.confidence

    # ==================== 网格点详情查询 ====================

    def get_point_detail(self, point, cameras):
        """
        获取某个空间点在各个相机中的颜色详情

        Args:
            point: 空间中的点 (x, y)
            cameras: Camera 对象列表

        Returns:
            dict: 包含置信度、有效相机数和各相机详情
        """
        inconsistency, num_valid, valid_colors = self._compute_color_consistency(point, cameras)

        if num_valid >= 2:
            sigma = self.consistency_threshold
            confidence = np.exp(-inconsistency ** 2 / (2 * sigma ** 2))
        else:
            confidence = 0.0

        camera_details = []
        for idx, cam in enumerate(cameras):
            pixel_idx, hit = cam.project(point)
            if hit:
                color = cam.image[pixel_idx]
            else:
                pixel_idx = None
                color = None
            camera_details.append({
                'cam_index': idx,
                'pixel_index': pixel_idx,
                'color': color,
            })

        return {
            'confidence': confidence,
            'num_valid': num_valid,
            'camera_details': camera_details,
        }

    def evaluate(self, curve_sample_points):
        """
        评估重建质量：计算准确度和完整度

        准确度：每个重建表面点到最近曲线采样点的平均距离（越小越好）
        完整度：每个曲线采样点到最近重建表面点的平均距离（越小越好）

        Args:
            curve_sample_points: 真实曲线的采样点坐标，形状为 (N, 2)

        Returns:
            (accuracy, completeness): 两个浮点数
        """
        if len(self.surface_points) == 0:
            return None, None

        # 提取重建表面点的坐标（去掉置信度）
        recon_pts = np.array([[p[0], p[1]] for p in self.surface_points])
        curve_pts = np.array(curve_sample_points)

        # 准确度：每个重建点到最近曲线点的距离
        acc_distances = []
        for rp in recon_pts:
            dists = np.linalg.norm(curve_pts - rp, axis=1)
            acc_distances.append(np.min(dists))
        accuracy = np.mean(acc_distances)

        # 完整度：每个曲线点到最近重建点的距离
        comp_distances = []
        for cp in curve_pts:
            dists = np.linalg.norm(recon_pts - cp, axis=1)
            comp_distances.append(np.min(dists))
        completeness = np.mean(comp_distances)

        return accuracy, completeness