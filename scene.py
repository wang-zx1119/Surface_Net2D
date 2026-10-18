"""
scene.py - 场景管理

整合所有组件：
  - 一条彩色曲线（场景表面）
  - 多个相机（拍摄曲线）
  - 一个重建器（从相机成像重建表面）
"""

import numpy as np
from curve import Curve
from camera import Camera
from reconstruction import Reconstructor


class Scene:
    """
    场景管理器

    负责创建、管理和协调场景中的所有对象。
    提供统一的接口给交互层和渲染层使用。

    Attributes:
        curve: 彩色曲线对象
        cameras: 相机列表
        reconstructor: 重建器对象
    """

    def __init__(self, grid_resolution=50):
        """
        初始化场景

        Args:
            grid_resolution: 重建网格的分辨率
        """
        # 创建默认的 S 形曲线
        self.curve = Curve()

        # 创建多个默认相机
        self.cameras = []
        self._create_default_cameras()

        # 创建重建器
        self.reconstructor = Reconstructor(resolution=grid_resolution)

        # 对所有相机进行初始成像
        self._capture_all()

        # 执行初始重建
        self._reconstruct()

    # ==================== 默认相机设置 ====================

    def _create_default_cameras(self):
        """
        创建默认的相机阵列

        6个相机均匀环绕在圆上，自动朝向场景中心 (0.5, 0.5)。
        """
        center = np.array([0.5, 0.5])

        # 6个相机均匀环绕：角度间隔 60°
        angles = [0, 60, 120, 180, 240, 300]  # 度数

        for deg in angles:
            rad = np.deg2rad(deg)
            # 在距中心0.5的圆周上放置相机
            pos = center + 0.5 * np.array([np.cos(rad), np.sin(rad)])
            # 朝向中心
            angle = np.arctan2(center[1] - pos[1], center[0] - pos[0])
            cam = Camera(center=pos, angle=angle)
            self.cameras.append(cam)

    # ==================== 成像与重建 ====================

    def _capture_all(self):
        """
        让所有相机对当前曲线进行成像
        """
        for cam in self.cameras:
            cam.capture(self.curve)

    def _reconstruct(self):
        """
        执行表面重建
        """
        self.reconstructor.reconstruct(self.cameras)

    def update(self):
        """
        完整更新流程：
        1. 所有相机重新成像
        2. 重新运行重建
        """
        self._capture_all()
        self._reconstruct()

    # ==================== 曲线操作 ====================

    def update_curve_control_point(self, index, new_position):
        """
        更新曲线的控制点并触发重建

        Args:
            index: 控制点索引
            new_position: 新位置 (x, y)
        """
        self.curve.update_control_point(index, new_position)
        self.update()

    # ==================== 相机操作 ====================

    def add_camera(self, position=None):
        """
        添加一个新相机

        Args:
            position: 光心位置，默认为随机位置

        Returns:
            新创建的 Camera 对象
        """
        if position is None:
            position = np.array([
                np.random.uniform(0.05, 0.95),
                np.random.uniform(0.05, 0.95)
            ])

        # 朝向场景中心
        center = np.array([0.5, 0.5])
        angle = np.arctan2(center[1] - position[1], center[0] - position[0])

        cam = Camera(center=position, angle=angle)
        self.cameras.append(cam)
        cam.capture(self.curve)
        self._reconstruct()

        return cam

    def remove_camera(self, index):
        """
        移除指定索引的相机

        Args:
            index: 相机在列表中的索引

        Returns:
            被移除的 Camera 对象，如果索引无效则返回 None
        """
        if 0 <= index < len(self.cameras) and len(self.cameras) > 2:
            cam = self.cameras.pop(index)
            self._reconstruct()
            return cam
        return None

    def update_camera(self, index, new_center=None, new_angle=None):
        """
        更新相机的位置或朝向

        Args:
            index: 相机索引
            new_center: 新光心位置（可选）
            new_angle: 新朝向角度（可选）
        """
        if 0 <= index < len(self.cameras):
            cam = self.cameras[index]

            if new_center is not None:
                cam.update_position(new_center)
            if new_angle is not None:
                cam.update_angle(new_angle)

            cam.capture(self.curve)
            self._reconstruct()

    # ==================== 查询接口 ====================

    def find_curve_control_point(self, pos):
        """
        查找鼠标位置附近的曲线控制点

        Args:
            pos: 鼠标位置 (x, y)

        Returns:
            控制点索引，如果没找到返回 None
        """
        return self.curve.find_nearest_control_point(pos)

    def find_camera_by_image_plane(self, pos):
        """
        查找鼠标位置附近的相机成像平面

        Args:
            pos: 鼠标位置 (x, y)

        Returns:
            相机索引，如果没找到返回 None
        """
        for idx, cam in enumerate(self.cameras):
            if cam.contains_point(pos):
                return idx
        return None

    def find_camera_by_center(self, pos):
        """
        查找鼠标位置附近的相机光心

        Args:
            pos: 鼠标位置 (x, y)

        Returns:
            相机索引，如果没找到返回 None
        """
        for idx, cam in enumerate(self.cameras):
            if cam.find_near_center(pos):
                return idx
        return None

    def get_camera_count(self):
        """返回相机数量"""
        return len(self.cameras)

    # ==================== 重建参数 ====================

    def set_grid_resolution(self, resolution):
        """
        设置重建网格分辨率

        Args:
            resolution: 新的分辨率
        """
        self.reconstructor.set_resolution(resolution)
        self._reconstruct()

    def set_consistency_threshold(self, threshold):
        """
        设置颜色一致性阈值

        Args:
            threshold: 新的阈值
        """
        self.reconstructor.set_consistency_threshold(threshold)
        self._reconstruct()

    def apply_thinning(self):
        """对重建表面进行细化"""
        self.reconstructor.thin_surface(self.cameras)

    def evaluate_reconstruction(self):
        """评估当前重建质量"""
        return self.reconstructor.evaluate(self.curve.sample_points)

    def run_batch_experiment_geometry(self, num_curves=100):
        """
        分析一：几何复杂度对重建质量的影响（完全随机版本）

        随机生成 num_curves 条曲线（固定纹理为彩虹渐变），
        每条曲线用当前相机布局做重建，记录相机平均可见度、准确度和完整度。

        Args:
            num_curves: 曲线数量

        Returns:
            list: 每行包含 (相机平均可见度, 准确度, 完整度)
        """
        results = []

        for i in range(num_curves):
            # 随机生成曲线
            self.curve = Curve.random_curve()

            # 计算相机平均可见度
            avg_visibility = self.curve.compute_average_visibility(self.cameras)

            # 运行重建
            self._capture_all()
            self._reconstruct()

            # 评估
            accuracy, completeness = self.evaluate_reconstruction()

            if accuracy is not None:
                results.append((avg_visibility, accuracy, completeness))
                print(f"曲线 {i + 1}/{num_curves}: "
                      f"可见度={avg_visibility:.2f}, "
                      f"准确度={accuracy:.4f}, 完整度={completeness:.4f}")

        return results
    def run_batch_experiment_texture(self, hue_ranges=None, repeats=3):
        """
        分析二：纹理丰富度对重建质量的影响

        固定曲线形状为默认S形，在不同色相范围下重复重建，
        每个色相范围重复 repeats 次取平均。

        Args:
            hue_ranges: 色相范围列表，默认 [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]
            repeats: 每个梯度的重复次数

        Returns:
            dict: {hue_range: [(accuracy, completeness), ...]}
        """
        if hue_ranges is None:
            hue_ranges = [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]

        all_results = {}

        for hue_range in hue_ranges:
            results_for_this_hue = []

            for rep in range(repeats):
                # 使用默认S形控制点 + 指定色相范围创建曲线
                default_cp = np.array([
                    [0.10, 0.50],
                    [0.35, 0.15],
                    [0.65, 0.85],
                    [0.90, 0.50],
                ])
                self.curve = Curve.with_hue_range(
                    hue_range=hue_range,
                    control_points=default_cp
                )

                # 运行重建
                self._capture_all()
                self._reconstruct()

                # 评估
                accuracy, completeness = self.evaluate_reconstruction()

                if accuracy is not None:
                    results_for_this_hue.append((accuracy, completeness))
                    print(f"色相范围={hue_range:.2f} 第{rep+1}/{repeats}次: "
                          f"准确度={accuracy:.4f}, 完整度={completeness:.4f}")

            all_results[hue_range] = results_for_this_hue

        return all_results

    def start_auto_vary(self, renderer, interval=1500):
        """
        启动控制点自动随机变化

        Args:
            renderer: Renderer 对象，用于刷新界面
            interval: 变化间隔（毫秒）
        """
        import threading
        import time

        def vary_loop():
            while True:
                time.sleep(interval / 1000.0)
                new_cp = np.random.uniform(0.05, 0.95, (4, 2))
                for i in range(4):
                    self.curve.update_control_point(i, new_cp[i])
                self.update()
                # 通过 matplotlib 的主线程更新渲染
                if renderer is not None:
                    renderer.update(self)

        t = threading.Thread(target=vary_loop, daemon=True)
        t.start()