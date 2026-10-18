"""
rendering.py - 渲染模块

负责将所有场景元素绘制到 matplotlib 画布上：
  - 曲线（彩色表面）
  - 相机（线段 + 光心 + 成像颜色条）
  - 重建结果（置信度热力图 + 提取的表面点）
  - CVC 可视化（论文核心：反投影条纹图案）
  - 选中网格点的多相机颜色详情
  - 准确度和完整度评估数据显示
"""

import numpy as np
import matplotlib.pyplot as plt


class Renderer:
    """
    渲染器

    管理 matplotlib 图形和子图布局，提供统一的绘制接口。
    """

    def __init__(self):
        """初始化渲染器，创建 2×2 四宫格布局"""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('SurfaceNet Demo - Multi-View Surface Reconstruction',
                          fontsize=14, fontweight='bold')

        # 2×2 布局
        self.axes = {
            'scene':    self.fig.add_subplot(2, 2, 1),  # 左上：场景
            'cvc':      self.fig.add_subplot(2, 2, 2),  # 右上：CVC可视化
            'heatmap':  self.fig.add_subplot(2, 2, 3),  # 左下：置信度热力图
            'surface':  self.fig.add_subplot(2, 2, 4),  # 右下：重建表面
        }

        self._setup_axes()

        # 被选中的网格点信息
        self.selected_point = None
        self.selected_detail = None
        self.cvc_camera_index = 0
        self._colorbar = None

        # 用于显示准确度和完整度的持久文本对象
        self.accuracy_text = None

        plt.tight_layout(pad=2.5)

    # ==================== 子图设置 ====================

    def _setup_axes(self):
        """设置各子图的标题、范围等"""
        titles = {
            'scene':   'Scene (Curve + Cameras)',
            'cvc':     'CVC: Colored Voxel Cube (View of Cam1)',
            'heatmap': 'Photo-Consistency Confidence',
            'surface': 'Reconstructed Surface',
        }
        for name, ax in self.axes.items():
            ax.set_title(titles[name], fontsize=11)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

    # ==================== 场景视图 ====================

    def draw_scene(self, scene):
        """左上图：场景视图"""
        ax = self.axes['scene']
        ax.clear()
        self._setup_single_axis(ax, 'Scene (Curve + Cameras)')

        curve = scene.curve

        # 1. 曲线
        ax.scatter(curve.sample_points[:, 0], curve.sample_points[:, 1],
                   c=curve.sample_colors, s=10, marker='o', zorder=5)

        # 2. 控制多边形
        cp = curve.control_points
        ax.plot(cp[:, 0], cp[:, 1], '--', color='gray',
                linewidth=1.5, alpha=0.7, zorder=4)
        ax.scatter(cp[:, 0], cp[:, 1], c='black', s=80,
                   marker='o', zorder=6)

        # 3. 相机
        for idx, cam in enumerate(scene.cameras):
            self._draw_camera(ax, cam, idx, scene)

        # 参考圆：以(0.5, 0.5)为圆心、半径0.5，供摆放相机参考
        from matplotlib.patches import Circle
        ref_circle = Circle((0.5, 0.5), 0.5, fill=False,
                            edgecolor='black', linewidth=0.8,
                            linestyle='-', alpha=0.3, zorder=1)
        ax.add_patch(ref_circle)

        ax.legend(loc='upper right', fontsize=7)

    def _draw_camera(self, ax, cam, idx, scene):
        """绘制单个相机"""
        # 光心
        ax.scatter(cam.center[0], cam.center[1],
                   c='red', s=60, marker='s', zorder=7)

        # 成像平面
        ax.plot([cam.endpoint1[0], cam.endpoint2[0]],
                [cam.endpoint1[1], cam.endpoint2[1]],
                '-', color='red', linewidth=2.5, zorder=6, alpha=0.8)

        # 朝向虚线
        ax.plot([cam.center[0], cam.image_center[0]],
                [cam.center[1], cam.image_center[1]],
                '--', color='red', linewidth=1, alpha=0.5, zorder=5)

        # 成像颜色条
        n_pixels = cam.resolution
        pixel_positions = []
        pixel_colors = []
        for px_idx in range(0, n_pixels, max(1, n_pixels // 50)):
            color = cam.image[px_idx]
            if np.any(color > 0.01):
                pixel_positions.append(cam.pixel_positions[px_idx])
                pixel_colors.append(color)
        if pixel_positions:
            pixel_positions = np.array(pixel_positions)
            ax.scatter(pixel_positions[:, 0], pixel_positions[:, 1],
                       c=np.array(pixel_colors), s=15, marker='s',
                       zorder=8, edgecolors='none')

        # 标签
        ax.text(cam.center[0] - 0.03, cam.center[1] + 0.03,
                f'Cam{idx+1}', fontsize=8, color='red',
                fontweight='bold', zorder=9)

    # ==================== CVC 可视化 ====================

    def draw_cvc(self, scene):
        """右上图：CVC 可视化"""
        ax = self.axes['cvc']
        ax.clear()

        cam_index = self.cvc_camera_index
        if cam_index >= len(scene.cameras):
            cam_index = 0

        cam = scene.cameras[cam_index]
        ax.set_title(f'CVC: View of Cam{cam_index+1} (反投影条纹)', fontsize=11)
        self._setup_single_axis(ax, f'CVC: View of Cam{cam_index+1}')

        recon = scene.reconstructor
        grid = recon.grid_points
        colors = np.zeros((len(grid), 3))

        for i, point in enumerate(grid):
            pixel_idx, hit = cam.project(point)
            if hit:
                colors[i] = cam.image[pixel_idx]
            else:
                colors[i] = [0, 0, 0]

        ax.scatter(grid[:, 0], grid[:, 1],
                   c=colors, s=10, marker='s',
                   edgecolors='none', alpha=0.8)

        curve = scene.curve
        ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1],
                'w--', linewidth=2, alpha=0.5, label='Ground Truth')

        ax.scatter(cam.center[0], cam.center[1],
                   c='red', s=80, marker='s', zorder=10, label='Camera')
        ax.plot([cam.center[0], cam.image_center[0]],
                [cam.center[1], cam.image_center[1]],
                'r-', linewidth=1, alpha=0.6)

        ax.legend(loc='upper right', fontsize=7)

    def set_cvc_camera(self, cam_index):
        """设置CVC视图用哪个相机"""
        self.cvc_camera_index = cam_index

    # ==================== 置信度热力图 ====================

    def draw_confidence(self, scene):
        """左下图：置信度热力图"""
        ax = self.axes['heatmap']
        ax.clear()
        self._setup_single_axis(ax, 'Photo-Consistency Confidence')

        recon = scene.reconstructor
        xx, yy, confidence = recon.get_confidence_heatmap()

        im = ax.pcolormesh(xx, yy, confidence,
                           cmap='hot', vmin=0, vmax=1,
                           shading='auto', alpha=0.8)

        curve = scene.curve
        ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1],
                'c-', linewidth=2, alpha=0.7, label='Ground Truth')

        if self.selected_point is not None:
            ax.scatter(self.selected_point[0], self.selected_point[1],
                       c='cyan', s=150, marker='o', edgecolors='white',
                       linewidths=2, zorder=10, label='Selected Point')

        if self._colorbar is None:
            self._colorbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
            self._colorbar.set_label('Confidence', fontsize=8)

        ax.legend(loc='upper right', fontsize=7)

    # ==================== 重建表面 ====================

    def draw_reconstruction(self, scene):
        """右下图：重建表面"""
        ax = self.axes['surface']
        ax.clear()
        self._setup_single_axis(ax, 'Reconstructed Surface')

        curve = scene.curve
        ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1],
                'c-', linewidth=2, alpha=0.5, label='Ground Truth')

        recon = scene.reconstructor
        if len(recon.surface_points) > 0:
            pts = np.array(recon.surface_points)
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=pts[:, 2], cmap='hot',
                       s=30, marker='o', edgecolors='black',
                       linewidths=0.5, vmin=0, vmax=1,
                       zorder=5, label='Reconstructed')

        if self.selected_point is not None:
            ax.scatter(self.selected_point[0], self.selected_point[1],
                       c='cyan', s=150, marker='o', edgecolors='white',
                       linewidths=2, zorder=10, label='Selected Point')

        ax.legend(loc='upper right', fontsize=7)

        # 显示准确度和完整度（在子图右侧的图外）
        if len(recon.surface_points) > 0:
            accuracy, completeness = scene.evaluate_reconstruction()
            if accuracy is not None:
                bbox = ax.get_position()
                x = bbox.x1 + 0.02
                y = bbox.y1

                if self.accuracy_text is None:
                    self.accuracy_text = self.fig.text(
                        x, y, '',
                        fontsize=9, fontfamily='monospace',
                        ha='left', va='top',
                        color='white',
                        bbox=dict(boxstyle='round',
                                  facecolor='#333333',
                                  alpha=0.85)
                    )

                self.accuracy_text.set_text(
                    f'Accuracy:      {accuracy:.4f}\nCompleteness: {completeness:.4f}'
                )
                self.accuracy_text.set_position((x, y))

    # ==================== 网格点查询信息显示 ====================

    def set_selected_point(self, point, detail):
        """设置当前选中的网格点及其详情"""
        self.selected_point = point
        self.selected_detail = detail

    def draw_point_info(self):
        """在热力图子图上叠加选中点的多相机颜色信息"""
        if self.selected_detail is None:
            return

        ax = self.axes['heatmap']

        cam_details = self.selected_detail['camera_details']
        text_lines = []
        for cd in cam_details:
            if cd['color'] is not None:
                r, g, b = cd['color']
                text_lines.append(
                    f"Cam{cd['cam_index']+1}: ({r:.2f}, {g:.2f}, {b:.2f})"
                )
            else:
                text_lines.append(f"Cam{cd['cam_index']+1}: (不可见)")

        conf = self.selected_detail['confidence']
        info_text = (
            f"Selected ({self.selected_point[0]:.3f}, {self.selected_point[1]:.3f})\n"
            f"Confidence: {conf:.3f}  |  Valid Cams: {self.selected_detail['num_valid']}\n"
            + "  |  ".join(text_lines)
        )

        ax.text(0.5, -0.15, info_text,
                transform=ax.transAxes, fontsize=8,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ==================== 完整更新 ====================

    def update(self, scene):
        """更新所有视图"""
        self.draw_scene(scene)
        self.draw_cvc(scene)
        self.draw_confidence(scene)
        self.draw_reconstruction(scene)
        self.draw_point_info()
        self.fig.canvas.draw_idle()

    # ==================== 辅助方法 ====================

    def _setup_single_axis(self, ax, title):
        """设置单个子图的基本属性"""
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

    def get_figure(self):
        """返回 matplotlib 图对象"""
        return self.fig