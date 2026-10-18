"""
interaction.py - 交互处理

支持的操作：
  - 左键拖动控制点：改变曲线形状
  - 左键拖动光心：移动相机（朝向不变）
  - 右键拖动成像平面：旋转相机（位置不变）
  - 中键点击：添加相机
  - A键：随机添加相机
  - D键：删除选中的相机
  - T键：表面细化
  - R键：重置场景
  - 滚轮：调整一致性阈值
  - 左键点击网格点：查看该点多相机颜色详情
"""

import numpy as np


class InteractionHandler:
    """交互处理器"""

    def __init__(self, scene, renderer):
        self.scene = scene
        self.renderer = renderer
        self.fig = renderer.get_figure()

        self.drag_target = None
        self.is_dragging = False
        self.selected_camera = None
        self.selected_grid_point = None  # 被点击查询的网格点

        self._connect_events()

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    # ==================== 鼠标按下 ====================

    def _on_press(self, event):
        if event.inaxes is None:
            return
        if self.fig.canvas.toolbar.mode != '':
            return

        pos = np.array([event.xdata, event.ydata])
        ax_name = self._get_axis_name(event.inaxes)

        if event.button == 1:  # 左键
            self._handle_left_press(pos, ax_name)
        elif event.button == 2:  # 中键
            self._handle_middle_press(pos)
        elif event.button == 3:  # 右键
            self._handle_right_press(pos)

    def _get_axis_name(self, inaxes):
        """判断点击的是哪个子图"""
        for name, ax in self.renderer.axes.items():
            if inaxes == ax:
                return name
        return None

    def _handle_left_press(self, pos, ax_name):
        # 1. 曲线控制点（在场景视图中）
        if ax_name == 'scene':
            cp_idx = self.scene.find_curve_control_point(pos)
            if cp_idx is not None:
                self.drag_target = {'type': 'curve_control_point', 'index': cp_idx}
                self.is_dragging = True
                self.selected_camera = None
                self.renderer.update(self.scene)
                return

            # 2. 相机光心 → 选中并拖动（只改位置）
            cam_idx = self.scene.find_camera_by_center(pos)
            if cam_idx is not None:
                self.selected_camera = cam_idx
                self.drag_target = {'type': 'camera_center', 'index': cam_idx}
                self.is_dragging = True
                self.renderer.update(self.scene)
                return

            # 3. 相机成像平面 → 选中
            cam_idx = self.scene.find_camera_by_image_plane(pos)
            if cam_idx is not None:
                self.selected_camera = cam_idx
                self.renderer.update(self.scene)
                return

            # 4. 点击空白 → 取消选中
            self.selected_camera = None
            self.renderer.update(self.scene)

        # 5. 在热力图或重建视图点击 → 查询该点多相机颜色详情
        if ax_name in ('heatmap', 'surface'):
            self._query_grid_point(pos)

    def _handle_middle_press(self, pos):
        """中键：添加相机"""
        self.scene.add_camera(position=pos)
        self.selected_camera = len(self.scene.cameras) - 1
        self.renderer.update(self.scene)

    def _handle_right_press(self, pos):
        """右键：旋转相机朝向"""
        cam_idx = self.scene.find_camera_by_image_plane(pos)
        if cam_idx is not None:
            self.selected_camera = cam_idx
            cam = self.scene.cameras[cam_idx]
            self.drag_target = {
                'type': 'camera_rotate',
                'index': cam_idx,
                'start_angle': cam.angle,
                'start_pos': pos,
            }
            self.is_dragging = True
            self.renderer.update(self.scene)

    # ==================== 鼠标释放 ====================

    def _on_release(self, event):
        self.is_dragging = False
        self.drag_target = None

    # ==================== 鼠标移动 ====================

    def _on_motion(self, event):
        if not self.is_dragging or self.drag_target is None:
            return
        if event.inaxes is None:
            return

        pos = np.array([event.xdata, event.ydata])
        target = self.drag_target

        if target['type'] == 'curve_control_point':
            self.scene.update_curve_control_point(target['index'], pos)
            self.renderer.update(self.scene)

        elif target['type'] == 'camera_center':
            # 改位置，同时自动朝向场景中心
            center = np.array([0.5, 0.5])
            new_angle = np.arctan2(center[1] - pos[1], center[0] - pos[0])
            self.scene.update_camera(target['index'],
                                     new_center=pos,
                                     new_angle=new_angle)
            self.renderer.update(self.scene)

        elif target['type'] == 'camera_rotate':
            # 只改朝向，位置保持不变
            cam = self.scene.cameras[target['index']]
            dx = pos[0] - cam.center[0]
            dy = pos[1] - cam.center[1]
            new_angle = np.arctan2(dy, dx)
            self.scene.update_camera(target['index'], new_angle=new_angle)
            self.renderer.update(self.scene)

    # ==================== 键盘事件 ====================

    # ==================== 键盘事件 ====================

    def _on_key(self, event):
        # ---- 数字键：切换CVC视图 ----
        if event.key in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            cam_idx = int(event.key) - 1
            total = self.scene.get_camera_count()
            if cam_idx < total:
                self.renderer.set_cvc_camera(cam_idx)
                self.renderer.update(self.scene)
                print(f"CVC 视图切换为 Cam{cam_idx+1}（共{total}个相机）")
            else:
                print(f"❌ 无法切换：当前只有 {total} 个相机，Cam{cam_idx+1} 不存在")

        # ---- A键：随机添加相机 ----
        elif event.key == 'a':
            self.scene.add_camera()
            self.selected_camera = len(self.scene.cameras) - 1
            self.renderer.update(self.scene)
            total = self.scene.get_camera_count()
            print(f"✅ 添加相机，当前共 {total} 个相机")
            if total <= 9:
                print(f"   按数字键 {total} 可在CVC图中查看该相机视角")

        # ---- D键：删除选中的相机 ----
        elif event.key == 'd':
            if self.selected_camera is not None:
                if self.scene.get_camera_count() > 2:
                    idx = self.selected_camera
                    self.scene.remove_camera(idx)
                    self.selected_camera = None
                    self.renderer.update(self.scene)
                    total = self.scene.get_camera_count()
                    print(f"🗑️  删除相机 #{idx+1}，当前共 {total} 个相机")
                else:
                    print("❌ 至少保留 2 个相机")
            else:
                print("❌ 请先点击一个相机来选中它（点击红色方块或红色线段）")

        # ---- T键：表面细化 ----
        elif event.key == 't':
            print("🔧 执行表面细化 (Thinning)...")
            self.scene.apply_thinning()
            self.renderer.update(self.scene)
            print("✅ 细化完成")

        # ---- R键：重置场景 ----
        elif event.key == 'r':
            print("🔄 重置场景...")
            from scene import Scene
            new_scene = Scene()
            self.scene.curve = new_scene.curve
            self.scene.cameras = new_scene.cameras
            self.scene.reconstructor = new_scene.reconstructor
            self.scene.update()
            self.selected_camera = None
            self.selected_grid_point = None
            self.renderer.update(self.scene)
            print("✅ 场景已重置")

    # ==================== 滚轮 ====================

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        current = self.scene.reconstructor.consistency_threshold
        delta = 0.01
        if event.button == 'up':
            new_threshold = max(0.01, current - delta)
        else:
            new_threshold = min(0.5, current + delta)
        self.scene.set_consistency_threshold(new_threshold)
        self.renderer.update(self.scene)
        print(f"一致性阈值: {new_threshold:.3f}")

    # ==================== 网格点查询（改进4） ====================

    def _query_grid_point(self, pos):
        """查询某个网格点的多相机颜色详情"""
        recon = self.scene.reconstructor

        # 找到最近的网格点
        distances = np.linalg.norm(recon.grid_points - pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_point = recon.grid_points[nearest_idx]

        self.selected_grid_point = nearest_point

        # 获取详情
        detail = recon.get_point_detail(nearest_point, self.scene.cameras)

        # 打印到终端
        print("\n" + "=" * 50)
        print(f"📌 网格点 ({nearest_point[0]:.3f}, {nearest_point[1]:.3f})")
        print(f"   置信度: {detail['confidence']:.3f}")
        print(f"   有效相机数: {detail['num_valid']}")
        print("-" * 50)
        for cam_info in detail['camera_details']:
            status = "✅" if cam_info['color'] is not None else "❌ 不可见"
            if cam_info['color'] is not None:
                color_str = f"({cam_info['color'][0]:.2f}, {cam_info['color'][1]:.2f}, {cam_info['color'][2]:.2f})"
            else:
                color_str = "无"
            print(f"   Cam{cam_info['cam_index']+1}: 像素{cam_info['pixel_index']} → {color_str} {status}")
        print("=" * 50 + "\n")

        # 更新渲染，高亮选中的网格点
        self.renderer.set_selected_point(nearest_point, detail)
        self.renderer.update(self.scene)
