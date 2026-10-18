"""
main.py - 程序入口

三种运行模式：
  1. 交互模式：python main.py
  2. 自动演示模式：python main.py --demo
  3. 分析一实验模式：python main.py --exp1
  4. 分析二实验模式：python main.py --exp2
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene
from rendering import Renderer
from interaction import InteractionHandler


def run_experiment_1():
    print("=" * 60)
    print("  分析一：几何复杂度对重建质量的影响")
    print("=" * 60)
    scene = Scene(grid_resolution=50)
    results = scene.run_batch_experiment_geometry(num_curves=300)
    results = np.array(results)
    np.savetxt('experiment_1_results.csv', results,
               delimiter=',', header='avg_visibility,accuracy,completeness',
               comments='')
    print(f"实验完成，共 {len(results)} 条曲线，数据已保存")


def run_experiment_2():
    print("=" * 60)
    print("  分析二：纹理丰富度对重建质量的影响")
    print("=" * 60)
    scene = Scene(grid_resolution=50)
    all_results = scene.run_batch_experiment_texture(
        hue_ranges=[0.01, 0.05, 0.10, 0.20, 0.40, 0.80],
        repeats=3
    )
    summary = []
    for hue_range in sorted(all_results.keys()):
        results = all_results[hue_range]
        accs = [r[0] for r in results]
        comps = [r[1] for r in results]
        summary.append((hue_range, np.mean(accs), np.mean(comps)))
    summary = np.array(summary)
    np.savetxt('experiment_2_results.csv', summary,
               delimiter=',', header='hue_range,mean_accuracy,mean_completeness',
               comments='')
    print("实验完成，数据已保存")

def run_demo2():
    from matplotlib.widgets import Button

    scene = Scene(grid_resolution=50)
    renderer = Renderer()
    renderer.update(scene)
    interaction = InteractionHandler(scene, renderer)

    hue_ranges = [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]
    current_idx = [0]
    is_running = [False]
    timer = [None]

    def update_to_hue(idx):
        from curve import Curve
        default_cp = np.array([
            [0.10, 0.50],
            [0.35, 0.15],
            [0.65, 0.85],
            [0.90, 0.50],
        ])
        scene.curve = Curve.with_hue_range(
            hue_range=hue_ranges[idx],
            control_points=default_cp
        )
        scene.update()
        renderer.update(scene)

    def auto_vary_texture():
        if not is_running[0]:
            return
        current_idx[0] = (current_idx[0] + 1) % len(hue_ranges)
        update_to_hue(current_idx[0])

    def toggle_demo(event):
        is_running[0] = not is_running[0]
        if is_running[0]:
            btn_start.label.set_text('Stop')
            timer[0] = fig.canvas.new_timer(interval=1500)
            timer[0].add_callback(auto_vary_texture)
            timer[0].start()
        else:
            btn_start.label.set_text('Start')
            if timer[0] is not None:
                timer[0].stop()

    # 初始显示第一个色相
    update_to_hue(0)

    fig = renderer.get_figure()
    ax_btn = fig.add_axes([0.02, 0.95, 0.05, 0.03])
    btn_start = Button(ax_btn, 'Start')
    btn_start.on_clicked(toggle_demo)

    plt.show()

def run_demo():
    from matplotlib.widgets import Button

    scene = Scene(grid_resolution=50)
    renderer = Renderer()
    renderer.update(scene)
    interaction = InteractionHandler(scene, renderer)

    # 状态控制
    is_running = [False]
    timer = [None]

    def auto_vary_controls():
        if not is_running[0]:
            return
        curve = scene.curve
        new_cp = np.random.uniform(0.05, 0.95, (4, 2))
        for i in range(4):
            curve.update_control_point(i, new_cp[i])
        scene.update()
        renderer.update(scene)

    def toggle_demo(event):
        is_running[0] = not is_running[0]
        if is_running[0]:
            btn_start.label.set_text('Stop')
            timer[0] = fig.canvas.new_timer(interval=1500)
            timer[0].add_callback(auto_vary_controls)
            timer[0].start()
        else:
            btn_start.label.set_text('Start')
            if timer[0] is not None:
                timer[0].stop()

    fig = renderer.get_figure()
    # 在左上角放一个启动/停止按钮（不影响四个子图）
    ax_btn = fig.add_axes([0.02, 0.95, 0.05, 0.03])
    btn_start = Button(ax_btn, 'Start')
    btn_start.on_clicked(toggle_demo)

    plt.show()


def main():
    scene = Scene(grid_resolution=50)
    renderer = Renderer()
    renderer.update(scene)
    interaction = InteractionHandler(scene, renderer)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exp1':
        run_experiment_1()
    elif len(sys.argv) > 1 and sys.argv[1] == '--exp2':
        run_experiment_2()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo2':
        run_demo2()
    else:
        main()