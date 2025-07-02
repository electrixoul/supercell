# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set matplotlib to use default fonts (English)
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']

def calculate_water_brake_depth():
    """
    通过数值模拟计算一个高速圆柱体在水中减速所需的深度。

    该函数模拟一个圆柱体垂直进入水池的过程，综合考虑重力、
    随深度变化的浮力以及与速度相关的流体阻力。
    """
    # --- 1. 参数定义 ---
    # 已知物理参数
    m = 13600  # 质量 (kg)
    L = 6.2    # 长度 (m)
    D = 0.8    # 直径 (m)

    # 初始和最终条件
    v_initial = 450.0  # 初始速度 (m/s)
    v_final = 100.0    # 目标速度 (m/s)

    # 物理常数和估算参数
    g = 9.81              # 重力加速度 (m/s^2)
    rho_water = 1000      # 水的密度 (kg/m^3)
    # 阻力系数 (为平头圆柱体设定的典型工程估算值)
    Cd = 0.82

    # --- 2. 计算衍生参数 ---
    A = np.pi * (D / 2)**2 # 横截面积 (m^2)
    V_obj = A * L          # 物体总体积 (m^3)
    
    print("--- Physical Parameters and Model Setup ---")
    print(f"Object mass (m): {m} kg")
    print(f"Object length (L): {L} m")
    print(f"Object diameter (D): {D} m")
    print(f"Cross-sectional area (A): {A:.4f} m^2")
    print(f"Drag coefficient (Cd): {Cd}")
    print("\n--- Simulation Conditions ---")
    print(f"Initial velocity: {v_initial} m/s")
    print(f"Target velocity: {v_final} m/s")

    # --- 3. 数值模拟设置 ---
    # 初始化状态变量
    depth = 0.0          # 当前深度 (m)
    velocity = v_initial   # 当前速度 (m/s)
    
    # 模拟步长 (步长越小，结果越精确)
    delta_x = 0.01  # 深度步长 (m), 设定为1厘米

    # 用于记录数据以供绘图
    depth_history = [depth]
    velocity_history = [velocity]

    # --- 4. 模拟循环 ---
    # 当速度仍然大于目标速度时，持续计算
    while velocity > v_final:
        # 计算当前状态下的动能
        kinetic_energy = 0.5 * m * velocity**2
        
        # (a) 计算重力 (方向向下，为正)
        force_gravity = m * g

        # (b) 计算浮力 (方向向上，为负)
        # 判断物体是否已完全浸没
        if depth < L:
            # 未完全浸没，浮力随浸没体积线性增加
            submerged_volume = A * depth
            force_buoyancy = rho_water * submerged_volume * g
        else:
            # 已完全浸没，浮力达到最大值并恒定
            force_buoyancy = rho_water * V_obj * g

        # (c) 计算流体阻力 (方向向上，为负)
        force_drag = 0.5 * Cd * rho_water * A * velocity**2

        # (d) 计算合力
        net_force = force_gravity - force_buoyancy - force_drag

        # (e) 根据功能原理，计算合力在 delta_x 步长内做的功，并更新动能
        work_done = net_force * delta_x
        new_kinetic_energy = kinetic_energy + work_done
        
        # (f) 检查动能是否有效 (如果合力做负功过大，动能可能小于0)
        if new_kinetic_energy <= 0:
            velocity = 0
            break

        # (g) 根据新动能计算新速度
        velocity = np.sqrt(2 * new_kinetic_energy / m)

        # (h) 更新深度
        depth += delta_x
        
        # 记录历史数据
        depth_history.append(depth)
        velocity_history.append(velocity)

    print(f"\n--- Calculation Results ---")
    print(f"To reduce velocity from {v_initial} m/s to {velocity:.2f} m/s, required water pool depth: {depth:.2f} meters")
    
    return depth_history, velocity_history

def plot_results(depth_data, velocity_data):
    """
    Visualize the simulation results.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot velocity vs depth curve
    color = 'tab:blue'
    ax1.set_xlabel('Water Pool Depth (meters)', fontsize=14)
    ax1.set_ylabel('Velocity (m/s)', color=color, fontsize=14)
    ax1.plot(depth_data, velocity_data, color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Mark initial and final points
    ax1.plot(depth_data[0], velocity_data[0], 'go', markersize=10, label=f'Initial Point ({velocity_data[0]} m/s)')
    ax1.plot(depth_data[-1], velocity_data[-1], 'ro', markersize=10, label=f'Target Point ({velocity_data[-1]:.1f} m/s)')

    plt.title('Cylinder Velocity vs Depth in Water Simulation', fontsize=16)
    plt.legend()
    fig.tight_layout()
    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 执行计算
    depth_history, velocity_history = calculate_water_brake_depth()
    
    # 可视化结果
    if depth_history and velocity_history:
        plot_results(depth_history, velocity_history)
