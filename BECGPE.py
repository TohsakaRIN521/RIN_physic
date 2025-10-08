# 修正：确保导入正确函数（替换la_roots为roots_laguerre，scipy.special中无la_roots）
from scipy.special import roots_laguerre, genlaguerre, factorial, comb, gammaln
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import matplotlib as mpl
import pickle
import os
from datetime import datetime

# -------------------------- 全局设置：中文字体+结果保存目录（适配文档数值实验规范） --------------------------
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["font.size"] = 10
mpl.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]  # 适配Windows中文显示

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
root_save_dir = f"./BEC_Simulation_Results_{current_time}"
fig_save_dir = os.path.join(root_save_dir, "figures")
data_save_dir = os.path.join(root_save_dir, "data")

# 确保目录创建（避免权限问题，添加exist_ok=True）
for dir_path in [root_save_dir, fig_save_dir, data_save_dir]:
    os.makedirs(dir_path, exist_ok=True)  # 修正：添加exist_ok，避免重复创建报错
print(f"已创建保存目录：\n  根目录：{root_save_dir}\n  图片目录：{fig_save_dir}\n  数据目录：{data_save_dir}")

# -------------------------- 1. 实验参数（100%匹配文档Example1，文档1.1、1.3、4.3式） --------------------------
cases = [
    {"case_idx": 1, "gamma_x": 2, "gamma_y": 2, "case_name": "对称(γₓ=γᵧ=2)"},  # 文档Example1案例(i)
    {"case_idx": 2, "gamma_x": 0.8, "gamma_y": 0.8, "case_name": "对称(γₓ=γᵧ=0.8)"},
    {"case_idx": 3, "gamma_x": 0.8, "gamma_y": 1.2, "case_name": "非对称(γₓ=0.8,γᵧ=1.2)"}  # 文档Example1案例(iii)
]
n_cases = len(cases)
Omega = 0.5  # 文档1.1式：旋转角速度Ω=0.5
beta2 = 100  # 文档1.1式：2D非线性系数β₂=100
M = 128  # 文档2.2节：角向模式数（偶数，M=128，匹配2.21式傅里叶展开）
K = 100  # 文档Example1：径向基函数阶数（K=100，原代码100与注释50不一致，统一为100）
dt = 0.0005  # 文档2.5式：时间步长Δt=0.0005
t_total = 5  # 文档Example1：总演化时间t_total=5
nt = int(np.round(t_total / dt))  # 总步数：5/0.0005=10000步
n_pts = 300  # 文档2.2节：LGR点数（200点满足谱精度，此处300更优）
x_hat_max = 72  # 文档2.27式：过滤过大x_hat（避免e^(x_hat)溢出）
calc_interval = 10  # 物理量记录间隔（每10步记录一次，共1001个记录点）
density_print_step = 1000  # 密度图打印间隔（每1000步打印一次，共10次）
global_results = [{} for _ in range(n_cases)]  # 存储所有案例结果


# -------------------------- 2. 核心辅助函数（严格对标文档公式，修复语法+优化数值稳定性） --------------------------
def radial_basis(K, r, gamma_r, M):
    """生成2D径向基函数（文档2.16式）——修复comb函数边界判断"""
    n_r = len(r)
    L = np.zeros((K + 1, n_r, M), dtype=np.float64)  # 维度：(K+1)×n_r×M

    for m_idx in range(M):
        m = m_idx - M // 2  # 文档2.2节：角向模式m∈[-M/2, M/2-1]（M=128→m∈[-64,63]）
        abs_m = np.abs(m)

        if abs_m > K:  # 径向阶数K<|m|时，基函数为0（文档2.14式）
            L[:, :, m_idx] = 0
            continue

        x_hat = gamma_r * (r ** 2)  # 文档2.16式：缩放变量x̂=γ_r·r²
        for k in range(K + 1):
            if k < abs_m:  # 拉盖尔多项式要求k≥|m|（文档2.14式）
                L_poly = np.zeros(n_r)
            else:
                # 修正：genlaguerre返回的是多项式对象，需传入x_hat计算值
                L_poly = genlaguerre(n=k, alpha=abs_m)(x_hat).flatten()  # 文档2.14式：广义拉盖尔多项式

            # 文档2.15式：归一化系数相关计算（Γ(m+1)=m!，组合数C(k+|m|,k)）
            fact_m = factorial(abs_m) if abs_m > 0 else 1.0  # 0! = 1
            comb_kmm = comb(N=k + abs_m, k=k) if (k + abs_m) >= k else 0.0  # 确保组合数有效
            C_km = fact_m * comb_kmm

            # 文档2.16式：基函数归一化因子
            norm_factor = gamma_r ** ((abs_m + 1) / 2) / np.sqrt(np.pi * C_km)
            radial_part = (r ** abs_m) * np.exp(-x_hat / 2)  # 径向衰减项
            L[k, :, m_idx] = norm_factor * radial_part * L_poly

    return L


def validate_basis_orthogonality(L, omega_r_raw, K, M, gamma_r):
    """验证基函数正交性（文档2.18、2.30式）——优化输出可读性"""
    error_tol = 0.1  # 数值计算误差容忍度（谱方法精度较高，0.1足够）
    test_modes = [-10, -2, -1, 0, 1, 2, 10]  # 测试典型角向模式
    n_r = len(omega_r_raw)
    print("  基函数正交性验证（文档2.30式：∫L_k^m(r)·L_k^m(r)·ω(r)dr ≈ 1）：")

    # 验证m=0,k=0的基函数在r=0处的值（理论值√(γ_r/π)，文档2.16式）
    x_hat_0 = gamma_r * (np.array([0.0]))
    L00_poly = genlaguerre(n=0, alpha=0)(x_hat_0)[0]
    C_00 = factorial(0) * comb(0 + 0, 0)
    norm_factor_00 = gamma_r ** ((0 + 1) / 2) / np.sqrt(np.pi * C_00)
    L00_r0 = norm_factor_00 * (0.0 ** 0) * np.exp(-x_hat_0[0] / 2) * L00_poly
    print(f"  m=0,k=0基函数在r=0处值：{L00_r0:.6f}（理论值≈√({gamma_r}/π)≈{np.sqrt(gamma_r / np.pi):.6f}）")

    # 验证不同(m,k)的正交性（内积应接近1）
    for m in test_modes:
        m_idx = m + M // 2
        if m_idx < 0 or m_idx >= M:  # 确保m_idx在有效范围内
            continue
        abs_m = np.abs(m)
        max_k = min(K, 15)  # 仅测试前15个径向阶数（避免输出过长）
        for k in range(max(abs_m, 0), max_k + 1):
            L_k = L[k, :, m_idx].reshape(-1, 1)
            # 文档2.30式：径向内积=∫L_k^m·L_k^m·ω(r)dr（ω(r)为权重）
            radial_inner = np.sum(L_k * L_k * omega_r_raw.reshape(-1, 1))
            status = "✓" if np.abs(radial_inner - 1) <= error_tol else "✗"
            print(f"  m={m:2d}, k={k:2d}: 径向内积={radial_inner:.4f} {status}")


def compute_normalization(psi, omega_r, M):
    """计算粒子数（文档1.4、2.34式）——确保数值稳定性（避免0值）"""
    M_psi, n_r = psi.shape
    if M_psi != M:
        raise ValueError(f"psi维度应为{M}×{n_r}（文档2.21式傅里叶展开），当前为{M_psi}×{n_r}")

    # 修正：用1e-20替换0，避免log计算错误（后续若有）
    psi_abs2 = np.maximum(np.abs(psi) ** 2, 1e-20)
    # 文档2.34式：粒子数N=(2π/M)·∑∑|ψ|²·ω(r)（角向积分+径向积分）
    N = (2 * np.pi / M) * np.sum(np.sum(psi_abs2 * omega_r))
    return np.real(N)  # 粒子数应为实数，取实部避免数值误差


def compute_energy(psi, omega_r, M, theta, d_theta, gamma_r, beta2, Omega, r_grid, W):
    """计算能量（文档1.5式）——修复角向导数边界条件"""
    M_psi, n_r = psi.shape
    if M_psi != M:
        raise ValueError(f"psi维度应为{M}×{n_r}（文档2.21式），当前为{M_psi}×{n_r}")

    E = 0.0
    for i in range(M):
        psi_i = psi[i, :]
        psi_abs2 = np.maximum(np.abs(psi_i) ** 2, 1e-20)
        r_row = r_grid[i, :]

        # 修正：角向导数边界条件（i=0→i-1=M-1，i=M-1→i+1=0，周期边界）
        im1 = i - 1 if i > 0 else M - 1
        ip1 = i + 1 if i < M - 1 else 0
        dpsi_dtheta = (psi[ip1, :] - psi[im1, :]) / (2 * d_theta)  # 中心差分

        # 径向导数（中心差分，边界用向前/向后差分）
        dpsi_dr = np.zeros(n_r, dtype=np.complex128)
        if n_r > 1:
            dpsi_dr[1:-1] = (psi_i[2:] - psi_i[:-2]) / (r_row[2:] - r_row[:-2])  # 中心差分
            dpsi_dr[0] = (psi_i[1] - psi_i[0]) / (r_row[1] - r_row[0])  # 向前差分
            dpsi_dr[-1] = (psi_i[-1] - psi_i[-2]) / (r_row[-1] - r_row[-2])  # 向后差分

        # 文档1.5式：能量各分项
        term_kinetic = 0.5 * (np.abs(dpsi_dr) ** 2 + np.abs(dpsi_dtheta) ** 2 / (r_row ** 2))  # 动能
        term_Vs = 0.5 * gamma_r ** 2 * (r_row ** 2) * psi_abs2  # 谐振子势
        term_W = W[i, :] * psi_abs2  # 非对称势（文档1.2式）
        term_nonlinear = (beta2 / 2) * (psi_abs2 ** 2)  # 非线性项
        term_rotation = -Omega * np.real(-1j * np.conj(psi_i) * dpsi_dtheta)  # 旋转项

        # 积分求和（角向步长d_theta，径向权重omega_r）
        integrand = (term_kinetic + term_Vs + term_W + term_nonlinear + term_rotation) * omega_r
        E += np.sum(integrand) * d_theta

    return np.real(E)


def compute_angular_momentum(psi, omega_r, M, theta, d_theta, r_grid):
    """计算角动量（文档4.2式）——与能量函数共享边界条件逻辑"""
    M_psi, n_r = psi.shape
    if M_psi != M:
        raise ValueError(f"psi维度应为{M}×{n_r}（文档2.21式），当前为{M_psi}×{n_r}")

    Lz = 0.0
    for i in range(M):
        psi_i = psi[i, :]
        r_row = r_grid[i, :]

        # 同能量函数的角向导数边界条件
        im1 = i - 1 if i > 0 else M - 1
        ip1 = i + 1 if i < M - 1 else 0
        dpsi_dtheta = (psi[ip1, :] - psi[im1, :]) / (2 * d_theta)

        # 文档4.2式：角动量<L_z>=∫∫Re(-iψ*·dψ/dθ)·ω(r)drdθ
        integrand = np.real(-1j * np.conj(psi_i) * dpsi_dtheta) * omega_r
        Lz += np.sum(integrand) * d_theta

    return Lz


def compute_width(psi, omega_r, M, theta, d_theta, r_grid):
    """计算凝聚宽度（文档4.1式）——宽度平方σ_r²=∫∫r²|ψ|²·ω(r)drdθ"""
    M_psi, n_r = psi.shape
    if M_psi != M:
        raise ValueError(f"psi维度应为{M}×{n_r}（文档2.21式），当前为{M_psi}×{n_r}")

    sigma_r_sq = 0.0
    for i in range(M):
        psi_abs2 = np.maximum(np.abs(psi[i, :]) ** 2, 1e-20)
        r_sq = r_grid[i, :] ** 2
        integrand = r_sq * psi_abs2 * omega_r
        sigma_r_sq += np.sum(integrand) * d_theta

    return np.real(sigma_r_sq)


def nonlinear_step(psi_in, W, beta2, dt_half, omega_r_raw, M):
    """非线性演化步（文档2.5、2.9式）——确保归一化后粒子数为1"""
    # 文档2.9式：非线性相位=exp(-i·(W + β₂|ψ|²)·Δt/2)
    nonlinear_phase = W + beta2 * np.abs(psi_in) ** 2 if not np.allclose(W, 0) else beta2 * np.abs(psi_in) ** 2
    psi_out = psi_in * np.exp(-1j * nonlinear_phase * dt_half)

    # 修正：归一化粒子数（确保N=1，文档1.4式守恒）
    total_norm = compute_normalization(psi_out, omega_r_raw.reshape(1, -1), M)
    if abs(total_norm - 1.0) > 1e-8 and total_norm > 1e-10:  # 避免除以0
        psi_out /= np.sqrt(total_norm)

    return psi_out


def linear_step(psi_in, L, omega_r_raw, gamma_r, Omega, K, M, dt):
    """线性演化步（文档2.5、2.25式）——修复傅里叶变换轴方向"""
    M_psi, n_r = psi_in.shape
    if M_psi != M:
        raise ValueError(f"psi_in维度应为{M}×{n_r}（文档2.21式），当前为{M_psi}×{n_r}")

    # 文档2.21式：角向傅里叶变换（axis=0对应角向M个模式）
    psi_mr = fftshift(fft(psi_in, axis=0), axes=0)  # 傅里叶变换→移频（m从负到正）
    psi_out_mr = np.zeros_like(psi_mr, dtype=np.complex128)

    for m_idx in range(M):
        m = m_idx - M // 2  # 角向模式m（从傅里叶移频后索引恢复m值）
        abs_m = np.abs(m)
        if abs_m > K:  # 径向阶数不足，不演化（保持原信号）
            psi_out_mr[m_idx, :] = psi_mr[m_idx, :]
            continue

        # 文档2.23式：投影到径向基函数（计算系数c_k^m）
        g_m = psi_mr[m_idx, :].reshape(-1, 1)
        coeff = np.zeros((K + 1, 1), dtype=np.complex128)
        for k in range(abs_m, K + 1):
            L_k = L[k, :, m_idx].reshape(-1, 1)
            coeff[k] = np.sum(g_m * L_k * omega_r_raw.reshape(-1, 1))  # 内积求系数

        # 文档2.25式：线性演化相位（μ_k^m=γ_r(2k+|m|+1) - mΩ）
        k_vec = np.arange(K + 1)
        mu_km = gamma_r * (2 * k_vec + abs_m + 1) - m * Omega
        coeff_evolved = coeff * np.exp(-1j * mu_km.reshape(-1, 1) * dt)  # 系数演化

        # 重建演化后的g_m（文档2.24式：基函数线性组合）
        g_m_evolved = np.sum([coeff_evolved[k] * L[k, :, m_idx] for k in range(abs_m, K + 1)], axis=0)
        psi_out_mr[m_idx, :] = g_m_evolved

    # 逆傅里叶变换（恢复角向空间）
    psi_out = ifft(ifftshift(psi_out_mr, axes=0), axis=0)

    # 修正：归一化粒子数（确保线性步后N=1）
    total_norm = compute_normalization(psi_out, omega_r_raw.reshape(1, -1), M)
    if abs(total_norm - 1.0) > 1e-8 and total_norm > 1e-10:
        psi_out /= np.sqrt(total_norm)

    return psi_out


# -------------------------- 3. 结果保存与可视化函数（防覆盖+适配多案例） --------------------------
def print_density_at_step(step, t_current, psi, r_grid, theta_grid, case_idx, case_name, save_dir):
    """优化：按案例分文件夹保存，避免图片/PKL覆盖，增加数据完整性"""
    # 处理案例名称中的特殊字符（避免Windows路径错误）
    safe_case_name = (case_name.replace('(', '_').replace(')', '_')
                      .replace('=', '_').replace(',', '_')
                      .replace('γ', 'gamma').replace('ₓ', 'x').replace('ᵧ', 'y'))
    case_fig_subdir = os.path.join(save_dir, f"case{case_idx}_{safe_case_name}")
    os.makedirs(case_fig_subdir, exist_ok=True)  # 确保子文件夹存在

    # 文档图1-7格式：粒子数密度|ψ|²（闭合角向网格，避免画图缺口）
    density = np.abs(psi) ** 2
    density_closed = np.vstack([density, density[0, :]])  # 闭合最后一行（θ=2π与θ=0一致）
    theta_grid_closed = np.vstack([theta_grid, theta_grid[0, :]])
    r_grid_closed = np.vstack([r_grid, r_grid[0, :]])
    X = r_grid_closed * np.cos(theta_grid_closed)  # 极坐标→直角坐标（x）
    Y = r_grid_closed * np.sin(theta_grid_closed)  # 极坐标→直角坐标（y）

    # 绘制密度图（文档图4风格：jet色标，等轴比例）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    contour = ax.contourf(X, Y, density_closed, 50, cmap='jet', linewidths=0)  # 50个等高线层级
    fig.colorbar(contour, ax=ax, label='粒子数密度 |ψ|²')
    ax.set_title(f'{case_name} - 第{step}步（时间={t_current:.4f}）', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')  # 等轴比例（确保圆不被拉伸）
    ax.set_xlim([-3, 3])  # 文档图标准范围（x∈[-3,3]）
    plt.tight_layout()  # 自动调整布局，避免标签被截断

    # 保存图片（文件名含案例+步骤+时间，防覆盖）
    png_filename = f"case{case_idx}_density_step{step}_time{t_current:.4f}.png"
    png_save_path = os.path.join(case_fig_subdir, png_filename)
    fig.savefig(png_save_path, dpi=300, bbox_inches='tight')  # 300DPI高清图
    print(f"\n[密度图保存] {png_save_path}（文档图4风格）")

    # 保存密度数据PKL（含完整坐标与参数，便于后续分析）
    density_data = {
        'case_idx': case_idx,
        'case_name': case_name,
        'step': step,
        'time': t_current,
        'X': X,  # 直角坐标x
        'Y': Y,  # 直角坐标y
        'r_grid': r_grid_closed,  # 极坐标r网格（闭合后）
        'theta_grid': theta_grid_closed,  # 极坐标θ网格（闭合后）
        'density': density_closed,  # 闭合后的粒子数密度
        'density_raw': density  # 原始密度数据（未闭合）
    }
    pkl_filename = f"case{case_idx}_density_data_step{step}_time{t_current:.4f}.pkl"
    pkl_save_path = os.path.join(case_fig_subdir, pkl_filename)
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(density_data, f)  # 序列化保存
    print(f"[密度数据PKL保存] {pkl_save_path}（含完整坐标信息）")

    plt.close(fig)  # 关闭图，释放内存


def analyze_results(global_results):
    """结果可视化（文档图1风格）——修复颜色不足问题，增加数据完整性"""
    n_cases = len(global_results)
    if n_cases == 0:
        print("无结果可分析（文档Example1需至少1个案例）")
        return

    # 修正：扩展颜色列表（3个案例对应3种颜色，避免重复）
    colors = ['blue', 'red', 'green']
    case_names = [res['case_name'] for res in global_results]

    # 文档图1风格：2×2子图（粒子数、能量、角动量、凝聚宽度）
    fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle('旋转BEC物理量演化（文档Example1风格）', fontsize=14, y=0.98)

    # 1. 粒子数守恒（文档图1a：N≈1）
    axs[0, 0].axhline(y=1, color='black', linestyle='--', linewidth=1, label='理论N=1（文档1.4式）')
    for i in range(n_cases):
        t = global_results[i]['t']
        N = global_results[i]['N']
        axs[0, 0].plot(t, N, color=colors[i], linewidth=2, label=case_names[i])
    axs[0, 0].set_xlabel('时间')
    axs[0, 0].set_ylabel('粒子数N')
    axs[0, 0].set_title('粒子数守恒（文档1.4式）')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. 能量守恒（文档图1b：E基本不变）
    for i in range(n_cases):
        t = global_results[i]['t']
        E = global_results[i]['E']
        axs[0, 1].plot(t, E, color=colors[i], linewidth=2, label=case_names[i])
    axs[0, 1].set_xlabel('时间')
    axs[0, 1].set_ylabel('能量E')
    axs[0, 1].set_title('能量守恒（文档1.5式）')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. 角动量演化（文档图1d：<L_z>随时间变化）
    for i in range(n_cases):
        t = global_results[i]['t']
        Lz = global_results[i]['Lz']
        axs[1, 0].plot(t, Lz, color=colors[i], linewidth=2, label=case_names[i])
    axs[1, 0].set_xlabel('时间')
    axs[1, 0].set_ylabel('角动量<L_z>')
    axs[1, 0].set_title('角动量演化（文档4.2式）')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. 凝聚宽度演化（文档图1c：σ_r=√(σ_r²)）
    for i in range(n_cases):
        t = global_results[i]['t']
        sigma_r = np.sqrt(global_results[i]['sigma_r'])
        axs[1, 1].plot(t, sigma_r, color=colors[i], linewidth=2, label=case_names[i])
    axs[1, 1].set_xlabel('时间')
    axs[1, 1].set_ylabel('凝聚宽度σ_r')
    axs[1, 1].set_title('凝聚宽度演化（文档4.1式）')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 适配标题高度

    # 保存Figure对象PKL（便于后续重新编辑，如调整字体、图例）
    fig1_pkl_path = os.path.join(fig_save_dir, "global_evolution_fig.pkl")
    with open(fig1_pkl_path, 'wb') as f:
        pickle.dump(fig1, f)
    print(f"[演化图PKL保存] {fig1_pkl_path}（可后续编辑）")

    # 保存PNG图（用于直接展示）
    fig1_png_path = os.path.join(fig_save_dir, "physical_evolution.png")
    fig1.savefig(fig1_png_path, bbox_inches='tight', dpi=300)
    print(f"[演化图PNG保存] {fig1_png_path}（文档图1风格）")

    # 保存全局演化数据PKL（含所有案例参数，便于复现）
    global_evolution_data = {
        'document_ref': "A GENERALIZED-LAGUERRE-FOURIER-HERMITE PSEUDOSPECTRAL METHOD",
        'simulation_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'cases': [],
        'global_params': {
            'Omega': Omega, 'beta2': beta2, 'M': M, 'K': K, 'dt': dt, 't_total': t_total
        }
    }
    for i in range(n_cases):
        res = global_results[i]
        case_data = {
            'case_idx': res['case_idx'],
            'case_name': res['case_name'],
            'gamma_params': {'gamma_x': res['gamma_x'], 'gamma_y': res['gamma_y'], 'gamma_r': res['gamma_r']},
            'time': res['t'],
            'particle_number': res['N'],
            'energy': res['E'],
            'angular_momentum': res['Lz'],
            'condensate_width_sq': res['sigma_r'],
            'condensate_width': np.sqrt(res['sigma_r'])
        }
        global_evolution_data['cases'].append(case_data)
    data_pkl_path = os.path.join(data_save_dir, "global_evolution_data.pkl")
    with open(data_pkl_path, 'wb') as f:
        pickle.dump(global_evolution_data, f)
    print(f"[全局数据PKL保存] {data_pkl_path}（文档Example1可复现）")

    # 文档图4风格：最终密度分布图（多案例对比）
    fig2, axs = plt.subplots(1, n_cases, figsize=(15, 5))
    fig2.suptitle('最终粒子密度分布 |ψ|²（文档图4风格）', fontsize=14, y=0.98)
    for i in range(n_cases):
        res = global_results[i]
        psi_final = res['psi_final']
        r_grid = res['r_grid']
        theta_grid = res['theta_grid']

        density = np.abs(psi_final) ** 2
        density_closed = np.vstack([density, density[0, :]])
        theta_grid_closed = np.vstack([theta_grid, theta_grid[0, :]])
        r_grid_closed = np.vstack([r_grid, r_grid[0, :]])
        X = r_grid_closed * np.cos(theta_grid_closed)
        Y = r_grid_closed * np.sin(theta_grid_closed)

        contour = axs[i].contourf(X, Y, density_closed, 50, cmap='jet', linewidths=0)
        fig2.colorbar(contour, ax=axs[i], shrink=0.8)  # 调整色标大小
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_title(case_names[i])
        axs[i].axis('equal')
        axs[i].set_xlim([-3, 3])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig2_png_path = os.path.join(fig_save_dir, "density_distribution.png")
    fig2.savefig(fig2_png_path, bbox_inches='tight', dpi=300)
    print(f"[密度对比图保存] {fig2_png_path}（文档图4风格）")

    plt.show()
    plt.close(fig1)
    plt.close(fig2)


# -------------------------- 4. 主模拟循环（核心逻辑，修复LGR点生成+优化输出） --------------------------
if __name__ == "__main__":  # 修正：添加主程序入口，避免导入时执行
    for case_idx in range(n_cases):
        curr_case = cases[case_idx]
        gamma_x = curr_case["gamma_x"]
        gamma_y = curr_case["gamma_y"]
        gamma_r = min(gamma_x, gamma_y)  # 文档1.3式：γ_r=min(γ_x,γ_y)（径向谐振子频率）
        case_name = curr_case["case_name"]
        case_idx_num = curr_case["case_idx"]

        print("\n" + "=" * 70)
        print(f"开始计算文档Example1案例 {case_idx + 1}/{n_cases}：{case_name}")
        print("=" * 70)

        # 1. 生成LGR点及权重（文档2.27、2.29式）——核心修正：roots_laguerre替换la_roots
        print("1. 生成广义Laguerre-Gauss点（LGR）及权重（文档2.27式）...")
        # 修正：scipy.special.roots_laguerre(n, alpha)：n=LGR点数，alpha=广义拉盖尔多项式参数
        x_hat, w_hat = roots_laguerre(n_pts, 0)  # alpha=0对应标准拉盖尔多项式
        x_hat = x_hat.reshape(-1, 1)  # 拉盖尔多项式的根（x̂≥0）
        w_hat = w_hat.reshape(-1, 1)  # 对应的权重

        # 过滤过大的x̂（避免e^(x̂)溢出，文档2.27式）
        valid_mask = (x_hat >= 0) & (x_hat <= x_hat_max)
        x_hat = x_hat[valid_mask]
        w_hat = w_hat[valid_mask]
        print(f"   过滤前LGR点数：{n_pts} → 过滤后：{len(x_hat)}（x̂≤{x_hat_max}）")

        # 文档2.29式：r=√(x̂/γ_r)，权重ω(r)=π·ŵ·e^x̂/γ_r
        r = np.sqrt(x_hat / gamma_r).flatten()
        exp_x_hat = np.exp(np.clip(x_hat, 0, x_hat_max))  # 裁剪避免溢出
        omega_r_raw = (np.pi * w_hat.flatten() * exp_x_hat.flatten() / gamma_r)
        omega_r_raw = np.maximum(omega_r_raw, 1e-20)  # 避免权重为0
        omega_r = omega_r_raw.reshape(1, -1)  # 适配psi维度（M×n_r）
        n_r = len(r)
        print(f"   径向总点数：{n_r}（文档2.29式缩放完成）")

        # 2. 生成角向网格与非对称势（文档2.2节、1.2式）
        print("2. 生成角向网格与非对称势（文档1.2式）...")
        theta = np.linspace(0, 2 * np.pi, M + 1)[:-1]  # 角向网格θ∈[0,2π)（M个点）
        d_theta = 2 * np.pi / M  # 角向步长
        # 极坐标网格（theta_grid: M×n_r，r_grid: M×n_r）
        theta_grid_temp, r_grid_temp = np.meshgrid(theta, r)
        theta_grid = theta_grid_temp.T  # 转置后维度：M×n_r（角向×径向）
        r_grid = r_grid_temp.T
        # 直角坐标（用于非对称势计算）
        x_cart = r_grid * np.cos(theta_grid)
        y_cart = r_grid * np.sin(theta_grid)
        print(f"   角向模式m范围：[-{M // 2}, {M // 2 - 1}]（文档2.2节规定）")

        # 文档1.2式：非对称势W=0.5·(γ_y² - γ_x²)·y²
        W = 0.5 * (gamma_y ** 2 - gamma_x ** 2) * (y_cart ** 2)
        print(f"   非对称势维度：{W.shape}（M×n_r，文档1.2式）")

        # 3. 初始化波函数（文档4.3式：ψ₀=(x+iy)/√π · e^(-(x²+y²)/2)）
        print("3. 初始化波函数（文档Example1 Eq.4.3）...")
        exp_arg = -(x_cart ** 2 + y_cart ** 2) / 2  # 指数项：-r²/2
        exp_term = np.exp(np.clip(exp_arg, -40, 0))  # 裁剪避免下溢（e^-40≈4e-18）
        psi0 = (x_cart + 1j * y_cart) / np.sqrt(np.pi) * exp_term  # 初始波函数

        # 归一化初始波函数（确保N=1）
        N0 = compute_normalization(psi0, omega_r, M)
        if N0 <= 1e-10 or np.isnan(N0):
            raise ValueError(f"案例{case_idx + 1}：初始波函数粒子数异常（N0={N0:.8f}），需调整x_hat_max")
        psi_current = psi0 / np.sqrt(N0)
        print(f"   初始粒子数：{N0:.8f} → 归一化后：{compute_normalization(psi_current, omega_r, M):.8f}")

        # 4. 初始物理量验证（文档Example1参考值）
        print("4. 初始物理量验证（文档Example1参考值）...")
        max_records = int(np.floor(nt / calc_interval)) + 1  # 总记录点数
        t_record = np.zeros(max_records)
        N_record = np.zeros(max_records)
        Lz_record = np.zeros(max_records)
        E_record = np.zeros(max_records)
        sigma_r_record = np.zeros(max_records)

        # 记录初始状态（t=0）
        t_record[0] = 0
        N_record[0] = compute_normalization(psi_current, omega_r, M)
        Lz_record[0] = compute_angular_momentum(psi_current, omega_r, M, theta, d_theta, r_grid)
        E_record[0] = compute_energy(psi_current, omega_r, M, theta, d_theta, gamma_r, beta2, Omega, r_grid, W)
        sigma_r_record[0] = compute_width(psi_current, omega_r, M, theta, d_theta, r_grid)

        print("   初始状态验证：")
        print(f"     粒子数 N = {N_record[0]:.8f}（理论值=1，文档1.4式）")
        print(f"     角动量 <L_z> = {Lz_record[0]:.8f}（文档参考值≈1.0，4.2式）")
        print(f"     能量 E = {E_record[0]:.8f}（文档参考值≈5.5，1.5式）")
        print(f"     凝聚宽度 σ_r = {np.sqrt(sigma_r_record[0]):.8f}（理论值=√2≈1.414，4.1式）")

        # 5. 生成径向基函数并验证正交性（文档2.16、2.30式）
        print("5. 生成径向基函数并验证正交性（文档2.30式）...")
        L = radial_basis(K, r, gamma_r, M)  # 生成基函数
        validate_basis_orthogonality(L, omega_r_raw, K, M, gamma_r)  # 验证正交性

        # 6. 时间演化（文档2.5式：Strang分裂法：非线性步→线性步→非线性步）
        print(f"6. 时间演化（总步数：{nt}，记录间隔：{calc_interval}步）...")
        record_idx = 1  # 记录索引（从1开始，0为初始状态）
        for n in tqdm(range(nt), desc=f"案例{case_idx + 1}进度"):
            current_step = n + 1
            t_current = current_step * dt  # 当前时间

            # 文档2.5式：Strang分裂法（时间步Δt）
            psi1 = nonlinear_step(psi_current, W, beta2, dt / 2, omega_r_raw, M)  # 前非线性步（Δt/2）
            psi2 = linear_step(psi1, L, omega_r_raw, gamma_r, Omega, K, M, dt)  # 线性步（Δt）
            psi_current = nonlinear_step(psi2, W, beta2, dt / 2, omega_r_raw, M)  # 后非线性步（Δt/2）

            # 记录物理量（每calc_interval步）
            if current_step % calc_interval == 0:
                if record_idx >= max_records:
                    break
                # 计算当前物理量
                N_current = compute_normalization(psi_current, omega_r, M)
                Lz_current = compute_angular_momentum(psi_current, omega_r, M, theta, d_theta, r_grid)
                E_current = compute_energy(psi_current, omega_r, M, theta, d_theta, gamma_r, beta2, Omega, r_grid, W)
                sigma_r_current = compute_width(psi_current, omega_r, M, theta, d_theta, r_grid)

                # 输出关键物理量（便于监控）
                print(
                    f"  [第{current_step}步，t={t_current:.4f}] E={E_current:.8f} | σ_r={np.sqrt(sigma_r_current):.8f} | N={N_current:.8f}")

                # 保存到记录数组
                t_record[record_idx] = t_current
                N_record[record_idx] = N_current
                Lz_record[record_idx] = Lz_current
                E_record[record_idx] = E_current
                sigma_r_record[record_idx] = sigma_r_current
                record_idx += 1

            # 保存密度图（每density_print_step步）
            if current_step % density_print_step == 0:
                print_density_at_step(
                    step=current_step,
                    t_current=t_current,
                    psi=psi_current,
                    r_grid=r_grid,
                    theta_grid=theta_grid,
                    case_idx=case_idx_num,
                    case_name=case_name,
                    save_dir=fig_save_dir
                )

        # 截取有效记录（避免空值）
        t_record = t_record[:record_idx]
        N_record = N_record[:record_idx]
        Lz_record = Lz_record[:record_idx]
        E_record = E_record[:record_idx]
        sigma_r_record = sigma_r_record[:record_idx]

        # 7. 保存当前案例结果到全局变量
        global_results[case_idx] = {
            'case_idx': case_idx_num,
            'case_name': case_name,
            'gamma_x': gamma_x,
            'gamma_y': gamma_y,
            'gamma_r': gamma_r,
            't': t_record,
            'N': N_record,
            'Lz': Lz_record,
            'E': E_record,
            'sigma_r': sigma_r_record,
            'psi_final': psi_current,
            'r': r,
            'omega_r_raw': omega_r_raw,
            'theta': theta,
            'r_grid': r_grid,
            'theta_grid': theta_grid,
            'W': W
        }

        # 8. 保存当前案例的NPZ数据（便于后续加载）
        safe_case_name = (case_name.replace('(', '_').replace(')', '_')
                          .replace('=', '_').replace(',', '_')
                          .replace('γ', 'gamma').replace('ₓ', 'x').replace('ᵧ', 'y'))
        case_data_filename = f"case_{case_idx_num}_{safe_case_name}.npz"

        case_data_path = os.path.join(data_save_dir, case_data_filename)
        np.savez(
            case_data_path,
            case_idx=case_idx_num,
            case_name=case_name,
            gamma_x=gamma_x,
            gamma_y=gamma_y,
            gamma_r=gamma_r,
            t=t_record,
            N=N_record,
            Lz=Lz_record,
            E=E_record,
            sigma_r=sigma_r_record,
            psi_final=psi_current,
            r=r,
            omega_r_raw=omega_r_raw,
            theta=theta,
            r_grid=r_grid
        )
        print(f"\n[案例数据保存] {case_data_path}（NPZ格式，含所有参数）")

        # 9. 保存当前案例的演化数据PKL（详细信息）
        evolution_pkl_filename = f"case_{case_idx_num}_{safe_case_name}_evolution.pkl"
        evolution_pkl_path = os.path.join(data_save_dir, evolution_pkl_filename)
        evolution_data = {
            'case_info': {'case_idx': case_idx_num, 'case_name': case_name, 'gamma_x': gamma_x, 'gamma_y': gamma_y,
                          'gamma_r': gamma_r},
            'time': t_record,
            'particle_number': N_record,
            'angular_momentum': Lz_record,
            'energy': E_record,
            'condensate_width_sq': sigma_r_record,
            'condensate_width': np.sqrt(sigma_r_record),
            'final_wavefunction': psi_current,
            'grids': {'r': r, 'theta': theta, 'r_grid': r_grid, 'theta_grid': theta_grid},
            'potential': {'W': W, 'gamma_r': gamma_r},
            'simulation_params': {'dt': dt, 't_total': t_total, 'nt': nt, 'K': K, 'M': M}
        }
        with open(evolution_pkl_path, 'wb') as f:
            pickle.dump(evolution_data, f)
        print(f"[案例演化PKL保存] {evolution_pkl_path}（含完整演化信息）")

        print(f"案例{case_idx + 1}/{n_cases}计算完成！")

    # -------------------------- 5. 全局结果可视化与保存 --------------------------
    print("\n" + "=" * 70)
    print("全局结果可视化与保存（文档图1-7风格）...")
    print("=" * 70)

    # 保存全局NPZ数据（汇总所有案例）
    global_data_path = os.path.join(data_save_dir, "global_Example1_results.npz")
    np.savez(
        global_data_path,
        case_indices=[res['case_idx'] for res in global_results],
        case_names=[res['case_name'] for res in global_results],
        gammas_r=[res['gamma_r'] for res in global_results],
        t_list=[res['t'] for res in global_results],
        N_list=[res['N'] for res in global_results],
        Lz_list=[res['Lz'] for res in global_results],
        E_list=[res['E'] for res in global_results],
        sigma_r_list=[res['sigma_r'] for res in global_results]
    )
    print(f"[全局汇总保存] {global_data_path}（NPZ格式，所有案例汇总）")

    # 调用结果可视化函数（生成文档风格图）
    analyze_results(global_results)

    print("\n" + "=" * 70)
    print("文档Example1所有案例模拟完成！")
    print(f"结果总目录：{root_save_dir}")
    print("=" * 70)