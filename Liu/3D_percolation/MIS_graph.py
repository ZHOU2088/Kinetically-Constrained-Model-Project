import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

def read_max_cluster_density_data(filename):
    """从max-cluster-density-stat文件中读取数据并返回一个字典"""
    data = {}
    
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在！")
        return data
    
    with open(filename, 'r') as f:
        # 跳过头部
        header = f.readline().strip()
        print(f"文件头部: {header}")
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) != 4:
                print(f"跳过格式错误的行: {line}")
                continue
                
            try:
                L = int(parts[0])
                rho = float(parts[1])
                mean_density = float(parts[2])
                std_density = float(parts[3])
                
                if L not in data:
                    data[L] = {'rho': [], 'mean': [], 'stddev': []}
                
                data[L]['rho'].append(rho)
                data[L]['mean'].append(mean_density)
                data[L]['stddev'].append(std_density)
                
            except ValueError as e:
                print(f"转换数据时出错: {line}, 错误: {e}")
                continue
    
    # 对每个L的数据按rho排序
    for L in data:
        rho = np.array(data[L]['rho'])
        mean = np.array(data[L]['mean'])
        stddev = np.array(data[L]['stddev'])
        sort_idx = np.argsort(rho)
        data[L]['rho'] = rho[sort_idx].tolist()
        data[L]['mean'] = mean[sort_idx].tolist()
        data[L]['stddev'] = stddev[sort_idx].tolist()
    
    return data

def compact_format(x, decimal_places):
    """紧凑格式化数字"""
    format_str = f"{{:.{decimal_places}f}}"
    formatted = format_str.format(x).rstrip('0').rstrip('.')
    return formatted

def filter_data_by_rho_range(data, rho_range=None):
    """
    根据指定的rho范围过滤数据
    
    参数:
    data: 原始数据字典
    rho_range: 元组 (rho_min, rho_max) 或 None
    
    返回:
    filtered_data: 过滤后的数据字典
    """
    if rho_range is None:
        return data
    
    rho_min, rho_max = rho_range
    filtered_data = {}
    
    for L in data:
        rho_values = np.array(data[L]['rho'])
        mean_values = np.array(data[L]['mean'])
        stddev_values = np.array(data[L]['stddev'])
        
        # 筛选在指定范围内的数据点
        mask = (rho_values >= rho_min) & (rho_values <= rho_max)
        
        if np.any(mask):
            filtered_data[L] = {
                'rho': rho_values[mask].tolist(),
                'mean': mean_values[mask].tolist(),
                'stddev': stddev_values[mask].tolist()
            }
    
    return filtered_data

def get_filename_suffix(rho_range):
    """根据rho范围生成文件名后缀"""
    if rho_range is None:
        return ""
    
    rho_min, rho_max = rho_range
    # 格式化数字，避免小数点引起的文件系统问题
    suffix = f"_rho{rho_min:.3f}to{rho_max:.3f}".replace('.', 'p')
    return suffix

def get_safe_title(rho_range):
    """生成安全的标题字符串，避免mathtext解析错误"""
    if rho_range is None:
        return 'Maximum Cluster Density vs Density Parameter'
    else:
        rho_min, rho_max = rho_range
        # 使用普通字符串而不是mathtext
        return f'Maximum Cluster Density vs Density Parameter (ρ: {rho_min:.3f}-{rho_max:.3f})'

def plot_max_cluster_density(data, rho_range=None, output_dir="./MIS-graph/figures"):
    """
    使用数据绘制最大连通图密度与rho的关系图
    
    参数:
    data: 原始数据
    rho_range: 元组 (rho_min, rho_max) 或 None
    output_dir: 输出目录
    """
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.6, 6.45), dpi=300)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据rho范围过滤数据
    if rho_range is not None:
        rho_min_filter, rho_max_filter = rho_range
        data = filter_data_by_rho_range(data, rho_range)
        if not data:
            print(f"警告: 在rho范围 [{rho_min_filter}, {rho_max_filter}] 内没有找到数据")
            return
    
    # 颜色和标记设置
    markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', 'H']
    L_values = sorted(data.keys())

    colors = [
        '#4E79A7', '#F28E2C', '#E15759', '#76B7B2', '#59A14F',
        '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB',
        '#D37295', '#B07AA1', '#9D7660', '#D7B5A6', '#8CD17D',
        '#86BCB6', '#F1CE63', '#499894', '#79706E', '#D4A6C8'
    ]
    
    # 找出所有数据中rho的范围
    all_rho = []
    all_mean = []
    for L in data:
        all_rho.extend(data[L]['rho'])
        all_mean.extend(data[L]['mean'])
    
    if not all_rho:
        print("没有找到有效数据！")
        return
    
    rho_min, rho_max = min(all_rho), max(all_rho)
    mean_max = max(all_mean)
    
    # 如果指定了rho范围，使用指定范围
    if rho_range is not None:
        rho_min, rho_max = rho_range
        plot_xlim = (rho_min, rho_max)
    else:
        plot_xlim = (rho_min - 0.01, rho_max + 0.01)
    
    print(f"数据范围: rho [{rho_min:.3f}, {rho_max:.3f}], 最大密度: {mean_max:.3f}")
    if rho_range is not None:
        print(f"使用指定rho范围: [{rho_range[0]:.3f}, {rho_range[1]:.3f}]")
    
    # 绘制数据
    for idx, L in enumerate(L_values):
        values = data[L]
        rho_values = values['rho']
        means = values['mean']
        stddevs = values['stddev']
        color = colors[idx % len(colors)]

        # 绘制主曲线
        ax.plot(rho_values, means, 
               label=f'L={L}', 
               color=color,
               linewidth=2)
        
        # 绘制误差带
        ax.fill_between(rho_values,
                       np.array(means) - np.array(stddevs),
                       np.array(means) + np.array(stddevs),
                       alpha=0.2,
                       color=color)
    
    # 设置x轴（rho）
    ax.set_xlabel(r'$\rho$', fontsize=15)
    ax.set_xlim(plot_xlim[0], plot_xlim[1])
    
    # 设置y轴
    ax.set_ylabel('Maximum cluster ratio', fontsize=15)
    ax.set_ylim(0, min(1.0, mean_max * 1.1))  # 确保y轴不超过1.0（密度的最大值）
    
    # 根据数据范围动态确定刻度间隔
    rho_range_plot = plot_xlim[1] - plot_xlim[0]
    if rho_range_plot > 0.5:
        major_step = 0.1
        decimal_places = 1
    elif rho_range_plot > 0.2:
        major_step = 0.05
        decimal_places = 2
    else:
        major_step = 0.02
        decimal_places = 3
    
    # 设置次刻度为主刻度的1/5
    minor_step = major_step / 5
    
    # 生成主刻度
    major_rho_ticks = np.arange(
        np.ceil(plot_xlim[0] / major_step) * major_step,
        np.floor(plot_xlim[1] / major_step) * major_step + major_step,
        major_step
    )
    
    # 生成次刻度
    minor_rho_ticks = np.arange(
        np.ceil(plot_xlim[0] / minor_step) * minor_step,
        np.floor(plot_xlim[1] / minor_step) * minor_step + minor_step,
        minor_step
    )
    
    # 设置刻度
    ax.set_xticks(major_rho_ticks)
    ax.set_xticklabels([compact_format(x, decimal_places) for x in major_rho_ticks])
    ax.set_xticks(minor_rho_ticks, minor=True)
    
    # 设置刻度样式
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置图例
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=False)
    
    # 设置网格
    # ax.grid(which='major', color='#A0C8E0', linestyle='-', linewidth=0.5, alpha=0.7)
    # ax.grid(which='minor', color='#A0C8E0', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # 使用安全的标题
    title = get_safe_title(rho_range)
    ax.set_title(title, fontsize=16, pad=20)
    
    # 设置数学文本
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    # 调整布局
    plt.tight_layout()
    
    # 生成文件名后缀
    suffix = get_filename_suffix(rho_range)
    
    # 保存图片
    base_filename = f"rho-max_cluster_density{suffix}"
    output_path_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
    output_path_png = os.path.join(output_dir, f"{base_filename}.png")
    
    try:
        plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
        print(f"图片已保存至: {output_path_pdf}")
    except Exception as e:
        print(f"保存PDF失败: {e}")
        output_path_pdf = output_path_pdf.replace('.pdf', '_backup.pdf')
        plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
        print(f"图片已保存至备用文件: {output_path_pdf}")
    
    try:
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        print(f"图片已保存至: {output_path_png}")
    except Exception as e:
        print(f"保存PNG失败: {e}")
        output_path_png = output_path_png.replace('.png', '_backup.png')
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        print(f"图片已保存至备用文件: {output_path_png}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    filename = f"C:/Users/admin/Downloads/MIS-3D-results/max-cluster-density-stat-3d-rho.txt"
    print(f"正在读取文件: {filename}")

    # 读取数据
    data = read_max_cluster_density_data(filename)

    if not data:
        print("没有读取到数据，请检查文件路径和格式。")
    else:
        print(f"成功读取数据，包含 {len(data)} 个不同的L值:")
        for L in sorted(data.keys()):
            print(f"  L={L}: {len(data[L]['rho'])} 个rho值")
        
        rho_range = (0.2, 0.31)

        if rho_range != None:

            print("\n=== 使用指定rho范围绘图 ===")
        
            plot_max_cluster_density(data, rho_range=rho_range)
        else:
            
            print("\n=== 使用全部数据绘图 ===")
            plot_max_cluster_density(data)
        
        
        
