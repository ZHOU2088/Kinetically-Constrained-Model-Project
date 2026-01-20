import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import re
from datetime import datetime

# 设置中文字体（如果需要）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def read_percolation_data(input_file: str) -> Dict[Tuple[int, float], Dict]:
    """
    读取渗流模拟数据
    
    参数:
    input_file: 输入数据文件路径
    
    返回:
    处理后的数据字典，键为 (L, rho)，值为统计量字典
    """
    statistics_dict = {}
    
    print(f"正在读取文件: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            # 读取头部
            header = f.readline().strip()
            print(f"文件头部: {header}")
            
            # 检查文件格式
            expected_headers = [
                "L,rho,mean_density,std_density,second_moment,fourth_moment,Binder_cumulant",
                "L,rho,mean_density,std_density,second_moment,fourth_moment,Binder_cumulant\n"
            ]
            
            if header not in expected_headers:
                print(f"警告: 文件头部与预期格式不符")
                print(f"预期: '{expected_headers[0]}'")
                print(f"实际: '{header}'")
            
            line_count = 0
            error_count = 0
            
            for line in f:
                line_count += 1
                line = line.strip()
                
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 分割行
                    parts = [p.strip() for p in line.split(',')]
                    
                    if len(parts) < 7:
                        print(f"警告: 第{line_count}行列数不足，跳过")
                        error_count += 1
                        continue
                    
                    # 解析数据
                    L = int(parts[0])
                    rho = float(parts[1])
                    mean_density = float(parts[2])
                    std_density = float(parts[3])
                    second_moment = float(parts[4])
                    fourth_moment = float(parts[5])
                    binder_cumulant = float(parts[6])
                    
                    # 存储结果
                    key = (L, rho)
                    statistics_dict[key] = {
                        'mean': mean_density,
                        'std': std_density,
                        'second_moment': second_moment,
                        'fourth_moment': fourth_moment,
                        'binder_cumulant': binder_cumulant
                    }
                    
                    if line_count % 1000 == 0:
                        print(f"已读取 {line_count} 行...")
                        
                except ValueError as e:
                    print(f"解析第{line_count}行时出错: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"处理第{line_count}行时发生未知错误: {e}")
                    error_count += 1
                    continue
            
            print(f"读取完成！")
            print(f"共读取 {line_count} 行，成功 {line_count - error_count} 行，失败 {error_count} 行")
            print(f"总数据点数: {len(statistics_dict)}")
            
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 不存在")
        return {}
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return {}
    
    return statistics_dict

def compact_format(x, decimal_places):
    """紧凑格式化数字"""
    format_str = f"{{:.{decimal_places}f}}"
    formatted = format_str.format(x).rstrip('0').rstrip('.')
    return formatted

def filter_data_by_rho_range(data: Dict, L_values: List[int], rho_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    根据L值和rho范围过滤数据
    """
    filtered_data = {}
    
    for (L, rho), stats in data.items():
        if L in L_values:
            if rho_range is None or (rho_range[0] <= rho <= rho_range[1]):
                if L not in filtered_data:
                    filtered_data[L] = {'rhos': [], 'stats': []}
                filtered_data[L]['rhos'].append(rho)
                filtered_data[L]['stats'].append(stats)
    
    # 对每个L的数据按rho排序
    for L in filtered_data:
        sort_idx = np.argsort(filtered_data[L]['rhos'])
        filtered_data[L]['rhos'] = [filtered_data[L]['rhos'][i] for i in sort_idx]
        filtered_data[L]['stats'] = [filtered_data[L]['stats'][i] for i in sort_idx]
    
    return filtered_data

def plot_max_cluster_density(data: Dict, L_values: List[int], 
                           rho_range: Optional[Tuple[float, float]] = None,
                           output_dir: str = "./plots",
                           base_filename: str = "max_cluster_density"):
    """
    绘制最大簇密度均值与rho的关系图
    
    参数:
    data: 原始数据字典
    L_values: 要绘制的系统尺寸列表
    rho_range: rho范围 (rho_min, rho_max) 或 None
    output_dir: 输出目录
    base_filename: 基础文件名
    """
    # 过滤数据
    filtered_data = filter_data_by_rho_range(data, L_values, rho_range)
    
    if not filtered_data:
        print("警告: 没有找到符合条件的数据")
        return
    
    # 创建图形
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.6, 6.45), dpi=300)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色设置
    colors = [
        '#4E79A7', '#F28E2C', '#E15759', '#76B7B2', '#59A14F',
        '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB',
        '#D37295', '#B07AA1', '#9D7660', '#D7B5A6', '#8CD17D',
        '#86BCB6', '#F1CE63', '#499894', '#79706E', '#D4A6C8'
    ]
    
    # 收集所有rhos用于确定坐标轴范围
    all_rhos = []
    all_means = []
    
    # 绘制数据
    for idx, L in enumerate(sorted(filtered_data.keys())):
        if L not in L_values:
            continue
            
        rhos = filtered_data[L]['rhos']
        stats_list = filtered_data[L]['stats']
        
        if not rhos:
            continue
            
        # 提取均值和标准差
        means = [s['mean'] for s in stats_list]
        stds = [s['std'] for s in stats_list]
        
        color = colors[idx % len(colors)]
        
        # 绘制主曲线
        ax.plot(rhos, means, 
               label=f'L={L}', 
               color=color,
               linewidth=2)
        
        # 绘制误差带
        ax.fill_between(rhos,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2,
                       color=color)
        
        all_rhos.extend(rhos)
        all_means.extend(means)
    
    if not all_rhos:
        return
    
    # 确定坐标轴范围
    rho_min, rho_max = min(all_rhos), max(all_rhos)
    mean_max = max(all_means)
    
    if rho_range is not None:
        plot_xlim = (rho_range[0], rho_range[1])
    else:
        plot_xlim = (rho_min - 0.01, rho_max + 0.01)
    
    # 设置x轴
    ax.set_xlabel(r'$\rho$', fontsize=15)
    ax.set_xlim(plot_xlim[0], plot_xlim[1])
    
    # 设置y轴
    ax.set_ylabel('Maximum cluster density', fontsize=15)
    ax.set_ylim(0, min(1.0, mean_max * 1.1))
    
    # 根据数据范围动态确定刻度间隔
    rho_range_plot = plot_xlim[1] - plot_xlim[0]
    if rho_range_plot > 0.5:
        major_step = 0.05
        decimal_places = 2
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
    
    # 设置标题
    if rho_range is not None:
        title = f'Maximum Cluster Density vs ρ (ρ: {rho_range[0]:.3f}-{rho_range[1]:.3f})'
    else:
        title = 'Maximum Cluster Density vs ρ'
    ax.set_title(title, fontsize=16, pad=20)
    
    # 设置数学文本
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    # 调整布局
    plt.tight_layout()
    
    # 生成文件名后缀
    L_str = "_".join(str(L) for L in sorted(L_values))
    suffix = ""
    if rho_range is not None:
        rho_min_str, rho_max_str = f"{rho_range[0]:.3f}", f"{rho_range[1]:.3f}"
        suffix = f"_L{L_str}_rho{rho_min_str}to{rho_max_str}".replace('.', 'p')
    else:
        suffix = f"_L{L_str}"
    
    # 保存图片
    output_path_pdf = os.path.join(output_dir, f"{base_filename}{suffix}.pdf")
    output_path_png = os.path.join(output_dir, f"{base_filename}{suffix}.png")
    
    plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"最大簇密度图已保存:")
    print(f"  PDF: {output_path_pdf}")
    print(f"  PNG: {output_path_png}")
    
    # 显示图片
    plt.show()

def plot_binder_cumulant(data: Dict, L_values: List[int],
                        rho_range: Optional[Tuple[float, float]] = None,
                        output_dir: str = "./plots",
                        base_filename: str = "binder_cumulant"):
    """
    绘制Binder累计量与rho的关系图
    
    参数:
    data: 原始数据字典
    L_values: 要绘制的系统尺寸列表
    rho_range: rho范围 (rho_min, rho_max) 或 None
    output_dir: 输出目录
    base_filename: 基础文件名
    """
    # 过滤数据
    filtered_data = filter_data_by_rho_range(data, L_values, rho_range)
    
    if not filtered_data:
        print("警告: 没有找到符合条件的数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8.6, 6.45), dpi=300)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色设置
    colors = [
        '#4E79A7', '#F28E2C', "#D11619", '#76B7B2', '#59A14F',
        '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB',
        '#D37295', '#B07AA1', '#9D7660', '#D7B5A6', '#8CD17D',
        '#86BCB6', '#F1CE63', '#499894', '#79706E', '#D4A6C8'
    ]
    
    # 收集所有rhos用于确定坐标轴范围
    all_rhos = []
    all_binders = []
    
    # 绘制数据
    for idx, L in enumerate(sorted(filtered_data.keys())):
        if L not in L_values:
            continue
            
        rhos = filtered_data[L]['rhos']
        stats_list = filtered_data[L]['stats']
        
        if not rhos:
            continue
            
        # 提取Binder累计量
        binders = [s['binder_cumulant'] for s in stats_list]
        
        color = colors[idx % len(colors)]
        
        # 绘制曲线
        ax.plot(rhos, binders, 
               label=f'L={L}', 
               color=color,
               linewidth=2,
               marker='o',
               markersize=4,
               markerfacecolor='none',
               markeredgecolor=color,
               markeredgewidth=1.5)
        
        all_rhos.extend(rhos)
        all_binders.extend(binders)
    
    if not all_rhos:
        return
    
    # 确定坐标轴范围
    rho_min, rho_max = min(all_rhos), max(all_rhos)
    
    if rho_range is not None:
        plot_xlim = (rho_range[0], rho_range[1])
    else:
        plot_xlim = (rho_min - 0.01, rho_max + 0.01)
    
    # 设置x轴
    ax.set_xlabel(r'$\rho$', fontsize=15)
    ax.set_xlim(plot_xlim[0], plot_xlim[1])
    
    # 设置y轴
    ax.set_ylabel('Binder cumulant', fontsize=15)
    
    # 根据数据范围自动调整y轴范围
    if all_binders:
        binder_min, binder_max = min(all_binders), max(all_binders)
        binder_range = binder_max - binder_min
        ax.set_ylim(binder_min - 0.1 * binder_range, binder_max + 0.1 * binder_range)
    
    # 根据数据范围动态确定刻度间隔
    rho_range_plot = plot_xlim[1] - plot_xlim[0]
    if rho_range_plot > 0.5:
        major_step = 0.1
        decimal_places = 2
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
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 设置标题
    if rho_range is not None:
        title = f'Binder Cumulant vs ρ (ρ: {rho_range[0]:.3f}-{rho_range[1]:.3f})'
    else:
        title = 'Binder Cumulant vs ρ'
    ax.set_title(title, fontsize=16, pad=20)
    
    # 添加参考线 y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 设置数学文本
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    # 调整布局
    plt.tight_layout()
    
    # 生成文件名后缀
    L_str = "_".join(str(L) for L in sorted(L_values))
    suffix = ""
    if rho_range is not None:
        rho_min_str, rho_max_str = f"{rho_range[0]:.3f}", f"{rho_range[1]:.3f}"
        suffix = f"_L{L_str}_rho{rho_min_str}to{rho_max_str}".replace('.', 'p')
    else:
        suffix = f"_L{L_str}"
    
    # 保存图片
    output_path_pdf = os.path.join(output_dir, f"{base_filename}{suffix}.pdf")
    output_path_png = os.path.join(output_dir, f"{base_filename}{suffix}.png")
    
    plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"Binder累计量图已保存:")
    print(f"  PDF: {output_path_pdf}")
    print(f"  PNG: {output_path_png}")
    
    # 显示图片
    plt.show()

def main():
    """主函数"""
    # 配置参数
    input_file = "C:/Users/admin/Downloads/percolation_batch_1/percolation_stat.txt"
    output_dir = "./percolation_plots"
    
    print("="*60)
    print("渗流模拟数据可视化工具")
    print("="*60)
    
    # 读取数据
    data = read_percolation_data(input_file)
    
    if not data:
        print("没有成功读取数据")
        return
    
    # 获取文件中所有可用的L值
    all_L_values = sorted(set(L for L, _ in data.keys()))
    print(f"文件中包含的系统尺寸: {all_L_values}")
    
    # 配置绘图参数
    L_values_all = all_L_values
    L_values_binder = [120,140,180,220,260,300,340,380,420]
    rho_range = (0.0,0.52)

    rho_range_binder = (0.245, 0.275)
    print("\n" + "="*60)
    print("绘制最大簇密度图")
    print("="*60)
    print(f"系统尺寸: {L_values_all}")
    if rho_range:
        print(f"rho范围: {rho_range}")
    
    plot_max_cluster_density(
        data=data,
        L_values=L_values_all,
        rho_range=rho_range,
        output_dir=output_dir,
        base_filename="max_cluster_density"
    )

    # 2. 绘制Binder累计量图
    print("\n" + "="*60)
    print("绘制Binder累计量图")
    print("="*60)
    print(f"系统尺寸: {L_values_all}")
    if rho_range:
        print(f"rho范围: {rho_range}")
    
    plot_binder_cumulant(
        data=data,
        L_values=L_values_binder,
        rho_range=rho_range_binder,
        output_dir=output_dir,
        base_filename="binder_cumulant"
    )


    print("\n" + "="*60)
    print("绘图完成！所有图形已保存到:")
    print(f"  {os.path.abspath(output_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()