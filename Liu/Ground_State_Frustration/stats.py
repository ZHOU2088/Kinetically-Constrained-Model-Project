import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import digamma, polygamma
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class ProportionStatistics:
    """专门处理比例数据的统计类"""
    
    def __init__(self):
        pass
    
    def dirichlet_log_likelihood(self, alpha, data):
        """计算Dirichlet分布的对数似然函数"""
        n, k = data.shape
        log_likelihood = 0
        
        for i in range(n):
            # 避免对0取对数
            safe_data = np.clip(data[i], 1e-10, 1-1e-10)
            log_likelihood += np.sum((alpha - 1) * np.log(safe_data))
        
        # 添加对数Gamma项
        log_likelihood -= n * (np.sum([np.log(np.math.gamma(a)) for a in alpha]) - 
                              np.log(np.math.gamma(np.sum(alpha))))
        
        return -log_likelihood  # 返回负对数似然用于最小化
    
    def dirichlet_mle(self, data, initial_alpha=None, prior_type='uniform'):
        """使用最大似然估计Dirichlet分布参数，支持不同先验"""
        n, k = data.shape
        
        if initial_alpha is None:
            if prior_type == 'uniform':
                # 均匀先验
                initial_alpha = np.ones(k)
            elif prior_type == 'jeffreys':
                # Jeffreys先验
                initial_alpha = np.full(k, 0.5)
            elif prior_type == 'adaptive':
                # 自适应先验
                mean_val = np.mean(data, axis=0)
                if n < 50:  # 小样本
                    initial_alpha = mean_val * 20 + 0.05
                else:  # 大样本
                    initial_alpha = mean_val * 5 + 0.01
            else:  # 默认使用改进的矩估计
                mean_val = np.mean(data, axis=0)
                var_val = np.var(data, axis=0)
                
                safe_var = np.where(var_val < 1e-10, 1e-10, var_val)
                alpha0 = mean_val * (mean_val * (1 - mean_val) / safe_var - 1)
                
                # 稳健化处理
                alpha0 = np.clip(alpha0, 0.1, 100)  # 设置合理范围
                initial_alpha = alpha0
        
        # 其余优化代码保持不变
        bounds = [(1e-10, None) for _ in range(k)]
        result = minimize(self.dirichlet_log_likelihood, initial_alpha, 
                        args=(data,), bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return result.x
        else:
            print("MLE optimization failed, using initial estimation")
            return initial_alpha
        
    def dirichlet_confidence_interval(self, data, alpha=0.05):
        """基于Dirichlet分布计算置信区间"""
        n, k = data.shape
        
        if n < 2:
            # 样本太少，使用传统方法
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            ci_lower = np.maximum(0, mean_val - 1.96 * std_val / np.sqrt(n))
            ci_upper = np.minimum(1, mean_val + 1.96 * std_val / np.sqrt(n))
            confidence_intervals = list(zip(ci_lower, ci_upper))
            return mean_val, confidence_intervals, None
        
        # 估计Dirichlet参数
        try:
            alpha_params = self.dirichlet_mle(data)
            alpha0 = np.sum(alpha_params)
            
            # 计算均值和方差
            mean_val = alpha_params / alpha0
            
            # 使用Beta分布近似计算每个分量的置信区间
            from scipy.stats import beta as beta_dist
            confidence_intervals = []
            for i in range(k):
                # 每个分量边缘分布是Beta分布
                a = alpha_params[i]
                b = alpha0 - alpha_params[i]
                
                # 计算置信区间
                if a > 0 and b > 0:
                    lower = beta_dist.ppf(alpha/2, a, b)
                    upper = beta_dist.ppf(1 - alpha/2, a, b)
                else:
                    # 如果参数无效，使用传统方法
                    lower = max(0, mean_val[i] - 1.96 * np.std(data[:, i]) / np.sqrt(n))
                    upper = min(1, mean_val[i] + 1.96 * np.std(data[:, i]) / np.sqrt(n))
                
                confidence_intervals.append((lower, upper))
            
            return mean_val, confidence_intervals, alpha_params
        except Exception as e:
            print(f"Dirichlet estimation error: {e}")
            # 回退到传统方法
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            ci_lower = np.maximum(0, mean_val - 1.96 * std_val / np.sqrt(n))
            ci_upper = np.minimum(1, mean_val + 1.96 * std_val / np.sqrt(n))
            confidence_intervals = list(zip(ci_lower, ci_upper))
            return mean_val, confidence_intervals, None
    
    def bootstrap_dirichlet_ci(self, data, n_bootstrap=1000, alpha=0.05):
        """使用Bootstrap方法计算Dirichlet分布的置信区间"""
        n, k = data.shape
        
        if n < 2:
            # 样本太少，使用传统方法
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            ci_lower = np.maximum(0, mean_val - 1.96 * std_val / np.sqrt(n))
            ci_upper = np.minimum(1, mean_val + 1.96 * std_val / np.sqrt(n))
            confidence_intervals = list(zip(ci_lower, ci_upper))
            return mean_val, confidence_intervals, None
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n, n, replace=True)
            bootstrap_sample = data[indices]
            
            # 估计Dirichlet参数
            try:
                alpha_params = self.dirichlet_mle(bootstrap_sample)
                alpha0 = np.sum(alpha_params)
                mean_val = alpha_params / alpha0
                bootstrap_means.append(mean_val)
            except:
                # 如果Dirichlet估计失败，使用样本均值
                bootstrap_means.append(np.mean(bootstrap_sample, axis=0))
        
        if not bootstrap_means:
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            ci_lower = np.maximum(0, mean_val - 1.96 * std_val / np.sqrt(n))
            ci_upper = np.minimum(1, mean_val + 1.96 * std_val / np.sqrt(n))
            confidence_intervals = list(zip(ci_lower, ci_upper))
            return mean_val, confidence_intervals, None
        
        bootstrap_means = np.array(bootstrap_means)
        mean_val = np.mean(bootstrap_means, axis=0)
        lower = np.percentile(bootstrap_means, 100 * alpha / 2, axis=0)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2), axis=0)
        
        return mean_val, list(zip(lower, upper)), None


class ScientificDataAnalyzer:
    def __init__(self, d=2, seed=114514, boundary_condition='Periodic'):
        """
        初始化数据分析器
        
        参数:
        d: 系统维度
        seed: 随机数种子
        boundary_condition: 边界条件
        """
        self.d = d
        self.seed = seed
        self.boundary_condition = boundary_condition
        self.results = {}  # 存储分析结果
        self.prop_stats = ProportionStatistics()  # 添加比例统计对象
        
    def process_folder(self, folder_path):
        """处理包含多个样本的文件夹"""
        # 查找所有样本文件夹
        sample_folders = [f for f in os.listdir(folder_path) 
                        if f.startswith('sample_') and os.path.isdir(os.path.join(folder_path, f))]
        
        if not sample_folders:
            print(f"Warning: No sample folders found in {folder_path}")
            return None
        
        sample_results = []
        
        for sample_folder in sample_folders:
            sample_path = os.path.join(folder_path, sample_folder)
            data_path = os.path.join(sample_path, "restricted_result.txt")
            
            if not os.path.exists(data_path):
                print(f"Warning: File not found {data_path}")
                continue
                
            try:
                # 读取单个样本数据
                df = pd.read_csv(data_path, header=None, skiprows=1, 
                            names=['index', 'occupation', 'change_number', 'change_energy', 'Is_in_largest_connected_component'])
                
                # 计算单个样本的统计量
                sample_stat = self._calculate_sample_statistics(df)
                sample_results.append(sample_stat)
                
            except Exception as e:
                print(f"Error processing sample {sample_folder}: {e}")
                continue
        
        if not sample_results:
            return None
    
        # 计算多个样本的均值和标准差（传统方法和Dirichlet方法）
        traditional_stats = self._calculate_ensemble_statistics(sample_results, folder_path)
        dirichlet_stats = self._calculate_ensemble_statistics_dirichlet(sample_results, folder_path)
        
        # 合并两种方法的统计结果
        if traditional_stats and dirichlet_stats:
            combined_stats = {**traditional_stats, **dirichlet_stats}
            return combined_stats
        elif traditional_stats:
            return traditional_stats
        else:
            return None
    
    def _calculate_sample_statistics(self, df):
        """计算单个样本的统计量"""
        total_points = len(df)
        
        # 1. 能量变化为0的格点比例
        energy_zero_points = len(df[df['change_energy'] == 0])
        ratio_energy_zero = energy_zero_points / total_points
        
        # 2. 能量变化不为0，占据为1的格点比例
        energy_nonzero = df[df['change_energy'] != 0]
        energy_nonzero_occ1 = len(energy_nonzero[energy_nonzero['occupation'] == 1])
        ratio_nonzero_occ1 = energy_nonzero_occ1 / total_points
        
        # 3. 能量变化不为0，占据为0的格点比例
        energy_nonzero_occ0 = len(energy_nonzero[energy_nonzero['occupation'] == 0])
        ratio_nonzero_occ0 = energy_nonzero_occ0 / total_points
        
        # 4. 最大联通分量内的格点统计（只考虑Is_in_largest_connected_component=1的数据）
        largest_component_df = df[df['Is_in_largest_connected_component'] == 1]
        largest_component_points = len(largest_component_df)
        
        if largest_component_points > 0:
            # 最大联通分量内的能量变化为0的格点比例
            lc_energy_zero_points = len(largest_component_df[largest_component_df['change_energy'] == 0])
            lc_ratio_energy_zero = lc_energy_zero_points / largest_component_points
            
            # 最大联通分量内能量变化不为0，占据为1的格点比例
            lc_energy_nonzero = largest_component_df[largest_component_df['change_energy'] != 0]
            lc_energy_nonzero_occ1 = len(lc_energy_nonzero[lc_energy_nonzero['occupation'] == 1])
            lc_ratio_nonzero_occ1 = lc_energy_nonzero_occ1 / largest_component_points
            
            # 最大联通分量内能量变化不为0，占据为0的格点比例
            lc_energy_nonzero_occ0 = len(lc_energy_nonzero[lc_energy_nonzero['occupation'] == 0])
            lc_ratio_nonzero_occ0 = lc_energy_nonzero_occ0 / largest_component_points
            
            # 最大联通分量内的其他统计量
            lc_mean_change_number = largest_component_df['change_number'].mean()
            lc_std_change_number = largest_component_df['change_number'].std()
            lc_mean_energy_change = largest_component_df['change_energy'].mean()
            lc_std_energy_change = largest_component_df['change_energy'].std()
        else:
            lc_ratio_energy_zero = 0
            lc_ratio_nonzero_occ1 = 0
            lc_ratio_nonzero_occ0 = 0
            lc_mean_change_number = 0
            lc_std_change_number = 0
            lc_mean_energy_change = 0
            lc_std_energy_change = 0
        
        # 其他统计量
        mean_change_number = df['change_number'].mean()
        std_change_number = df['change_number'].std()
        mean_energy_change = df['change_energy'].mean()
        std_energy_change = df['change_energy'].std()
        
        return {
            'total_points': total_points,
            'ratio_energy_zero': ratio_energy_zero,
            'ratio_nonzero_occ1': ratio_nonzero_occ1,
            'ratio_nonzero_occ0': ratio_nonzero_occ0,
            'mean_change_number': mean_change_number,
            'std_change_number': std_change_number,
            'mean_energy_change': mean_energy_change,
            'std_energy_change': std_energy_change,
            'largest_component_points': largest_component_points,
            'lc_ratio_energy_zero': lc_ratio_energy_zero,
            'lc_ratio_nonzero_occ1': lc_ratio_nonzero_occ1,
            'lc_ratio_nonzero_occ0': lc_ratio_nonzero_occ0,
            'lc_mean_change_number': lc_mean_change_number,
            'lc_std_change_number': lc_std_change_number,
            'lc_mean_energy_change': lc_mean_energy_change,
            'lc_std_energy_change': lc_std_energy_change
        }

    def _calculate_ensemble_statistics(self, sample_results, folder_path):
        """计算多个样本的集成统计量（均值和标准差）"""
        # 解析文件夹名获取参数
        folder_name = os.path.basename(folder_path)
        params = self._parse_folder_name(folder_name)
        
        # 将样本结果转换为DataFrame以便计算
        df_samples = pd.DataFrame(sample_results)
        
        ensemble_stats = {
            'L': params.get('L'),
            'W': params.get('W'),
            'rho': params.get('rho'),
            'n_samples': len(sample_results)
        }
        
        # 计算每个统计量的均值和标准差
        for stat in ['ratio_energy_zero', 'ratio_nonzero_occ1', 'ratio_nonzero_occ0', 
                    'mean_change_number', 'mean_energy_change']:
            values = df_samples[stat].values
            ensemble_stats[f'{stat}_mean'] = np.mean(values)
            ensemble_stats[f'{stat}_std'] = np.std(values)
        
        # 计算最大联通分量内统计量的均值和标准差
        for stat in ['lc_ratio_energy_zero', 'lc_ratio_nonzero_occ1', 'lc_ratio_nonzero_occ0',
                    'lc_mean_change_number', 'lc_mean_energy_change']:
            values = df_samples[stat].values
            ensemble_stats[f'{stat}_mean'] = np.mean(values)
            ensemble_stats[f'{stat}_std'] = np.std(values)
        
        # 计算最大联通分量内格点数量的统计
        lc_points_values = df_samples['largest_component_points'].values
        ensemble_stats['largest_component_points_mean'] = np.mean(lc_points_values)
        ensemble_stats['largest_component_points_std'] = np.std(lc_points_values)
        
        return ensemble_stats

    def _calculate_ensemble_statistics_dirichlet(self, sample_results, folder_path):
        """使用Dirichlet分布计算集成统计量"""
        # 解析文件夹名获取参数
        folder_name = os.path.basename(folder_path)
        params = self._parse_folder_name(folder_name)
        
        # 将样本结果转换为DataFrame以便计算
        df_samples = pd.DataFrame(sample_results)
        
        ensemble_stats = {
            'L': params.get('L'),
            'W': params.get('W'),
            'rho': params.get('rho'),
            'n_samples': len(sample_results)
        }
        
        # 准备比例数据（三个比例）
        ratio_data = df_samples[['ratio_energy_zero', 'ratio_nonzero_occ1', 'ratio_nonzero_occ0']].values
        
        # 使用Dirichlet分布估计
        try:
            mean_val, ci_intervals, alpha_params = self.prop_stats.dirichlet_confidence_interval(ratio_data)
            
            # 存储Dirichlet估计结果
            for i, ratio_name in enumerate(['ratio_energy_zero', 'ratio_nonzero_occ1', 'ratio_nonzero_occ0']):
                ensemble_stats[f'{ratio_name}_dirichlet_mean'] = mean_val[i]
                ensemble_stats[f'{ratio_name}_dirichlet_ci_lower'] = ci_intervals[i][0]
                ensemble_stats[f'{ratio_name}_dirichlet_ci_upper'] = ci_intervals[i][1]
            
            # 存储Dirichlet参数
            if alpha_params is not None:
                ensemble_stats['dirichlet_alpha_params'] = alpha_params
            
        except Exception as e:
            print(f"Dirichlet estimation failed: {e}")
            # 回退到传统方法
            for ratio_name in ['ratio_energy_zero', 'ratio_nonzero_occ1', 'ratio_nonzero_occ0']:
                values = df_samples[ratio_name].values
                ensemble_stats[f'{ratio_name}_dirichlet_mean'] = np.mean(values)
                ensemble_stats[f'{ratio_name}_dirichlet_ci_lower'] = max(0, np.mean(values) - 1.96 * np.std(values))
                ensemble_stats[f'{ratio_name}_dirichlet_ci_upper'] = min(1, np.mean(values) + 1.96 * np.std(values))
        
        # 同时计算Bootstrap Dirichlet CI作为比较
        try:
            mean_val_bs, ci_intervals_bs, _ = self.prop_stats.bootstrap_dirichlet_ci(ratio_data)
            for i, ratio_name in enumerate(['ratio_energy_zero', 'ratio_nonzero_occ1', 'ratio_nonzero_occ0']):
                ensemble_stats[f'{ratio_name}_bootstrap_dirichlet_mean'] = mean_val_bs[i]
                ensemble_stats[f'{ratio_name}_bootstrap_dirichlet_ci_lower'] = ci_intervals_bs[i][0]
                ensemble_stats[f'{ratio_name}_bootstrap_dirichlet_ci_upper'] = ci_intervals_bs[i][1]
        except Exception as e:
            print(f"Bootstrap Dirichlet estimation failed: {e}")
        
        return ensemble_stats

    def _parse_folder_name(self, folder_name):
        """解析文件夹名称获取参数"""
        params = {}
        parts = folder_name.split('-')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=')
                if key in ['L', 'W']:
                    params[key] = int(value)
                elif key == 'rho':
                    params[key] = float(value)
                else:
                    params[key] = value
        
        return params
    
    def collect_data(self, L_range, rho_range, base_path='.'):
        """收集所有指定参数的数据（支持多个样本）"""
        all_results = {}
        
        for L in L_range:
            for rho in rho_range:
                # 构建文件夹名
                # 智能处理浮点数格式化，避免精度问题
                if abs(rho - round(rho, 2)) < 1e-10:
                    rho_str = f"{rho:.2f}"  # 两位小数
                elif abs(rho - round(rho, 1)) < 1e-10:
                    rho_str = f"{rho:.1f}"  # 一位小数
                else:
                    rho_str = str(rho)
                
                # 去除可能的尾随零
                if '.' in rho_str:
                    rho_str = rho_str.rstrip('0').rstrip('.')
                folder_name = f"d={self.d}-L={L}-W={L}-{self.boundary_condition}-rho={rho_str}-main_seed={self.seed}"
                folder_path = os.path.join(base_path, folder_name)
                
                if os.path.exists(folder_path):
                    print(f"Processing folder: {folder_name}")
                    result = self.process_folder(folder_path)
                    
                    if result is not None:
                        key = (L, rho)
                        all_results[key] = result
                        print(f"  Processed {result['n_samples']} samples")
                else:
                    # 尝试其他可能的格式化方式
                    alternative_formats = [
                        f"{rho:.2f}",  # 两位小数
                        f"{rho:.1f}",  # 一位小数
                        str(rho),      # 原始字符串
                    ]
                    
                    found = False
                    for fmt in alternative_formats:
                        alt_folder_name = f"d={self.d}-L={L}-W={L}-{self.boundary_condition}-rho={fmt}-main_seed={self.seed}"
                        alt_folder_path = os.path.join(base_path, alt_folder_name)
                        
                        if os.path.exists(alt_folder_path):
                            print(f"Found alternative format: {alt_folder_name}")
                            result = self.process_folder(alt_folder_path)
                            
                            if result is not None:
                                key = (L, rho)
                                all_results[key] = result
                                print(f"  Processed {result['n_samples']} samples")
                                found = True
                                break
                    
                    if not found:
                        print(f"Warning: Folder not found for rho={rho} (tried: {rho_str} and alternatives)")
        
        self.results = all_results
        return all_results
    
    def generate_summary_statistics(self):
        """生成数据汇总统计（传统方法）"""
        if not self.results:
            print("No data available")
            return None
        
        summary_data = []
        
        for (L, rho), result in self.results.items():
            # 从集成统计量中获取数据
            summary_data.append({
                'L': L,
                'rho': rho,
                'Total Points': result.get('n_samples', 0) * (L * L),
                'No Prefer Ratio': result.get('ratio_energy_zero_mean', 0),
                'Prefer Occupation Ratio': result.get('ratio_nonzero_occ1_mean', 0),
                'Prefer Unoccupation Ratio': result.get('ratio_nonzero_occ0_mean', 0),
                'Mean Change Number': result.get('mean_change_number_mean', 0),
                'Std Change Number': result.get('mean_change_number_std', 0),
                'Mean Energy Change': result.get('mean_energy_change_mean', 0),
                'Std Energy Change': result.get('mean_energy_change_std', 0),
                'Largest Component Size': result.get('largest_component_points_mean', 0),
                'Largest Component Ratio': result.get('largest_component_points_mean', 0) / (L * L),
                'LC No Prefer Ratio': result.get('lc_ratio_energy_zero_mean', 0),
                'LC Prefer Occupation Ratio': result.get('lc_ratio_nonzero_occ1_mean', 0),
                'LC Prefer Unoccupation Ratio': result.get('lc_ratio_nonzero_occ0_mean', 0),
                'Number of Samples': result.get('n_samples', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def generate_dirichlet_summary_statistics(self):
        """生成Dirichlet方法的数据汇总统计"""
        if not self.results:
            print("No data available")
            return None
        
        # 创建三个不同的DataFrame：传统方法、Dirichlet MLE、Bootstrap Dirichlet
        traditional_data = []
        dirichlet_data = []
        bootstrap_dirichlet_data = []
        
        for (L, rho), result in self.results.items():
            # 传统方法数据
            traditional_data.append({
                'L': L,
                'rho': rho,
                'No Prefer Ratio': result.get('ratio_energy_zero_mean', 0),
                'No Prefer Std': result.get('ratio_energy_zero_std', 0),
                'Prefer Occupation Ratio': result.get('ratio_nonzero_occ1_mean', 0),
                'Prefer Occupation Std': result.get('ratio_nonzero_occ1_std', 0),
                'Prefer Unoccupation Ratio': result.get('ratio_nonzero_occ0_mean', 0),
                'Prefer Unoccupation Std': result.get('ratio_nonzero_occ0_std', 0),
                'Number of Samples': result.get('n_samples', 0)
            })
            
            # Dirichlet MLE方法数据
            dirichlet_data.append({
                'L': L,
                'rho': rho,
                'No Prefer Ratio (Dirichlet)': result.get('ratio_energy_zero_dirichlet_mean', 0),
                'No Prefer CI Lower': result.get('ratio_energy_zero_dirichlet_ci_lower', 0),
                'No Prefer CI Upper': result.get('ratio_energy_zero_dirichlet_ci_upper', 0),
                'Prefer Occupation Ratio (Dirichlet)': result.get('ratio_nonzero_occ1_dirichlet_mean', 0),
                'Prefer Occupation CI Lower': result.get('ratio_nonzero_occ1_dirichlet_ci_lower', 0),
                'Prefer Occupation CI Upper': result.get('ratio_nonzero_occ1_dirichlet_ci_upper', 0),
                'Prefer Unoccupation Ratio (Dirichlet)': result.get('ratio_nonzero_occ0_dirichlet_mean', 0),
                'Prefer Unoccupation CI Lower': result.get('ratio_nonzero_occ0_dirichlet_ci_lower', 0),
                'Prefer Unoccupation CI Upper': result.get('ratio_nonzero_occ0_dirichlet_ci_upper', 0),
                'Number of Samples': result.get('n_samples', 0)
            })
            
            # Bootstrap Dirichlet方法数据
            bootstrap_dirichlet_data.append({
                'L': L,
                'rho': rho,
                'No Prefer Ratio (Bootstrap)': result.get('ratio_energy_zero_bootstrap_dirichlet_mean', 0),
                'No Prefer CI Lower (Bootstrap)': result.get('ratio_energy_zero_bootstrap_dirichlet_ci_lower', 0),
                'No Prefer CI Upper (Bootstrap)': result.get('ratio_energy_zero_bootstrap_dirichlet_ci_upper', 0),
                'Prefer Occupation Ratio (Bootstrap)': result.get('ratio_nonzero_occ1_bootstrap_dirichlet_mean', 0),
                'Prefer Occupation CI Lower (Bootstrap)': result.get('ratio_nonzero_occ1_bootstrap_dirichlet_ci_lower', 0),
                'Prefer Occupation CI Upper (Bootstrap)': result.get('ratio_nonzero_occ1_bootstrap_dirichlet_ci_upper', 0),
                'Prefer Unoccupation Ratio (Bootstrap)': result.get('ratio_nonzero_occ0_bootstrap_dirichlet_mean', 0),
                'Prefer Unoccupation CI Lower (Bootstrap)': result.get('ratio_nonzero_occ0_bootstrap_dirichlet_ci_lower', 0),
                'Prefer Unoccupation CI Upper (Bootstrap)': result.get('ratio_nonzero_occ0_bootstrap_dirichlet_ci_upper', 0),
                'Number of Samples': result.get('n_samples', 0)
            })
        
        traditional_df = pd.DataFrame(traditional_data)
        dirichlet_df = pd.DataFrame(dirichlet_data)
        bootstrap_dirichlet_df = pd.DataFrame(bootstrap_dirichlet_data)
        
        return {
            'traditional': traditional_df,
            'dirichlet': dirichlet_df,
            'bootstrap_dirichlet': bootstrap_dirichlet_df
        }
    
    def print_summary_statistics(self):
        """打印所有方法的汇总统计"""
        if not self.results:
            print("No data available")
            return
        
        # 生成传统方法统计
        print("\n" + "="*80)
        print("TRADITIONAL METHOD SUMMARY STATISTICS")
        print("="*80)
        traditional_df = self.generate_summary_statistics()
        if traditional_df is not None:
            print(traditional_df.to_string(index=False))
            traditional_df.to_csv('traditional_summary.csv', index=False)
            print("Traditional summary statistics saved to traditional_summary.csv")
        
        # 生成Dirichlet方法统计
        dirichlet_stats = self.generate_dirichlet_summary_statistics()
        if dirichlet_stats is not None:
            print("\n" + "="*80)
            print("DIRICHLET MLE METHOD SUMMARY STATISTICS")
            print("="*80)
            print(dirichlet_stats['dirichlet'].to_string(index=False))
            dirichlet_stats['dirichlet'].to_csv('dirichlet_summary.csv', index=False)
            print("Dirichlet MLE summary statistics saved to dirichlet_summary.csv")
            
            print("\n" + "="*80)
            print("BOOTSTRAP DIRICHLET METHOD SUMMARY STATISTICS")
            print("="*80)
            print(dirichlet_stats['bootstrap_dirichlet'].to_string(index=False))
            dirichlet_stats['bootstrap_dirichlet'].to_csv('bootstrap_dirichlet_summary.csv', index=False)
            print("Bootstrap Dirichlet summary statistics saved to bootstrap_dirichlet_summary.csv")
    
    def plot_ratios_vs_rho_dirichlet(self, save_path=None, largest_component=False, method='traditional', add_jitter=True):
        """统一绘图函数：绘制三种比值与密度ρ的关系图，支持三种方法"""
        if not self.results:
            print("No data available")
            return
        
        L_values = sorted(set([key[0] for key in self.results.keys()]))
        
        for L in L_values:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取该L值下所有rho的结果
            L_results = {}
            for key, result in self.results.items():
                if key[0] == L:
                    L_results[key[1]] = result
            
            if not L_results:
                continue
                
            sorted_rhos = sorted(L_results.keys())
            prefix = 'lc_' if largest_component else ''
            title_suffix = ' (Largest Component)' if largest_component else ''
            method_suffix = f' ({method.capitalize()} Method)' if method != 'traditional' else ''
            
            # 提取三个比值的均值和置信区间
            ratios_data = {}
            for ratio_name in ['energy_zero', 'nonzero_occ1', 'nonzero_occ0']:
                full_name = f'{prefix}ratio_{ratio_name}'
                
                if method == 'dirichlet':
                    # Dirichlet MLE方法
                    means = [L_results[rho].get(f'{full_name}_dirichlet_mean', np.nan) for rho in sorted_rhos]
                    lowers = [L_results[rho].get(f'{full_name}_dirichlet_ci_lower', np.nan) for rho in sorted_rhos]
                    uppers = [L_results[rho].get(f'{full_name}_dirichlet_ci_upper', np.nan) for rho in sorted_rhos]
                elif method == 'bootstrap_dirichlet':
                    # Bootstrap Dirichlet方法
                    means = [L_results[rho].get(f'{full_name}_bootstrap_dirichlet_mean', np.nan) for rho in sorted_rhos]
                    lowers = [L_results[rho].get(f'{full_name}_bootstrap_dirichlet_ci_lower', np.nan) for rho in sorted_rhos]
                    uppers = [L_results[rho].get(f'{full_name}_bootstrap_dirichlet_ci_upper', np.nan) for rho in sorted_rhos]
                else:  # 传统方法
                    means = [L_results[rho].get(f'{full_name}_mean', np.nan) for rho in sorted_rhos]
                    stds = [L_results[rho].get(f'{full_name}_std', np.nan) for rho in sorted_rhos]
                    
                    # 取消误差棒限制，直接计算置信区间
                    lowers = [means[i] - 1.96 * stds[i] for i in range(len(means))]
                    uppers = [means[i] + 1.96 * stds[i] for i in range(len(means))]
                
                ratios_data[ratio_name] = {
                    'means': means,
                    'lowers': lowers,
                    'uppers': uppers
                }
            
            # 为不同类型格点添加微小扰动（仅对传统方法）
            if add_jitter and method == 'traditional':
                # 计算rho的最小间隔，用于确定扰动大小
                if len(sorted_rhos) > 1:
                    min_interval = min([sorted_rhos[i+1] - sorted_rhos[i] for i in range(len(sorted_rhos)-1)])
                    jitter_amount = min_interval * 0.1  # 扰动大小为最小间隔的10%
                else:
                    jitter_amount = 0.001  # 默认扰动大小
                
                # 为每个类别添加不同的扰动
                jitter_energy_zero = [rho - jitter_amount for rho in sorted_rhos]      # 向左偏移
                jitter_nonzero_occ1 = sorted_rhos                                      # 不偏移
                jitter_nonzero_occ0 = [rho + jitter_amount for rho in sorted_rhos]     # 向右偏移
            else:
                jitter_energy_zero = sorted_rhos
                jitter_nonzero_occ1 = sorted_rhos
                jitter_nonzero_occ0 = sorted_rhos
            
            # 绘制带置信区间的数据点
            colors = ['blue', 'red', 'green']
            markers = ['o', 's', '^']
            labels = ['No Prefer', 'Prefer Occupation', 'Prefer Unoccupation']
            
            for i, (ratio_name, color, marker, label) in enumerate(
                zip(['energy_zero', 'nonzero_occ1', 'nonzero_occ0'], colors, markers, labels)):
                
                data = ratios_data[ratio_name]
                means = np.array(data['means'])
                lowers = np.array(data['lowers'])
                uppers = np.array(data['uppers'])
                
                # 选择x轴坐标（传统方法使用扰动坐标，其他方法使用原始坐标）
                if method == 'traditional':
                    if ratio_name == 'energy_zero':
                        x_coords = jitter_energy_zero
                    elif ratio_name == 'nonzero_occ1':
                        x_coords = jitter_nonzero_occ1
                    else:  # nonzero_occ0
                        x_coords = jitter_nonzero_occ0
                else:
                    x_coords = sorted_rhos
                
                # 绘制均值和置信区间
                ax.errorbar(x_coords, means, 
                        yerr=[means - lowers, uppers - means],
                        fmt=marker, markersize=8, capsize=5, capthick=2, elinewidth=2,
                        label=label, color=color, alpha=0.8)
            
            # 设置图形属性
            ax.set_title(f'Ratios vs Density ρ (L={L}{title_suffix}{method_suffix})', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Density ρ', fontsize=14)
            ax.set_ylabel('Ratio', fontsize=14)
            
            # 设置x轴刻度
            ax.set_xticks(sorted_rhos)
            if len(sorted_rhos) > 10:
                ax.set_xticks(sorted_rhos[::2])
            
            # 设置y轴范围（传统方法允许负值，其他方法限制在[0,1]）
            if method == 'traditional':
                # 计算y轴范围，考虑可能的负值
                all_means = np.concatenate([np.array(ratios_data[ratio]['means']) for ratio in ratios_data])
                all_lowers = np.concatenate([np.array(ratios_data[ratio]['lowers']) for ratio in ratios_data])
                all_uppers = np.concatenate([np.array(ratios_data[ratio]['uppers']) for ratio in ratios_data])
                
                y_min = min(np.nanmin(all_lowers), -0.05)  # 允许负值，但最小不超过-0.05
                y_max = max(np.nanmax(all_uppers), 1.05)   # 允许超过1，但最大不超过1.05
                
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(-0.05, 1.05)  # Dirichlet方法限制在[0,1]范围内
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=12, loc='best')
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            
            if save_path:
                suffix = '_largest_component' if largest_component else ''
                method_str = f'_{method}' if method != 'traditional' else ''
                file_path = f"{save_path.replace('.png', '')}_L{L}{suffix}{method_str}.png"
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to: {file_path}")
            
            plt.show()

def main():
    """主函数：演示如何使用数据分析器"""
    # 创建分析器实例
    analyzer = ScientificDataAnalyzer(d=2, seed= 0, boundary_condition='Periodic')
    
    # 定义参数范围
    L_range = [40, 80, 100, 160]  # 系统尺寸范围
    rho_range = np.linspace(0.1, 0.2, 11)  # 占据密度范围
    
    # 注意：这里需要根据实际数据路径修改base_path
    base_path = "D:\\Program Files\\MIS-graph\\mis-data"  # 数据文件夹的基础路径
    
    print("Collecting data...")
    results = analyzer.collect_data(L_range, rho_range, base_path)
    
    if not results:
        print("No data files found, generating demo data for illustration...")
        # 如果没有找到数据，生成模拟数据用于演示
        analyzer = generate_demo_data(analyzer, L_range, rho_range)
    
    print(f"Successfully processed {len(analyzer.results)} data files")
    
    # 打印所有方法的汇总统计
    analyzer.print_summary_statistics()
    
    # 绘制图形 - 比较不同方法
    print("\nGenerating figures with different statistical methods...")
    
    # 1. 传统方法
    analyzer.plot_ratios_vs_rho_dirichlet(save_path='ratios_vs_rho_d=2.png', 
                                         method='traditional')
    
    # 2. Dirichlet MLE方法
    analyzer.plot_ratios_vs_rho_dirichlet(save_path='ratios_vs_rho_dirichlet_d=2.png', 
                                         method='dirichlet')
    
    # 3. Bootstrap Dirichlet方法
    analyzer.plot_ratios_vs_rho_dirichlet(save_path='ratios_vs_rho_bootstrap_dirichlet_d=2.png', 
                                         method='bootstrap_dirichlet')


def generate_demo_data(analyzer, L_range, rho_range):
    """生成演示数据（当没有真实数据时使用）"""
    import numpy as np
    
    print("Generating demo data...")
    
    demo_results = {}
    
    for L in L_range:
        for rho in rho_range:
            # 生成模拟数据
            total_points = L * L  # 假设是二维系统
            
            # 模拟数据生成
            np.random.seed(analyzer.seed + L + int(rho*100))
            
            # 生成随机数据
            n_points = 100  # 每个文件模拟100个数据点
            
            # 生成随机索引
            indices = np.arange(n_points)
            
            # 生成占据状态（0或1），密度为rho
            occupation = np.random.binomial(1, rho, n_points)
            
            # 生成变化格点数量（1到L^2之间）
            change_number = np.random.randint(1, min(10, L*L//2) + 1, n_points)
            
            # 生成能量变化，大部分接近0，少数有较大变化
            # 能量变化为0的比例随rho变化
            energy_zero_prob = 0.3 + 0.4 * rho
            energy_changes = np.zeros(n_points)
            
            # 部分点能量变化为0
            zero_mask = np.random.rand(n_points) < energy_zero_prob
            energy_changes[zero_mask] = 0
            
            # 非零能量变化
            nonzero_mask = ~zero_mask
            # 根据占据状态和rho决定能量变化的正负和大小
            for i in np.where(nonzero_mask)[0]:
                if occupation[i] == 1:
                    # 占据为1的点，能量变化通常为负
                    energy_changes[i] = -np.random.exponential(1.0) * (1 + rho)
                else:
                    # 占据为0的点，能量变化通常为正
                    energy_changes[i] = np.random.exponential(1.0) * (1 + rho)
            
            # 生成最大联通分量标记（模拟，大约30%的点在最大联通分量内）
            largest_component = np.random.binomial(1, 0.3, n_points)
            
            # 创建DataFrame
            df = pd.DataFrame({
                'index': indices,
                'occupation': occupation,
                'change_number': change_number,
                'change_energy': energy_changes,
                'Is_in_largest_connected_component': largest_component
            })
            
            # 计算统计量
            total_points = len(df)
            energy_zero_points = len(df[df['change_energy'] == 0])
            ratio_energy_zero = energy_zero_points / total_points
            
            energy_nonzero = df[df['change_energy'] != 0]
            energy_nonzero_occ1 = len(energy_nonzero[energy_nonzero['occupation'] == 1])
            ratio_nonzero_occ1 = energy_nonzero_occ1 / total_points
            
            energy_nonzero_occ0 = len(energy_nonzero[energy_nonzero['occupation'] == 0])
            ratio_nonzero_occ0 = energy_nonzero_occ0 / total_points
            
            # 最大联通分量内的统计量
            largest_component_df = df[df['Is_in_largest_connected_component'] == 1]
            largest_component_points = len(largest_component_df)
            
            if largest_component_points > 0:
                lc_energy_zero_points = len(largest_component_df[largest_component_df['change_energy'] == 0])
                lc_ratio_energy_zero = lc_energy_zero_points / largest_component_points
                
                lc_energy_nonzero = largest_component_df[largest_component_df['change_energy'] != 0]
                lc_energy_nonzero_occ1 = len(lc_energy_nonzero[lc_energy_nonzero['occupation'] == 1])
                lc_ratio_nonzero_occ1 = lc_energy_nonzero_occ1 / largest_component_points
                
                lc_energy_nonzero_occ0 = len(lc_energy_nonzero[lc_energy_nonzero['occupation'] == 0])
                lc_ratio_nonzero_occ0 = lc_energy_nonzero_occ0 / largest_component_points
                
                lc_mean_change_number = largest_component_df['change_number'].mean()
                lc_mean_energy_change = largest_component_df['change_energy'].mean()
            else:
                lc_ratio_energy_zero = 0
                lc_ratio_nonzero_occ1 = 0
                lc_ratio_nonzero_occ0 = 0
                lc_mean_change_number = 0
                lc_mean_energy_change = 0
            
            # 保存结果
            result = {
                'L': L,
                'W': L,
                'rho': rho,
                'n_samples': 1,
                'total_points': total_points,
                'ratio_energy_zero_mean': ratio_energy_zero,
                'ratio_nonzero_occ1_mean': ratio_nonzero_occ1,
                'ratio_nonzero_occ0_mean': ratio_nonzero_occ0,
                'mean_change_number_mean': df['change_number'].mean(),
                'mean_energy_change_mean': df['change_energy'].mean(),
                'largest_component_points_mean': largest_component_points,
                'lc_ratio_energy_zero_mean': lc_ratio_energy_zero,
                'lc_ratio_nonzero_occ1_mean': lc_ratio_nonzero_occ1,
                'lc_ratio_nonzero_occ0_mean': lc_ratio_nonzero_occ0,
                # 添加Dirichlet方法的结果（模拟）
                'ratio_energy_zero_dirichlet_mean': ratio_energy_zero,
                'ratio_energy_zero_dirichlet_ci_lower': max(0, ratio_energy_zero - 0.05),
                'ratio_energy_zero_dirichlet_ci_upper': min(1, ratio_energy_zero + 0.05),
                'ratio_nonzero_occ1_dirichlet_mean': ratio_nonzero_occ1,
                'ratio_nonzero_occ1_dirichlet_ci_lower': max(0, ratio_nonzero_occ1 - 0.05),
                'ratio_nonzero_occ1_dirichlet_ci_upper': min(1, ratio_nonzero_occ1 + 0.05),
                'ratio_nonzero_occ0_dirichlet_mean': ratio_nonzero_occ0,
                'ratio_nonzero_occ0_dirichlet_ci_lower': max(0, ratio_nonzero_occ0 - 0.05),
                'ratio_nonzero_occ0_dirichlet_ci_upper': min(1, ratio_nonzero_occ0 + 0.05),
                # 添加Bootstrap Dirichlet方法的结果（模拟）
                'ratio_energy_zero_bootstrap_dirichlet_mean': ratio_energy_zero,
                'ratio_energy_zero_bootstrap_dirichlet_ci_lower': max(0, ratio_energy_zero - 0.04),
                'ratio_energy_zero_bootstrap_dirichlet_ci_upper': min(1, ratio_energy_zero + 0.04),
                'ratio_nonzero_occ1_bootstrap_dirichlet_mean': ratio_nonzero_occ1,
                'ratio_nonzero_occ1_bootstrap_dirichlet_ci_lower': max(0, ratio_nonzero_occ1 - 0.04),
                'ratio_nonzero_occ1_bootstrap_dirichlet_ci_upper': min(1, ratio_nonzero_occ1 + 0.04),
                'ratio_nonzero_occ0_bootstrap_dirichlet_mean': ratio_nonzero_occ0,
                'ratio_nonzero_occ0_bootstrap_dirichlet_ci_lower': max(0, ratio_nonzero_occ0 - 0.04),
                'ratio_nonzero_occ0_bootstrap_dirichlet_ci_upper': min(1, ratio_nonzero_occ0 + 0.04),
            }
            
            key = (L, rho)
            demo_results[key] = result
    
    analyzer.results = demo_results
    return analyzer


if __name__ == "__main__":
    main()