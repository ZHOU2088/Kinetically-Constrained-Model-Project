import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 三维渗流临界指数
THREE_D_PERCOLATION_NU = 0.8765  # 三维渗流临界指数理论值

def read_all_data(filename, min_L=None, max_L=None, selected_Ls=None):
    """
    读取包含所有系统尺寸数据的文件
    可以限制系统尺寸范围
    """
    try:
        # 尝试用pandas读取
        data = pd.read_csv(filename)
        print("数据列名:", data.columns.tolist())
        
        # 检查列名
        if 'L' in data.columns and 'rho' in data.columns and 'mean_density' in data.columns:
            # 过滤系统尺寸
            if min_L is not None:
                data = data[data['L'] >= min_L]
            if max_L is not None:
                data = data[data['L'] <= max_L]
            
            if selected_Ls is not None:
                # 确保selected_Ls是整数列表
                selected_Ls = [int(L) for L in selected_Ls]
                data = data[data['L'].isin(selected_Ls)]
            
            L_values = np.sort(data['L'].unique())
            
            if len(L_values) == 0:
                print("警告: 没有找到任何符合条件的数据")
                return None, []
            
            all_data = {}
            
            for L in L_values:
                subset = data[data['L'] == L]
                all_data[L] = {
                    'rhos': subset['rho'].values,
                    'densities': subset['mean_density'].values,
                    'stds': subset['std_density'].values if 'std_density' in subset.columns else None
                }
                print(f"L={L}: {len(subset)} 个数据点")
            
            return all_data, L_values
            
        else:
            # 尝试用numpy读取
            print("使用numpy读取...")
            raw_data = np.loadtxt(filename, delimiter=',', skiprows=1)
            
            raw_data[:, 0] = raw_data[:, 0].astype(int)

            # 过滤系统尺寸
            if min_L is not None:
                mask = raw_data[:, 0] >= min_L
                raw_data = raw_data[mask]
            if max_L is not None:
                mask = raw_data[:, 0] <= max_L
                raw_data = raw_data[mask]
            
            if selected_Ls is not None:
                # 确保selected_Ls是整数列表
                selected_Ls = [int(L) for L in selected_Ls]
                mask = np.isin(raw_data[:, 0], selected_Ls)
                raw_data = raw_data[mask]
            
            L_values = np.sort(np.unique(raw_data[:, 0]))
            
            if len(L_values) == 0:
                print("警告: 没有找到任何符合条件的数据")
                return None, []
            
            all_data = {}
            
            for L in L_values:
                mask = raw_data[:, 0] == L
                subset = raw_data[mask]
                all_data[L] = {
                    'rhos': subset[:, 1],
                    'densities': subset[:, 3],
                    'stds': subset[:, 4] if subset.shape[1] > 4 else None
                }
                print(f"L={L}: {len(subset)} 个数据点")
            
            return all_data, L_values.astype(int)
            
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None, []

def poly_interpolate(rho, density, density_target, degree=4):
    """
    通过多项式插值找到给定最大簇密度对应的rho
    """
    # 对密度排序
    sort_idx = np.argsort(density)
    density_sorted = density[sort_idx]
    rho_sorted = rho[sort_idx]
    
    # 检查目标密度是否在范围内
    if density_target < density_sorted[0] or density_target > density_sorted[-1]:
        print(f"  警告: 目标密度 {density_target:.4f} 超出范围 [{density_sorted[0]:.4f}, {density_sorted[-1]:.4f}]")
        return None
    
    # 多项式拟合
    coeffs = np.polyfit(density_sorted, rho_sorted, degree)
    poly = np.poly1d(coeffs)
    
    return poly(density_target)

def finite_size_scaling_improved(
    rho_L: np.ndarray, 
    L: np.ndarray, 
    sigma: np.ndarray = None,
    nu_guess: float = THREE_D_PERCOLATION_NU,  # 使用三维值
    method: str = 'trf',
    bounds: tuple = None,
    max_nfev: int = 10000
):
    """
    改进的有限尺度分析拟合函数（针对三维模型）
    
    参数:
    ----------
    rho_L : np.ndarray
        不同系统尺寸对应的rho值
    L : np.ndarray
        系统尺寸数组
    sigma : np.ndarray, optional
        各数据点的误差（用于加权拟合）
    nu_guess : float, default=THREE_D_PERCOLATION_NU
        ν的初始猜测值，使用三维渗流理论值
    method : str, default='trf'
        拟合方法：'lm'（Levenberg-Marquardt）或 'trf'（信赖域反射）
    bounds : tuple, optional
        参数边界：((rho_infty_min, C_min, nu_min), (rho_infty_max, C_max, nu_max))
    max_nfev : int, default=10000
        最大函数调用次数
    
    返回:
    -------
    dict : 包含拟合结果的字典
    """
    # 定义拟合函数
    def func(L_val, rho_infty, C, nu):
        """rho(L) = rho_infty + C * L^(-1/nu)"""
        return rho_infty + C * np.power(L_val, -1.0/nu)
    
    # 改进的初始猜测
    rho_max = np.max(rho_L)
    rho_min = np.min(rho_L)
    
    # 智能初始猜测
    p0 = [
        rho_max * 0.9,  # rho_infty: 略低于最大值
        0.1 * (rho_max - rho_min),  # C: 基于数据范围
        nu_guess
    ]
    
    # 默认边界 - 针对三维模型的合理边界
    if bounds is None:
        # 设置合理的物理边界
        bounds = (
            [0.0, -np.inf, 0.5],   # 下限: rho_infty≥0, ν≥0.5
            [1.0, np.inf, 2.0]     # 上限: rho_infty≤1, ν≤2.0
        )
    
    # 执行拟合
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Covariance of the parameters could not be estimated')
            
            if method == 'lm' and bounds is None:
                # LM方法不支持边界
                popt, pcov = curve_fit(
                    func, L, rho_L, 
                    p0=p0, 
                    sigma=sigma,
                    maxfev=max_nfev
                )
            else:
                # TRF方法支持边界
                popt, pcov = curve_fit(
                    func, L, rho_L, 
                    p0=p0, 
                    sigma=sigma,
                    method=method,
                    bounds=bounds,
                    max_nfev=max_nfev
                )
        
        # 提取参数
        rho_infty, C, nu = popt
        
        # 计算参数误差
        if pcov is not None and not np.any(np.isnan(pcov)):
            perr = np.sqrt(np.diag(pcov))
            rho_infty_err, C_err, nu_err = perr
        else:
            rho_infty_err = C_err = nu_err = np.nan
            print("警告: 无法计算参数协方差")
        
        # 计算拟合质量指标
        residuals = rho_L - func(L, *popt)
        if sigma is not None:
            # 加权残差
            chi_squared = np.sum((residuals / sigma)**2)
        else:
            # 使用标准差估计误差
            if len(rho_L) > 1:
                error_estimate = np.std(rho_L, ddof=1) * 0.1
            else:
                error_estimate = 0.001
            chi_squared = np.sum((residuals / error_estimate)**2)
        
        dof = len(rho_L) - 3  # 自由度
        chi_squared_per_dof = chi_squared / dof if dof > 0 else np.nan
        
        # 计算R平方
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((rho_L - np.mean(rho_L))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        # 创建结果字典
        result = {
            'success': True,
            'params': {
                'rho_infty': rho_infty,
                'C': C,
                'nu': nu
            },
            'errors': {
                'rho_infty': rho_infty_err,
                'C': C_err,
                'nu': nu_err
            },
            'goodness_of_fit': {
                'chi_squared': chi_squared,
                'chi_squared_per_dof': chi_squared_per_dof,
                'r_squared': r_squared,
                'dof': dof
            },
            'residuals': residuals,
            'fitted_values': func(L, *popt),
            'covariance_matrix': pcov,
            'fit_function': func
        }
        
        return result
        
    except Exception as e:
        print(f"  拟合失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'params': None,
            'errors': None,
            'goodness_of_fit': None
        }

def finite_size_scaling_improved_global(
    rho_L: np.ndarray, 
    L: np.ndarray, 
    sigma: np.ndarray = None,
    nu_guess: float = None,
    method: str = 'trf',
    bounds: tuple = None,
    max_nfev: int = 20000,
    use_global_optimization: bool = True,
    global_opt_strategy: str = 'best1bin',
    global_opt_population: int = 20,  # 增加种群大小
    global_opt_maxiter: int = 1000,
    refine_with_lm: bool = True
):
    """
    改进的有限尺度分析拟合函数，使用差分进化进行全局优化，再用LM算法细化
    """
    # 定义拟合函数
    def func(L_val, rho_infty, C, nu):
        """rho(L) = rho_infty + C * L^(-1/nu)"""
        return rho_infty + C * np.power(L_val, -1.0/nu)
    
    # 定义误差函数
    def error_func(params, L_vals, rho_vals, sigma_vals=None):
        """计算拟合误差"""
        rho_infty, C, nu = params
        
        # 检查参数有效性
        if nu <= 0 or rho_infty < 0 or rho_infty > 1:
            return 1e10  # 返回很大的误差
        
        predicted = func(L_vals, rho_infty, C, nu)
        residuals = rho_vals - predicted
        
        if sigma_vals is not None:
            return np.sum((residuals / sigma_vals)**2)
        return np.sum(residuals**2)
    
    # 改进边界设置
    if bounds is None:
        # 基于数据范围设置合理的边界
        rho_max = np.max(rho_L)
        rho_min = np.min(rho_L)
        rho_range = rho_max - rho_min
        
        # 针对三维渗流的合理物理边界
        bounds_for_curve_fit = (
            [0.0, -10.0 , 0.5],   # 下限: ρ∞, C, ν
            [1.0, 10.0, 2.0]      # 上限: ρ∞, C, ν
        )
        bounds_for_de = [
            (max(0.0, rho_min - 0.1), min(1.0, rho_max + 0.1)),  # ρ∞的范围
            (-10.0 , 10.0),  # C的范围，基于数据范围
            (0.5, 1.5)  # ν的范围，针对三维渗流
        ]
    else:
        # 如果用户提供了边界，转换为两种格式
        bounds_for_curve_fit = bounds
        bounds_for_de = [
            (bounds[0][0], bounds[1][0]),  # ρ∞
            (bounds[0][1], bounds[1][1]),  # C
            (bounds[0][2], bounds[1][2])   # ν
        ]
    
    # 改进初始猜测策略
    L_log = np.log(L)
    rho_L_array = np.array(rho_L)
    
    # 尝试线性拟合log-log图以获得更好的初始猜测
    if len(L) >= 3:
        try:
            # 对log(rho_L - rho_infty_guess) ~ -1/nu * log(L)进行拟合
            # 先估计rho_infty
            rho_infty_guess = rho_L_array[-1]  # 最大的L对应的值
            y = np.log(np.abs(rho_L_array - rho_infty_guess))
            x = np.log(L)
            
            # 移除无穷大和NaN
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid_mask) >= 2:
                slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)
                nu_initial = -1.0 / slope
                C_initial = np.exp(intercept) * np.sign(np.mean(rho_L_array - rho_infty_guess))
                
                # 检查参数合理性
                if 0.5 <= nu_initial <= 2.0 and -10 <= C_initial <= 10:
                    p0 = [rho_infty_guess, C_initial, nu_initial]
                    print(f"  通过log-log分析得到初始猜测: ρ∞={rho_infty_guess:.4f}, C={C_initial:.4f}, ν={nu_initial:.4f}")
                else:
                    raise ValueError("初始猜测超出合理范围")
            else:
                raise ValueError("有效数据点不足")
        except:
            # 回退到简单猜测
            rho_infty_guess = np.mean(rho_L_array)
            C_initial = 0.1
            nu_initial = THREE_D_PERCOLATION_NU
            p0 = [rho_infty_guess, C_initial, nu_initial]
    else:
        rho_infty_guess = np.mean(rho_L_array)
        C_initial = 0.1
        nu_initial = THREE_D_PERCOLATION_NU
        p0 = [rho_infty_guess, C_initial, nu_initial]
    
    # 第一步：使用差分进化进行全局优化
    if use_global_optimization and len(rho_L) >= 3:
        print(f"  使用差分进化算法进行全局优化...")
        print(f"  参数边界: ρ∞∈[{bounds_for_de[0][0]:.3f}, {bounds_for_de[0][1]:.3f}], "
              f"C∈[{bounds_for_de[1][0]:.3f}, {bounds_for_de[1][1]:.3f}], "
              f"ν∈[{bounds_for_de[2][0]:.3f}, {bounds_for_de[2][1]:.3f}]")
        
        try:
            # 添加惩罚项到误差函数
            def error_func_with_penalty(params, L_vals, rho_vals, sigma_vals=None):
                """带有惩罚项的误差函数，避免边界解"""
                rho_infty, C, nu = params
                
                # 基础误差
                base_error = error_func(params, L_vals, rho_vals, sigma_vals)
                
                # 添加边界惩罚
                penalty = 0.0
                
                # 对接近边界的参数添加惩罚
                for i, (param, bounds) in enumerate(zip(params, bounds_for_de)):
                    lower, upper = bounds
                    if param < lower or param > upper:
                        penalty += 1e5
                    else:
                        # 对接近边界但未越界的参数添加轻微惩罚
                        range_width = upper - lower
                        distance_to_boundary = min(param - lower, upper - param)
                        if distance_to_boundary < 0.1 * range_width:
                            penalty += 1e3 * (0.1 - distance_to_boundary/range_width)**2
                
                return base_error + penalty
            
            # 执行差分进化优化
            de_result = differential_evolution(
                error_func_with_penalty,
                bounds_for_de,
                args=(L, rho_L, sigma),
                strategy=global_opt_strategy,
                maxiter=global_opt_maxiter,
                popsize=global_opt_population,
                tol=1e-8,
                mutation=(0.5, 1.0),  # 减小变异范围
                recombination=0.7,
                seed=42,
                disp=False,
                workers=1,
                updating='immediate',
                atol=0,
                polish=False  # 先不抛光，后续用curve_fit细化
            )
            
            if de_result.success or de_result.nfev > 0:  # 即使未完全收敛，也接受结果
                global_opt_params = de_result.x
                global_opt_error = de_result.fun
                print(f"  差分进化完成: 误差 = {global_opt_error:.6e}")
                print(f"  全局最优参数: ρ∞={global_opt_params[0]:.6f}, "
                      f"C={global_opt_params[1]:.6f}, ν={global_opt_params[2]:.6f}")
                
                # 检查参数是否合理
                rho_infty_de, C_de, nu_de = global_opt_params
                is_reasonable = (
                    bounds_for_de[0][0] <= rho_infty_de <= bounds_for_de[0][1] and
                    bounds_for_de[1][0] <= C_de <= bounds_for_de[1][1] and
                    bounds_for_de[2][0] <= nu_de <= bounds_for_de[2][1] and
                    abs(C_de) < 0.5 * (bounds_for_de[1][1] - bounds_for_de[1][0])
                )
                
                if is_reasonable:
                    p0 = global_opt_params
                else:
                    print(f"  警告: 差分进化结果不合理，使用初始猜测")
            else:
                print(f"  差分进化失败，使用初始猜测")
                
        except Exception as e:
            print(f"  差分进化失败: {e}")
    
    # 第二步：使用局部优化算法细化结果
    best_params = p0
    best_cov = None
    best_residuals = np.inf
    best_method = ""
    
    # 尝试不同的优化方法
    optimization_methods = [('trf', bounds_for_curve_fit)]
    
    if refine_with_lm:
        optimization_methods.append(('lm', None))
    
    for opt_method, opt_bounds in optimization_methods:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                if opt_method == 'lm':
                    # LM算法不支持边界
                    popt, pcov = curve_fit(
                        func, L, rho_L, 
                        p0=best_params,
                        sigma=sigma,
                        method='lm',
                        maxfev=max_nfev,
                        ftol=1e-12,
                        gtol=1e-12,
                        xtol=1e-12
                    )
                else:
                    # TRF算法支持边界
                    popt, pcov = curve_fit(
                        func, L, rho_L, 
                        p0=best_params,
                        sigma=sigma,
                        method='trf',
                        bounds=opt_bounds,
                        maxfev=max_nfev,
                        ftol=1e-12,
                        gtol=1e-12,
                        xtol=1e-12
                    )
            
            # 计算残差
            residuals = rho_L - func(L, *popt)
            if sigma is not None:
                chi_squared = np.sum((residuals / sigma)**2)
            else:
                # 估计误差
                if len(rho_L) > 1:
                    error_estimate = np.std(rho_L, ddof=1) * 0.1
                else:
                    error_estimate = 0.001
                chi_squared = np.sum((residuals / error_estimate)**2)
            
            # 检查参数合理性
            rho_infty, C, nu = popt
            
            if opt_bounds is not None:
                lower_bounds, upper_bounds = opt_bounds
                within_bounds = (
                    lower_bounds[0] <= rho_infty <= upper_bounds[0] and
                    lower_bounds[1] <= C <= upper_bounds[1] and
                    lower_bounds[2] <= nu <= upper_bounds[2]
                )
            else:
                within_bounds = True
            
            if within_bounds and chi_squared < best_residuals:
                best_residuals = chi_squared
                best_params = popt
                best_cov = pcov
                best_method = opt_method
                print(f"  {opt_method.upper()}优化成功: χ²={chi_squared:.4e}, "
                      f"ρ∞={rho_infty:.6f}, C={C:.6f}, ν={nu:.6f}")
                    
        except Exception as e:
            print(f"  {opt_method.upper()}优化失败: {e}")
            continue
    
    if best_params is None:
        print("  所有优化方法都失败，使用初始猜测")
        best_params = p0
        best_method = "initial_guess"
    
    # 提取最佳参数
    rho_infty, C, nu = best_params
    
    # 计算参数误差
    if best_cov is not None and not np.any(np.isnan(best_cov)):
        perr = np.sqrt(np.diag(best_cov))
        rho_infty_err, C_err, nu_err = perr
    else:
        rho_infty_err = C_err = nu_err = np.nan
        print("  警告: 无法计算参数协方差")
    
    # 计算拟合质量指标
    residuals = rho_L - func(L, rho_infty, C, nu)
    if sigma is not None:
        chi_squared = np.sum((residuals / sigma)**2)
    else:
        if len(rho_L) > 1:
            error_estimate = np.std(rho_L, ddof=1) * 0.1
        else:
            error_estimate = 0.001
        chi_squared = np.sum((residuals / error_estimate)**2)
    
    dof = len(rho_L) - 3
    chi_squared_per_dof = chi_squared / dof if dof > 0 else np.nan
    
    # 计算R平方
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((rho_L - np.mean(rho_L))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # 计算AIC和BIC
    n = len(rho_L)
    k = 3
    aic = n * np.log(chi_squared/n) + 2 * k if chi_squared > 0 else np.nan
    bic = n * np.log(chi_squared/n) + k * np.log(n) if chi_squared > 0 else np.nan
    
    # 计算与理论值的偏差
    nu_deviation_pct = (nu - THREE_D_PERCOLATION_NU) / THREE_D_PERCOLATION_NU * 100
    
    # 检查参数是否在合理范围内
    if abs(C) > 10:  # 如果C太大，警告
        print(f"  警告: 参数C={C:.4f}可能不合理")
    
    # 创建结果字典
    result = {
        'success': True,
        'params': {
            'rho_infty': rho_infty,
            'C': C,
            'nu': nu
        },
        'errors': {
            'rho_infty': rho_infty_err,
            'C': C_err,
            'nu': nu_err
        },
        'goodness_of_fit': {
            'chi_squared': chi_squared,
            'chi_squared_per_dof': chi_squared_per_dof,
            'r_squared': r_squared,
            'dof': dof,
            'aic': aic,
            'bic': bic
        },
        'residuals': residuals,
        'fitted_values': func(L, rho_infty, C, nu),
        'covariance_matrix': best_cov,
        'fit_function': func,
        'optimization_info': {
            'used_global_optimization': use_global_optimization,
            'refined_with_lm': refine_with_lm,
            'nu_deviation_pct': nu_deviation_pct,
            'final_method': best_method
        }
    }
    
    return result

def analyze_density_targets(all_data, L_values, density_targets, min_fit_points=3, is_global = False):
    """
    对每个目标密度进行分析
    min_fit_points: 最小拟合点数
    """
    print(f"\n使用的系统尺寸: {L_values}")
    print(f"三维渗流理论ν值: {THREE_D_PERCOLATION_NU:.4f}")
    print(f"最小拟合点数: {min_fit_points}")
    
    rho_L_list = []
    L_values_list = []
    valid_density_targets = []
    fitting_results = []
    
    for density_target in density_targets:
        print(f"\n{'='*50}")
        print(f"分析目标最大簇密度: {density_target:.4f}")
        print('-'*50)
        
        rho_L = []
        current_L_values = []
        
        for L in L_values:
            data = all_data[L]
            rho_val = poly_interpolate(data['rhos'], data['densities'], density_target, degree=4)
            
            if rho_val is not None:
                rho_L.append(rho_val)
                current_L_values.append(L)
                print(f"  L={int(L):4d}: ρ = {rho_val:.6f}")
            else:
                print(f"  L={int(L):4d}: 超出插值范围")
        
        if len(rho_L) >= min_fit_points:
            L_array = np.array(current_L_values)
            rho_array = np.array(rho_L)
            
            if(is_global):
                result = finite_size_scaling_improved_global(
                rho_array, L_array, 
                nu_guess=None,  # 不提供猜测值
                use_global_optimization=True,  # 使用全局优化
                refine_with_lm=True,  # 使用LM算法细化
                global_opt_population=20,  # 增加种群大小
                global_opt_maxiter=500,  # 最大迭代次数
                bounds=None  # 使用默认边界
            )
                if result['success']:
                    params = result['params']
                    errors = result['errors']
                    gof = result['goodness_of_fit']
                    func = result['fit_function']
                    opt_info = result['optimization_info']
                    
                    print(f"  拟合结果:")
                    print(f"  ρ∞ = {params['rho_infty']:.6f} ± {errors['rho_infty']:.6f}")
                    print(f"  C  = {params['C']:.6f} ± {errors['C']:.6f}")
                    print(f"  ν  = {params['nu']:.6f} ± {errors['nu']:.6f}")
                    print(f"  与理论值偏差: {opt_info['nu_deviation_pct']:+.1f}%")
                    print(f"  拟合质量:")
                    print(f"    χ²/dof = {gof['chi_squared_per_dof']:.3f}")
                    print(f"    R²     = {gof['r_squared']:.4f}")
                    print(f"    AIC    = {gof['aic']:.2f}")
                    print(f"    BIC    = {gof['bic']:.2f}")
                    print(f"  优化方法: {'差分进化+LM' if opt_info['used_global_optimization'] and opt_info['refined_with_lm'] else '直接拟合'}")
                    print(f"  使用系统尺寸: {current_L_values}")
                    
                    rho_L_list.append(rho_array)
                    L_values_list.append(L_array)
                    valid_density_targets.append(density_target)
                    fitting_results.append((
                        params['rho_infty'], params['C'], params['nu'],
                        errors['rho_infty'], errors['C'], errors['nu'],
                        func
                    ))
                else:
                    print(f"  只有 {len(rho_L)} 个有效数据点，需要至少 {min_fit_points} 个点")
            else:
                result = finite_size_scaling_improved( rho_array, L_array, 
                nu_guess=THREE_D_PERCOLATION_NU,
                method='trf')

                if result['success']:
                    params = result['params']
                    errors = result['errors']
                    gof = result['goodness_of_fit']
                    func = result['fit_function']
                    
                    print(f"  拟合结果:")
                    print(f"  ρ∞ = {params['rho_infty']:.6f} ± {errors['rho_infty']:.6f}")
                    print(f"  C  = {params['C']:.6f} ± {errors['C']:.6f}")
                    print(f"  ν  = {params['nu']:.6f} ± {errors['nu']:.6f}")
                    print(f"  与理论值偏差: {(params['nu'] - THREE_D_PERCOLATION_NU)/THREE_D_PERCOLATION_NU*100:.1f}%")
                    print(f"  拟合质量:")
                    print(f"    χ²/dof = {gof['chi_squared_per_dof']:.3f}")
                    print(f"    R²     = {gof['r_squared']:.4f}")
                    print(f"  使用系统尺寸: {current_L_values}")
                    
                    rho_L_list.append(rho_array)
                    L_values_list.append(L_array)
                    valid_density_targets.append(density_target)
                    fitting_results.append((
                        params['rho_infty'], params['C'], params['nu'],
                        errors['rho_infty'], errors['C'], errors['nu'],
                        func
                    ))
                else:
                    print(f"  只有 {len(rho_L)} 个有效数据点，需要至少 {min_fit_points} 个点")

            
    
    return L_values_list, rho_L_list, valid_density_targets, fitting_results

def plot_individual_fit_curves(L_values_list, rho_L_list, density_targets, fitting_results, 
                              output_dir=".", filename_suffix=""):
    """
    单独生成每个密度的拟合曲线图
    
    参数:
    ----------
    L_values_list : list
        每个密度对应的L值数组列表
    rho_L_list : list
        每个密度对应的rho(L)值列表
    density_targets : list
        密度值列表
    fitting_results : list
        拟合结果列表
    output_dir : str
        输出目录
    filename_suffix : str
        文件名后缀
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色设置
    colors = plt.cm.tab20(np.linspace(0, 1, len(density_targets)))
    
    # 创建单独的拟合曲线图
    fig_individual, ax_individual = plt.subplots(figsize=(10, 8))
    
    L_min_all = min([min(L) for L in L_values_list])
    L_max_all = max([max(L) for L in L_values_list])
    L_fine = np.logspace(np.log10(L_min_all), np.log10(L_max_all * 1.2), 200)
    
    # 为每个密度绘制拟合曲线
    for i, (L_vals, rho_vals, density, result) in enumerate(zip(L_values_list, rho_L_list, 
                                                               density_targets, fitting_results)):
        rho_infty, C, nu, rho_infty_err, C_err, nu_err, func = result
        
        # 数据点
        ax_individual.plot(L_vals, rho_vals, 'o', color=colors[i], markersize=8,
                          label=f'$f_{{giant}}$={density:.4f}\nρ(L) = {rho_infty:.3f} + {C:.3f}·L^( -1/{nu:.3f} )')
        
        # 拟合曲线
        ax_individual.plot(L_fine, func(L_fine, rho_infty, C, nu), 
                          color=colors[i], linewidth=2, alpha=0.7)
    
    ax_individual.set_xlabel('System Size L', fontsize=14)
    ax_individual.set_ylabel('ρ(L)', fontsize=14)
    ax_individual.set_title('3D Finite Size Scaling Fits\nρ(L) = ρ∞ + C·L^(-1/ν)', fontsize=16)
    ax_individual.legend(loc='best', fontsize=10, ncol=2)
    ax_individual.grid(True, alpha=0.3)
    ax_individual.set_xscale('log')
    
    # 设置x轴刻度
    ax_individual.minorticks_on()
    ax_individual.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax_individual.tick_params(axis='both', which='major', labelsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存单独的拟合曲线图
    if filename_suffix:
        base_name = f"3D_fit_curves_{filename_suffix}"
    else:
        base_name = "3D_fit_curves"
    
    output_filename_png = os.path.join(output_dir, f"{base_name}.png")
    output_filename_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    
    plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_filename_pdf, dpi=300, bbox_inches='tight', format='pdf')
    
    print(f"单独的拟合曲线图已保存:")
    print(f"  PNG: {output_filename_png}")
    print(f"  PDF: {output_filename_pdf}")
    
    plt.show()
    
    return fig_individual, output_filename_png

def plot_comprehensive_results(L_values_list, rho_L_list, density_targets, fitting_results, 
                              min_L=None, max_L=None, filename_suffix="", output_dir = ""):
    """
    绘制综合结果图（原来的四子图）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # 颜色设置
    colors = plt.cm.tab20(np.linspace(0, 1, len(density_targets)))
    
    # 1. 原始数据点
    for i, (L_vals, rho_vals, density) in enumerate(zip(L_values_list, rho_L_list, density_targets)):
        ax1.plot(L_vals, rho_vals, 'o-', color=colors[i], linewidth=2, markersize=8,
                label=f'$P_\\infty$={density:.4f}')
    
    ax1.set_xlabel('System Size L', fontsize=12)
    ax1.set_ylabel('ρ(L)', fontsize=12)
    title = '3D Finite Size Scaling: ρ(L) for Different $P_\\infty$'
    if min_L is not None:
        title += f' (L ≥ {min_L})'
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. 拟合曲线
    L_min = min([min(L) for L in L_values_list])
    L_max = max([max(L) for L in L_values_list])
    L_fine = np.logspace(np.log10(L_min), np.log10(L_max*2), 200)
    
    for i, (L_vals, rho_vals, density, result) in enumerate(zip(L_values_list, rho_L_list, 
                                                                density_targets, fitting_results)):
        rho_infty, C, nu, rho_infty_err, C_err, nu_err, func = result
        
        # 数据点
        ax2.plot(L_vals, rho_vals, 'o', color=colors[i], markersize=8)
        
        # 拟合曲线
        ax2.plot(L_fine, func(L_fine, rho_infty, C, nu), 
                color=colors[i], linewidth=2, alpha=0.7,
                label=f'$P_\\infty$={density:.4f}\nν={nu:.3f}±{nu_err:.3f}')
    
    ax2.set_xlabel('L', fontsize=12)
    ax2.set_ylabel('ρ(L)', fontsize=12)
    ax2.set_title('3D Finite Size Scaling Fits: ρ(L) = ρ∞ + C·L^(-1/ν)', fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. 对数坐标图
    for i, (L_vals, rho_vals, density, result) in enumerate(zip(L_values_list, rho_L_list, 
                                                                density_targets, fitting_results)):
        rho_infty, C, nu, rho_infty_err, C_err, nu_err, func = result
        
        ax3.plot(L_vals, rho_vals - rho_infty, 's', color=colors[i], markersize=8)
        ax3.plot(L_fine, C * np.power(L_fine, -1.0/nu), 
                color=colors[i], linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('L', fontsize=12)
    ax3.set_ylabel('ρ(L) - ρ$_\\infty$', fontsize=12)
    ax3.set_title('3D Scaling Function: ρ(L) - ρ$_\\infty$ ∝ L^(-1/ν)', fontsize=14)
    ax3.legend([f'$P_\\infty$={d:.4f}' for d in density_targets], loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # 添加理论参考线
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    for i, (result, color) in enumerate(zip(fitting_results, colors)):
        _, _, nu, _, _, _, _ = result
        if i == 0:  # 只标注第一个
            ax3.plot([L_min, L_max], [0.1, 0.1 * (L_min/L_max)**(1/nu)], 'k--', alpha=0.5, 
                    label=f'Slope = -1/{nu:.3f}')
    
    # 4. 参数表格
    ax4.axis('off')
    table_data = []
    
    for i, (density, result) in enumerate(zip(density_targets, fitting_results)):
        rho_infty, C, nu, rho_infty_err, C_err, nu_err, func = result
        L_vals = L_values_list[i]
        
        # 计算卡方
        predicted = func(L_vals, rho_infty, C, nu)
        residuals = rho_L_list[i] - predicted
        if len(residuals) > 1:
            error_estimate = np.std(rho_L_list[i], ddof=1) * 0.1
        else:
            error_estimate = 0.001
        chi2 = np.sum((residuals / error_estimate)**2)
        dof = len(L_vals) - 3
        chi2_per_dof = chi2 / dof if dof > 0 else np.nan
        
        # 计算R平方
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((rho_L_list[i] - np.mean(rho_L_list[i]))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        # 计算与理论值的偏差
        nu_deviation = (nu - THREE_D_PERCOLATION_NU) / THREE_D_PERCOLATION_NU * 100
        
        table_data.append([
            f'{density:.4f}',
            f'{rho_infty:.4f}±{rho_infty_err:.4f}',
            f'{nu:.4f}±{nu_err:.4f}',
            f'{nu_deviation:+.1f}%',
            f'{chi2_per_dof:.2f}',
            f'{r_squared:.4f}'
        ])
    
    # 创建表格
    table = ax4.table(cellText=table_data,
                     colLabels=['$P_\\infty$', 'ρ$_\\infty$', 'ν', 'ν偏差', 'χ²/dof', 'R²'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1, 0.2, 0.15, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    ax4.set_title(f'3D Percolation: Fitting Parameters (Theoretical ν={THREE_D_PERCOLATION_NU:.4f})', fontsize=14, pad=20)
    
    # 主标题
    suptitle = f'3D Finite Size Scaling Analysis for Percolation Model'
    if min_L is not None:
        suptitle += f' (L ≥ {min_L})'
    plt.suptitle(suptitle, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图形
    if filename_suffix:
        base_name = f"3D_finite_size_scaling_comprehensive_{filename_suffix}"
    else:
        base_name = "3D_finite_size_scaling_comprehensive"
    
    if min_L is not None:
        base_name += f"_Lmin{min_L}"
    
    output_filename = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n综合结果图已保存到: {os.path.abspath(output_filename)}")
    
    output_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"综合结果图已保存到: {os.path.abspath(output_pdf)}")
    
    plt.show()
    
    return fig, output_filename

def save_detailed_results(all_data, density_targets, L_values_list, rho_L_list, 
                         fitting_results, min_L=None, max_L=None, output_dir = ""):
    """保存详细结果到文件"""
    if min_L is not None:
        filename = os.path.join(output_dir,f"3D_finite_size_scaling_results_Lmin{min_L}.txt")
    else:
        filename = os.path.join(output_dir,"3D_finite_size_scaling_results.txt")
    
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("3D FINITE SIZE SCALING ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"3D Percolation Theoretical ν = {THREE_D_PERCOLATION_NU:.4f}\n")
        if min_L is not None:
            f.write(f"Minimum L = {min_L}\n")
        if max_L is not None:
            f.write(f"Maximum L = {max_L}\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        for L, data in all_data.items():
            f.write(f"L = {L}: {len(data['rhos'])} data points\n")
            f.write(f"  ρ range: [{data['rhos'].min():.4f}, {data['rhos'].max():.4f}]\n")
            f.write(f"  P∞ range: [{data['densities'].min():.4f}, {data['densities'].max():.4f}]\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FITTING RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, (density, L_vals, rho_vals, result) in enumerate(zip(density_targets, L_values_list, 
                                                                   rho_L_list, fitting_results)):
            rho_infty, C, nu, rho_infty_err, C_err, nu_err, func = result
            
            f.write(f"\nTarget P∞ = {density:.4f}\n")
            f.write("-"*50 + "\n")
            
            f.write(f"System sizes used: {list(L_vals.astype(int))}\n")
            f.write("Interpolated ρ(L) values:\n")
            f.write("L        ρ(L)\n")
            f.write("-"*20 + "\n")
            for L_val, rho_val in zip(L_vals, rho_vals):
                f.write(f"{L_val:<8} {rho_val:.6f}\n")
            
            f.write(f"\nFitting function: ρ(L) = ρ∞ + C·L^(-1/ν)\n")
            f.write(f"ρ∞ = {rho_infty:.6f} ± {rho_infty_err:.6f}\n")
            f.write(f"C  = {C:.6f} ± {C_err:.6f}\n")
            f.write(f"ν  = {nu:.6f} ± {nu_err:.6f}\n")
            f.write(f"ν deviation from 3D theory: {(nu - THREE_D_PERCOLATION_NU)/THREE_D_PERCOLATION_NU*100:+.1f}%\n")
            
            # 计算拟合质量指标
            predicted = func(L_vals, rho_infty, C, nu)
            residuals = rho_vals - predicted
            if len(residuals) > 1:
                error_estimate = np.std(rho_vals, ddof=1) * 0.1
            else:
                error_estimate = 0.001
            chi2 = np.sum((residuals / error_estimate)**2)
            dof = len(L_vals) - 3
            chi2_per_dof = chi2 / dof if dof > 0 else np.nan
            
            # 计算R平方
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((rho_vals - np.mean(rho_vals))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            f.write(f"χ²/dof = {chi2_per_dof:.3f}\n")
            f.write(f"R²     = {r_squared:.4f}\n")
            f.write(f"Final function: ρ(L) = {rho_infty:.6f} + {C:.6f} * L^(-1/{nu:.6f})\n\n")
    
    print(f"详细结果已保存到: {os.path.abspath(filename)}")

def main():
    """主函数"""
    # 配置参数
    data_filename = "C:/Users/admin/Downloads/percolation_batch_1/max-cluster-density-stat-3d-rho_b_1_120-420.txt"  # 你的数据文件名
    output_dir = "./3D_finite_size_scaling_results"  # 输出目录
    
    # 目标最大簇密度
    density_targets = [0.0012]
    
    Max_density = max(density_targets)
    Min_density = min(density_targets)

    # 系统尺寸限制
    min_L = None  # 最小系统尺寸，设为None表示不限制
    max_L = None  # 最大系统尺寸，设为None表示不限制
    
    selected_Ls = None

    # 最小拟合点数
    min_fit_points = 3
    
    print("="*70)
    print("三维渗流有限尺度分析 (3D Finite Size Scaling Analysis)")
    print("="*70)
    print(f"数据文件: {data_filename}")
    print(f"三维渗流理论ν值: {THREE_D_PERCOLATION_NU:.4f}")
    print(f"目标最大簇密度: {density_targets}")
    if min_L is not None:
        print(f"最小系统尺寸: L ≥ {min_L}")
    if max_L is not None:
        print(f"最大系统尺寸: L ≤ {max_L}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("\n读取数据...")
    all_data, L_values = read_all_data(data_filename, min_L=min_L, max_L=max_L, selected_Ls=selected_Ls)
    
    if all_data is None or len(all_data) < min_fit_points:
        print(f"错误: 需要至少{min_fit_points}个系统尺寸的数据，当前只有 {len(all_data) if all_data else 0} 个")
        return
    
    # 分析数据
    L_values_list, rho_L_list, valid_density_targets, fitting_results = analyze_density_targets(
        all_data, L_values, density_targets, min_fit_points=min_fit_points, is_global= True
    )
    
    if not valid_density_targets:
        print("\n错误: 没有成功拟合任何密度值")
        return
    
    # 绘制综合结果图（原来的四子图）
    print(f"\n{'='*70}")
    print(f"成功分析 {len(valid_density_targets)} 个最大簇密度值")
    print(f"生成综合结果图...")
    
    filename_suffix = f"Lmin{min_L}_P_inf_{Min_density:.3f}-{Max_density:.3f}" if min_L is not None else f"P_inf_{Min_density:.3f}-{Max_density:.3f}"
    
    # 绘制综合结果图
    fig_comprehensive, output_comprehensive = plot_comprehensive_results(
        L_values_list, rho_L_list, valid_density_targets, 
        fitting_results, min_L=min_L, max_L=max_L,
        filename_suffix=filename_suffix, output_dir=output_dir
    )
    
    # 绘制单独的拟合曲线图
    print(f"\n{'='*70}")
    print(f"生成单独的拟合曲线图...")
    
    fig_individual, output_individual = plot_individual_fit_curves(
        L_values_list, rho_L_list, valid_density_targets,
        fitting_results, output_dir=output_dir,
        filename_suffix=filename_suffix
    )
    
    # 保存详细结果
    save_detailed_results(all_data, valid_density_targets, L_values_list, rho_L_list, 
                         fitting_results, min_L=min_L, max_L=max_L, output_dir=output_dir)
    
    print(f"\n{'='*70}")
    print("三维渗流有限尺度分析完成!")
    print(f"理论ν值: {THREE_D_PERCOLATION_NU:.4f}")
    print(f"输出文件:")
    print(f"  1. 综合结果图: {output_comprehensive}")
    print(f"  2. 拟合曲线图: {output_individual}")
    print(f"  3. 详细结果文件: 3D_finite_size_scaling_results*.txt")
    print("="*70)

if __name__ == "__main__":
    main()