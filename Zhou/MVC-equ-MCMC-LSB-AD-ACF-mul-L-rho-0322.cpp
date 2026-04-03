
// ---------------------------- 程序总览（核心流水线） ----------------------------
// 1) 生成初始随机晶格并施加局域规则，保留最大连通分量（LCC）。
// 2) 将晶格有效点映射为二分图，求最大匹配（Hopcroft-Karp）。
// 3) 依据 Kőnig 相关结构做“冻结传播”：确定必选/必不选的 MVC 节点。
// 4) 对未冻结核边构建依赖有向图，做 SCC 缩点得到 DAG。
// 5) 在 DAG 上按连通分量执行均匀采样：
//    - 小分量：精确枚举合法态并均匀抽样；
//    - 大分量：Level-Set Block Gibbs（可选 ACF 自动估计 sweeps）。
// 6) 组装 MVC 序参量 R 与高阶统计，输出原始样本和统计汇总文件。
//
// 性能设计主线：
// - thread_local 工作区复用，避免高频分配；
// - 邻居查询快路径 + interior_mask 减少除法/取模；
// - 多自旋编码 MSC64 批量并行 64 条链；
// - 断点续跑 + tau 缓存减少重复计算。

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <array>
#include <random>
#include <cmath>
#include <complex>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <type_traits>
#if __cplusplus >= 201703L
#include <filesystem>
#endif
#include <omp.h>

// ========================== 编译期配置 ==========================
// 关键参数改为 constexpr，避免通过 argv + stoi/stod 在运行期注入。
constexpr int d = 2;
static_assert(d >= 1, "d must be >= 1");
constexpr bool kPeriodicBoundaryCompileTime = true;

// 随机数引擎选择：true=Xoshiro256++，false=std::mt19937_64
constexpr bool use_xoshiro256pp = true;

// 邻居查询快路径开关：true=启用 if constexpr + interior_mask 快路径；false=回退通用方案（用于 A/B 对照）
constexpr bool use_neighbor_fastpath = true;

#ifndef MVC_HK_QUEUE_OPTIMIZED
#define MVC_HK_QUEUE_OPTIMIZED 1
#endif

#ifndef MVC_HK_SHORTEST_LAYER_CUTOFF
#define MVC_HK_SHORTEST_LAYER_CUTOFF 0
#endif

#ifndef MVC_HK_ITERATIVE_DFS
#define MVC_HK_ITERATIVE_DFS 1
#endif

#ifndef MVC_HK_FREE_U_LIST
#define MVC_HK_FREE_U_LIST 1
#endif

#ifndef MVC_HK_LAZY_CUR_INIT
#define MVC_HK_LAZY_CUR_INIT 0
#endif

#ifndef MVC_HK_LAZY_DIST_RESET
#define MVC_HK_LAZY_DIST_RESET 0
#endif

#ifndef MVC_HK_SKIP_UNREACHABLE_ROOTS
#define MVC_HK_SKIP_UNREACHABLE_ROOTS 1
#endif

// 小 WCC 精确枚举阈值：K <= EXACT_ENUM_THRESHOLD 时直接物化全部合法态。
constexpr int EXACT_ENUM_THRESHOLD = 16;
// 19~24：优先 DFS 物化 exact；25~63：优先 frontier-DP exact；更大则回退 MCMC。
constexpr int DFS_EXACT_MAX_K = 24;
constexpr int FRONTIER_DP_MAX_K = 32;
// DFS 物化 exact 时，允许存入 exact_states_data 的“合法状态总数”上限。
// 若合法态数超过该值，就停止继续物化全部状态，转而尝试 frontier-DP 或回退 MCMC。
constexpr int EXACT_ENUM_VALID_STATE_CAP = 600000;
// DFS 物化 exact 时，递归搜索过程中允许访问的搜索树节点总数上限。
// 这个阈值主要控制预处理时间，防止某些约束较弱的分量在 DFS 上耗时过长。
constexpr int EXACT_ENUM_NODE_CAP = 4000000;
// frontier-DP exact 允许的最大 frontier 宽度。
// frontier 表示“已经处理过、但仍会影响后续决策”的活跃节点集合；其状态数大致按 2^width 增长。
// 一旦宽度超过该阈值，frontier-DP 的常数和内存会迅速变大，因此直接回退 MCMC。
constexpr int FRONTIER_DP_WIDTH_CAP = 12;
// frontier-DP exact 允许的 DP 总状态数上限（对所有层的状态数求和）。
// 即使 frontier 宽度没有超标，若整张 DP 表累计状态数仍然过大，也会放弃 frontier-DP，
// 以避免在 exact 预处理上投入过多时间和内存。
constexpr int FRONTIER_DP_STATE_CAP = 200000;

// ========================== 参数结构体 ==========================

// 运行参数总表：既包含模型几何参数，也包含采样策略和诊断开关。
struct Params {
	int d;
	int L;
	int W;
	double rho;
	uint64_t seed;
	bool periodic_boundary;
	bool debug_output;
	int mcmc_factor; // 固定模式下的 Level-Set Sweep 次数（每个 Sweep 含 forward+backward）
	bool use_acf_auto_sweeps; // true=先做短链并基于 ACF 自动估计 sweeps
	int acf_probe_sweeps;     // 自动模式：用于估计 ACF 的采样段 sweeps 数（仅该段参与 ACF）
	int acf_probe_burnin_sweeps; // 自动模式：ACF 探针链热化 sweeps 数（不参与 ACF）
	int acf_max_sweeps;       // 自动模式：总 sweeps 上限（含 probe）
	double acf_tau_multiplier; // 自动模式：目标平衡时间 = acf_tau_multiplier * tau_int
	int acf_tau_estimator;    // 自动模式：0=Sokal 自洽窗, 1=Geyer-IPS + Sokal(取较大值)
	double cached_tau_int;    // 自动模式：若 >0，则直接使用该 tau_int 并跳过 ACF 探针
	int init_mode;      // 初态模式：0=随机全0/全1, 1=BP采样, 2=雪崩动力学, 3=独立均匀随机
	int bp_iters;       // BP 消息传递迭代次数（仅 init_mode=1 时生效）
	int num_thermal_samples; // 同一张图上的热采样次数
	bool multi_start_mode; // true=每个热样本独立随机重启并burn-in后采样
	bool multi_spin_coding_mode; // true=当 thermal_samples 为 64 的倍数时启用 MSC64
	double decorrelation_multiplier; // 相邻热样本间隔倍数：auto 模式乘 tau_int；固定模式乘 mcmc_factor
	bool hk_use_greedy_init; // Hopcroft-Karp 前是否启用 O(E) 贪心预匹配
	std::vector<long long> L_multipliers;
	const std::vector<long long>* L_multipliers_ref = nullptr; // 可选：指向共享的乘子表，避免频繁拷贝 vector
	const std::vector<uint64_t>* interior_mask = nullptr; // 几何内部点位图（可选）
	const std::vector<uint8_t>* parity_cache = nullptr; // 节点奇偶性缓存（可选）
};

// 线程本地工作区：集中缓存所有中间数组，避免在热点循环中反复分配内存。
// 字段命名约定：
// - hk_*：Hopcroft-Karp 最大匹配；
// - bip_*：二分图构图过程；
// - dep_*/tarjan_*：依赖图与 SCC 缩点；
// - mcmc_*：DAG 采样、ACF 探针、BP/雪崩初始化；
// - mark_* 与 *_stamp：时间戳打标，减少 O(N) 清空次数。
struct ThreadLocalWorkspace {
	std::vector<int> hk_match_u;
	std::vector<int> hk_match_v;
	std::vector<int> hk_dist;
	std::vector<int> hk_dist_stamp;
	std::vector<int> hk_cur;
	std::vector<int> hk_queue;
	std::vector<int> hk_stack_u;
	std::vector<int> hk_stack_v;
	std::vector<int> hk_free_u;
	std::vector<int> hk_cur_stamp;
	std::vector<int> bip_mark_v;
	std::vector<int> bip_deg_v;
	std::vector<int> bip_uniq_v;
	std::vector<int> bip_cur_v;
	std::vector<int> gtu_stamp;
	std::vector<int> gtu_local;
	std::vector<int> gtv_stamp;
	std::vector<int> gtv_local;
	std::vector<uint8_t> lattice;
	std::vector<char> flags_b;
	std::vector<int> rule_input_zero_sites;
	std::vector<int> rule_restore_sites;
	std::vector<int> rule_zero_sites;
	std::vector<int> queue;
	std::vector<int> component;
	std::vector<int> best_component;
	std::vector<int> active_sites;
	std::vector<int> mark_a;
	std::vector<int> mark_b;
	std::vector<int> temp_ids;
	std::vector<int> dep_offsets;
	std::vector<int> dep_data;
	std::vector<int> super_deg;
	std::vector<int> pred_deg;
	std::vector<int> tarjan_dfn;
	std::vector<int> tarjan_low;
	std::vector<int> tarjan_in_stack;
	std::vector<int> tarjan_scc_id;
	std::vector<int> tarjan_stk;
	std::vector<int> tarjan_parent;
	std::vector<int> tarjan_explicit_u;
	std::vector<int> tarjan_explicit_next_ei;
	std::vector<int> dag_super_cur;
	std::vector<int> dag_pred_cur;
	std::vector<char> mcmc_visited;
	std::vector<int> mcmc_comp_offsets;
	std::vector<int> mcmc_comp_nodes;
	std::vector<int> mcmc_comp_order;
	std::vector<int> mcmc_bfs_queue;
	std::vector<int> mcmc_indeg_scratch;
	std::vector<int> mcmc_topo_order;
	std::vector<int> mcmc_kahn_q;
	std::vector<int> mcmc_level;
	std::vector<int> mcmc_level_offsets;
	std::vector<int> mcmc_level_data;
	std::vector<int> mcmc_level_counts;
	std::vector<int> mcmc_level_write;
	std::vector<int> mcmc_num_zero_parents;
	std::vector<int> mcmc_num_one_children;
	std::vector<int> mcmc_visit_mark;
	std::vector<int> mcmc_bp_local_index;
	std::vector<int> mcmc_bp_parent_offsets;
	std::vector<int> mcmc_bp_child_offsets;
	std::vector<int> mcmc_bp_parent_data;
	std::vector<int> mcmc_bp_child_data;
	std::vector<int> mcmc_bp_parent_count;
	std::vector<int> mcmc_bp_child_count;
	std::vector<int> mcmc_bp_parent_cursor;
	std::vector<int> mcmc_bp_child_cursor;
	std::vector<int> mcmc_bp_edge_parent;
	std::vector<int> mcmc_bp_edge_child;
	std::vector<int> mcmc_bp_edge_parent_local;
	std::vector<int> mcmc_bp_edge_child_local;
	std::vector<double> mcmc_bp_mu;
	std::vector<double> mcmc_bp_eta;
	std::vector<double> mcmc_bp_new_mu;
	std::vector<double> mcmc_bp_new_eta;
	std::vector<int> mcmc_avalanche_queue;
	std::vector<double> mcmc_x_traj;
	std::vector<double> mcmc_acf;
	std::vector<std::complex<double>> mcmc_fft_buffer;
	std::vector<double> mcmc_gauss_weights; // ACF 探针用随机高斯投影权重
	int gtu_cur_stamp = 1;
	int gtv_cur_stamp = 1;
	int bip_mark_v_stamp = 1;
	int hk_cur_phase = 1;
	int hk_dist_phase = 1;
	int mcmc_visit_token = 1;
	int mark_a_stamp = 1;
	int mark_b_stamp = 1;
};

static thread_local ThreadLocalWorkspace tls_ws;

// ========================== 工具函数 ==========================

// 条件调试输出：仅在 enabled=true 时打印，避免常态运行下 I/O 干扰性能。
void debug_log(bool enabled, const std::string& message) {
	if (!enabled) return;
	std::cout << "[DEBUG] " << message << std::endl;
}

constexpr double PI_CONST = 3.141592653589793238462643383279502884;

#if defined(__GNUC__) || defined(__clang__)
#define MVC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MVC_UNLIKELY(x) (x)
#endif

enum class BenchmarkPhase : int {
	GenerateInitialLattice = 0,
	ApplyRules,
	CollectActiveSites,
	ExtractLargestConnectedComponent,
	BuildBipartiteGraph,
	HopcroftKarp,
	Freezing,
	BuildDependencySccDag,
	PrecomputeMcmcDag,
	SampleMvcMcmc,
	Count
};

constexpr int kBenchmarkPhaseCount = static_cast<int>(BenchmarkPhase::Count);

struct BenchmarkTrace {
	std::array<double, kBenchmarkPhaseCount> seconds{};
	std::array<long long, kBenchmarkPhaseCount> calls{};

	void add(BenchmarkPhase phase, double elapsed_seconds) {
		const int idx = static_cast<int>(phase);
		seconds[idx] += elapsed_seconds;
		calls[idx] += 1;
	}

	void merge_from(const BenchmarkTrace& other) {
		for (int i = 0; i < kBenchmarkPhaseCount; ++i) {
			seconds[i] += other.seconds[i];
			calls[i] += other.calls[i];
		}
	}
};

static thread_local BenchmarkTrace* tls_benchmark_trace = nullptr;

class ScopedBenchmarkPhase {
public:
	explicit ScopedBenchmarkPhase(BenchmarkPhase phase)
		: phase_(phase),
		  trace_(tls_benchmark_trace),
		  active_(trace_ != nullptr)
	{
		if (active_) start_ = Clock::now();
	}

	~ScopedBenchmarkPhase() {
		if (!active_) return;
		const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
		trace_->add(phase_, elapsed);
	}

private:
	using Clock = std::chrono::steady_clock;

	BenchmarkPhase phase_;
	BenchmarkTrace* trace_ = nullptr;
	Clock::time_point start_{};
	bool active_ = false;
};

class BenchmarkTraceGuard {
public:
	explicit BenchmarkTraceGuard(BenchmarkTrace* trace)
		: previous_(tls_benchmark_trace)
	{
		tls_benchmark_trace = trace;
	}

	~BenchmarkTraceGuard() {
		tls_benchmark_trace = previous_;
	}

private:
	BenchmarkTrace* previous_ = nullptr;
};

inline const char* benchmark_phase_name(BenchmarkPhase phase) {
	switch (phase) {
	case BenchmarkPhase::GenerateInitialLattice: return "generate_initial_lattice";
	case BenchmarkPhase::ApplyRules: return "apply_rules";
	case BenchmarkPhase::CollectActiveSites: return "collect_active_sites";
	case BenchmarkPhase::ExtractLargestConnectedComponent: return "extract_lcc";
	case BenchmarkPhase::BuildBipartiteGraph: return "build_bipartite_graph";
	case BenchmarkPhase::HopcroftKarp: return "hopcroft_karp";
	case BenchmarkPhase::Freezing: return "freezing";
	case BenchmarkPhase::BuildDependencySccDag: return "build_dependency_scc_dag";
	case BenchmarkPhase::PrecomputeMcmcDag: return "precompute_mcmc_dag";
	case BenchmarkPhase::SampleMvcMcmc: return "sample_mvc_mcmc";
	case BenchmarkPhase::Count: break;
	}
	return "unknown";
}

// 32 位整数扰动哈希：用于把近邻种子打散，降低相关性。
inline unsigned int hash_seed(unsigned int input) {
	unsigned int z = input + 0x9e3779b9u;
	z ^= z >> 15;
	z *= 0x85ebca6bu;
	z ^= z >> 13;
	z *= 0xc2b2ae35u;
	z ^= z >> 16;
	return z;
}

// 多参数 64 位混合哈希，用于消除线性叠加种子的碰撞风险
inline uint64_t hash_seed_64(uint64_t seed1, uint64_t seed2, uint64_t seed3, uint64_t seed4) {
	// 这里采用类似 boost::hash_combine 的“加常数+移位反馈”混合方式：
	// 每引入一个分量都让当前 h 参与新混合，避免简单异或导致的信息丢失。
	uint64_t h = seed1;
	h ^= seed2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
	h ^= seed3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
	h ^= seed4 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
	// Splitmix64 终结器
	h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
	h ^= h >> 27; h *= 0x94d049bb133111ebULL;
	h ^= h >> 31;
	return h;
}

// SplitMix64 迭代：把单个 64 位输入扩展成高质量伪随机序列。
inline uint64_t splitmix64_next(uint64_t& x) {
	x += 0x9e3779b97f4a7c15ULL;
	uint64_t z = x;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

// 循环左移：xoshiro 的核心位操作。
inline uint64_t rotl64(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

// xoshiro256++ 引擎封装（URBG 接口）：
// - 通过 seed_engine 将 64 位种子扩展到 4x64 状态；
// - operator() 生成一个 64 位随机数；
// - 对全零状态做保护，避免退化序列。
class Xoshiro256PlusPlus {
public:
	using result_type = uint64_t;

	explicit Xoshiro256PlusPlus(uint64_t seed = 1ULL) {
		seed_engine(seed);
	}

	static constexpr result_type min() { return 0ULL; }
	static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

	result_type operator()() {
		// xoshiro256++ 输出函数：
		// result = rotl(s0+s3,23)+s0，再做一轮线性状态跃迁。
		const uint64_t result = rotl64(state_[0] + state_[3], 23) + state_[0];
		const uint64_t t = state_[1] << 17;

		state_[2] ^= state_[0];
		state_[3] ^= state_[1];
		state_[1] ^= state_[2];
		state_[0] ^= state_[3];

		state_[2] ^= t;
		state_[3] = rotl64(state_[3], 45);

		return result;
	}

private:
	uint64_t state_[4] = { 0ULL, 0ULL, 0ULL, 0ULL };

	void seed_engine(uint64_t seed) {
		uint64_t x = seed;
		// 用 SplitMix64 把单种子扩展为 4 个状态字，降低相邻 seed 的相关性风险。
		for (int i = 0; i < 4; ++i) state_[i] = splitmix64_next(x);
		// xoshiro 的状态不能全零；若命中则注入固定非零常量。
		if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0ULL) {
			state_[0] = 0x9e3779b97f4a7c15ULL;
		}
	}
};

using RNG = std::conditional_t<use_xoshiro256pp, Xoshiro256PlusPlus, std::mt19937_64>;

constexpr const char* kRngName = use_xoshiro256pp ? "xoshiro256++" : "mt19937_64";

// 位缓存硬币采样器：一次 RNG 调用缓存 64 个随机 bit，随后按位消费。
struct FastCoinFlip {
	uint64_t bit_buffer = 0;
	int bits_left = 0;

	inline int next(RNG& gen) {
		if (MVC_UNLIKELY(bits_left == 0)) {
			// 位缓存耗尽时才访问底层 RNG，一次填满 64 位以摊薄调用开销。
			bit_buffer = gen();
			bits_left = 64;
		}
		// 始终从最低位取样，再右移到下一位。
		int bit = static_cast<int>(bit_buffer & 1ULL);
		bit_buffer >>= 1;
		--bits_left;
		return bit;
	}
};

// 时间戳打标工具：
// 若 stamp 溢出则清零重置；否则递增并返回新戳。
// 用于“逻辑清空”数组，避免每轮都 O(N) memset。
inline int next_mark_stamp(std::vector<int>& marks, int& stamp) {
	if (stamp == std::numeric_limits<int>::max()) {
		std::fill(marks.begin(), marks.end(), 0);
		stamp = 1;
	}
	return ++stamp;
}

// 快速生成 [0, n) 均匀整数：
// - 2 的幂用按位与；
// - 其他情形优先用 128 位乘高位法，减少模除偏差和开销。
inline int fast_rand_below(RNG& gen, int n) {
	if (n <= 1) return 0;
	uint64_t un = static_cast<uint64_t>(n);
	if ((un & (un - 1ULL)) == 0ULL) {
		// n 为 2^k 时，低 k 位天然均匀，直接掩码最快。
		return static_cast<int>(gen() & (un - 1ULL));
	}
#if defined(__SIZEOF_INT128__)
	// “乘高位”技巧：floor((U*n)/2^64)。
	// 当 U 在 [0,2^64) 均匀时，此映射较取模更均匀，且通常更快。
	return static_cast<int>(
		(static_cast<unsigned __int128>(gen()) * static_cast<unsigned __int128>(un)) >> 64);
#else
	return static_cast<int>(gen() % un);
#endif
}

// 64 位版本的均匀整数采样：frontier-DP exact 需要对计数表做无偏抽样。
inline uint64_t fast_rand_below_u64(RNG& gen, uint64_t n) {
	if (n <= 1ULL) return 0ULL;
	if ((n & (n - 1ULL)) == 0ULL) {
		return static_cast<uint64_t>(gen()) & (n - 1ULL);
	}
#if defined(__SIZEOF_INT128__)
	while (true) {
		__uint128_t wide = static_cast<__uint128_t>(gen()) * static_cast<__uint128_t>(n);
		uint64_t low = static_cast<uint64_t>(wide);
		uint64_t high = static_cast<uint64_t>(wide >> 64);
		uint64_t threshold = static_cast<uint64_t>(-n) % n;
		if (low >= threshold) return high;
	}
#else
	std::uniform_int_distribution<uint64_t> dist(0ULL, n - 1ULL);
	return dist(gen);
#endif
}

// 获取线性索引乘子表（支持共享引用，减少 Params 拷贝成本）。
inline const std::vector<long long>& get_l_multipliers(const Params& p) {
	return (p.L_multipliers_ref != nullptr) ? *p.L_multipliers_ref : p.L_multipliers;
}

// 快速计算节点奇偶性（不分配内存）
inline int index_parity_uncached(int index, const Params& p) {
	if constexpr (d == 1) {
		return index % p.L & 1;
	} else if constexpr (d == 2) {
		int row = index / p.W;
		int col = index - row * p.W;
		return (row + col) & 1;
	} else if constexpr (d == 3) {
		const int W = p.W;
		const int W2 = W * W;
		int z = index / W2;
		int rem = index - z * W2;
		int y = rem / W;
		int x = rem - y * W;
		return (x + y + z) & 1;
	} else {
		if (p.d <= 0) return 0;
		int parity = 0;
		for (int i = p.d - 1; i >= 1; --i) {
			parity += index % p.W;
			index /= p.W;
		}
		parity += index % p.L;
		return parity & 1;
	}
}

inline int index_parity(int index, const Params& p) {
	if (p.parity_cache != nullptr) {
		return static_cast<int>((*p.parity_cache)[static_cast<size_t>(index)]);
	}
	return index_parity_uncached(index, p);
}

void build_parity_cache(const Params& p, long long total_nodes, std::vector<uint8_t>& parity_cache) {
	const size_t total_nodes_sz = static_cast<size_t>(std::max<long long>(0, total_nodes));
	parity_cache.resize(total_nodes_sz);
	for (int i = 0; i < total_nodes; ++i) {
		parity_cache[static_cast<size_t>(i)] = static_cast<uint8_t>(index_parity_uncached(i, p));
	}
}

// 栈上邻居列表（避免堆分配）
struct NeighborList {
	int data[2 * d];
	int count = 0;
	void push(int v) { data[count++] = v; }
	int* begin() { return data; }
	int* end() { return data + count; }
	const int* begin() const { return data; }
	const int* end() const { return data + count; }
	int size() const { return count; }
	bool empty() const { return count == 0; }
	int operator[](int i) const { return data[i]; }
};

// 通用邻居构造（慢路径）：
// 适用于任意维度 d，先解码坐标再按边界条件回写邻居索引。
NeighborList compute_neighbors_slow_once(int index, const Params& p) {
	NeighborList neighbors;
	if (p.d == 0 || p.L <= 0 || (p.d > 1 && p.W <= 0)) return neighbors;

	int tmp = index;
	int coords[d];
	if (p.d == 1) {
		coords[0] = tmp % p.L;
	} else {
		for (int i = p.d - 1; i >= 1; --i) {
			coords[i] = tmp % p.W;
			tmp /= p.W;
		}
		coords[0] = tmp % p.L;
	}

	for (int i = 0; i < p.d; ++i) {
		int dim_size = (i == 0) ? p.L : p.W;
		if (dim_size <= 1) continue;

		const auto& multipliers = get_l_multipliers(p);
		long long multiplier = multipliers[i];
		int original_coord_val = coords[i];

		if (p.periodic_boundary) {
			if (original_coord_val == dim_size - 1)
				neighbors.push(index - original_coord_val * static_cast<int>(multiplier));
			else
				neighbors.push(index + static_cast<int>(multiplier));
		} else {
			if (original_coord_val + 1 < dim_size)
				neighbors.push(index + static_cast<int>(multiplier));
		}

		if (p.periodic_boundary) {
			if (original_coord_val == 0)
				neighbors.push(index + (dim_size - 1) * static_cast<int>(multiplier));
			else
				neighbors.push(index - static_cast<int>(multiplier));
		} else {
			if (original_coord_val > 0)
				neighbors.push(index - static_cast<int>(multiplier));
		}
	}

	if ((p.L == 2 || p.W == 2) && p.d > 1) {
		std::sort(neighbors.begin(), neighbors.end());
		neighbors.count = static_cast<int>(std::unique(neighbors.begin(), neighbors.end()) - neighbors.begin());
	}
	return neighbors;
}

// 邻居查询总入口：
// - d=1/2/3 时使用手写快路径；
// - 对内部点直接常数偏移，不做除法和取模；
// - 其余回退到通用慢路径。
NeighborList get_neighbors(int index, const Params& p) {
	if constexpr (!use_neighbor_fastpath) {
		return compute_neighbors_slow_once(index, p);
	}

	NeighborList nb;

	if constexpr (d == 1) {
		if (p.L <= 0) return nb;
		if constexpr (kPeriodicBoundaryCompileTime) {
			nb.push(index == p.L - 1 ? 0 : index + 1);
			nb.push(index == 0 ? p.L - 1 : index - 1);
			nb.count = 2;
		} else {
			if (index + 1 < p.L) nb.push(index + 1);
			if (index - 1 >= 0) nb.push(index - 1);
		}
		return nb;
	}

	if constexpr (d == 2) {
		if (p.L <= 0 || p.W <= 0) return nb;
		const int W = p.W;

		// Fast Path: 内部节点（无需除法/取模）
		if (p.interior_mask != nullptr) {
			const auto& mask = *p.interior_mask;
			// bitset 编码：第 index 个点位于 mask[index>>6] 的第 (index&63) 位。
			// 若该位为 1，说明是内部点，四个邻居可直接常数偏移得到。
			if (!mask.empty() && ((mask[index >> 6] >> (index & 63)) & 1ULL)) {
				nb.push(index + 1);
				nb.push(index - 1);
				nb.push(index + W);
				nb.push(index - W);
				nb.count = 4;
				return nb;
			}
		}

		// Slow Path: 边界节点
		int row = index / W;
		int col = index % W;
		if constexpr (kPeriodicBoundaryCompileTime) {
			nb.push(col == W - 1 ? index - (W - 1) : index + 1);
			nb.push(col == 0 ? index + (W - 1) : index - 1);
			nb.push(row == p.L - 1 ? index - (p.L - 1) * W : index + W);
			nb.push(row == 0 ? index + (p.L - 1) * W : index - W);
			nb.count = 4;
			if (p.L == 2 || p.W == 2) {
				// 当边长为 2 时，周期边界可能让“前后邻居”重合，需要去重。
				std::sort(nb.begin(), nb.end());
				nb.count = static_cast<int>(std::unique(nb.begin(), nb.end()) - nb.begin());
			}
		} else {
			if (col + 1 < W) nb.push(index + 1);
			if (col > 0) nb.push(index - 1);
			if (row + 1 < p.L) nb.push(index + W);
			if (row > 0) nb.push(index - W);
		}
		return nb;
	}

	if constexpr (d == 3) {
		if (p.L <= 0 || p.W <= 0) return nb;
		const int W = p.W;
		const int W2 = W * W;

		// Fast Path: 内部节点（无需除法/取模）
		if (p.interior_mask != nullptr) {
			const auto& mask = *p.interior_mask;
			if (!mask.empty() && ((mask[index >> 6] >> (index & 63)) & 1ULL)) {
				nb.push(index + 1);
				nb.push(index - 1);
				nb.push(index + W);
				nb.push(index - W);
				nb.push(index + W2);
				nb.push(index - W2);
				nb.count = 6;
				return nb;
			}
		}

		// Slow Path: 边界节点
		int z = index / W2;
		int rem = index - z * W2;
		int y = rem / W;
		int x = rem - y * W;

		if constexpr (kPeriodicBoundaryCompileTime) {
			nb.push(x == W - 1 ? index - (W - 1) : index + 1);
			nb.push(x == 0 ? index + (W - 1) : index - 1);
			nb.push(y == W - 1 ? index - (W - 1) * W : index + W);
			nb.push(y == 0 ? index + (W - 1) * W : index - W);
			nb.push(z == p.L - 1 ? index - (p.L - 1) * W2 : index + W2);
			nb.push(z == 0 ? index + (p.L - 1) * W2 : index - W2);
			nb.count = 6;
			if (p.L == 2 || W == 2) {
				std::sort(nb.begin(), nb.end());
				nb.count = static_cast<int>(std::unique(nb.begin(), nb.end()) - nb.begin());
			}
		} else {
			if (x + 1 < W) nb.push(index + 1);
			if (x > 0) nb.push(index - 1);
			if (y + 1 < W) nb.push(index + W);
			if (y > 0) nb.push(index - W);
			if (z + 1 < p.L) nb.push(index + W2);
			if (z > 0) nb.push(index - W2);
		}
		return nb;
	}

	// d > 3 通用后备路径
	return compute_neighbors_slow_once(index, p);
}

// 返回不小于 x 的最小 2 次幂（FFT 填充长度）。
int next_pow2(int x) {
	int n = 1;
	while (n < x) n <<= 1;
	return n;
}

// Cooley-Tukey 迭代 FFT（原地）：
// invert=false 正变换，invert=true 逆变换并除以 n。
void fft(std::vector<std::complex<double>>& a, bool invert) {
	int n = static_cast<int>(a.size());
	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) std::swap(a[i], a[j]);
	}

	for (int len = 2; len <= n; len <<= 1) {
		double ang = 2.0 * PI_CONST / len * (invert ? -1.0 : 1.0);
		std::complex<double> wlen(std::cos(ang), std::sin(ang));
		for (int i = 0; i < n; i += len) {
			std::complex<double> w(1.0, 0.0);
			for (int j = 0; j < len / 2; ++j) {
				std::complex<double> u = a[i + j];
				std::complex<double> v = a[i + j + len / 2] * w;
				a[i + j] = u + v;
				a[i + j + len / 2] = u - v;
				w *= wlen;
			}
		}
	}

	if (invert) {
		for (auto& x : a) x /= static_cast<double>(n);
	}
}

// 基于 FFT 的偏置自相关估计：
// 1) 去均值；
// 2) 频域平方幅值；
// 3) 逆变换回时域并按 c0 归一化得到 rho(k)。
void acf_fft_biased_inplace(
	const std::vector<double>& x,
	std::vector<double>& rho,
	std::vector<std::complex<double>>& fft_buffer)
{
	int N = static_cast<int>(x.size());
	rho.assign(N, 0.0);
	if (N == 0) return;

	double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(N);
	int M = next_pow2(2 * N);
	fft_buffer.resize(M);
	std::fill(fft_buffer.begin(), fft_buffer.end(), std::complex<double>(0.0, 0.0));
	for (int i = 0; i < N; ++i) fft_buffer[i] = {x[i] - mean, 0.0};

	fft(fft_buffer, false);
	for (int i = 0; i < M; ++i) fft_buffer[i] *= std::conj(fft_buffer[i]);
	fft(fft_buffer, true);

	double c0 = fft_buffer[0].real() / static_cast<double>(N);
	// if (std::abs(c0) < 1e-6) {
	// 	std::fill(rho.begin(), rho.end(), 1.0);
	// 	return;
	// }

	if (std::abs(c0) < 1e-6) {
		std::fill(rho.begin(), rho.end(), 0.0);
		rho[0] = 1.0;
		return;
	}


	for (int k = 0; k < N; ++k) {
		double ck = fft_buffer[k].real() / static_cast<double>(N);
		rho[k] = ck / c0;
	}
	rho[0] = 1.0;
}

// ACF 推导出的积分相关时间及其截断窗口信息。
struct TauResult {
	double tau_int = 1.0;
	int window_M = 0;
};

// Sokal 自洽窗估计 tau_int：
// 满足 M >= c * tau(M) 的首个窗口作为截断点。
TauResult estimate_tau_sokal(const std::vector<double>& acf, double c = 5.0) {
	TauResult out;
	if (acf.size() <= 1) return out;

	double sum = 0.0;
	int N = static_cast<int>(acf.size());
	int M_cap = std::max(1, N / 10);
	for (int M = 1; M <= M_cap; ++M) {
		sum += acf[M];
		double tauM = 1.0 + 2.0 * sum;
		if (tauM < 1.0) tauM = 1.0;
		if (M >= c * tauM) {
			out.tau_int = tauM;
			out.window_M = M;
			return out;
		}
	}
	out.tau_int = 1.0 + 2.0 * sum;
	if (out.tau_int < 1.0) out.tau_int = 1.0;
	out.window_M = M_cap;
	return out;
}

// Geyer Initial Positive Sequence (IPS) 估计：
// 对成对自协方差和做非增包络并截断到正区间。
TauResult estimate_tau_geyer_ips(const std::vector<double>& acf) {
	TauResult out;
	int N = static_cast<int>(acf.size());
	if (N <= 1) return out;
	int M_cap = std::max(1, N / 10);

	double sum = 0.0;
	int last_k = 0;
	double min_gamma = std::numeric_limits<double>::max();
	for (int k = 0; 2 * k + 1 <= M_cap; ++k) {
		double gamma_k = acf[2 * k] + acf[2 * k + 1];

		if (k == 0) {
			sum += acf[1];
			min_gamma = gamma_k;
			if (min_gamma <= 0.0) {
				last_k = 0;
				break;
			}
			last_k = 0;
			continue;
		}

		gamma_k = std::min(gamma_k, min_gamma);
		if (gamma_k <= 0.0) break;
		min_gamma = gamma_k;
		sum += gamma_k;
		last_k = k;
	}

	out.tau_int = std::max(1.0, 1.0 + 2.0 * sum);
	out.window_M = std::min(M_cap, 2 * last_k + 1);
	return out;
}

// 断点键：唯一标识一个 (d, L, W, rho) 任务点。
struct CheckpointKey {
	int d = 0;
	int L = 0;
	int W = 0;
	long long rho_key = 0;

	bool operator==(const CheckpointKey& other) const {
		return d == other.d && L == other.L && W == other.W && rho_key == other.rho_key;
	}
};

// CheckpointKey 的组合哈希，便于 unordered_* 容器查询。
struct CheckpointKeyHash {
	std::size_t operator()(const CheckpointKey& k) const {
		std::size_t h1 = std::hash<int>{}(k.d);
		std::size_t h2 = std::hash<int>{}(k.L);
		std::size_t h3 = std::hash<int>{}(k.W);
		std::size_t h4 = std::hash<long long>{}(k.rho_key);
		std::size_t h = h1;
		h ^= h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		h ^= h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		h ^= h4 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		return h;
	}
};

// 将 rho 量化为整数键，规避浮点比较误差。
inline long long rho_to_key(double rho) {
	return static_cast<long long>(std::llround(rho * 1e10));
}

// 样本种子构造：
// 可选 CRN（跨 rho 共享随机数）以降低差分估计方差。
inline uint64_t make_sample_seed(
	uint64_t seed_base,
	int L,
	double rho,
	int sample_index,
	bool use_common_random_numbers_across_rho)
{
	const uint64_t rho_component = use_common_random_numbers_across_rho
		? 0xD1B54A32D192ED03ULL
		: static_cast<uint64_t>(rho_to_key(rho));
	return hash_seed_64(seed_base,
		static_cast<uint64_t>(L),
		rho_component,
		static_cast<uint64_t>(sample_index));
}

// 从统计文件读取已完成断点，用于 resume 模式跳过重复计算。
std::unordered_set<CheckpointKey, CheckpointKeyHash> load_completed_checkpoints_from_stat(
	const std::string& stat_file)
{
	std::unordered_set<CheckpointKey, CheckpointKeyHash> completed;
	std::ifstream fin(stat_file);
	if (!fin.is_open()) return completed;

	int d = 0, L = 0, W = 0;
	double rho = 0.0;
	std::string line;
	while (std::getline(fin, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		if (!(iss >> d >> L >> W >> rho)) continue;
		CheckpointKey key;
		key.d = d;
		key.L = L;
		key.W = W;
		key.rho_key = rho_to_key(rho);
		completed.insert(key);
	}
	return completed;
}

// 读取 tau 缓存：键为 (d,L,W,rho)，值为估计到的 tau_int。
std::unordered_map<CheckpointKey, double, CheckpointKeyHash> load_tau_cache(
	const std::string& tau_cache_file)
{
	std::unordered_map<CheckpointKey, double, CheckpointKeyHash> cache;
	std::ifstream fin(tau_cache_file);
	if (!fin.is_open()) return cache;

	int d = 0, L = 0, W = 0;
	double rho = 0.0, tau = 0.0;
	std::string line;
	while (std::getline(fin, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		if (!(iss >> d >> L >> W >> rho >> tau)) continue;
		if (!(tau > 0.0) || !std::isfinite(tau)) continue;
		CheckpointKey key;
		key.d = d;
		key.L = L;
		key.W = W;
		key.rho_key = rho_to_key(rho);
		cache[key] = tau;
	}
	return cache;
}

// 追加写入一条 tau 缓存记录，失败返回 false 交由调用方告警。
bool append_tau_cache_entry(
	const std::string& tau_cache_file,
	int d, int L, int W, double rho, double tau)
{
	if (!(tau > 0.0) || !std::isfinite(tau)) return false;
	std::ofstream fout(tau_cache_file, std::ios::app);
	if (!fout.is_open()) return false;
	fout << d << " " << L << " " << W << " "
		 << std::fixed << std::setprecision(10) << rho << " "
		 << std::fixed << std::setprecision(10) << tau << "\n";
	return true;
}

// 重建 tau 缓存文件并写入表头（用于强制刷新缓存）。
bool reset_tau_cache_file_with_header(const std::string& tau_cache_file) {
#if __cplusplus >= 201703L
	{
		std::error_code ec;
		std::filesystem::remove(tau_cache_file, ec);
		if (ec) {
			std::cerr << "Warning: failed to remove old tau cache file "
					  << tau_cache_file << " : " << ec.message() << std::endl;
		}
	}
#endif

	std::ofstream fout(tau_cache_file, std::ios::trunc);
	if (!fout.is_open()) return false;
	fout << "# d L W rho tau_int\n";
	return true;
}

// 仅在“新文件”或“非续跑覆盖模式”下写表头，避免重复 header。
bool should_write_header(const std::string& file_path, bool resume_mode) {
	if (!resume_mode) return true;
	std::ifstream fin(file_path, std::ios::binary);
	if (!fin.is_open()) return true;
	return fin.peek() == std::ifstream::traits_type::eof();
}

// ========================== 模块一：随机图生成 ==========================

// 生成 Bernoulli 初始晶格：
// 0 表示空位，1 表示占据；P(0)=rho。
void generate_initial_lattice_inplace(std::vector<uint8_t>& lattice, long long total_nodes, double rho, RNG& gen) {
	ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::GenerateInitialLattice);
	const size_t total_nodes_sz = static_cast<size_t>(std::max<long long>(0, total_nodes));
	lattice.resize(total_nodes_sz);
	if (total_nodes_sz == 0) return;
	if (rho <= 0.0) {
		std::fill(lattice.begin(), lattice.end(), 1);
		return;
	}
	if (rho >= 1.0) {
		std::fill(lattice.begin(), lattice.end(), 0);
		return;
	}

	constexpr long double kUInt64Range =
		static_cast<long double>(std::numeric_limits<uint64_t>::max()) + 1.0L; // 2^64
	const uint64_t threshold = static_cast<uint64_t>(rho * kUInt64Range);

	uint8_t* lattice_data = lattice.data();
	for (size_t i = 0; i < total_nodes_sz; ++i) {
		lattice_data[i] = (static_cast<uint64_t>(gen()) < threshold) ? uint8_t{0} : uint8_t{1};
	}
}

// 对初始晶格施加两条局域规则：
// - 规则1：去除孤立空位（恢复为 1）；
// - 规则2：空位向其最近邻扩张（邻居置 0）。
void apply_rules(
	std::vector<uint8_t>& lattice,
	const Params& p,
	long long total_nodes)
{
	ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::ApplyRules);
	if (total_nodes == 0) return;
	ThreadLocalWorkspace& ws = tls_ws;
	auto& to_be_zeroed = ws.flags_b;
	auto& zero_sites = ws.rule_input_zero_sites;
	auto& isolated_zero_sites = ws.rule_restore_sites;
	auto& zeroed_sites = ws.rule_zero_sites;
	const size_t total_nodes_sz = static_cast<size_t>(total_nodes);

	if (to_be_zeroed.size() < total_nodes_sz) to_be_zeroed.resize(total_nodes_sz, 0);
	zero_sites.clear();
	isolated_zero_sites.clear();
	zeroed_sites.clear();

	// 先收集原始零点集合，后续两条规则都只在这批点上工作。
	for (long long i = 0; i < total_nodes; ++i) {
		if (lattice[i] == 0) zero_sites.push_back(static_cast<int>(i));
	}

	// 每个原始零点只查询一次邻居：
	// - 若是孤立零点，则加入恢复列表；
	// - 否则立即把其邻居加入“待清零”集合。
	for (int idx : zero_sites) {
		auto neighbors = get_neighbors(idx, p);
		bool all_neighbors_one = !neighbors.empty();
		for (int neighbor_idx : neighbors) {
			if (lattice[neighbor_idx] == 0) {
				all_neighbors_one = false;
				break;
			}
		}
		if (all_neighbors_one) {
			isolated_zero_sites.push_back(idx);
			continue;
		}
		for (int neighbor_idx : neighbors) {
			size_t nb_idx = static_cast<size_t>(neighbor_idx);
			if (to_be_zeroed[nb_idx] == 0) {
				to_be_zeroed[nb_idx] = 1;
				zeroed_sites.push_back(neighbor_idx);
			}
		}
	}

	for (int idx : isolated_zero_sites) lattice[idx] = 1;
	for (int idx : zeroed_sites) {
		lattice[idx] = 0;
		to_be_zeroed[static_cast<size_t>(idx)] = 0;
	}
}

void collect_active_sites(
	const std::vector<uint8_t>& lattice,
	std::vector<int>& active_sites)
{
	ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::CollectActiveSites);
	active_sites.clear();
	active_sites.reserve(lattice.size());
	for (int i = 0; i < static_cast<int>(lattice.size()); ++i) {
		if (lattice[static_cast<size_t>(i)] == 1) active_sites.push_back(i);
	}
}

// 从 lattice==1 的子图中提取最大连通分量（BFS），其余点清零。
// 该步骤可显著减小后续匹配与 MCMC 规模。
void extract_largest_connected_component(
	std::vector<uint8_t>& lattice,
	const Params& p,
	long long total_nodes,
	const std::vector<int>& active_sites,
	std::vector<int>& current,
	std::vector<int>& q,
	std::vector<int>& best_component,
	long long* lcc_size_out = nullptr)
{
	ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::ExtractLargestConnectedComponent);
	ThreadLocalWorkspace& ws = tls_ws;
	auto& visited_mark = ws.mark_a;
	auto& keep_mark = ws.mark_b;
	if (static_cast<long long>(visited_mark.size()) < total_nodes) {
		visited_mark.resize(static_cast<size_t>(total_nodes), 0);
	}
	if (static_cast<long long>(keep_mark.size()) < total_nodes) {
		keep_mark.resize(static_cast<size_t>(total_nodes), 0);
	}
	int visited_stamp = next_mark_stamp(visited_mark, ws.mark_a_stamp);
	int keep_stamp = next_mark_stamp(keep_mark, ws.mark_b_stamp);

	best_component.clear();
	current.clear();
	q.clear();

	for (int start : active_sites) {
		if (visited_mark[start] == visited_stamp) continue;

		current.clear();
		q.clear();
		q.push_back(start);
		visited_mark[start] = visited_stamp;
		int head = 0;

		while (head < static_cast<int>(q.size())) {
			int v = q[head++];
			current.push_back(v);
			auto neighbors = get_neighbors(v, p);
			for (int nb : neighbors) {
				if (nb >= 0 && nb < total_nodes && lattice[nb] == 1 && visited_mark[nb] != visited_stamp) {
					visited_mark[nb] = visited_stamp;
					q.push_back(nb);
				}
			}
		}

		if (current.size() > best_component.size()) best_component.swap(current);
	}

	for (int v : best_component) keep_mark[v] = keep_stamp;
	for (int v : active_sites) {
		if (keep_mark[v] != keep_stamp) lattice[v] = 0;
	}

	if (lcc_size_out != nullptr) *lcc_size_out = static_cast<long long>(best_component.size());
}

// ========================== 模块二：二分图映射 ==========================

// 二分图 CSR 表示：
// U/V 为奇偶划分后的点集；adjU 与 adjV 同时保留便于后续双向传播。
struct BipartiteGraph {
	int nU = 0;
	int nV = 0;
	std::vector<int> U_global;  // U 局部 ID -> 全局 ID
	std::vector<int> V_global;  // V 局部 ID -> 全局 ID
	std::vector<int> adjU_offsets; // CSR: size nU+1
	std::vector<int> adjU_data;    // CSR data: V ids
	std::vector<int> adjV_offsets; // CSR: size nV+1
	std::vector<int> adjV_data;    // CSR data: U ids
};

// 由晶格态构建二分图：
// 1) 按 index_parity 分到 U/V；
// 2) 构建 U->V 的 CSR；
// 3) 由 U->V 反填 V->U 的 CSR。
void build_bipartite_graph(
	const std::vector<uint8_t>& lattice,
	const std::vector<int>* active_sites,
	const Params& p,
	long long total_nodes,
	BipartiteGraph& g)
{
	g.nU = 0;
	g.nV = 0;
	g.U_global.clear();
	g.V_global.clear();
	g.adjU_offsets.clear();
	g.adjU_data.clear();
	g.adjV_offsets.clear();
	g.adjV_data.clear();
	ThreadLocalWorkspace& ws = tls_ws;

	const size_t total_nodes_sz = static_cast<size_t>(std::max<long long>(0, total_nodes));
	// 时间戳映射：用数组 + stamp 代替哈希表，避免构图热点中的动态分配。
	auto prepare_stamp_map = [&](std::vector<int>& stamp_vec, std::vector<int>& local_vec, int& cur_stamp) {
		if (stamp_vec.size() < total_nodes_sz) {
			stamp_vec.resize(total_nodes_sz, 0);
			local_vec.resize(total_nodes_sz, 0);
		}
		if (cur_stamp == std::numeric_limits<int>::max()) {
			std::fill(stamp_vec.begin(), stamp_vec.end(), 0);
			cur_stamp = 1;
		} else {
			++cur_stamp;
		}
	};
	prepare_stamp_map(ws.gtu_stamp, ws.gtu_local, ws.gtu_cur_stamp);
	prepare_stamp_map(ws.gtv_stamp, ws.gtv_local, ws.gtv_cur_stamp);
	const int gtu_stamp = ws.gtu_cur_stamp;
	const int gtv_stamp = ws.gtv_cur_stamp;

	// 首次遍历：只保留 lattice==1 的点，并按二分性写入 U/V 侧连续局部编号。
	if (active_sites != nullptr) {
		g.U_global.reserve(active_sites->size());
		g.V_global.reserve(active_sites->size());
		for (int i : *active_sites) {
			if (index_parity(i, p) == 0) {
				ws.gtu_stamp[i] = gtu_stamp;
				ws.gtu_local[i] = g.nU;
				g.U_global.push_back(i);
				g.nU++;
			} else {
				ws.gtv_stamp[i] = gtv_stamp;
				ws.gtv_local[i] = g.nV;
				g.V_global.push_back(i);
				g.nV++;
			}
		}
	} else {
		for (int i = 0; i < total_nodes; ++i) {
			if (lattice[i] != 1) continue;
			if (index_parity(i, p) == 0) {
				ws.gtu_stamp[i] = gtu_stamp;
				ws.gtu_local[i] = g.nU;
				g.U_global.push_back(i);
				g.nU++;
			} else {
				ws.gtv_stamp[i] = gtv_stamp;
				ws.gtv_local[i] = g.nV;
				g.V_global.push_back(i);
				g.nV++;
			}
		}
	}

	g.adjU_offsets.assign(g.nU + 1, 0);
	g.adjU_data.clear();
	g.adjV_offsets.assign(g.nV + 1, 0);
	g.adjV_data.clear();

	auto& mark_v = ws.bip_mark_v;
	auto& degV = ws.bip_deg_v;
	auto& uniq_v = ws.bip_uniq_v;
	auto& curV = ws.bip_cur_v;
	if (static_cast<int>(mark_v.size()) < g.nV) mark_v.resize(g.nV, 0);
	degV.assign(g.nV, 0);
	uniq_v.clear();
	uniq_v.reserve(2 * d);
	int stamp = ws.bip_mark_v_stamp;

	// 第二次遍历：构建 U->V 邻接（CSR），并在每个 u 内去重以处理边界重合邻居。
	for (int u = 0; u < g.nU; ++u) {
		g.adjU_offsets[u] = static_cast<int>(g.adjU_data.size());
		int ug = g.U_global[u];
		auto neighbors = get_neighbors(ug, p);
		uniq_v.clear();

		if (stamp == std::numeric_limits<int>::max()) {
			std::fill(mark_v.begin(), mark_v.end(), 0);
			stamp = 1;
		}
		++stamp;

		for (int vg : neighbors) {
			if (vg < 0 || vg >= total_nodes || lattice[vg] != 1) continue;
			int v_local = (ws.gtv_stamp[vg] == gtv_stamp) ? ws.gtv_local[vg] : -1;
			if (v_local != -1 && mark_v[v_local] != stamp) {
				mark_v[v_local] = stamp;
				uniq_v.push_back(v_local);
			}
		}

		std::sort(uniq_v.begin(), uniq_v.end());
		for (int v_local : uniq_v) {
			g.adjU_data.push_back(v_local);
			degV[v_local]++;
		}
	}
	ws.bip_mark_v_stamp = stamp;
	g.adjU_offsets[g.nU] = static_cast<int>(g.adjU_data.size());

	// 第三次遍历：由 U->V 反向回填得到 V->U（CSR），便于后续双向传播。
	for (int v = 0; v < g.nV; ++v) {
		g.adjV_offsets[v + 1] = g.adjV_offsets[v] + degV[v];
	}
	g.adjV_data.assign(g.adjV_offsets[g.nV], -1);
	curV = g.adjV_offsets;

	for (int u = 0; u < g.nU; ++u) {
		for (int ei = g.adjU_offsets[u]; ei < g.adjU_offsets[u + 1]; ++ei) {
			int v = g.adjU_data[ei];
			g.adjV_data[curV[v]++] = u;
		}
	}
}

// ========================== 模块三：Hopcroft-Karp 最大匹配 ==========================

// Hopcroft-Karp 最大匹配：
// BFS 分层找“最短增广路层级”，DFS 在层级图上并行找可增广路。
struct HopcroftKarpSolver {
	static constexpr int INF = std::numeric_limits<int>::max();
	int nU_hk, nV_hk;

	// BFS：从所有未匹配 U 出发，构建交替路层级。
	bool bfs(
		const std::vector<int>& adjU_offsets,
		const std::vector<int>& adjU_data,
		const std::vector<int>& free_u,
		std::vector<int>& match_U,
		std::vector<int>& match_V,
		std::vector<int>& dist,
		std::vector<int>& q) {
		(void)match_U;
#if MVC_HK_LAZY_DIST_RESET
		ThreadLocalWorkspace& ws = tls_ws;
		auto& dist_stamp = ws.hk_dist_stamp;
		if (ws.hk_dist_phase == std::numeric_limits<int>::max()) {
			std::fill(dist_stamp.begin(), dist_stamp.end(), 0);
			ws.hk_dist_phase = 1;
		} else {
			++ws.hk_dist_phase;
		}
		const int dist_phase = ws.hk_dist_phase;
		auto get_dist = [&](int node_u) -> int {
			return (dist_stamp[node_u] == dist_phase) ? dist[node_u] : INF;
		};
		auto set_dist = [&](int node_u, int value) {
			dist_stamp[node_u] = dist_phase;
			dist[node_u] = value;
		};
#else
		std::fill(dist.begin(), dist.end(), INF);
		auto get_dist = [&](int node_u) -> int {
			return dist[node_u];
		};
		auto set_dist = [&](int node_u, int value) {
			dist[node_u] = value;
		};
#endif
#if MVC_HK_QUEUE_OPTIMIZED
		q.resize(static_cast<size_t>(nU_hk));
		int tail = 0;
#else
		q.clear();
#endif
		for (int u : free_u) {
#if MVC_HK_QUEUE_OPTIMIZED
			q[tail++] = u;
#else
			q.push_back(u);
#endif
			set_dist(u, 0);
		}
		int head = 0;
		bool found = false;
		int dist_nil = INF;
#if MVC_HK_QUEUE_OPTIMIZED
		while (head < tail) {
			int u = q[head++];
#if MVC_HK_SHORTEST_LAYER_CUTOFF
			if (get_dist(u) >= dist_nil) continue;
#endif
			const int begin = adjU_offsets[u];
			const int end = adjU_offsets[u + 1];
			const int next_dist = get_dist(u) + 1;
			for (int ei = begin; ei < end; ++ei) {
				const int v = adjU_data[ei];
				const int matched_u = match_V[v];
				if (matched_u == -1) {
					// 触达自由 V，说明本轮层级图存在最短增广路。
					found = true;
#if MVC_HK_SHORTEST_LAYER_CUTOFF
					if (next_dist < dist_nil) dist_nil = next_dist;
#endif
				} else if (get_dist(matched_u) == INF) {
					// 只沿匹配边向下一层 U 扩展，保证 DFS 时遵循最短层级。
					set_dist(matched_u, next_dist);
					q[tail++] = matched_u;
				}
			}
		}
#else
		while (head < static_cast<int>(q.size())) {
			int u = q[head++];
#if MVC_HK_SHORTEST_LAYER_CUTOFF
			if (get_dist(u) >= dist_nil) continue;
#endif
			const int next_dist = get_dist(u) + 1;
			for (int ei = adjU_offsets[u]; ei < adjU_offsets[u + 1]; ++ei) {
				int v = adjU_data[ei];
				if (match_V[v] == -1) {
					// 触达自由 V，说明本轮层级图存在最短增广路。
					found = true;
#if MVC_HK_SHORTEST_LAYER_CUTOFF
					if (next_dist < dist_nil) dist_nil = next_dist;
#endif
				}
				if (match_V[v] == -1 || get_dist(match_V[v]) == INF) {
					if (match_V[v] != -1) {
						// 只沿匹配边向下一层 U 扩展，保证 DFS 时遵循最短层级。
						set_dist(match_V[v], next_dist);
						q.push_back(match_V[v]);
					}
				}
			}
		}
#endif
		return found;
	}

	// DFS：沿层级图寻找并落实一条增广路。
	bool dfs(
		int u,
		const std::vector<int>& adjU_offsets,
		const std::vector<int>& adjU_data,
		std::vector<int>& match_U,
		std::vector<int>& match_V,
		std::vector<int>& dist,
		std::vector<int>& cur) {
		ThreadLocalWorkspace& ws = tls_ws;
#if MVC_HK_LAZY_CUR_INIT
		auto& cur_stamp = ws.hk_cur_stamp;
		const int cur_phase = ws.hk_cur_phase;
		auto ensure_cur_initialized = [&](int node_u) -> int& {
			if (cur_stamp[node_u] != cur_phase) {
				cur_stamp[node_u] = cur_phase;
				cur[node_u] = adjU_offsets[node_u];
			}
			return cur[node_u];
		};
#endif
#if MVC_HK_LAZY_DIST_RESET
		auto& dist_stamp = ws.hk_dist_stamp;
		const int dist_phase = ws.hk_dist_phase;
		auto get_dist = [&](int node_u) -> int {
			return (dist_stamp[node_u] == dist_phase) ? dist[node_u] : INF;
		};
		auto set_dist = [&](int node_u, int value) {
			dist_stamp[node_u] = dist_phase;
			dist[node_u] = value;
		};
#else
		auto get_dist = [&](int node_u) -> int {
			return dist[node_u];
		};
		auto set_dist = [&](int node_u, int value) {
			dist[node_u] = value;
		};
#endif
#if MVC_HK_ITERATIVE_DFS
		auto& stack_u = ws.hk_stack_u;
		auto& stack_v = ws.hk_stack_v;
		stack_u.clear();
		stack_v.clear();

		int current_u = u;
		while (true) {
			const int next_dist = get_dist(current_u) + 1;
#if MVC_HK_LAZY_CUR_INIT
			int& ei = ensure_cur_initialized(current_u);
#else
			int& ei = cur[current_u];
#endif
			const int end = adjU_offsets[current_u + 1];
			bool advanced = false;

			while (ei < end) {
				const int v = adjU_data[ei];
				const int matched_u = match_V[v];
				if (matched_u == -1) {
					match_V[v] = current_u;
					match_U[current_u] = v;
					while (!stack_u.empty()) {
						const int parent_u = stack_u.back();
						stack_u.pop_back();
						const int parent_v = stack_v.back();
						stack_v.pop_back();
						match_V[parent_v] = parent_u;
						match_U[parent_u] = parent_v;
					}
					return true;
				}

				++ei; // 该边已尝试；若子递归失败，回到父节点时继续下一条边。
				if (get_dist(matched_u) == next_dist) {
					stack_u.push_back(current_u);
					stack_v.push_back(v);
					current_u = matched_u;
					advanced = true;
					break;
				}
			}

			if (advanced) continue;

			set_dist(current_u, INF);
			if (stack_u.empty()) return false;

			current_u = stack_u.back();
			stack_u.pop_back();
			stack_v.pop_back();
		}
#else
#if MVC_HK_QUEUE_OPTIMIZED
		const int next_dist = get_dist(u) + 1;
		#if MVC_HK_LAZY_CUR_INIT
		for (int& ei = ensure_cur_initialized(u); ei < adjU_offsets[u + 1]; ++ei) {
		#else
		for (int& ei = cur[u]; ei < adjU_offsets[u + 1]; ++ei) {
		#endif
			const int v = adjU_data[ei];
			const int matched_u = match_V[v];
			if (matched_u == -1 ||
				(get_dist(matched_u) == next_dist &&
					dfs(matched_u, adjU_offsets, adjU_data, match_U, match_V, dist, cur))) {
				// 回溯翻转交替路上的匹配关系。
				match_V[v] = u;
				match_U[u] = v;
				return true;
			}
		}
#else
		#if MVC_HK_LAZY_CUR_INIT
		for (int& ei = ensure_cur_initialized(u); ei < adjU_offsets[u + 1]; ++ei) {
		#else
		for (int& ei = cur[u]; ei < adjU_offsets[u + 1]; ++ei) {
		#endif
			int v = adjU_data[ei];
			if (match_V[v] == -1 ||
				(get_dist(match_V[v]) == get_dist(u) + 1 &&
					dfs(match_V[v], adjU_offsets, adjU_data, match_U, match_V, dist, cur))) {
				// 回溯翻转交替路上的匹配关系。
				match_V[v] = u;
				match_U[u] = v;
				return true;
			}
		}
#endif
		set_dist(u, INF);
		return false;
#endif
	}

	// run：可选 O(E) 贪心预匹配 + HK 主循环，返回最大匹配大小。
	int run(int num_U, int num_V, const std::vector<int>& adjU_offsets, const std::vector<int>& adjU_data, bool use_greedy_init = true) {
		nU_hk = num_U;
		nV_hk = num_V;
		ThreadLocalWorkspace& ws = tls_ws;
		auto& match_U = ws.hk_match_u;
		auto& match_V = ws.hk_match_v;
		auto& dist = ws.hk_dist;
		auto& dist_stamp = ws.hk_dist_stamp;
		auto& cur = ws.hk_cur;
		auto& q = ws.hk_queue;
		auto& free_u = ws.hk_free_u;
		auto& cur_stamp = ws.hk_cur_stamp;

		match_U.assign(nU_hk, -1);
		match_V.assign(nV_hk, -1);
#if MVC_HK_QUEUE_OPTIMIZED
		cur.resize(nU_hk);
#else
		cur.assign(nU_hk, 0);
#endif
		if (static_cast<int>(cur_stamp.size()) < nU_hk) cur_stamp.resize(nU_hk, 0);
		if (nU_hk > 0) {
#if MVC_HK_LAZY_DIST_RESET
			dist.resize(nU_hk);
			if (static_cast<int>(dist_stamp.size()) < nU_hk) dist_stamp.resize(nU_hk, 0);
#else
			dist.assign(nU_hk, INF);
#endif
		} else {
			dist.clear();
		}

		int matching_size = 0;

		if (use_greedy_init) {
			// O(E) 贪心初始化：先消除大量长度为 1 的增广路，减少后续 HK 迭代开销。
			for (int u = 0; u < nU_hk; ++u) {
				for (int ei = adjU_offsets[u]; ei < adjU_offsets[u + 1]; ++ei) {
					int v = adjU_data[ei];
					if (match_V[v] == -1) {
						match_U[u] = v;
						match_V[v] = u;
						matching_size++;
						break;
					}
				}
			}
		}

		free_u.clear();
		free_u.reserve(static_cast<size_t>(nU_hk));
		for (int u = 0; u < nU_hk; ++u) {
			if (match_U[u] == -1) free_u.push_back(u);
		}

		while (!free_u.empty() && bfs(adjU_offsets, adjU_data, free_u, match_U, match_V, dist, q)) {
#if MVC_HK_LAZY_CUR_INIT
			if (ws.hk_cur_phase == std::numeric_limits<int>::max()) {
				std::fill(cur_stamp.begin(), cur_stamp.end(), 0);
				ws.hk_cur_phase = 1;
			} else {
				++ws.hk_cur_phase;
			}
#else
#if MVC_HK_QUEUE_OPTIMIZED
			std::copy(adjU_offsets.begin(), adjU_offsets.begin() + nU_hk, cur.begin());
#else
			cur = adjU_offsets;
#endif
#endif
			size_t write_pos = 0;
			for (int u : free_u) {
#if MVC_HK_SKIP_UNREACHABLE_ROOTS
				bool root_reachable;
				#if MVC_HK_LAZY_DIST_RESET
				root_reachable = (dist_stamp[u] == ws.hk_dist_phase && dist[u] != INF);
				#else
				root_reachable = (dist[u] != INF);
				#endif
				if (root_reachable && dfs(u, adjU_offsets, adjU_data, match_U, match_V, dist, cur)) {
#else
				if (dfs(u, adjU_offsets, adjU_data, match_U, match_V, dist, cur)) {
#endif
					matching_size++;
				} else {
					free_u[write_pos++] = u;
				}
			}
			free_u.resize(write_pos);
		}
		return matching_size;
	}
};

// ========================== 模块四：冻结传播 ==========================

enum NodeState {
	UNASSIGNED = -1,
	NOT_IN_MVC = 0,
	IN_MVC = 1
};

// 冻结传播结果：
// - stateU/stateV：三值状态（在 MVC / 不在 MVC / 未定）；
// - core_edges：仍未冻结的“核心匹配边”，后续交由 DAG 采样处理。
struct FreezingResult {
	std::vector<int> stateU;
	std::vector<int> stateV;
	std::vector<int> fixed_mvc_globals;  // 被冻结为 IN_MVC 的全局 ID
	std::vector<std::pair<int, int>> core_edges; // 未冻结核匹配边 (u_local, v_local)
	std::vector<int> core_edge_of_u; // U局部ID -> 所属核边ID（-1表示不属于核）
	std::vector<int> core_edge_of_v; // V局部ID -> 所属核边ID
};

static thread_local FreezingResult tls_freezing_result;
static thread_local std::vector<int> tls_freezing_queue;

// 冻结传播（基于匹配与最小覆盖互补约束）：
// 从未匹配点出发在交替结构上做 BFS，把可确定的节点状态尽量冻结。
FreezingResult& run_freezing_bfs(
	const BipartiteGraph& g,
	const std::vector<int>& match_U,
	const std::vector<int>& match_V)
{
	FreezingResult& result = tls_freezing_result;
	result.stateU.assign(g.nU, UNASSIGNED);
	result.stateV.assign(g.nV, UNASSIGNED);
	result.core_edge_of_u.assign(g.nU, -1);
	result.core_edge_of_v.assign(g.nV, -1);
	result.fixed_mvc_globals.clear();
	result.core_edges.clear();

	// BFS 队列：位打包编码，最高位表示 side(1=V,0=U)，其余位存 local_id
	static constexpr uint32_t SIDE_MASK = 0x80000000u;
	std::vector<int>& q = tls_freezing_queue;
	q.clear();
	q.reserve(static_cast<size_t>(g.nU + g.nV));
	int q_head = 0;

	auto assign_u = [&](int u, int val) {
		if (result.stateU[u] == UNASSIGNED) {
			result.stateU[u] = val;
			q.push_back(u);
		}
	};
	auto assign_v = [&](int v, int val) {
		if (result.stateV[v] == UNASSIGNED) {
			result.stateV[v] = val;
			// V 侧节点入队时打 side 标记，避免维护两个独立队列。
			q.push_back(static_cast<int>(static_cast<uint32_t>(v) | SIDE_MASK));
		}
	};

	// 初始化：未匹配节点设为 NOT_IN_MVC
	for (int u = 0; u < g.nU; ++u) {
		if (match_U[u] == -1) assign_u(u, NOT_IN_MVC);
	}
	for (int v = 0; v < g.nV; ++v) {
		if (match_V[v] == -1) assign_v(v, NOT_IN_MVC);
	}

	// BFS 传播
	while (q_head < static_cast<int>(q.size())) {
		uint32_t packed = static_cast<uint32_t>(q[q_head++]);
		// 反解码：
		// - is_v_side 判断属于 U 还是 V；
		// - id 取回局部索引。
		bool is_v_side = (packed & SIDE_MASK) != 0u;
		int id = static_cast<int>(packed & ~SIDE_MASK);
		if (!is_v_side) { // U 节点
			int st = result.stateU[id];
			if (st == NOT_IN_MVC) {
				// U不在MVC => 所有邻居V必须在MVC
				for (int ei = g.adjU_offsets[id]; ei < g.adjU_offsets[id + 1]; ++ei) {
					assign_v(g.adjU_data[ei], IN_MVC);
				}
			} else if (st == IN_MVC) {
				// U在MVC => 其匹配的V可以不在MVC
				int mv = match_U[id];
				if (mv != -1) assign_v(mv, NOT_IN_MVC);
			}
		} else { // V 节点
			int st = result.stateV[id];
			if (st == NOT_IN_MVC) {
				for (int ei = g.adjV_offsets[id]; ei < g.adjV_offsets[id + 1]; ++ei) {
					assign_u(g.adjV_data[ei], IN_MVC);
				}
			} else if (st == IN_MVC) {
				int mu = match_V[id];
				if (mu != -1) assign_u(mu, NOT_IN_MVC);
			}
		}
	}

	// 收集冻结的 MVC 节点
	for (int u = 0; u < g.nU; ++u) {
		if (result.stateU[u] == IN_MVC) result.fixed_mvc_globals.push_back(g.U_global[u]);
	}
	for (int v = 0; v < g.nV; ++v) {
		if (result.stateV[v] == IN_MVC) result.fixed_mvc_globals.push_back(g.V_global[v]);
	}

	// 收集未冻结核匹配边
	for (int u = 0; u < g.nU; ++u) {
		int v = match_U[u];
		if (v == -1) continue;
		// 仅把“U/V 两端都未定”的匹配边保留为核心自由边，交给后续 DAG 采样决定取向。
		if (result.stateU[u] == UNASSIGNED && result.stateV[v] == UNASSIGNED && result.core_edge_of_u[u] == -1 && result.core_edge_of_u[u] == -1 && result.core_edge_of_v[v] == -1) {
			int eid = static_cast<int>(result.core_edges.size());
			result.core_edges.push_back({u, v});
			result.core_edge_of_u[u] = eid;
			result.core_edge_of_v[v] = eid;
		}
	}

	return result;
}

// ========================== 模块五：依赖图构建与 SCC 缩点 ==========================

// 依赖图缩点结果：
// 核边 -> SCC 超点 -> DAG，后续 MCMC 只在超点层面进行。
struct SccDagResult {
	int nSuper = 0;
	std::vector<int> edge_to_super;          // 核边 ID -> 超级节点 ID
	std::vector<int> super_to_edges_offsets;  // CSR: size nSuper+1
	std::vector<int> super_to_edges_data;     // 所含核边 ID 列表
	std::vector<int> super_weights;           // 每个超级节点包含的核边数
	std::vector<int> dag_offsets;             // CSR: size nSuper+1
	std::vector<int> dag_data;                // DAG 邻接表（子节点）
	std::vector<int> dag_pred_offsets;        // CSR: size nSuper+1
	std::vector<int> dag_pred_data;           // DAG 逆邻接表（父节点）
};

// 构建“核边依赖图”并做 Tarjan SCC 缩点：
// - 原图节点是核心匹配边；
// - 有向边表示“若一边取 1 则另一边受约束”；
// - 缩点后得到 DAG，便于分层 Gibbs 更新。
void build_dependency_scc_dag(
	const BipartiteGraph& g,
	const std::vector<int>& match_U,
	const FreezingResult& fr,
	SccDagResult& out)
{
	out.nSuper = 0;
	out.edge_to_super.clear();
	out.super_to_edges_offsets.clear();
	out.super_to_edges_data.clear();
	out.super_weights.clear();
	out.dag_offsets.clear();
	out.dag_data.clear();
	out.dag_pred_offsets.clear();
	out.dag_pred_data.clear();

	int c = static_cast<int>(fr.core_edges.size());
	ThreadLocalWorkspace& ws = tls_ws;
	auto& dep_offsets = ws.dep_offsets;
	auto& dep_data = ws.dep_data;
	auto& super_deg = ws.super_deg;
	auto& pred_deg = ws.pred_deg;
	auto& dfn = ws.tarjan_dfn;
	auto& low = ws.tarjan_low;
	auto& in_stack = ws.tarjan_in_stack;
	auto& scc_id = ws.tarjan_scc_id;
	auto& stk = ws.tarjan_stk;
	auto& parent = ws.tarjan_parent;
	auto& explicit_u = ws.tarjan_explicit_u;
	auto& explicit_next_ei = ws.tarjan_explicit_next_ei;
	auto& super_cur = ws.dag_super_cur;
	auto& pred_cur = ws.dag_pred_cur;

	// 步骤1：构建原始有向图 D（CSR）
	if (static_cast<int>(ws.mark_a.size()) < c) ws.mark_a.assign(c, 0);
	dep_offsets.assign(c + 1, 0);
	dep_data.clear();
	// dep_data.reserve(static_cast<size_t>(c) * 2u);
	dep_data.reserve(static_cast<size_t>(c) * static_cast<size_t>(2 * d));
	for (int eid = 0; eid < c; ++eid) {
		int u = fr.core_edges[eid].first;
		int dep_stamp = next_mark_stamp(ws.mark_a, ws.mark_a_stamp);
		for (int ei = g.adjU_offsets[u]; ei < g.adjU_offsets[u + 1]; ++ei) {
			int v = g.adjU_data[ei];
			if (fr.core_edge_of_v[v] == -1) continue;
			int eid2 = fr.core_edge_of_v[v];
			if (match_U[u] == v) continue; // 排除自身匹配边
			if (eid != eid2 && ws.mark_a[eid2] != dep_stamp) {
				// 依赖语义：当前核边的取值会约束通过同一 V 侧连接到的另一条核边。
				ws.mark_a[eid2] = dep_stamp;
				dep_data.push_back(eid2);
			}
		}
		dep_offsets[eid + 1] = static_cast<int>(dep_data.size());
	}

	// 步骤2：Tarjan SCC（这里用“显式栈”模拟递归，避免深递归爆栈）
	dfn.assign(c, -1);
	low.assign(c, -1);
	in_stack.assign(c, 0);
	scc_id.assign(c, -1);
	stk.clear();
	stk.reserve(c);
	int timer = 0, scc_cnt = 0;

	parent.assign(c, -1);
	explicit_u.clear();
	explicit_next_ei.clear();
	explicit_u.reserve(c);
	explicit_next_ei.reserve(c);

	for (int start = 0; start < c; ++start) {
		if (dfn[start] != -1) continue;

		parent[start] = -1;
		dfn[start] = low[start] = timer++;
		stk.push_back(start);
		in_stack[start] = 1;
		explicit_u.push_back(start);
		explicit_next_ei.push_back(dep_offsets[start]);

		while (!explicit_u.empty()) {
			int u = explicit_u.back();
			int& next_ei = explicit_next_ei.back();

			if (next_ei < dep_offsets[u + 1]) {
				int v = dep_data[next_ei++];
				if (dfn[v] == -1) {
					// 相当于递归调用 tarjan(v)：先记录父子关系，再入显式调用栈。
					parent[v] = u;
					dfn[v] = low[v] = timer++;
					stk.push_back(v);
					in_stack[v] = 1;
					explicit_u.push_back(v);
					explicit_next_ei.push_back(dep_offsets[v]);
				} else if (in_stack[v]) {
					// 返祖边：用 dfn[v] 更新 low[u]。
					low[u] = std::min(low[u], dfn[v]);
				}
				continue;
			}

			if (low[u] == dfn[u]) {
				// u 是 SCC 根：从 Tarjan 栈顶持续弹出，直到回到 u。
				while (true) {
					int x = stk.back();
					stk.pop_back();
					in_stack[x] = 0;
					scc_id[x] = scc_cnt;
					if (x == u) break;
				}
				scc_cnt++;
			}

			explicit_u.pop_back();
			explicit_next_ei.pop_back();
			int pu = parent[u];
			if (pu != -1) {
				low[pu] = std::min(low[pu], low[u]);
			}
		}
	}

	// Tarjan 生成的 SCC 编号天然是逆拓扑序：这里原地翻转为拓扑序。
	for (int i = 0; i < c; ++i) {
		scc_id[i] = (scc_cnt - 1) - scc_id[i];
	}

	// 步骤3：图缩点（构建 DAG）
	out.nSuper = scc_cnt;
	out.edge_to_super = scc_id;
	super_deg.assign(scc_cnt, 0);
	for (int i = 0; i < c; ++i) super_deg[scc_id[i]]++;
	out.super_weights = super_deg;
	out.super_to_edges_offsets.assign(scc_cnt + 1, 0);
	for (int s = 0; s < scc_cnt; ++s) {
		out.super_to_edges_offsets[s + 1] = out.super_to_edges_offsets[s] + super_deg[s];
	}
	out.super_to_edges_data.assign(c, -1);
	super_cur = out.super_to_edges_offsets;
	for (int i = 0; i < c; ++i) {
		int s = scc_id[i];
		out.super_to_edges_data[super_cur[s]++] = i;
	}

	if (static_cast<int>(ws.mark_b.size()) < scc_cnt) ws.mark_b.assign(scc_cnt, 0);
	ws.temp_ids.clear();
	out.dag_offsets.assign(scc_cnt + 1, 0);
	out.dag_data.clear();
	out.dag_data.reserve(dep_data.size());
	pred_deg.assign(scc_cnt, 0);
	for (int s = 0; s < scc_cnt; ++s) {
		int super_stamp = next_mark_stamp(ws.mark_b, ws.mark_b_stamp);
		ws.temp_ids.clear();
		for (int ei = out.super_to_edges_offsets[s]; ei < out.super_to_edges_offsets[s + 1]; ++ei) {
			int e = out.super_to_edges_data[ei];
			for (int di = dep_offsets[e]; di < dep_offsets[e + 1]; ++di) {
				int v = dep_data[di];
				int sv = scc_id[v];
				if (sv == s) continue;
				if (ws.mark_b[sv] != super_stamp) {
					ws.mark_b[sv] = super_stamp;
					ws.temp_ids.push_back(sv);
				}
			}
		}
		out.dag_offsets[s] = static_cast<int>(out.dag_data.size());
		for (int t : ws.temp_ids) {
			out.dag_data.push_back(t);
			pred_deg[t]++;
		}
		out.dag_offsets[s + 1] = static_cast<int>(out.dag_data.size());
	}

	out.dag_pred_offsets.assign(scc_cnt + 1, 0);
	for (int s = 0; s < scc_cnt; ++s) {
		out.dag_pred_offsets[s + 1] = out.dag_pred_offsets[s] + pred_deg[s];
	}
	out.dag_pred_data.assign(out.dag_pred_offsets[scc_cnt], -1);
	pred_cur = out.dag_pred_offsets;
	for (int s = 0; s < scc_cnt; ++s) {
		for (int ei = out.dag_offsets[s]; ei < out.dag_offsets[s + 1]; ++ei) {
			int t = out.dag_data[ei];
			out.dag_pred_data[pred_cur[t]++] = s;
		}
	}
}

// ========================== 模块六：DAG 分解与 Component-wise MCMC 均匀采样 ==========================

// 单链采样返回值（一个热样本或一个 burn-in 过程）。
struct McmcResult {
	std::vector<int8_t> super_state; // 每个超级节点的状态（0 或 1）
	double mean_sweeps_used = 0.0; // 该样本中各 WCC 实际执行 sweeps 的平均值
	int max_wcc_size = 0;          // 该样本中最大 WCC 的节点数
	int shared_sweeps = 0;         // 该样本中复用的全局 sweeps（自动模式下由最大 WCC 估计）
	double estimated_tau_int = 0.0; // 自动模式下估计到的 tau_int（若使用缓存则为缓存值）
	bool acf_probe_too_short = false;
};

// MSC64 批量采样返回值：用 64 位位平面同时表示 64 条链。
struct McmcBatch64Result {
	std::vector<uint64_t> super_state64; // 每个超级节点的 64 链打包状态
	double mean_sweeps_used = 0.0;
	int max_wcc_size = 0;
	int shared_sweeps = 0;
	double estimated_tau_int = 0.0;
	bool acf_probe_too_short = false;
};

// 位计数：MSC64 中用于快速统计 64 条链的 1 个数。
inline int popcount_u64(uint64_t x) {
	return __builtin_popcountll(static_cast<unsigned long long>(x));
}

enum class ExactSamplingMode : uint8_t {
	Mcmc = 0,
	Materialized = 1,
	FrontierDp = 2
};

// 每个 DAG 连通分量的预计算采样计划（偏移量形式，减少重复构造）。
struct McmcComponentPlan {
	int size = 0;
	int node_begin = 0;
	int node_end = 0;
	ExactSamplingMode exact_mode = ExactSamplingMode::Mcmc;
	int valid_states_begin = 0;
	int valid_states_end = 0;
	int topo_begin = 0;
	int topo_end = 0;
	int level_offsets_begin = 0;
	int level_offsets_end = 0;
	int level_data_begin = 0;
	int level_data_end = 0;
	int frontier_step_begin = 0;
	int frontier_step_end = 0;
	int frontier_layer_begin = 0;
	int frontier_layer_end = 0;
};

// DAG 预计算缓存：
// 预先准备每个分量的节点序、拓扑序、分层序，以及两类 exact 采样所需缓存。
struct McmcDagPrecomputed {
	int nSuper = 0;
	int max_wcc_size = 0;
	std::vector<McmcComponentPlan> components; // 已按规模降序排列
	std::vector<int> component_nodes;          // 各分量节点（BFS 顺序）
	std::vector<uint32_t> exact_states_data;   // 小分量合法状态集合（按分量展平存储）
	std::vector<int> topo_data;                // 各分量拓扑序
	std::vector<int> level_offsets_data;       // 每个分量局部 level offsets（起点为0）
	std::vector<int> level_data;               // 每个分量分层节点序列
	std::vector<int> frontier_parent_masks_data; // 每一步在 frontier 上的父约束掩码
	std::vector<int> frontier_keep_masks_data;   // frontier 压缩到下一层时保留的位掩码
	std::vector<uint8_t> frontier_add_current_data; // 当前节点是否进入下一层 frontier
	std::vector<int> frontier_layer_widths_data; // 每一层 frontier 宽度
	std::vector<int> frontier_count_offsets_data; // 每一层计数表在 counts 中的绝对偏移
	std::vector<int> frontier_base_states_data;   // 每层按 frontier_state 查到的压缩后状态
	std::vector<int> frontier_next_state_one_data; // 每层取 1 时的下一状态
	std::vector<uint64_t> frontier_counts_data;   // frontier-DP exact 的后缀计数表
};

static thread_local BipartiteGraph tls_bipartite_graph;
static thread_local SccDagResult tls_scc_dag;
static thread_local McmcDagPrecomputed tls_mcmc_dag_plan;
static thread_local std::vector<int> tls_precompute_topo_local;
static thread_local std::vector<uint64_t> tls_precompute_parent_mask_local;
static thread_local std::vector<uint32_t> tls_precompute_materialized_states;
static thread_local std::vector<uint32_t> tls_materialize_state_stack;
static thread_local std::vector<uint8_t> tls_materialize_stage_stack;

inline int compress_frontier_state(int state, int keep_mask) {
	int out = 0;
	int out_bit = 0;
	while (keep_mask != 0) {
		int lsb = keep_mask & -keep_mask;
		if (state & lsb) out |= (1 << out_bit);
		keep_mask ^= lsb;
		++out_bit;
	}
	return out;
}

bool try_materialize_exact_states_from_topo(
	const std::vector<int>& topo_local,
	const std::vector<uint64_t>& parent_mask_local,
	int valid_state_cap,
	int dfs_node_cap,
	std::vector<uint32_t>& out_states)
{
	out_states.clear();
	if (topo_local.empty()) {
		out_states.push_back(0u);
		return true;
	}

	const int K = static_cast<int>(topo_local.size());
	long long dfs_nodes = 0;
	auto& state_stack = tls_materialize_state_stack;
	auto& stage_stack = tls_materialize_stage_stack;
	state_stack.resize(static_cast<size_t>(K) + 1u);
	stage_stack.resize(static_cast<size_t>(K) + 1u);

	int pos = 0;
	state_stack[0] = 0u;
	stage_stack[0] = 0u;

	while (pos >= 0) {
		uint8_t& stage = stage_stack[static_cast<size_t>(pos)];
		if (stage == 0u) {
			if (++dfs_nodes > static_cast<long long>(dfs_node_cap)) {
				out_states.clear();
				return false;
			}
			if (pos == K) {
				out_states.push_back(state_stack[static_cast<size_t>(pos)]);
				if (static_cast<int>(out_states.size()) > valid_state_cap) {
					out_states.clear();
					return false;
				}
				--pos;
				continue;
			}

			const int li = topo_local[static_cast<size_t>(pos)];
			stage = 1u;
			state_stack[static_cast<size_t>(pos) + 1u] =
				state_stack[static_cast<size_t>(pos)] & ~(1u << li);
			stage_stack[static_cast<size_t>(pos) + 1u] = 0u;
			++pos;
			continue;
		}

		if (stage == 1u) {
			stage = 2u;
			const int li = topo_local[static_cast<size_t>(pos)];
			const uint32_t state_mask = state_stack[static_cast<size_t>(pos)];
			if ((static_cast<uint64_t>(state_mask) & parent_mask_local[li]) == parent_mask_local[li]) {
				state_stack[static_cast<size_t>(pos) + 1u] = state_mask | (1u << li);
				stage_stack[static_cast<size_t>(pos) + 1u] = 0u;
				++pos;
				continue;
			}
		}

		--pos;
	}
	return true;
}

struct FrontierDpExactPlan {
	std::vector<int> step_parent_masks;
	std::vector<int> step_keep_masks;
	std::vector<uint8_t> step_add_current;
	std::vector<int> layer_widths;
	std::vector<int> count_offsets;
	std::vector<int> base_states;
	std::vector<int> next_state_one;
	std::vector<uint64_t> counts;
	int max_frontier_width = 0;
};

static thread_local FrontierDpExactPlan tls_frontier_dp_exact_plan;
static thread_local std::vector<int> tls_frontier_topo_pos;
static thread_local std::vector<int> tls_frontier_last_child_pos;
static thread_local std::vector<int> tls_frontier_nodes_a;
static thread_local std::vector<int> tls_frontier_nodes_b;

bool try_build_frontier_dp_exact_plan(
	const SccDagResult& scc,
	const std::vector<int>& topo_order,
	const std::vector<int>& topo_local,
	const std::vector<uint64_t>& parent_mask_local,
	const std::vector<int>& local_index_of_global,
	int frontier_width_cap,
	int dp_state_cap,
	FrontierDpExactPlan& out)
{
	const int K = static_cast<int>(topo_local.size());
	if (K <= 0) return false;

	auto& topo_pos = tls_frontier_topo_pos;
	topo_pos.resize(K);
	for (int ti = 0; ti < K; ++ti) {
		topo_pos[topo_local[ti]] = ti;
	}

	auto& last_child_pos = tls_frontier_last_child_pos;
	last_child_pos.resize(K);
	std::fill(last_child_pos.begin(), last_child_pos.end(), -1);
	for (int ti = 0; ti < K; ++ti) {
		int u = topo_order[ti];
		int u_local = topo_local[ti];
		for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
			int child = scc.dag_data[ei];
			int child_local = local_index_of_global[child];
			if (child_local < 0) continue;
			last_child_pos[u_local] = std::max(last_child_pos[u_local], topo_pos[child_local]);
		}
	}

	out.step_parent_masks.resize(K);
	out.step_keep_masks.resize(K);
	out.step_add_current.resize(K);
	out.layer_widths.resize(K + 1);
	out.count_offsets.resize(K + 1);
	out.max_frontier_width = 0;

	auto& frontier = tls_frontier_nodes_a;
	auto& next_frontier = tls_frontier_nodes_b;
	frontier.clear();
	next_frontier.clear();
	if (static_cast<int>(frontier.capacity()) < K) frontier.reserve(K);
	if (static_cast<int>(next_frontier.capacity()) < K) next_frontier.reserve(K);
	int total_dp_states = 0;

	for (int ti = 0; ti < K; ++ti) {
		int width = static_cast<int>(frontier.size());
		out.layer_widths[ti] = width;
		out.max_frontier_width = std::max(out.max_frontier_width, width);
		if (width > frontier_width_cap) return false;

		int layer_states = 1 << width;
		total_dp_states += layer_states;
		if (total_dp_states > dp_state_cap) return false;

		int u_local = topo_local[ti];
		int parent_mask = 0;
		for (int fp = 0; fp < width; ++fp) {
			int frontier_local = frontier[fp];
			if (parent_mask_local[u_local] & (1ULL << frontier_local)) {
				parent_mask |= (1 << fp);
			}
		}
		out.step_parent_masks[ti] = parent_mask;

		int keep_mask = 0;
		next_frontier.clear();
		for (int fp = 0; fp < width; ++fp) {
			int frontier_local = frontier[fp];
			if (last_child_pos[frontier_local] > ti) {
				keep_mask |= (1 << fp);
				next_frontier.push_back(frontier_local);
			}
		}
		out.step_keep_masks[ti] = keep_mask;
		out.step_add_current[ti] = (last_child_pos[u_local] > ti) ? 1 : 0;
		if (out.step_add_current[ti] != 0) next_frontier.push_back(u_local);
		frontier.swap(next_frontier);
	}

	out.layer_widths[K] = static_cast<int>(frontier.size());
	out.max_frontier_width = std::max(out.max_frontier_width, out.layer_widths[K]);
	if (out.layer_widths[K] > frontier_width_cap) return false;
	if (!frontier.empty()) return false;

	total_dp_states += (1 << out.layer_widths[K]);
	if (total_dp_states > dp_state_cap) return false;

	int total_count_entries = 0;
	for (int li = 0; li <= K; ++li) {
		out.count_offsets[li] = total_count_entries;
		total_count_entries += (1 << out.layer_widths[li]);
	}
	out.base_states.resize(static_cast<size_t>(total_count_entries));
	out.next_state_one.resize(static_cast<size_t>(total_count_entries));
	out.counts.resize(static_cast<size_t>(total_count_entries));
	out.counts[static_cast<size_t>(out.count_offsets[K])] = 1ULL;

	for (int ti = K - 1; ti >= 0; --ti) {
		const int width = out.layer_widths[ti];
		const int keep_mask = out.step_keep_masks[ti];
		const int kept_width = out.layer_widths[ti + 1] - static_cast<int>(out.step_add_current[ti]);
		const int parent_mask = out.step_parent_masks[ti];
		const uint64_t* next_counts = out.counts.data() + out.count_offsets[ti + 1];
		uint64_t* cur_counts = out.counts.data() + out.count_offsets[ti];
		const int num_states = 1 << width;

		for (int state = 0; state < num_states; ++state) {
			int base_state = compress_frontier_state(state, keep_mask);
			out.base_states[static_cast<size_t>(out.count_offsets[ti] + state)] = base_state;
			uint64_t count0 = next_counts[base_state];
			int next_state_one = base_state;
			if (out.step_add_current[ti] != 0) next_state_one |= (1 << kept_width);
			out.next_state_one[static_cast<size_t>(out.count_offsets[ti] + state)] = next_state_one;
			if ((state & parent_mask) == parent_mask) {
				cur_counts[state] = count0 + next_counts[next_state_one];
			} else {
				cur_counts[state] = count0;
			}
		}
	}

	return out.counts[0] > 0ULL;
}

// 对缩点 DAG 做预处理：
// - 无向连通分量分解并按规模降序；
// - <=18 直接物化合法态；
// - 19~24 优先 DFS materialized exact（带 cap）；
// - 25~63 优先 frontier-DP exact（带 cap）；
// - 其余再计算 level 分层并交给 LSB Gibbs。
void precompute_mcmc_dag_structure(const SccDagResult& scc, McmcDagPrecomputed& out) {
	out.nSuper = scc.nSuper;
	out.max_wcc_size = 0;
	out.components.clear();
	out.component_nodes.clear();
	out.exact_states_data.clear();
	out.topo_data.clear();
	out.level_offsets_data.clear();
	out.level_data.clear();
	out.frontier_parent_masks_data.clear();
	out.frontier_keep_masks_data.clear();
	out.frontier_add_current_data.clear();
	out.frontier_layer_widths_data.clear();
	out.frontier_count_offsets_data.clear();
	out.frontier_base_states_data.clear();
	out.frontier_next_state_one_data.clear();
	out.frontier_counts_data.clear();
	int N = scc.nSuper;
	if (N == 0) return;

	ThreadLocalWorkspace& ws = tls_ws;
	auto& visited = ws.mcmc_visited;
	auto& comp_offsets = ws.mcmc_comp_offsets;
	auto& comp_nodes = ws.mcmc_comp_nodes;
	auto& comp_order = ws.mcmc_comp_order;
	auto& bfs_queue = ws.mcmc_bfs_queue;
	auto& indeg_scratch = ws.mcmc_indeg_scratch;
	auto& topo_order = ws.mcmc_topo_order;
	auto& kahn_q = ws.mcmc_kahn_q;
	auto& level = ws.mcmc_level;
	auto& level_offsets = ws.mcmc_level_offsets;
	auto& level_data = ws.mcmc_level_data;
	auto& level_counts = ws.mcmc_level_counts;
	auto& level_write = ws.mcmc_level_write;
	auto& bp_local_index = ws.mcmc_bp_local_index;

	visited.assign(N, 0);
	comp_offsets.clear();
	comp_nodes.clear();
	comp_order.clear();
	comp_offsets.push_back(0);
	bfs_queue.clear();

	for (int i = 0; i < N; ++i) {
		if (visited[i]) continue;
		bfs_queue.clear();
		bfs_queue.push_back(i);
		visited[i] = 1;
		int head = 0;
		while (head < static_cast<int>(bfs_queue.size())) {
			int u = bfs_queue[head++];
			for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
				int c = scc.dag_data[ei];
				if (!visited[c]) {
					visited[c] = 1;
					bfs_queue.push_back(c);
				}
			}
			// 按无向连通分量划分：不仅走子边，也沿父边回溯，确保组件划分完整。
			for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
				int pr = scc.dag_pred_data[ei];
				if (!visited[pr]) {
					visited[pr] = 1;
					bfs_queue.push_back(pr);
				}
			}
		}
		comp_nodes.insert(comp_nodes.end(), bfs_queue.begin(), bfs_queue.end());
		comp_offsets.push_back(static_cast<int>(comp_nodes.size()));
	}

	int num_components = static_cast<int>(comp_offsets.size()) - 1;
	comp_order.resize(num_components);
	std::iota(comp_order.begin(), comp_order.end(), 0);
	std::sort(comp_order.begin(), comp_order.end(),
		[&](int a, int b) {
			return (comp_offsets[a + 1] - comp_offsets[a]) > (comp_offsets[b + 1] - comp_offsets[b]);
		});
	out.max_wcc_size = (num_components > 0)
		? (comp_offsets[comp_order[0] + 1] - comp_offsets[comp_order[0]])
		: 0;
	out.components.reserve(num_components);

	indeg_scratch.assign(N, 0);
	level.assign(N, 0);

	for (int comp_rank = 0; comp_rank < num_components; ++comp_rank) {
		int cid = comp_order[comp_rank];
		int comp_begin = comp_offsets[cid];
		int comp_end = comp_offsets[cid + 1];
		int K = comp_end - comp_begin;
		if (K <= 0) continue;

		McmcComponentPlan cp;
		cp.size = K;
		cp.node_begin = static_cast<int>(out.component_nodes.size());
		out.component_nodes.insert(out.component_nodes.end(), comp_nodes.begin() + comp_begin, comp_nodes.begin() + comp_end);
		cp.node_end = static_cast<int>(out.component_nodes.size());

		if (K <= EXACT_ENUM_THRESHOLD) {
			cp.exact_mode = ExactSamplingMode::Materialized;
			cp.valid_states_begin = static_cast<int>(out.exact_states_data.size());

			if (static_cast<int>(bp_local_index.size()) < N) bp_local_index.resize(N, -1);
			for (int li = 0; li < K; ++li) {
				int u = out.component_nodes[cp.node_begin + li];
				bp_local_index[u] = li;
			}

			for (int ci = comp_begin; ci < comp_end; ++ci) {
				int u = comp_nodes[ci];
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int v = scc.dag_data[ei];
					indeg_scratch[v]++;
				}
			}

			topo_order.clear();
			topo_order.reserve(K);
			kahn_q.clear();
			kahn_q.reserve(K);
			for (int ci = comp_begin; ci < comp_end; ++ci) {
				int u = comp_nodes[ci];
				if (indeg_scratch[u] == 0) kahn_q.push_back(u);
			}
			for (int head = 0; head < static_cast<int>(kahn_q.size()); ++head) {
				int u = kahn_q[head];
				topo_order.push_back(u);
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int v = scc.dag_data[ei];
					if (--indeg_scratch[v] == 0) kahn_q.push_back(v);
				}
			}

			auto& topo_local = tls_precompute_topo_local;
			auto& parent_mask_local = tls_precompute_parent_mask_local;
			topo_local.resize(K);
			parent_mask_local.resize(K);
			for (int ti = 0; ti < K; ++ti) {
				int u = topo_order[ti];
				int u_local = bp_local_index[u];
				topo_local[ti] = u_local;
				uint64_t pm = 0ULL;
				for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
					int parent = scc.dag_pred_data[ei];
					int parent_local = bp_local_index[parent];
					if (parent_local >= 0) pm |= (1ULL << parent_local);
				}
				parent_mask_local[u_local] = pm;
			}

			auto& materialized_states = tls_precompute_materialized_states;
			if (try_materialize_exact_states_from_topo(
				topo_local,
				parent_mask_local,
				EXACT_ENUM_VALID_STATE_CAP,
				EXACT_ENUM_NODE_CAP,
				materialized_states)) {
				out.exact_states_data.insert(
					out.exact_states_data.end(),
					materialized_states.begin(),
					materialized_states.end());
			} else {
				// K<=18 时理论上极少触发；保留 brute-force 回退路径以保证行为稳定。
				uint32_t parent_mask[EXACT_ENUM_THRESHOLD] = {0};
				for (int li = 0; li < K; ++li) {
					int u = out.component_nodes[cp.node_begin + li];
					uint32_t pm = 0u;
					for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
						int parent = scc.dag_pred_data[ei];
						int pl = bp_local_index[parent];
						if (pl >= 0) pm |= (1u << pl);
					}
					parent_mask[li] = pm;
				}

				uint32_t total_states = 1u << K;
				for (uint32_t state = 0; state < total_states; ++state) {
					bool ok = true;
					uint32_t inv = ~state;
					for (int li = 0; li < K; ++li) {
						if ((state & (1u << li)) && (inv & parent_mask[li])) {
							ok = false;
							break;
						}
					}
					if (ok) out.exact_states_data.push_back(state);
				}
			}

			cp.valid_states_end = static_cast<int>(out.exact_states_data.size());

			for (int li = 0; li < K; ++li) {
				int u = out.component_nodes[cp.node_begin + li];
				bp_local_index[u] = -1;
			}

			out.components.push_back(cp);
			continue;
		}

		if (static_cast<int>(bp_local_index.size()) < N) bp_local_index.resize(N, -1);
		for (int li = 0; li < K; ++li) {
			int u = out.component_nodes[cp.node_begin + li];
			bp_local_index[u] = li;
		}

		for (int ci = comp_begin; ci < comp_end; ++ci) {
			int u = comp_nodes[ci];
			for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
				int v = scc.dag_data[ei];
				indeg_scratch[v]++;
			}
		}

		// 大分量先求拓扑序（Kahn），供后续 BP 初始采样与 level 分层复用。
		topo_order.clear();
		topo_order.reserve(K);
		kahn_q.clear();
		kahn_q.reserve(K);
		for (int ci = comp_begin; ci < comp_end; ++ci) {
			int u = comp_nodes[ci];
			if (indeg_scratch[u] == 0) kahn_q.push_back(u);
		}
		for (int head = 0; head < static_cast<int>(kahn_q.size()); ++head) {
			int u = kahn_q[head];
			topo_order.push_back(u);
			for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
				int v = scc.dag_data[ei];
				if (--indeg_scratch[v] == 0) kahn_q.push_back(v);
			}
		}

		cp.topo_begin = static_cast<int>(out.topo_data.size());
		out.topo_data.insert(out.topo_data.end(), topo_order.begin(), topo_order.end());
		cp.topo_end = static_cast<int>(out.topo_data.size());

		bool use_exact = false;
		if (K <= FRONTIER_DP_MAX_K) {
			auto& topo_local = tls_precompute_topo_local;
			auto& parent_mask_local = tls_precompute_parent_mask_local;
			topo_local.resize(K);
			parent_mask_local.resize(K);
			for (int ti = 0; ti < K; ++ti) {
				int u = topo_order[ti];
				int u_local = bp_local_index[u];
				topo_local[ti] = u_local;
				uint64_t pm = 0ULL;
				for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
					int parent = scc.dag_pred_data[ei];
					int parent_local = bp_local_index[parent];
					if (parent_local >= 0) pm |= (1ULL << parent_local);
				}
				parent_mask_local[u_local] = pm;
			}

			if (K <= DFS_EXACT_MAX_K) {
				auto& materialized_states = tls_precompute_materialized_states;
				if (try_materialize_exact_states_from_topo(
					topo_local,
					parent_mask_local,
					EXACT_ENUM_VALID_STATE_CAP,
					EXACT_ENUM_NODE_CAP,
					materialized_states)) {
					cp.exact_mode = ExactSamplingMode::Materialized;
					cp.valid_states_begin = static_cast<int>(out.exact_states_data.size());
					out.exact_states_data.insert(
						out.exact_states_data.end(),
						materialized_states.begin(),
						materialized_states.end());
					cp.valid_states_end = static_cast<int>(out.exact_states_data.size());
					use_exact = true;
				}
			}

			if (!use_exact) {
				FrontierDpExactPlan& frontier_plan = tls_frontier_dp_exact_plan;
				if (try_build_frontier_dp_exact_plan(
					scc,
					topo_order,
					topo_local,
					parent_mask_local,
					bp_local_index,
					FRONTIER_DP_WIDTH_CAP,
					FRONTIER_DP_STATE_CAP,
					frontier_plan)) {
					cp.exact_mode = ExactSamplingMode::FrontierDp;
					cp.frontier_step_begin = static_cast<int>(out.frontier_parent_masks_data.size());
					out.frontier_parent_masks_data.insert(
						out.frontier_parent_masks_data.end(),
						frontier_plan.step_parent_masks.begin(),
						frontier_plan.step_parent_masks.end());
					out.frontier_keep_masks_data.insert(
						out.frontier_keep_masks_data.end(),
						frontier_plan.step_keep_masks.begin(),
						frontier_plan.step_keep_masks.end());
					out.frontier_add_current_data.insert(
						out.frontier_add_current_data.end(),
						frontier_plan.step_add_current.begin(),
						frontier_plan.step_add_current.end());
					cp.frontier_step_end = static_cast<int>(out.frontier_parent_masks_data.size());

					cp.frontier_layer_begin = static_cast<int>(out.frontier_layer_widths_data.size());
					out.frontier_layer_widths_data.insert(
						out.frontier_layer_widths_data.end(),
						frontier_plan.layer_widths.begin(),
						frontier_plan.layer_widths.end());
					const int count_base = static_cast<int>(out.frontier_counts_data.size());
					for (int offset : frontier_plan.count_offsets) {
						out.frontier_count_offsets_data.push_back(count_base + offset);
					}
					cp.frontier_layer_end = static_cast<int>(out.frontier_layer_widths_data.size());
					out.frontier_base_states_data.insert(
						out.frontier_base_states_data.end(),
						frontier_plan.base_states.begin(),
						frontier_plan.base_states.end());
					out.frontier_next_state_one_data.insert(
						out.frontier_next_state_one_data.end(),
						frontier_plan.next_state_one.begin(),
						frontier_plan.next_state_one.end());
					out.frontier_counts_data.insert(
						out.frontier_counts_data.end(),
						frontier_plan.counts.begin(),
						frontier_plan.counts.end());
					use_exact = true;
				}
			}
		}

		if (use_exact) {
			for (int li = 0; li < K; ++li) {
				int u = out.component_nodes[cp.node_begin + li];
				bp_local_index[u] = -1;
			}
			out.components.push_back(cp);
			continue;
		}

		int max_level = -1;
		for (int u : topo_order) {
			int lu = 0;
			for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
				int parent = scc.dag_pred_data[ei];
				lu = std::max(lu, level[parent] + 1);
			}
			level[u] = lu;
			if (lu > max_level) max_level = lu;
		}

		level_counts.assign(max_level + 1, 0);
		for (int u : topo_order) {
			level_counts[level[u]]++;
		}
		level_offsets.assign(max_level + 2, 0);
		for (int L = 0; L <= max_level; ++L) {
			level_offsets[L + 1] = level_offsets[L] + level_counts[L];
		}
		level_data.assign(level_offsets[max_level + 1], -1);
		level_write = level_offsets;
		for (int u : topo_order) {
			int L = level[u];
			level_data[level_write[L]++] = u;
		}

		cp.level_offsets_begin = static_cast<int>(out.level_offsets_data.size());
		out.level_offsets_data.insert(out.level_offsets_data.end(), level_offsets.begin(), level_offsets.end());
		cp.level_offsets_end = static_cast<int>(out.level_offsets_data.size());

		cp.level_data_begin = static_cast<int>(out.level_data.size());
		out.level_data.insert(out.level_data.end(), level_data.begin(), level_data.end());
		// cp.level_data_end = static_cast<int>(out.level_data.size());  // 死赋值

		for (int li = 0; li < K; ++li) {
			int u = out.component_nodes[cp.node_begin + li];
			bp_local_index[u] = -1;
		}

		out.components.push_back(cp);
	}
}

inline void sample_component_materialized_exact_single(
	const McmcDagPrecomputed& dag_plan,
	const McmcComponentPlan& comp,
	RNG& gen,
	std::vector<int8_t>& super_state)
{
	int valid_count = comp.valid_states_end - comp.valid_states_begin;
	if (valid_count <= 0) return;
	uint32_t mask = dag_plan.exact_states_data[
		comp.valid_states_begin + fast_rand_below(gen, valid_count)];
	for (int li = 0; li < comp.size; ++li) {
		int u = dag_plan.component_nodes[comp.node_begin + li];
		super_state[u] = static_cast<int8_t>((mask >> li) & 1u);
	}
}

inline void sample_component_frontier_dp_single(
	const McmcDagPrecomputed& dag_plan,
	const McmcComponentPlan& comp,
	RNG& gen,
	std::vector<int8_t>& super_state)
{
	const int K = comp.size;
	const int* topo = dag_plan.topo_data.data() + comp.topo_begin;
	const int* parent_masks = dag_plan.frontier_parent_masks_data.data() + comp.frontier_step_begin;
	const int* count_offsets = dag_plan.frontier_count_offsets_data.data() + comp.frontier_layer_begin;
	const int* base_states = dag_plan.frontier_base_states_data.data();
	const int* next_state_one = dag_plan.frontier_next_state_one_data.data();

	int frontier_state = 0;
	for (int ti = 0; ti < K; ++ti) {
		const int lut_idx = count_offsets[ti] + frontier_state;
		const int base_state = base_states[lut_idx];
		const uint64_t* next_counts = dag_plan.frontier_counts_data.data() + count_offsets[ti + 1];
		const uint64_t count0 = next_counts[base_state];
		uint64_t count1 = 0ULL;
		if ((frontier_state & parent_masks[ti]) == parent_masks[ti]) {
			count1 = next_counts[next_state_one[lut_idx]];
		}

		int sampled = 0;
		if (count1 > 0ULL) {
			uint64_t threshold = count0 + count1;
			sampled = (fast_rand_below_u64(gen, threshold) >= count0) ? 1 : 0;
		}

		frontier_state = (sampled != 0) ? next_state_one[lut_idx] : base_state;
		super_state[topo[ti]] = static_cast<int8_t>(sampled);
	}
}

inline void sample_component_materialized_exact_msc64(
	const McmcDagPrecomputed& dag_plan,
	const McmcComponentPlan& comp,
	RNG& gen,
	std::vector<uint64_t>& super_state64)
{
	int valid_count = comp.valid_states_end - comp.valid_states_begin;
	if (comp.size == 1) {
		int u = dag_plan.component_nodes[comp.node_begin];
		if (valid_count <= 0) {
			super_state64[u] = 0ULL;
		} else if (valid_count == 1) {
			uint32_t only_mask = dag_plan.exact_states_data[comp.valid_states_begin];
			super_state64[u] = (only_mask & 1u) ? ~0ULL : 0ULL;
		} else {
			super_state64[u] = static_cast<uint64_t>(gen());
		}
		return;
	}

	for (int li = 0; li < comp.size; ++li) {
		int u = dag_plan.component_nodes[comp.node_begin + li];
		super_state64[u] = 0ULL;
	}
	if (valid_count <= 0) return;

	for (int bit = 0; bit < 64; ++bit) {
		uint32_t mask = dag_plan.exact_states_data[
			comp.valid_states_begin + fast_rand_below(gen, valid_count)];
		for (int li = 0; li < comp.size; ++li) {
			if ((mask & (1u << li)) == 0u) continue;
			int u = dag_plan.component_nodes[comp.node_begin + li];
			super_state64[u] |= (1ULL << bit);
		}
	}
}

inline void sample_component_frontier_dp_exact_msc64(
	const McmcDagPrecomputed& dag_plan,
	const McmcComponentPlan& comp,
	RNG& gen,
	std::vector<uint64_t>& super_state64)
{
	const int K = comp.size;
	const int* topo = dag_plan.topo_data.data() + comp.topo_begin;
	const int* parent_masks = dag_plan.frontier_parent_masks_data.data() + comp.frontier_step_begin;
	const int* count_offsets = dag_plan.frontier_count_offsets_data.data() + comp.frontier_layer_begin;
	const int* base_states = dag_plan.frontier_base_states_data.data();
	const int* next_state_one = dag_plan.frontier_next_state_one_data.data();

	for (int li = 0; li < comp.size; ++li) {
		int u = dag_plan.component_nodes[comp.node_begin + li];
		super_state64[u] = 0ULL;
	}

	for (int bit = 0; bit < 64; ++bit) {
		int frontier_state = 0;
		for (int ti = 0; ti < K; ++ti) {
			const int lut_idx = count_offsets[ti] + frontier_state;
			const int base_state = base_states[lut_idx];
			const uint64_t* next_counts = dag_plan.frontier_counts_data.data() + count_offsets[ti + 1];
			const uint64_t count0 = next_counts[base_state];
			uint64_t count1 = 0ULL;
			if ((frontier_state & parent_masks[ti]) == parent_masks[ti]) {
				count1 = next_counts[next_state_one[lut_idx]];
			}

			int sampled = 0;
			if (count1 > 0ULL) {
				uint64_t threshold = count0 + count1;
				sampled = (fast_rand_below_u64(gen, threshold) >= count0) ? 1 : 0;
			}

			frontier_state = (sampled != 0) ? next_state_one[lut_idx] : base_state;
			if (sampled != 0) super_state64[topo[ti]] |= (1ULL << bit);
		}
	}
}

// 在“超点 DAG”上执行单链 MCMC 采样：
// - 输入可为新链或续跑链（inout_super_state）；
// - 小分量直接精确抽样；
// - 大分量执行初始化 + Level-Set Block Gibbs；
// - 自动模式下可先做 ACF 探针估计 tau，再决定 sweeps。
McmcResult sample_super_states_mcmc(
	const SccDagResult& scc,
	const McmcDagPrecomputed& dag_plan,
	const Params& p,
	RNG& gen,
	std::vector<int8_t>* inout_super_state = nullptr)
{
	McmcResult result;
	int N = scc.nSuper;
	bool resume_chain = (inout_super_state != nullptr && static_cast<int>(inout_super_state->size()) == N);
	if (resume_chain) {
		result.super_state = *inout_super_state;
	} else {
		result.super_state.assign(N, 0); // 初始化为全 0
	}

	if (N == 0) return result;

	ThreadLocalWorkspace& ws = tls_ws;
	auto& num_zero_parents = ws.mcmc_num_zero_parents;
	auto& num_one_children = ws.mcmc_num_one_children;
	auto& visit_mark = ws.mcmc_visit_mark;
	auto& bp_local_index = ws.mcmc_bp_local_index;
	auto& bp_parent_offsets = ws.mcmc_bp_parent_offsets;
	auto& bp_child_offsets = ws.mcmc_bp_child_offsets;
	auto& bp_parent_data = ws.mcmc_bp_parent_data;
	auto& bp_child_data = ws.mcmc_bp_child_data;
	auto& bp_parent_count = ws.mcmc_bp_parent_count;
	auto& bp_child_count = ws.mcmc_bp_child_count;
	auto& bp_parent_cursor = ws.mcmc_bp_parent_cursor;
	auto& bp_child_cursor = ws.mcmc_bp_child_cursor;
	auto& bp_edge_parent = ws.mcmc_bp_edge_parent;
	auto& bp_edge_child = ws.mcmc_bp_edge_child;
	auto& bp_edge_parent_local = ws.mcmc_bp_edge_parent_local;
	auto& bp_edge_child_local = ws.mcmc_bp_edge_child_local;
	auto& bp_mu = ws.mcmc_bp_mu;
	auto& bp_eta = ws.mcmc_bp_eta;
	auto& bp_new_mu = ws.mcmc_bp_new_mu;
	auto& bp_new_eta = ws.mcmc_bp_new_eta;
	auto& avalanche_q = ws.mcmc_avalanche_queue;
	auto& x_traj = ws.mcmc_x_traj;
	auto& acf_buf = ws.mcmc_acf;
	auto& fft_buffer = ws.mcmc_fft_buffer;

	int num_components = static_cast<int>(dag_plan.components.size());
	result.max_wcc_size = dag_plan.max_wcc_size;

	// ---- 步骤 2：对每个 Component 独立采样（BP / 对称热启动 + Level-Set Block Gibbs） ----
	if (static_cast<int>(num_zero_parents.size()) < N) num_zero_parents.resize(N);
	if (static_cast<int>(num_one_children.size()) < N) num_one_children.resize(N);
	if (!resume_chain && p.init_mode == 2 && static_cast<int>(visit_mark.size()) < N) {
		visit_mark.resize(N, 0);
	}
	if (!resume_chain && p.init_mode == 1 && static_cast<int>(bp_local_index.size()) < N) {
		bp_local_index.resize(N, -1);
	}
	long long sample_total_sweeps = 0;
	long long sample_num_components = 0;
	int shared_auto_sweeps = std::max(0, p.mcmc_factor);
	bool shared_auto_sweeps_ready = !p.use_acf_auto_sweeps;
	bool has_mcmc_component = false;

	bp_parent_offsets.clear();
	bp_child_offsets.clear();
	bp_parent_data.clear();
	bp_child_data.clear();
	bp_parent_count.clear();
	bp_child_count.clear();
	bp_parent_cursor.clear();
	bp_child_cursor.clear();
	bp_edge_parent.clear();
	bp_edge_child.clear();
	bp_edge_parent_local.clear();
	bp_edge_child_local.clear();
	bp_mu.clear();
	bp_eta.clear();
	bp_new_mu.clear();
	bp_new_eta.clear();
	avalanche_q.clear();
	x_traj.clear();
	acf_buf.clear();
	fft_buffer.clear();

	for (int comp_rank = 0; comp_rank < num_components; ++comp_rank) {
		const McmcComponentPlan& comp = dag_plan.components[comp_rank];
		int K = comp.size;
		if (K <= 0) continue;
		sample_num_components++;
		const int topo_begin = comp.topo_begin;
		const int topo_end = comp.topo_end;
		const int node_begin = comp.node_begin;
		const int node_end = comp.node_end;
		const int level_offsets_begin = comp.level_offsets_begin;
		const int level_data_begin = comp.level_data_begin;
		const int max_level = (comp.level_offsets_end - comp.level_offsets_begin) - 2;

		// exact 分量：优先复用预处理阶段已确定的 materialized / frontier-DP 方案。
		if (comp.exact_mode == ExactSamplingMode::Materialized) {
			sample_component_materialized_exact_single(dag_plan, comp, gen, result.super_state);
			continue;
		}
		if (comp.exact_mode == ExactSamplingMode::FrontierDp) {
			sample_component_frontier_dp_single(dag_plan, comp, gen, result.super_state);
			continue;
		}
		has_mcmc_component = true;

		// ---- 步骤 2a：初始化合法初态（BP / Avalanche / 随机极端 / 独立均匀） ----
		// 目标：给大分量提供“尽量接近平衡且满足约束”的起点，缩短 burn-in。
		if (!resume_chain && p.init_mode == 1) {
			// BP 初始化：在分量内部迭代近似消息，随后按拓扑顺序做条件采样。
			constexpr double kBpProdCap = 1e150;

			auto mul_sat = [&](double acc, double factor) {
				if (!(acc > 0.0) || !(factor > 0.0)) return 0.0;
				if (!std::isfinite(acc) || !std::isfinite(factor)) return kBpProdCap;
				if (acc > kBpProdCap / factor) return kBpProdCap;
				double v = acc * factor;
				if (!std::isfinite(v) || v > kBpProdCap) return kBpProdCap;
				return v;
			};

			auto safe_prob_from_ratio = [&](double ratio) {
				// 将“几率比” ratio 转成概率 p=ratio/(1+ratio)，并处理数值溢出边界。
				if (!(ratio > 0.0)) return 0.0;
				if (!std::isfinite(ratio) || ratio >= kBpProdCap) return 1.0;
				return ratio / (1.0 + ratio);
			};

			bp_edge_parent.clear();
			bp_edge_child.clear();
			bp_edge_parent_local.clear();
			bp_edge_child_local.clear();
			bp_edge_parent.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_child.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_parent_local.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_child_local.reserve(static_cast<size_t>(K) * 2u);

			for (int li = 0; li < K; ++li) {
				int u = dag_plan.component_nodes[node_begin + li];
				bp_local_index[u] = li;
			}

			bp_parent_count.assign(K, 0);
			bp_child_count.assign(K, 0);

			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				int u_local = bp_local_index[u];
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int c = scc.dag_data[ei];
					int c_local = bp_local_index[c];
					if (c_local < 0) continue;
					bp_edge_parent.push_back(u);
					bp_edge_child.push_back(c);
					bp_edge_parent_local.push_back(u_local);
					bp_edge_child_local.push_back(c_local);
					bp_child_count[u_local]++;
					bp_parent_count[c_local]++;
				}
			}
			int nBpEdges = static_cast<int>(bp_edge_parent.size());
			// 将边列表重排为按“节点入边/出边段”的 CSR 结构，后续消息更新可顺序扫描。

			bp_parent_offsets.assign(K + 1, 0);
			bp_child_offsets.assign(K + 1, 0);
			for (int i = 0; i < K; ++i) {
				bp_parent_offsets[i + 1] = bp_parent_offsets[i] + bp_parent_count[i];
				bp_child_offsets[i + 1] = bp_child_offsets[i] + bp_child_count[i];
			}
			bp_parent_data.assign(bp_parent_offsets[K], -1);
			bp_child_data.assign(bp_child_offsets[K], -1);
			bp_parent_cursor = bp_parent_offsets;
			bp_child_cursor = bp_child_offsets;
			for (int eid = 0; eid < nBpEdges; ++eid) {
				int parent_local = bp_edge_parent_local[eid];
				int child_local = bp_edge_child_local[eid];
				bp_child_data[bp_child_cursor[parent_local]++] = eid;
				bp_parent_data[bp_parent_cursor[child_local]++] = eid;
			}

			if (nBpEdges == 0) {
				for (int ni = node_begin; ni < node_end; ++ni) {
					int u = dag_plan.component_nodes[ni];
					result.super_state[u] = static_cast<int>(gen() & 1);
				}
			} else {
				// mu/eta 为 BP 消息（边上两个方向的近似边缘信息）。
				bp_mu.assign(nBpEdges, 0.5);
				bp_eta.assign(nBpEdges, 2.0);
				bp_new_mu.assign(nBpEdges, 0.0);
				bp_new_eta.assign(nBpEdges, 0.0);

				for (int iter = 0; iter < p.bp_iters; ++iter) {
					for (int eid = 0; eid < nBpEdges; ++eid) {
						int par = bp_edge_parent[eid];
						int chi = bp_edge_child[eid];
						int par_local = bp_edge_parent_local[eid];
						int chi_local = bp_edge_child_local[eid];

						double R = 1.0;
						for (int ii = bp_parent_offsets[par_local]; ii < bp_parent_offsets[par_local + 1]; ++ii) {
							int pe = bp_parent_data[ii];
							R = mul_sat(R, bp_mu[pe]);
							if (R >= kBpProdCap) break;
						}
						if (R < kBpProdCap) {
							for (int ii = bp_child_offsets[par_local]; ii < bp_child_offsets[par_local + 1]; ++ii) {
								int ce = bp_child_data[ii];
								if (ce != eid) {
									R = mul_sat(R, bp_eta[ce]);
									if (R >= kBpProdCap) break;
								}
							}
						}
						bp_new_mu[eid] = safe_prob_from_ratio(R);

						R = 1.0;
						for (int ii = bp_parent_offsets[chi_local]; ii < bp_parent_offsets[chi_local + 1]; ++ii) {
							int pe = bp_parent_data[ii];
							if (pe != eid) {
								R = mul_sat(R, bp_mu[pe]);
								if (R >= kBpProdCap) break;
							}
						}
						if (R < kBpProdCap) {
							for (int ii = bp_child_offsets[chi_local]; ii < bp_child_offsets[chi_local + 1]; ++ii) {
								int ce = bp_child_data[ii];
								R = mul_sat(R, bp_eta[ce]);
								if (R >= kBpProdCap) break;
							}
						}
						bp_new_eta[eid] = (R >= kBpProdCap) ? kBpProdCap : (1.0 + R);
					}
					// 同步迭代：新消息整体替换旧消息，避免更新顺序带来的偏置。
					bp_mu.swap(bp_new_mu);
					bp_eta.swap(bp_new_eta);
				}

				std::uniform_real_distribution<double> unif01(0.0, 1.0);
				for (int ti = topo_begin; ti < topo_end; ++ti) {
					int u = dag_plan.topo_data[ti];
					bool any_parent_zero = false;
					for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
						int pred = scc.dag_pred_data[ei];
						if (result.super_state[pred] == 0) {
							any_parent_zero = true;
							break;
						}
					}
					if (any_parent_zero) {
						// 约束：若存在父节点为 0，则当前节点必须为 0。
						result.super_state[u] = 0;
					} else {
						int u_local = bp_local_index[u];
						double R_cond = 1.0;
						for (int ii = bp_child_offsets[u_local]; ii < bp_child_offsets[u_local + 1]; ++ii) {
							int ce = bp_child_data[ii];
							R_cond = mul_sat(R_cond, bp_eta[ce]);
							if (R_cond >= kBpProdCap) break;
						}
						double p1 = safe_prob_from_ratio(R_cond);
						result.super_state[u] = (unif01(gen) < p1) ? 1 : 0;
					}
				}
			}

			for (int li = 0; li < K; ++li) {
				int u = dag_plan.component_nodes[node_begin + li];
				bp_local_index[u] = -1;
			}

		} else if (!resume_chain && p.init_mode == 2) {
			// 雪崩初始化：随机点触发向父/子方向传播，快速形成大尺度相关结构。
			int extreme_state = static_cast<int>(gen() & 1);
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				result.super_state[u] = extreme_state;
			}

			long long avalanche_steps = std::max(10LL, 2LL * static_cast<long long>(K));
			std::uniform_int_distribution<int> idx_dist_comp(0, K - 1);
			avalanche_q.clear();
			avalanche_q.reserve(K);

			auto next_visit_token = [&]() {
				if (ws.mcmc_visit_token == std::numeric_limits<int>::max()) {
					// token 溢出时统一清零，随后从 1 重新开始。
					std::fill(visit_mark.begin(), visit_mark.end(), 0);
					ws.mcmc_visit_token = 1;
				}
				return ws.mcmc_visit_token++;
			};

			for (long long st = 0; st < avalanche_steps; ++st) {
				int start = dag_plan.component_nodes[node_begin + idx_dist_comp(gen)];
				bool force_one = (gen() & 1) == 1;
				int mark = next_visit_token();

				avalanche_q.clear();
				avalanche_q.push_back(start);
				visit_mark[start] = mark;

				for (int head2 = 0; head2 < static_cast<int>(avalanche_q.size()); ++head2) {
					int u = avalanche_q[head2];
					if (force_one) {
						result.super_state[u] = 1;
						// 置 1 时向“父方向”传播，修复可能受影响的上游约束。
						for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
							int parent = scc.dag_pred_data[ei];
							if (visit_mark[parent] != mark) {
								visit_mark[parent] = mark;
								avalanche_q.push_back(parent);
							}
						}
					} else {
						result.super_state[u] = 0;
						// 置 0 时向“子方向”传播，修复可能受影响的下游约束。
						for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
							int child = scc.dag_data[ei];
							if (visit_mark[child] != mark) {
								visit_mark[child] = mark;
								avalanche_q.push_back(child);
							}
						}
					}
				}
			}

		} else if (!resume_chain && p.init_mode == 3) {
			// 独立均匀随机初始化：每个节点独立以 1/2 概率取 0 或 1。
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				result.super_state[u] = static_cast<int>(gen() & 1);
			}
		} else if (!resume_chain) {
			// 对称极端初始化（默认 / init_mode==0）：全分量随机取全 0 或全 1。
			int extreme_state = static_cast<int>(gen() & 1);
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				result.super_state[u] = extreme_state;
			}
		}

		// ---- 维护 O(1) 约束计数缓存 ----
		// num_zero_parents[u]：u 的父节点中状态为 0 的计数；
		// num_one_children[u]：u 的子节点中状态为 1 的计数。
		// 这两个缓存让每次更新节点时无需重复遍历完整邻居表。
		for (int ni = node_begin; ni < node_end; ++ni) {
			int u = dag_plan.component_nodes[ni];
			num_zero_parents[u] = 0;
			num_one_children[u] = 0;
		}
		for (int ni = node_begin; ni < node_end; ++ni) {
			int u = dag_plan.component_nodes[ni];
			for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
				int c = scc.dag_data[ei];
				if (result.super_state[u] == 0) ++num_zero_parents[c];
				if (result.super_state[c] == 1) ++num_one_children[u];
			}
		}
		// ACF 探针使用随机高斯投影（Random Gaussian Projection）来估计 tau_int
		// 相比 running_ones（1-密度），高斯投影能耦合所有慢模式，避免低估相关时间
		bool need_acf_projection = p.use_acf_auto_sweeps && !shared_auto_sweeps_ready
			&& !(p.cached_tau_int > 0.0) && K > EXACT_ENUM_THRESHOLD;
		double running_projection = 0.0;
		if (need_acf_projection) {
			auto& gauss_w = ws.mcmc_gauss_weights;
			if (static_cast<int>(gauss_w.size()) < N) gauss_w.resize(N, 0.0);
			std::normal_distribution<double> ndist(0.0, 1.0);
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				gauss_w[u] = ndist(gen);
			}
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				if (result.super_state[u] == 1) running_projection += gauss_w[u];
			}
		}

		// 局部状态翻转时，增量维护 parent/child 计数，保证后续判定 O(1)。
		auto apply_state_change = [&](int u, int new_state) {
			int old_state = result.super_state[u];
			if (old_state == new_state) return;
			result.super_state[u] = new_state;

			if (old_state == 0 && new_state == 1) {
				if (need_acf_projection) running_projection += ws.mcmc_gauss_weights[u];
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int child = scc.dag_data[ei];
					--num_zero_parents[child];
				}
				for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
					int parent = scc.dag_pred_data[ei];
					++num_one_children[parent];
				}
			} else if (old_state == 1 && new_state == 0) {
				if (need_acf_projection) running_projection -= ws.mcmc_gauss_weights[u];
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int child = scc.dag_data[ei];
					++num_zero_parents[child];
				}
				for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
					int parent = scc.dag_pred_data[ei];
					--num_one_children[parent];
				}
			}
		};
		if (max_level < 0) continue;
		const int* local_level_offsets = dag_plan.level_offsets_data.data() + level_offsets_begin;

		// ---- 步骤 B + C：Level-Set Sweep（Forward + Backward）逐层更新 ----
		// 注意：main() 外层已对 num_samples 并行，这里避免嵌套 OpenMP 带来的 Fork-Join/Barrier 开销。
		FastCoinFlip coin_flip;

		auto update_one_level = [&](int L) {
			for (int i = local_level_offsets[L]; i < local_level_offsets[L + 1]; ++i) {
				int u = dag_plan.level_data[level_data_begin + i];

				// 规则A：任一父节点为 0 -> u 只能为 0。
				if (num_zero_parents[u] > 0) {
					if (result.super_state[u] != 0) apply_state_change(u, 0);
					continue;
				}

				// 规则B：若某个子节点为 1，为满足单调约束 u 必须为 1。
				if (num_one_children[u] > 0) {
					if (result.super_state[u] != 1) apply_state_change(u, 1);
				} else {
					// 规则C：若 A/B 都不触发，则 0/1 等概率采样（局部自由度）。
					int sampled_state = coin_flip.next(gen);
					if (result.super_state[u] != sampled_state) apply_state_change(u, sampled_state);
				}
			}
		};

		auto one_full_sweep = [&]() {
			for (int L = 0; L <= max_level; ++L) {
				update_one_level(L); // forward
			}
			for (int L = max_level; L >= 0; --L) {
				update_one_level(L); // backward
			}
		};

		auto comp_projection = [&]() {
			return running_projection;
		};

		int total_target_sweeps = std::max(0, p.mcmc_factor);
		int probe_done_sweeps = 0;

		// 自动 sweeps 策略：
		// - 若给定 cached_tau_int，直接换算目标 sweeps；
		// - 否则先跑 probe 链估计 tau_int，再按倍数放大。
		if (p.use_acf_auto_sweeps) {
			if (!shared_auto_sweeps_ready) {
				if (p.cached_tau_int > 0.0) {
					int est_sweeps = static_cast<int>(std::ceil(p.acf_tau_multiplier * p.cached_tau_int));
					est_sweeps = std::max(20, est_sweeps);
					if (p.acf_max_sweeps > 0) est_sweeps = std::min(est_sweeps, p.acf_max_sweeps);
					shared_auto_sweeps = est_sweeps;
					shared_auto_sweeps_ready = true;
					result.estimated_tau_int = p.cached_tau_int;
				} else {
					int probe_burnin_sweeps = std::max(0, p.acf_probe_burnin_sweeps);
					int probe_sampling_sweeps = std::max(8, p.acf_probe_sweeps);
					x_traj.clear();
					x_traj.reserve(probe_sampling_sweeps);

					for (int s = 0; s < probe_burnin_sweeps; ++s) {
						one_full_sweep();
						probe_done_sweeps++;
					}

					for (int s = 0; s < probe_sampling_sweeps; ++s) {
						one_full_sweep();
						x_traj.push_back(comp_projection());
						probe_done_sweeps++;
					}

					acf_fft_biased_inplace(x_traj, acf_buf, fft_buffer);
					TauResult tau = estimate_tau_sokal(acf_buf, 5.0);
					if (p.acf_tau_estimator == 1) {
						TauResult tau_geyer = estimate_tau_geyer_ips(acf_buf);
						if (tau_geyer.tau_int > tau.tau_int) tau = tau_geyer;
					}

					if (tau.tau_int > static_cast<double>(probe_sampling_sweeps) / 50.0) {
						result.acf_probe_too_short = true;
						debug_log(p.debug_output,
							"Warning: Probe length may be too short for accurate tau estimation.");
					}
					result.estimated_tau_int = tau.tau_int;

					int est_sweeps = static_cast<int>(std::ceil(p.acf_tau_multiplier * tau.tau_int));
					est_sweeps = std::max(20, est_sweeps); // 至少 20 次
					if (p.acf_max_sweeps > 0) est_sweeps = std::min(est_sweeps, p.acf_max_sweeps);

					shared_auto_sweeps = est_sweeps;
					shared_auto_sweeps_ready = true;
				}
			}
			total_target_sweeps = shared_auto_sweeps;
		} else {
			total_target_sweeps = std::max(0, p.mcmc_factor);
		}

		int remain_sweeps = std::max(0, total_target_sweeps - probe_done_sweeps);
		// 注意：probe 已执行的 sweeps 也计入总开销，因此这里只补剩余部分。
		sample_total_sweeps += static_cast<long long>(probe_done_sweeps + remain_sweeps);
		need_acf_projection = false; // 探针结束，剩余 sweeps 无需维护投影

		// 正式阶段：执行剩余 sweep，输出用于测量的近平衡状态。
		for (int sweep = 0; sweep < remain_sweeps; ++sweep) {
			one_full_sweep();
		}
	}

	result.mean_sweeps_used = (sample_num_components > 0)
		? static_cast<double>(sample_total_sweeps) / static_cast<double>(sample_num_components)
		: 0.0;
	result.shared_sweeps = has_mcmc_component
		? (p.use_acf_auto_sweeps ? shared_auto_sweeps : std::max(0, p.mcmc_factor))
		: 0;

	if (inout_super_state != nullptr) {
		*inout_super_state = result.super_state;
	}

	return result;
}

// 生成 64 个独立 Bernoulli(p) 位，打包为 uint64_t。
// 每次 gen() 产出 64 bit，拆为高低 32-bit 各与阈值比较，得 2 个结果位。
inline uint64_t bernoulli_bits_64(double prob, RNG& gen) {
	if (prob <= 0.0) return 0ULL;
	if (prob >= 1.0) return ~0ULL;
	// uint32_t threshold = static_cast<uint32_t>(prob * 4294967296.0); // prob * 2^32
	uint64_t thresh64 = static_cast<uint64_t>(prob * 4294967296.0);
	if (thresh64 > 0xFFFFFFFFULL) thresh64 = 0xFFFFFFFFULL;
	uint32_t threshold = static_cast<uint32_t>(thresh64);
	uint64_t bits = 0ULL;
	for (int i = 0; i < 64; i += 2) {
		uint64_t r = gen();
		uint32_t lo = static_cast<uint32_t>(r);
		uint32_t hi = static_cast<uint32_t>(r >> 32);
		if (lo < threshold) bits |= (1ULL << i);
		if (hi < threshold) bits |= (1ULL << (i + 1));
	}
	return bits;
}

// MSC64 版本：把 64 条链打包在 uint64_t 位平面中同步更新。
// 仅在 multi_start 且热样本数为 64 倍数时启用，用于吞吐量优化。
McmcBatch64Result sample_super_states_mcmc_msc64(
	const SccDagResult& scc,
	const McmcDagPrecomputed& dag_plan,
	const Params& p,
	RNG& gen)
{
	McmcBatch64Result result;
	int N = scc.nSuper;
	result.super_state64.assign(N, 0ULL);
	if (N == 0) return result;

	ThreadLocalWorkspace& ws = tls_ws;
	auto& x_traj = ws.mcmc_x_traj;
	auto& acf_buf = ws.mcmc_acf;
	auto& fft_buffer = ws.mcmc_fft_buffer;
	auto& gauss_w = ws.mcmc_gauss_weights;
	auto& visit_mark = ws.mcmc_visit_mark;
	auto& bp_local_index = ws.mcmc_bp_local_index;
	auto& bp_parent_offsets = ws.mcmc_bp_parent_offsets;
	auto& bp_child_offsets = ws.mcmc_bp_child_offsets;
	auto& bp_parent_data = ws.mcmc_bp_parent_data;
	auto& bp_child_data = ws.mcmc_bp_child_data;
	auto& bp_parent_count = ws.mcmc_bp_parent_count;
	auto& bp_child_count = ws.mcmc_bp_child_count;
	auto& bp_parent_cursor = ws.mcmc_bp_parent_cursor;
	auto& bp_child_cursor = ws.mcmc_bp_child_cursor;
	auto& bp_edge_parent = ws.mcmc_bp_edge_parent;
	auto& bp_edge_child = ws.mcmc_bp_edge_child;
	auto& bp_edge_parent_local = ws.mcmc_bp_edge_parent_local;
	auto& bp_edge_child_local = ws.mcmc_bp_edge_child_local;
	auto& bp_mu = ws.mcmc_bp_mu;
	auto& bp_eta = ws.mcmc_bp_eta;
	auto& bp_new_mu = ws.mcmc_bp_new_mu;
	auto& bp_new_eta = ws.mcmc_bp_new_eta;
	auto& avalanche_q = ws.mcmc_avalanche_queue;

	int num_components = static_cast<int>(dag_plan.components.size());
	result.max_wcc_size = dag_plan.max_wcc_size;
	long long sample_total_sweeps = 0;
	long long sample_num_components = 0;
	int shared_auto_sweeps = std::max(0, p.mcmc_factor);
	bool shared_auto_sweeps_ready = !p.use_acf_auto_sweeps;
	bool has_mcmc_component = false;

	if (p.init_mode == 2 && static_cast<int>(visit_mark.size()) < N) {
		visit_mark.resize(N, 0);
	}
	if (p.init_mode == 1 && static_cast<int>(bp_local_index.size()) < N) {
		bp_local_index.resize(N, -1);
	}

	x_traj.clear();
	acf_buf.clear();
	fft_buffer.clear();

	for (int comp_rank = 0; comp_rank < num_components; ++comp_rank) {
		const McmcComponentPlan& comp = dag_plan.components[comp_rank];
		int K = comp.size;
		if (K <= 0) continue;
		sample_num_components++;
		const int node_begin = comp.node_begin;
		const int node_end = comp.node_end;
		const int level_offsets_begin = comp.level_offsets_begin;
		const int level_data_begin = comp.level_data_begin;
		const int max_level = (comp.level_offsets_end - comp.level_offsets_begin) - 2;

		if (comp.exact_mode == ExactSamplingMode::Materialized) {
			sample_component_materialized_exact_msc64(dag_plan, comp, gen, result.super_state64);
			continue;
		}
		if (comp.exact_mode == ExactSamplingMode::FrontierDp) {
			sample_component_frontier_dp_exact_msc64(dag_plan, comp, gen, result.super_state64);
			continue;
		}
		has_mcmc_component = true;

		bool need_acf_projection = p.use_acf_auto_sweeps && !shared_auto_sweeps_ready
			&& !(p.cached_tau_int > 0.0) && K > EXACT_ENUM_THRESHOLD;
		if (need_acf_projection) {
			if (static_cast<int>(gauss_w.size()) < N) gauss_w.resize(N, 0.0);
			std::normal_distribution<double> ndist(0.0, 1.0);
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				gauss_w[u] = ndist(gen);
			}
		}

		// ---- 步骤 MSC64-2a：初始化合法初态（BP / Avalanche / 对称极端 / 独立均匀） ----
		if (p.init_mode == 1) {
			// BP 初始化（位并行版）：消息传递与单链完全相同，采样阶段按拓扑序逐节点
			// 调用 bernoulli_bits_64(p1, gen) 生成 64 个独立 Bernoulli 位。
			constexpr double kBpProdCap = 1e150;

			auto mul_sat = [&](double acc, double factor) {
				if (!(acc > 0.0) || !(factor > 0.0)) return 0.0;
				if (!std::isfinite(acc) || !std::isfinite(factor)) return kBpProdCap;
				if (acc > kBpProdCap / factor) return kBpProdCap;
				double v = acc * factor;
				if (!std::isfinite(v) || v > kBpProdCap) return kBpProdCap;
				return v;
			};

			auto safe_prob_from_ratio = [&](double ratio) {
				if (!(ratio > 0.0)) return 0.0;
				if (!std::isfinite(ratio) || ratio >= kBpProdCap) return 1.0;
				return ratio / (1.0 + ratio);
			};

			bp_edge_parent.clear();
			bp_edge_child.clear();
			bp_edge_parent_local.clear();
			bp_edge_child_local.clear();
			bp_edge_parent.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_child.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_parent_local.reserve(static_cast<size_t>(K) * 2u);
			bp_edge_child_local.reserve(static_cast<size_t>(K) * 2u);

			for (int li = 0; li < K; ++li) {
				int u = dag_plan.component_nodes[node_begin + li];
				bp_local_index[u] = li;
			}

			bp_parent_count.assign(K, 0);
			bp_child_count.assign(K, 0);

			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				int u_local = bp_local_index[u];
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int c = scc.dag_data[ei];
					int c_local = bp_local_index[c];
					if (c_local < 0) continue;
					bp_edge_parent.push_back(u);
					bp_edge_child.push_back(c);
					bp_edge_parent_local.push_back(u_local);
					bp_edge_child_local.push_back(c_local);
					bp_child_count[u_local]++;
					bp_parent_count[c_local]++;
				}
			}
			int nBpEdges = static_cast<int>(bp_edge_parent.size());

			bp_parent_offsets.assign(K + 1, 0);
			bp_child_offsets.assign(K + 1, 0);
			for (int i = 0; i < K; ++i) {
				bp_parent_offsets[i + 1] = bp_parent_offsets[i] + bp_parent_count[i];
				bp_child_offsets[i + 1] = bp_child_offsets[i] + bp_child_count[i];
			}
			bp_parent_data.assign(bp_parent_offsets[K], -1);
			bp_child_data.assign(bp_child_offsets[K], -1);
			bp_parent_cursor = bp_parent_offsets;
			bp_child_cursor = bp_child_offsets;
			for (int eid = 0; eid < nBpEdges; ++eid) {
				int parent_local = bp_edge_parent_local[eid];
				int child_local = bp_edge_child_local[eid];
				bp_child_data[bp_child_cursor[parent_local]++] = eid;
				bp_parent_data[bp_parent_cursor[child_local]++] = eid;
			}

			if (nBpEdges == 0) {
				// 无内部边：每个节点 64 位独立均匀随机
				for (int ni = node_begin; ni < node_end; ++ni) {
					int u = dag_plan.component_nodes[ni];
					result.super_state64[u] = static_cast<uint64_t>(gen());
				}
			} else {
				bp_mu.assign(nBpEdges, 0.5);
				bp_eta.assign(nBpEdges, 2.0);
				bp_new_mu.assign(nBpEdges, 0.0);
				bp_new_eta.assign(nBpEdges, 0.0);

				for (int iter = 0; iter < p.bp_iters; ++iter) {
					for (int eid = 0; eid < nBpEdges; ++eid) {
						int par_local = bp_edge_parent_local[eid];
						int chi_local = bp_edge_child_local[eid];

						double R = 1.0;
						for (int ii = bp_parent_offsets[par_local]; ii < bp_parent_offsets[par_local + 1]; ++ii) {
							int pe = bp_parent_data[ii];
							R = mul_sat(R, bp_mu[pe]);
							if (R >= kBpProdCap) break;
						}
						if (R < kBpProdCap) {
							for (int ii = bp_child_offsets[par_local]; ii < bp_child_offsets[par_local + 1]; ++ii) {
								int ce = bp_child_data[ii];
								if (ce != eid) {
									R = mul_sat(R, bp_eta[ce]);
									if (R >= kBpProdCap) break;
								}
							}
						}
						bp_new_mu[eid] = safe_prob_from_ratio(R);

						R = 1.0;
						for (int ii = bp_parent_offsets[chi_local]; ii < bp_parent_offsets[chi_local + 1]; ++ii) {
							int pe = bp_parent_data[ii];
							if (pe != eid) {
								R = mul_sat(R, bp_mu[pe]);
								if (R >= kBpProdCap) break;
							}
						}
						if (R < kBpProdCap) {
							for (int ii = bp_child_offsets[chi_local]; ii < bp_child_offsets[chi_local + 1]; ++ii) {
								int ce = bp_child_data[ii];
								R = mul_sat(R, bp_eta[ce]);
								if (R >= kBpProdCap) break;
							}
						}
						bp_new_eta[eid] = (R >= kBpProdCap) ? kBpProdCap : (1.0 + R);
					}
					bp_mu.swap(bp_new_mu);
					bp_eta.swap(bp_new_eta);
				}

				// 拓扑序条件采样：按拓扑顺序生成 64 个独立样本
				const int topo_begin = comp.topo_begin;
				const int topo_end = comp.topo_end;
				for (int ti = topo_begin; ti < topo_end; ++ti) {
					int u = dag_plan.topo_data[ti];
					// z_mask：标记"有父为0"的链位
					uint64_t z_mask = 0ULL;
					for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
						int pred = scc.dag_pred_data[ei];
						z_mask |= ~result.super_state64[pred];
					}
					// z_mask 位为 1 的链强制 0
					int u_local = bp_local_index[u];
					double R_cond = 1.0;
					for (int ii = bp_child_offsets[u_local]; ii < bp_child_offsets[u_local + 1]; ++ii) {
						int ce = bp_child_data[ii];
						R_cond = mul_sat(R_cond, bp_eta[ce]);
						if (R_cond >= kBpProdCap) break;
					}
					double p1 = safe_prob_from_ratio(R_cond);
					uint64_t sample_bits = bernoulli_bits_64(p1, gen);
					result.super_state64[u] = (~z_mask) & sample_bits;
				}
			}

			for (int li = 0; li < K; ++li) {
				int u = dag_plan.component_nodes[node_begin + li];
				bp_local_index[u] = -1;
			}

		} else if (p.init_mode == 2) {
			// 雪崩初始化（串行 64 次单链雪崩逐位打包）：
			// 每条链独立执行完整单链雪崩流程（独立底色 + BFS 传播），
			// 保证与单链版语义完全等价。
			std::vector<int8_t> tmp_state(N, 0);
			std::uniform_int_distribution<int> idx_dist_comp(0, K - 1);
			long long avalanche_steps = std::max(10LL, 2LL * static_cast<long long>(K));
			if (static_cast<int>(visit_mark.size()) < N) visit_mark.resize(N, 0);
			avalanche_q.clear();
			avalanche_q.reserve(K);

			auto next_visit_token = [&]() {
				if (ws.mcmc_visit_token == std::numeric_limits<int>::max()) {
					std::fill(visit_mark.begin(), visit_mark.end(), 0);
					ws.mcmc_visit_token = 1;
				}
				return ws.mcmc_visit_token++;
			};

			for (int bit = 0; bit < 64; ++bit) {
				// 每条链独立随机底色
				int extreme_state = static_cast<int>(gen() & 1);
				for (int ni = node_begin; ni < node_end; ++ni) {
					int u = dag_plan.component_nodes[ni];
					tmp_state[u] = static_cast<int8_t>(extreme_state);
				}

				for (long long st = 0; st < avalanche_steps; ++st) {
					int start = dag_plan.component_nodes[node_begin + idx_dist_comp(gen)];
					bool force_one = (gen() & 1) == 1;
					int mark = next_visit_token();

					avalanche_q.clear();
					avalanche_q.push_back(start);
					visit_mark[start] = mark;

					for (int head2 = 0; head2 < static_cast<int>(avalanche_q.size()); ++head2) {
						int u = avalanche_q[head2];
						if (force_one) {
							tmp_state[u] = 1;
							for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
								int parent = scc.dag_pred_data[ei];
								if (visit_mark[parent] != mark) {
									visit_mark[parent] = mark;
									avalanche_q.push_back(parent);
								}
							}
						} else {
							tmp_state[u] = 0;
							for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
								int child = scc.dag_data[ei];
								if (visit_mark[child] != mark) {
									visit_mark[child] = mark;
									avalanche_q.push_back(child);
								}
							}
						}
					}
				}

				// 把这条链的结果打包进对应 bit
				uint64_t bit_mask = 1ULL << bit;
				for (int ni = node_begin; ni < node_end; ++ni) {
					int u = dag_plan.component_nodes[ni];
					if (tmp_state[u]) {
						result.super_state64[u] |= bit_mask;
					} else {
						result.super_state64[u] &= ~bit_mask;
					}
				}
			}

		} else if (p.init_mode == 3) {
			// 独立均匀随机初始化：每个节点 64 位独立随机
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				result.super_state64[u] = static_cast<uint64_t>(gen());
			}

		} else {
			// 对称极端初始化（默认 / init_mode==0）：
			// 64 条链各自独立选择全 0 或全 1 底色。
			// extreme_mask 的每一位决定对应链的底色。
			uint64_t extreme_mask = static_cast<uint64_t>(gen());
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				result.super_state64[u] = extreme_mask;
			}
		}

		// 统一计算 ACF 投影（bit-0 链）
		double running_projection = 0.0;
		if (need_acf_projection) {
			for (int ni = node_begin; ni < node_end; ++ni) {
				int u = dag_plan.component_nodes[ni];
				if (result.super_state64[u] & 1ULL) {
					running_projection += gauss_w[u];
				}
			}
		}

		if (max_level < 0) continue;
		const int* local_level_offsets = dag_plan.level_offsets_data.data() + level_offsets_begin;

		// 位并行更新规则：
		// z_mask 标记“有父为0”的链位，o_mask 标记“有子为1”的链位，
		// 再与随机位 r_mask 组合得到每个位链的新状态。
		auto update_one_level = [&](int L) {
			for (int i = local_level_offsets[L]; i < local_level_offsets[L + 1]; ++i) {
				int u = dag_plan.level_data[level_data_begin + i];
				uint64_t z_mask = 0ULL;
				uint64_t o_mask = 0ULL;
				for (int ei = scc.dag_pred_offsets[u]; ei < scc.dag_pred_offsets[u + 1]; ++ei) {
					int parent = scc.dag_pred_data[ei];
					z_mask |= ~result.super_state64[parent];
				}
				for (int ei = scc.dag_offsets[u]; ei < scc.dag_offsets[u + 1]; ++ei) {
					int child = scc.dag_data[ei];
					o_mask |= result.super_state64[child];
				}
				uint64_t r_mask = static_cast<uint64_t>(gen());
				// new_val 每一位的逻辑与标量版一致：
				// - z_mask 位为1 => 该链有“父为0”，故该位强制 0（由 ~z_mask 实现）；
				// - 否则若 o_mask 位为1（有子为1）则强制 1；
				// - 两者都不触发时，用 r_mask 的随机位决定。
				uint64_t old_val = result.super_state64[u];
				uint64_t new_val = (~z_mask) & (o_mask | r_mask);
				if (need_acf_projection) {
					int old_bit0 = static_cast<int>(old_val & 1ULL);
					int new_bit0 = static_cast<int>(new_val & 1ULL);
					if (old_bit0 != new_bit0) {
						running_projection += (new_bit0 ? gauss_w[u] : -gauss_w[u]);
					}
				}
				result.super_state64[u] = new_val;
			}
		};

		auto one_full_sweep = [&]() {
			for (int L = 0; L <= max_level; ++L) update_one_level(L);
			for (int L = max_level; L >= 0; --L) update_one_level(L);
		};

		auto comp_projection = [&]() -> double {
			return running_projection;
		};

		int total_target_sweeps = std::max(0, p.mcmc_factor);
		int probe_done_sweeps = 0;
		if (p.use_acf_auto_sweeps) {
			if (!shared_auto_sweeps_ready) {
				if (p.cached_tau_int > 0.0) {
					int est_sweeps = static_cast<int>(std::ceil(p.acf_tau_multiplier * p.cached_tau_int));
					est_sweeps = std::max(20, est_sweeps);
					if (p.acf_max_sweeps > 0) est_sweeps = std::min(est_sweeps, p.acf_max_sweeps);
					shared_auto_sweeps = est_sweeps;
					shared_auto_sweeps_ready = true;
					result.estimated_tau_int = p.cached_tau_int;
				} else {
					int probe_burnin_sweeps = std::max(0, p.acf_probe_burnin_sweeps);
					int probe_sampling_sweeps = std::max(8, p.acf_probe_sweeps);
					x_traj.clear();
					x_traj.reserve(probe_sampling_sweeps);

					for (int s = 0; s < probe_burnin_sweeps; ++s) {
						one_full_sweep();
						probe_done_sweeps++;
					}

					for (int s = 0; s < probe_sampling_sweeps; ++s) {
						one_full_sweep();
						x_traj.push_back(comp_projection());
						probe_done_sweeps++;
					}

					acf_fft_biased_inplace(x_traj, acf_buf, fft_buffer);
					TauResult tau = estimate_tau_sokal(acf_buf, 5.0);
					if (p.acf_tau_estimator == 1) {
						TauResult tau_geyer = estimate_tau_geyer_ips(acf_buf);
						if (tau_geyer.tau_int > tau.tau_int) tau = tau_geyer;
					}

					if (tau.tau_int > static_cast<double>(probe_sampling_sweeps) / 50.0) {
						result.acf_probe_too_short = true;
						debug_log(p.debug_output,
							"Warning: Probe length may be too short for accurate tau estimation.");
					}
					result.estimated_tau_int = tau.tau_int;

					int est_sweeps = static_cast<int>(std::ceil(p.acf_tau_multiplier * tau.tau_int));
					est_sweeps = std::max(20, est_sweeps);
					if (p.acf_max_sweeps > 0) est_sweeps = std::min(est_sweeps, p.acf_max_sweeps);

					shared_auto_sweeps = est_sweeps;
					shared_auto_sweeps_ready = true;
				}
			}
			total_target_sweeps = shared_auto_sweeps;
		} else {
			total_target_sweeps = std::max(0, p.mcmc_factor);
		}

		int remain_sweeps = std::max(0, total_target_sweeps - probe_done_sweeps);
		sample_total_sweeps += static_cast<long long>(probe_done_sweeps + remain_sweeps);
		need_acf_projection = false; // 探针结束，剩余 sweeps 无需维护投影

		for (int sweep = 0; sweep < remain_sweeps; ++sweep) {
			one_full_sweep();
		}
	}

	result.mean_sweeps_used = (sample_num_components > 0)
		? static_cast<double>(sample_total_sweeps) / static_cast<double>(sample_num_components)
		: 0.0;
	result.shared_sweeps = has_mcmc_component
		? (p.use_acf_auto_sweeps ? shared_auto_sweeps : std::max(0, p.mcmc_factor))
		: 0;

	return result;
}

// ========================== 模块七：结果组装与序参量计算 ==========================

// 单张图聚合结果（对该图上的所有热样本统计）。
struct MvcResult {
	std::vector<double> R_thermal_samples;
	int N_1A = 0;
	int N_1B = 0;
	int N_0A = 0;
	int N_0B = 0;
	int N_starA = 0;
	int N_starB = 0;
	std::vector<int> all_wcc_sizes;
	double mean_R_thermal = 0.0;
	double mean_absR_thermal = 0.0;
	double mean_R2_thermal = 0.0;
	double mean_R3_thermal = 0.0;
	double mean_absR3_thermal = 0.0;
	double mean_R4_thermal = 0.0;
	int core_edges = 0;
	double sweeps_used = 0.0;
	int max_wcc_size = 0;
	int shared_sweeps = 0;
	double estimated_tau_int = 0.0;
	bool acf_probe_too_short = false;
};

// run_single_sample 的返回值：用于外层无序平均和写盘。
struct SingleSampleResult {
	std::vector<double> R_thermal_samples;
	int N_1A = 0;
	int N_1B = 0;
	int N_0A = 0;
	int N_0B = 0;
	int N_starA = 0;
	int N_starB = 0;
	std::vector<int> all_wcc_sizes;
	double mean_R_thermal = 0.0;
	double mean_absR_thermal = 0.0;
	double mean_R2_thermal = 0.0;
	double mean_R3_thermal = 0.0;
	double mean_absR3_thermal = 0.0;
	double mean_R4_thermal = 0.0;
	double sweeps_used = 0.0;
	int max_wcc_size = 0;
	int shared_sweeps = 0;
	double estimated_tau_int = 0.0;
	bool acf_probe_too_short = false;
};

// 根据“冻结贡献 + 核边超点状态”计算单次样本的序参量 R。
double compute_R_from_super_state(
	const SccDagResult& scc,
	int core_edges,
	int frozen_u_in,
	int frozen_v_in,
	const std::vector<int8_t>& super_state)
{
	long long total_mvc_nodes = static_cast<long long>(frozen_u_in) +
		static_cast<long long>(frozen_v_in) +
		static_cast<long long>(core_edges);
	if (total_mvc_nodes <= 0) return 0.0;

	long long sum_R = static_cast<long long>(frozen_u_in) - static_cast<long long>(frozen_v_in);
	for (int sid = 0; sid < scc.nSuper; ++sid) {
		long long weight = static_cast<long long>(scc.super_weights[sid]);
		int val = super_state[sid];
		sum_R += (val == 1) ? weight : -weight;
	}
	return static_cast<double>(sum_R) / static_cast<double>(total_mvc_nodes);
}

// 在一张固定晶格上执行完整 MVC 采样流程并返回热统计量。
MvcResult sample_mvc_mcmc(
	const std::vector<uint8_t>& lattice,
	const std::vector<int>* active_sites,
	const Params& p,
	long long total_nodes,
	RNG& gen)
{
	MvcResult out;
	int thermal_samples = std::max(1, p.num_thermal_samples);
	out.R_thermal_samples.clear();
	out.R_thermal_samples.reserve(static_cast<size_t>(thermal_samples));

	// Step 1: 构建二分图
	BipartiteGraph& g = tls_bipartite_graph;
	{
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::BuildBipartiteGraph);
		build_bipartite_graph(lattice, active_sites, p, total_nodes, g);
	}

	if (g.nU == 0 && g.nV == 0) {
		out.R_thermal_samples.assign(static_cast<size_t>(thermal_samples), 0.0);
		return out;
	}

	// Step 2: Hopcroft-Karp 最大匹配
	HopcroftKarpSolver hk;
	{
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::HopcroftKarp);
		hk.run(g.nU, g.nV, g.adjU_offsets, g.adjU_data, p.hk_use_greedy_init);
	}
	const std::vector<int>& match_U = tls_ws.hk_match_u;
	const std::vector<int>& match_V = tls_ws.hk_match_v;

	// Step 3: 冻结传播
	FreezingResult& fr = [&]() -> FreezingResult& {
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::Freezing);
		return run_freezing_bfs(g, match_U, match_V);
	}();
	int frozen_u_in = 0;
	int frozen_v_in = 0;
	for (int u = 0; u < g.nU; ++u) {
		if (fr.stateU[u] == IN_MVC) {
			frozen_u_in++;
			out.N_1A++;
		} else if (fr.stateU[u] == NOT_IN_MVC) {
			out.N_0A++;
		} else {
			out.N_starA++;
		}
	}
	for (int v = 0; v < g.nV; ++v) {
		if (fr.stateV[v] == IN_MVC) {
			frozen_v_in++;
			out.N_1B++;
		} else if (fr.stateV[v] == NOT_IN_MVC) {
			out.N_0B++;
		} else {
			out.N_starB++;
		}
	}

	// Step 4: SCC 缩点
	SccDagResult& scc = tls_scc_dag;
	{
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::BuildDependencySccDag);
		build_dependency_scc_dag(g, match_U, fr, scc);
	}
	McmcDagPrecomputed& dag_plan = tls_mcmc_dag_plan;
	{
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::PrecomputeMcmcDag);
		precompute_mcmc_dag_structure(scc, dag_plan);
	}
	out.all_wcc_sizes.clear();
	out.all_wcc_sizes.reserve(dag_plan.components.size());
	for (const auto& comp : dag_plan.components) {
		out.all_wcc_sizes.push_back(comp.size);
	}
	out.core_edges = static_cast<int>(fr.core_edges.size());

	// Step 5: MCMC 采样
	// 统一累计每个热样本的矩，后续可直接得到均值、Binder 累积量与磁化率分解所需量。
	double sum_R = 0.0;
	double sum_absR = 0.0;
	double sum_R2 = 0.0;
	double sum_R3 = 0.0;
	double sum_absR3 = 0.0;
	double sum_R4 = 0.0;
	double sum_total_sweeps_per_restart = 0.0;
	std::vector<int8_t> chain_super_state;

	// 三种热采样模式：
	// 1) MSC64 多重重启（最快，64 条链并行）；
	// 2) 普通多重重启（每个热样本独立 burn-in）；
	// 3) 单链 decorrelation 采样（先 burn-in，再按间隔采样）。
	bool use_msc64_multi_start = p.multi_start_mode && p.multi_spin_coding_mode && (thermal_samples % 64 == 0);

	{
		ScopedBenchmarkPhase benchmark_scope(BenchmarkPhase::SampleMvcMcmc);
		if (use_msc64_multi_start) {
		// 位并行批处理：每个 batch 产出 64 个热样本。
		int batch_count = thermal_samples / 64;
		long long total_mvc_nodes = static_cast<long long>(frozen_u_in) +
			static_cast<long long>(frozen_v_in) +
			static_cast<long long>(out.core_edges);

		for (int b = 0; b < batch_count; ++b) {
			McmcBatch64Result batch_res = sample_super_states_mcmc_msc64(scc, dag_plan, p, gen);

			if (b == 0) {
				out.max_wcc_size = batch_res.max_wcc_size;
				out.shared_sweeps = batch_res.shared_sweeps;
				out.estimated_tau_int = batch_res.estimated_tau_int;
				out.acf_probe_too_short = batch_res.acf_probe_too_short;
			}
			// 同一张图结构 → 后续 batch 的 wcc_size/sweeps/tau/probe 相同，无需再聚合

			std::array<long long, 64> sum_R_bits{};
			if (total_mvc_nodes > 0) {
				const long long base_sum_R =
					static_cast<long long>(frozen_u_in) -
					static_cast<long long>(frozen_v_in) -
					static_cast<long long>(out.core_edges);
				sum_R_bits.fill(base_sum_R);
				for (int sid = 0; sid < scc.nSuper; ++sid) {
					uint64_t bits = batch_res.super_state64[sid];
					const long long delta = 2LL * static_cast<long long>(scc.super_weights[sid]);
					while (bits != 0ULL) {
						unsigned long bit = static_cast<unsigned long>(__builtin_ctzll(bits));
						sum_R_bits[static_cast<size_t>(bit)] += delta;
						bits &= (bits - 1ULL);
					}
				}
			}

			for (int bit = 0; bit < 64; ++bit) {
				double R = 0.0;
				if (total_mvc_nodes > 0) {
					R = static_cast<double>(sum_R_bits[static_cast<size_t>(bit)]) /
						static_cast<double>(total_mvc_nodes);
				}
				double absR = std::abs(R);
				double R2 = R * R;
				double R3 = R2 * R;
				double absR3 = R2 * absR;
				double R4 = R2 * R2;
				out.R_thermal_samples.push_back(R);
				sum_R += R;
				sum_absR += absR;
				sum_R2 += R2;
				sum_R3 += R3;
				sum_absR3 += absR3;
				sum_R4 += R4;
			}

			sum_total_sweeps_per_restart += batch_res.mean_sweeps_used * 64.0;
		}

		out.mean_R_thermal = sum_R / static_cast<double>(thermal_samples);
		out.mean_absR_thermal = sum_absR / static_cast<double>(thermal_samples);
		out.mean_R2_thermal = sum_R2 / static_cast<double>(thermal_samples);
		out.mean_R3_thermal = sum_R3 / static_cast<double>(thermal_samples);
		out.mean_absR3_thermal = sum_absR3 / static_cast<double>(thermal_samples);
		out.mean_R4_thermal = sum_R4 / static_cast<double>(thermal_samples);
		out.sweeps_used = sum_total_sweeps_per_restart / static_cast<double>(thermal_samples);
	} else if (p.multi_start_mode) {
		// 标准多重重启：每次从新初态进入，降低链间相关。
		for (int t = 0; t < thermal_samples; ++t) {
			// 每个热样本独立随机重启：新初态 -> burn-in -> 直接取样
			chain_super_state.clear();
			McmcResult burnin_res = sample_super_states_mcmc(scc, dag_plan, p, gen, &chain_super_state);

			if (t == 0) {
				out.max_wcc_size = burnin_res.max_wcc_size;
				out.shared_sweeps = burnin_res.shared_sweeps;
				out.estimated_tau_int = burnin_res.estimated_tau_int;
				out.acf_probe_too_short = burnin_res.acf_probe_too_short;
			}
			// 同一张图结构 → 后续 t 的 wcc_size/sweeps/tau/probe 相同，无需再聚合

			double R = compute_R_from_super_state(
				scc,
				out.core_edges,
				frozen_u_in,
				frozen_v_in,
				burnin_res.super_state);
			double absR = std::abs(R);
			double R2 = R * R;
			double R3 = R2 * R;
			double absR3 = R2 * absR;
			double R4 = R2 * R2;
			out.R_thermal_samples.push_back(R);
			sum_R += R;
			sum_absR += absR;
			sum_R2 += R2;
			sum_R3 += R3;
			sum_absR3 += absR3;
			sum_R4 += R4;
			sum_total_sweeps_per_restart += burnin_res.mean_sweeps_used;
		}

		out.mean_R_thermal = sum_R / static_cast<double>(thermal_samples);
		out.mean_absR_thermal = sum_absR / static_cast<double>(thermal_samples);
		out.mean_R2_thermal = sum_R2 / static_cast<double>(thermal_samples);
		out.mean_R3_thermal = sum_R3 / static_cast<double>(thermal_samples);
		out.mean_absR3_thermal = sum_absR3 / static_cast<double>(thermal_samples);
		out.mean_R4_thermal = sum_R4 / static_cast<double>(thermal_samples);
		out.sweeps_used = sum_total_sweeps_per_restart / static_cast<double>(thermal_samples);
	} else {
		// 单链模式：先一次 burn-in，再按 decorrelation_sweeps 抽样。
		McmcResult burnin_res = sample_super_states_mcmc(scc, dag_plan, p, gen, &chain_super_state);
		out.max_wcc_size = burnin_res.max_wcc_size;
		out.shared_sweeps = burnin_res.shared_sweeps;
		out.estimated_tau_int = burnin_res.estimated_tau_int;
		out.acf_probe_too_short = burnin_res.acf_probe_too_short;

		double decorrelation_base = 0.0;
		if (p.use_acf_auto_sweeps) {
			if (burnin_res.estimated_tau_int > 0.0 && std::isfinite(burnin_res.estimated_tau_int)) {
				decorrelation_base = burnin_res.estimated_tau_int;
			} else if (burnin_res.shared_sweeps > 0 && p.acf_tau_multiplier > 0.0) {
				decorrelation_base = static_cast<double>(burnin_res.shared_sweeps) / p.acf_tau_multiplier;
			} else {
				decorrelation_base = static_cast<double>(std::max(1, burnin_res.shared_sweeps));
			}
		} else {
			decorrelation_base = static_cast<double>(std::max(1, p.mcmc_factor));
		}

		int decorrelation_sweeps = static_cast<int>(std::ceil(
			std::max(0.0, p.decorrelation_multiplier) * decorrelation_base));
		decorrelation_sweeps = std::max(1, decorrelation_sweeps);

		Params p_measure = p;
		p_measure.use_acf_auto_sweeps = false;
		p_measure.mcmc_factor = decorrelation_sweeps;
		p_measure.cached_tau_int = 0.0;

		double sum_measure_sweeps = 0.0;
		for (int t = 0; t < thermal_samples; ++t) {
			McmcResult step_res = sample_super_states_mcmc(scc, dag_plan, p_measure, gen, &chain_super_state);
			double R = compute_R_from_super_state(
				scc,
				out.core_edges,
				frozen_u_in,
				frozen_v_in,
				step_res.super_state);
			double absR = std::abs(R);
			double R2 = R * R;
			double R3 = R2 * R;
			double absR3 = R2 * absR;
			double R4 = R2 * R2;
			out.R_thermal_samples.push_back(R);
			sum_R += R;
			sum_absR += absR;
			sum_R2 += R2;
			sum_R3 += R3;
			sum_absR3 += absR3;
			sum_R4 += R4;
			sum_measure_sweeps += step_res.mean_sweeps_used;
		}

		out.mean_R_thermal = sum_R / static_cast<double>(thermal_samples);
		out.mean_absR_thermal = sum_absR / static_cast<double>(thermal_samples);
		out.mean_R2_thermal = sum_R2 / static_cast<double>(thermal_samples);
		out.mean_R3_thermal = sum_R3 / static_cast<double>(thermal_samples);
		out.mean_absR3_thermal = sum_absR3 / static_cast<double>(thermal_samples);
		out.mean_R4_thermal = sum_R4 / static_cast<double>(thermal_samples);
		out.sweeps_used = burnin_res.mean_sweeps_used +
			(sum_measure_sweeps / static_cast<double>(thermal_samples));
	}
	}

	return out;
}
// 运行单次采样：生成晶格 -> 局域规则 -> 提取 LCC -> MVC 采样与统计。
// 该函数是外层并行循环的“最小工作单元”。
SingleSampleResult run_single_sample(
	const Params& p_base,
	long long total_nodes,
	uint64_t sample_seed,
	const std::vector<uint64_t>& interior_mask,
	double rho_override,
	int num_thermal_samples_override,
	bool force_fixed_sweeps,
	int fixed_sweeps,
	double cached_tau_override)
{
	Params p;
	// 显式逐字段复制模板参数，再按本次任务覆盖局部字段，避免影响外层模板状态。
	p.d = p_base.d;
	p.L = p_base.L;
	p.W = p_base.W;
	p.rho = rho_override;
	p.seed = p_base.seed;
	p.periodic_boundary = p_base.periodic_boundary;
	p.debug_output = false;
	p.mcmc_factor = force_fixed_sweeps ? fixed_sweeps : p_base.mcmc_factor;
	p.use_acf_auto_sweeps = force_fixed_sweeps ? false : p_base.use_acf_auto_sweeps;
	p.acf_probe_sweeps = p_base.acf_probe_sweeps;
	p.acf_probe_burnin_sweeps = p_base.acf_probe_burnin_sweeps;
	p.acf_max_sweeps = p_base.acf_max_sweeps;
	p.acf_tau_multiplier = p_base.acf_tau_multiplier;
	p.acf_tau_estimator = p_base.acf_tau_estimator;
	p.cached_tau_int = cached_tau_override;
	p.init_mode = p_base.init_mode;
	p.bp_iters = p_base.bp_iters;
	p.num_thermal_samples = (num_thermal_samples_override > 0)
		? num_thermal_samples_override
		: p_base.num_thermal_samples;
	p.multi_start_mode = p_base.multi_start_mode;
	p.multi_spin_coding_mode = p_base.multi_spin_coding_mode;
	p.decorrelation_multiplier = p_base.decorrelation_multiplier;
	p.hk_use_greedy_init = p_base.hk_use_greedy_init;
	p.L_multipliers_ref = (p_base.L_multipliers_ref != nullptr)
		? p_base.L_multipliers_ref
		: &p_base.L_multipliers;
	p.interior_mask = &interior_mask;
	p.parity_cache = p_base.parity_cache;

	RNG gen(sample_seed);
	ThreadLocalWorkspace& ws = tls_ws;

	generate_initial_lattice_inplace(ws.lattice, total_nodes, p.rho, gen);
	apply_rules(ws.lattice, p, total_nodes);
	collect_active_sites(ws.lattice, ws.active_sites);
	extract_largest_connected_component(
		ws.lattice,
		p,
		total_nodes,
		ws.active_sites,
		ws.component,
		ws.queue,
		ws.best_component);
	MvcResult mvc = sample_mvc_mcmc(ws.lattice, &ws.best_component, p, total_nodes, gen);
	SingleSampleResult out;
	out.R_thermal_samples = std::move(mvc.R_thermal_samples);
	out.N_1A = mvc.N_1A;
	out.N_1B = mvc.N_1B;
	out.N_0A = mvc.N_0A;
	out.N_0B = mvc.N_0B;
	out.N_starA = mvc.N_starA;
	out.N_starB = mvc.N_starB;
	out.all_wcc_sizes = std::move(mvc.all_wcc_sizes);
	out.mean_R_thermal = mvc.mean_R_thermal;
	out.mean_absR_thermal = mvc.mean_absR_thermal;
	out.mean_R2_thermal = mvc.mean_R2_thermal;
	out.mean_R3_thermal = mvc.mean_R3_thermal;
	out.mean_absR3_thermal = mvc.mean_absR3_thermal;
	out.mean_R4_thermal = mvc.mean_R4_thermal;
	out.sweeps_used = mvc.sweeps_used;
	out.max_wcc_size = mvc.max_wcc_size;
	out.shared_sweeps = mvc.shared_sweeps;
	out.estimated_tau_int = mvc.estimated_tau_int;
	out.acf_probe_too_short = mvc.acf_probe_too_short;
	return out;
}

// ========================== 主函数 ==========================

// PGO:
// g++ -O3 -march=native -fopenmp  -fprofile-generate "MVC-equ-MCMC-LSB-AD-ACF-mul-L-rho-0308-2.cpp" -o MVC_MCMC; ./MVC_MCMC
// g++ -O3 -march=native -fopenmp -fprofile-use "MVC-equ-MCMC-LSB-AD-ACF-mul-L-rho-0308-2.cpp" -o MVC_MCMC; ./MVC_MCMC

// "-Wl,--stack,536870912"
// g++ -O3 -march=native -fopenmp -std=c++17 "MVC-equ-MCMC-LSB-AD-ACF-mul-L-rho-2d-0322.cpp" -o MVC_MCMC; ./MVC_MCMC

// 主程序职责：
// - 扫描 L 与 rho 网格；
// - 处理断点续跑、tau 缓存、pilot 估计；
// - 并行运行样本并写出多份结果文件。
int main() {
	// 默认参数
	// 在此处设置需要遍历的 L 列表（不再从 argc 读取单个 L）
	const std::vector<int> L_list = { 300,400, 500,600, 700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000};//{ 200, 300,400, 500,600, 700,800,900,1000,1100};//,1200,1300,1400,1500,1600,1700,1800,1900,2000}; // 400,800,1000,1200,1400,1600,1800,
	 // {60,80,100,120,140,160,180,200}
	constexpr int W = -1; // W <= 0 时自动令 W = L（各向同性）
	uint64_t base_seed = 114514ULL;
	constexpr bool periodic_boundary = kPeriodicBoundaryCompileTime; // 周期性边界
	constexpr int num_samples = 5000;    // 无序系统采样数
	constexpr int num_thermal_samples = 64*2; // 每张图上的热采样个数
	constexpr bool multi_start_mode = true; // true=多重独立重启，降低初态记忆;每个热样本独立随机重启并 burn-in 后采样；=0：单链连续热采样
	constexpr bool multi_spin_coding_mode = true; // true=允许在 multi_start 且热样本数为64倍数时启用 MSC64
	constexpr bool use_common_random_numbers_across_rho = true; // true=同一 sample_id 在所有 rho 下复用同一随机数序列（CRN）
	constexpr int num_rho = 90;
	constexpr double rho_min =  0.08;//0.13;// 0.08;//
	constexpr double rho_max =  0.2475;//0.20;// 0.2475;//
	// 密集区参数：rho_dense_min/max < 0 表示不启用密集区，按原来的均匀分布
	constexpr double rho_dense_min = 0.08; // 密集区左端点（开区间）
	constexpr double rho_dense_max = 0.16; // 密集区右端点（开区间）
	constexpr int num_rho_dense = 60;       // 密集区内的点数
	constexpr int mcmc_factor = 10000; // Level-Set Sweep 次数
	//----------------------------------------------------------------------------------------------
	constexpr bool write_R_mcmc_all = false;  // 是否写入 R_mcmc_all-d-{d}.txt（每样本全部热样本 R 值）
	constexpr bool write_WCC_sizes = false;   // 是否写入 WCC_sizes-d-{d}.txt（每样本 WCC 尺寸列表）
	constexpr bool write_absR_thermal = true; // 是否写入 absR_thermal-d-{d}.txt（每行: d L W rho + N_samples个<|R|>热平均值）
	constexpr bool resume_mode = true; // 断点续跑开关： resume=1: 从统计文件加载已完成点并跳过；resume=0: 覆盖重跑
	constexpr bool use_tau_cache = true;        // 自动模式：是否读取/使用 tau 缓存
	constexpr bool write_tau_cache = true;      // 自动模式：是否写入 tau 缓存文件
	constexpr bool force_rebuild_tau_cache = false; // true=强制重建 tau_cache（删除旧文件并重写表头）
	std::string tau_cache_file;        // tau 缓存文件路径
	//----------------------------------------------------------------------------------------------
	constexpr bool use_acf_auto_sweeps = true; // false=固定 sweeps, true=ACF 自动 sweeps ; 固定 sweeps（mcmc_factor）；=1: 先短链估计 ACF 再自动决定 sweeps
	constexpr int acf_probe_sweeps = 2000;       // 自动模式：ACF 采样段长度（仅该段用于估计 tau）
	constexpr int acf_probe_burnin_sweeps = 1000; // 自动模式：ACF 探针热化长度（不进入 ACF）
	constexpr int acf_max_sweeps = 10000;        // 自动模式：总 sweeps 上限（含 probe）
	constexpr double acf_tau_multiplier = 50.0; // 自动模式：总 sweeps ~= acf_tau_multiplier * tau_int（更保守）
	constexpr double decorrelation_multiplier = 4.0; // 热采样间隔倍数。采样间隔约为 2*tau_int（固定模式为 2*mcmc_factor）//   固定模式：decorrelation_sweeps = decorrelation_multiplier * mcmc_factor
	constexpr int acf_tau_estimator = 1;         // 0=Sokal, 1=Geyer-IPS + Sokal(取较大)
	constexpr int acf_estimation_samples = 24*2;   // 自动模式：每个 rho 仅前若干样本计算 ACF，用于估计 sweeps
	constexpr double pilot_tau_percentile = 0.95; // pilot 聚合分位数（如 0.85 / 0.90 / 0.95）
	//-----------------------------------------------------------------------------------------------------
	constexpr bool hk_use_greedy_init = true; // HK 前是否启用 O(E) 贪心预匹配
	constexpr int init_mode = 2;    // 0=随机全0/全1, 1=BP采样, 2=雪崩动力学, 3=独立均匀随机
	constexpr int bp_iters = 100;   // BP 消息传递迭代次数（仅 init_mode=1 时生效）
    // Level-Set Block Gibbs：mcmc_factor 表示 Sweep 数。
    // 每个 Sweep = forward(0->max_level) + backward(max_level->0)。


	if (tau_cache_file.empty()) {
		tau_cache_file = "./data/MVS-MCMC/tau_cache-d-" + std::to_string(d) + ".txt";
	}

#if __cplusplus >= 201703L
	{
		std::error_code ec;
		std::filesystem::create_directories("./data/MVS-MCMC/", ec);
		if (ec) {
			std::cerr << "Warning: failed to create output directory ./data/MVS-MCMC/ : "
					  << ec.message() << std::endl;
		}
	}
#else
	std::cerr << "Warning: C++17 filesystem unavailable; please ensure ./data/MVS-MCMC/ exists before running." << std::endl;
#endif

	if (base_seed == 0) {
		std::random_device rd;
		base_seed = rd();
		std::cout << "Using random_device generated seed: " << base_seed << std::endl;
	} else {
		std::cout << "Using provided base seed: " << base_seed << std::endl;
	}

	// 构建 rho 列表
	std::vector<double> rho_list;
	rho_list.reserve(num_rho);
	if (rho_dense_min < 0 || rho_dense_max < 0 || num_rho_dense <= 0) {
		// 不启用密集区：全局均匀分布
		rho_list.resize(num_rho);
		for (int i = 0; i < num_rho; ++i) {
			rho_list[i] = (num_rho > 1) ? rho_min + i * (rho_max - rho_min) / (num_rho - 1) : rho_min;
		}
	} else {
		// 外侧区间 [rho_min, rho_dense_min] ∪ [rho_dense_max, rho_max]：均匀分布，包含两侧端点
		const int num_outer = num_rho - num_rho_dense;
		const double L1 = rho_dense_min - rho_min;
		const double L2 = rho_max - rho_dense_max;
		const double L_total = L1 + L2;
		// 按长度比例分配外侧点数（每段至少 2 个点以包含两端）
		int n1 = std::max(2, (int)std::round((double)num_outer * L1 / L_total));
		int n2 = num_outer - n1;
		if (n2 < 2) { n2 = 2; n1 = num_outer - 2; }
		// 左侧区间 [rho_min, rho_dense_min]
		for (int i = 0; i < n1; ++i) {
			rho_list.push_back(rho_min + i * L1 / (n1 - 1));
		}
		// 右侧区间 [rho_dense_max, rho_max]
		for (int i = 0; i < n2; ++i) {
			rho_list.push_back(rho_dense_max + i * L2 / (n2 - 1));
		}
		// 密集区 (rho_dense_min, rho_dense_max)：均匀分布，不含两侧端点
		const double L_dense = rho_dense_max - rho_dense_min;
		for (int i = 1; i <= num_rho_dense; ++i) {
			rho_list.push_back(rho_dense_min + i * L_dense / (num_rho_dense + 1));
		}
		// 排序
		std::sort(rho_list.begin(), rho_list.end());
	}

	if (L_list.empty()) {
		std::cerr << "Error: L_list is empty." << std::endl;
		return 1;
	}

	std::ostringstream lss;
	for (size_t i = 0; i < L_list.size(); ++i) {
		if (i) lss << ",";
		lss << L_list[i];
	}
	const std::string l_list_str = lss.str();

	std::string parameter_file = "./data/MVS-MCMC/parameter-d-" + std::to_string(d) + ".txt";
	std::ofstream fout_param(parameter_file, std::ios::trunc);
	if (!fout_param.is_open()) {
		std::cerr << "Error: cannot open parameter file: " << parameter_file << std::endl;
		return 1;
	}
	const auto run_epoch_seconds = std::chrono::duration_cast<std::chrono::seconds>(
		std::chrono::system_clock::now().time_since_epoch()
	).count();
	fout_param << "run_epoch_seconds=" << run_epoch_seconds << "\n";
	fout_param << "d=" << d << "\n";
	fout_param << "rng=" << kRngName << "\n";
	fout_param << "use_neighbor_fastpath=" << (use_neighbor_fastpath ? 1 : 0) << "\n";
	fout_param << "EXACT_ENUM_THRESHOLD=" << EXACT_ENUM_THRESHOLD << "\n";
	fout_param << "DFS_EXACT_MAX_K=" << DFS_EXACT_MAX_K << "\n";
	fout_param << "FRONTIER_DP_MAX_K=" << FRONTIER_DP_MAX_K << "\n";
	fout_param << "EXACT_ENUM_VALID_STATE_CAP=" << EXACT_ENUM_VALID_STATE_CAP << "\n";
	fout_param << "EXACT_ENUM_NODE_CAP=" << EXACT_ENUM_NODE_CAP << "\n";
	fout_param << "FRONTIER_DP_WIDTH_CAP=" << FRONTIER_DP_WIDTH_CAP << "\n";
	fout_param << "FRONTIER_DP_STATE_CAP=" << FRONTIER_DP_STATE_CAP << "\n";
	fout_param << "L_list=[" << l_list_str << "]\n";
	fout_param << "W=" << W << "\n";
	fout_param << "base_seed=" << base_seed << "\n";
	fout_param << "periodic_boundary=" << (periodic_boundary ? 1 : 0) << "\n";
	fout_param << "num_samples=" << num_samples << "\n";
	fout_param << "num_thermal_samples=" << num_thermal_samples << "\n";
	fout_param << "multi_start_mode=" << (multi_start_mode ? 1 : 0) << "\n";
	fout_param << "multi_spin_coding_mode=" << (multi_spin_coding_mode ? 1 : 0) << "\n";
	fout_param << "use_common_random_numbers_across_rho=" << (use_common_random_numbers_across_rho ? 1 : 0) << "\n";
	fout_param << "num_rho=" << num_rho << "\n";
	fout_param << "rho_min=" << rho_min << "\n";
	fout_param << "rho_max=" << rho_max << "\n";
	fout_param << "rho_dense_min=" << rho_dense_min << "\n";
	fout_param << "rho_dense_max=" << rho_dense_max << "\n";
	fout_param << "num_rho_dense=" << num_rho_dense << "\n";
	fout_param << "mcmc_factor=" << mcmc_factor << "\n";
	fout_param << "use_acf_auto_sweeps=" << (use_acf_auto_sweeps ? 1 : 0) << "\n";
	fout_param << "acf_probe_sweeps=" << acf_probe_sweeps << "\n";
	fout_param << "acf_probe_burnin_sweeps=" << acf_probe_burnin_sweeps << "\n";
	fout_param << "acf_max_sweeps=" << acf_max_sweeps << "\n";
	fout_param << "acf_tau_multiplier=" << acf_tau_multiplier << "\n";
	fout_param << "acf_tau_estimator=" << acf_tau_estimator << "\n";
	fout_param << "acf_estimation_samples=" << acf_estimation_samples << "\n";
	fout_param << "pilot_tau_percentile=" << pilot_tau_percentile << "\n";
	fout_param << "decorrelation_multiplier=" << decorrelation_multiplier << "\n";
	fout_param << "use_tau_cache=" << (use_tau_cache ? 1 : 0) << "\n";
	fout_param << "write_tau_cache=" << (write_tau_cache ? 1 : 0) << "\n";
	fout_param << "force_rebuild_tau_cache=" << (force_rebuild_tau_cache ? 1 : 0) << "\n";
	fout_param << "tau_cache_file=" << tau_cache_file << "\n";
	fout_param << "write_R_mcmc_all=" << (write_R_mcmc_all ? 1 : 0) << "\n";
	fout_param << "write_WCC_sizes=" << (write_WCC_sizes ? 1 : 0) << "\n";
	fout_param << "write_absR_thermal=" << (write_absR_thermal ? 1 : 0) << "\n";
	fout_param << "resume_mode=" << (resume_mode ? 1 : 0) << "\n";
	fout_param << "init_mode=" << init_mode << "\n";
	fout_param << "bp_iters=" << bp_iters << "\n";
	fout_param << "hk_use_greedy_init=" << (hk_use_greedy_init ? 1 : 0) << "\n";
	fout_param << "parameter_file=" << parameter_file << "\n";
	fout_param.flush();

	std::cout << "Parameters: d=" << d
			  << ", rng=" << kRngName
			  << ", use_neighbor_fastpath=" << (use_neighbor_fastpath ? 1 : 0)
			  << ", EXACT_ENUM_THRESHOLD=" << EXACT_ENUM_THRESHOLD
			  << ", DFS_EXACT_MAX_K=" << DFS_EXACT_MAX_K
			  << ", FRONTIER_DP_MAX_K=" << FRONTIER_DP_MAX_K
			  << ", EXACT_ENUM_VALID_STATE_CAP=" << EXACT_ENUM_VALID_STATE_CAP
			  << ", EXACT_ENUM_NODE_CAP=" << EXACT_ENUM_NODE_CAP
			  << ", FRONTIER_DP_WIDTH_CAP=" << FRONTIER_DP_WIDTH_CAP
			  << ", FRONTIER_DP_STATE_CAP=" << FRONTIER_DP_STATE_CAP
			  << ", L_list=[" << l_list_str << "]"
			  << ", W=" << W << " (<=0 means W=L)"
			  << ", base_seed=" << base_seed
			  << ", periodic_boundary=" << periodic_boundary
			  << ", num_samples=" << num_samples
			  << ", num_rho=" << num_rho
			  << ", rho_range=[" << rho_min << ", " << rho_max << "]"
			  << ", rho_dense=[" << rho_dense_min << ", " << rho_dense_max << "], num_rho_dense=" << num_rho_dense
			  << ", mcmc_factor=" << mcmc_factor
			  << ", use_acf_auto_sweeps=" << (use_acf_auto_sweeps ? 1 : 0)
			  << ", acf_probe_sweeps=" << acf_probe_sweeps
			  << ", acf_probe_burnin_sweeps=" << acf_probe_burnin_sweeps
			  << ", acf_max_sweeps=" << acf_max_sweeps
			  << ", acf_tau_multiplier=" << acf_tau_multiplier
			  << ", acf_tau_estimator=" << acf_tau_estimator
			  << ", acf_estimation_samples=" << acf_estimation_samples
			  << ", pilot_tau_percentile=" << pilot_tau_percentile
			  << ", use_tau_cache=" << (use_tau_cache ? 1 : 0)
			  << ", write_tau_cache=" << (write_tau_cache ? 1 : 0)
			  << ", force_rebuild_tau_cache=" << (force_rebuild_tau_cache ? 1 : 0)
			  << ", tau_cache_file=" << tau_cache_file
			  << ", init_mode=" << init_mode
			  << ", bp_iters=" << bp_iters
			  << ", num_thermal_samples=" << num_thermal_samples
			  << ", multi_start_mode=" << (multi_start_mode ? 1 : 0)
			  << ", multi_spin_coding_mode=" << (multi_spin_coding_mode ? 1 : 0)
			  << ", use_common_random_numbers_across_rho=" << (use_common_random_numbers_across_rho ? 1 : 0)
			  << ", decorrelation_multiplier=" << decorrelation_multiplier
			  << ", hk_use_greedy_init=" << (hk_use_greedy_init ? 1 : 0)
			  << ", resume_mode=" << (resume_mode ? 1 : 0)
			  << std::endl;

	// 输出文件约定：
	// - fn_all：每个无序样本的全部热样本 R；
	// - fn_stat：按 (L,rho) 聚合后的统计量；
	// - fn_frozen：冻结态计数；
	// - fn_wcc：各样本 WCC 分量规模。
	std::string fn_all = "./data/MVS-MCMC/R_mcmc_all-d-" + std::to_string(d) + ".txt";
	std::string fn_stat = "./data/MVS-MCMC/R_mcmc_stat-d-" + std::to_string(d) + ".txt";
	std::string fn_frozen = "./data/MVS-MCMC/Frozen_stat-d-" + std::to_string(d) + ".txt";
	std::string fn_wcc = "./data/MVS-MCMC/WCC_sizes-d-" + std::to_string(d) + ".txt";
	std::string fn_absR = "./data/MVS-MCMC/absR_thermal-d-" + std::to_string(d) + ".txt";

	std::unordered_set<CheckpointKey, CheckpointKeyHash> completed_checkpoints;
	if (resume_mode) {
		completed_checkpoints = load_completed_checkpoints_from_stat(fn_stat);
		std::cout << "Loaded " << completed_checkpoints.size()
				  << " completed checkpoints from: " << fn_stat << std::endl;
	}

	if (force_rebuild_tau_cache && use_acf_auto_sweeps && (use_tau_cache || write_tau_cache)) {
		if (!reset_tau_cache_file_with_header(tau_cache_file)) {
			std::cerr << "Error: failed to rebuild tau cache file: " << tau_cache_file << std::endl;
			return 1;
		}
		std::cout << "Force rebuilt tau cache file with header: " << tau_cache_file << std::endl;
	}

	std::unordered_map<CheckpointKey, double, CheckpointKeyHash> tau_cache_map;
	if ((use_tau_cache || write_tau_cache) && use_acf_auto_sweeps) {
		tau_cache_map = load_tau_cache(tau_cache_file);
		std::cout << "Loaded " << tau_cache_map.size()
				  << " tau cache entries from: " << tau_cache_file << std::endl;
	}

	// 启动前进度汇总：统计各 L 下未完成 rho 任务，用于估算剩余计算量。
	{
		long long total_tasks = static_cast<long long>(L_list.size()) * static_cast<long long>(num_rho);
		long long done_tasks = 0;
		std::vector<int> remaining_per_L(L_list.size(), 0);

		for (size_t li = 0; li < L_list.size(); ++li) {
			int L = L_list[li];
			int W_current = (W > 0) ? W : L;
			for (int ri = 0; ri < num_rho; ++ri) {
				double rho = rho_list[ri];
				CheckpointKey key{ d, L, W_current, rho_to_key(rho) };
				bool is_done = resume_mode && (completed_checkpoints.find(key) != completed_checkpoints.end());
				if (is_done) {
					done_tasks++;
				} else {
					remaining_per_L[li]++;
				}
			}
		}

		long long remaining_tasks = total_tasks - done_tasks;
		std::cout << "\n=== Resume Progress Summary ===" << std::endl;
		std::cout << "Total (L, rho) tasks: " << total_tasks
				  << ", done: " << done_tasks
				  << ", remaining: " << remaining_tasks << std::endl;
		for (size_t li = 0; li < L_list.size(); ++li) {
			std::cout << "  L=" << L_list[li]
					  << " -> remaining rho: " << remaining_per_L[li]
					  << "/" << num_rho << std::endl;
		}
		std::cout << "================================\n" << std::endl;
	}

	std::ofstream fout_all;
	if (write_R_mcmc_all) {
		fout_all.open(fn_all, resume_mode ? std::ios::app : std::ios::trunc);
	}
	std::ofstream fout_stat(fn_stat, resume_mode ? std::ios::app : std::ios::trunc);
	std::ofstream fout_frozen(fn_frozen, resume_mode ? std::ios::app : std::ios::trunc);
	std::ofstream fout_wcc;
	if (write_WCC_sizes) {
		fout_wcc.open(fn_wcc, resume_mode ? std::ios::app : std::ios::trunc);
	}
	std::ofstream fout_absR;
	if (write_absR_thermal) {
		fout_absR.open(fn_absR, resume_mode ? std::ios::app : std::ios::trunc);
	}
	if ((write_R_mcmc_all && !fout_all.is_open()) || !fout_stat.is_open() || !fout_frozen.is_open() || (write_WCC_sizes && !fout_wcc.is_open()) || (write_absR_thermal && !fout_absR.is_open())) {
		std::cerr << "Error: cannot open output files." << std::endl;
		return 1;
	}

	if (write_R_mcmc_all && should_write_header(fn_all, resume_mode)) {
		fout_all << "# d L W rho num_samples_index R_sample_1 ... R_sample_num_thermal_samples\n";
	}
	if (should_write_header(fn_stat, resume_mode)) {
		fout_stat << "# d L W rho N_samples N_thermal mean_sweeps sweeps_std dis_mR dis_mAbsR dis_mR2 dis_mR3 dis_mAbsR3 dis_mR4 q_EA dis_mAbsR_sq dis_mR_p4 dis_mAbsR_p4 dis_mR2_sq chi_th_abs chi_dis_abs chi_tot_abs chi_tot_abs_sem_jk chi_th_R chi_dis_R chi_tot_R U4 U_EA U_EA_abs U22 acf_probe_warn_count acf_probe_warn_frac\n";
	}
	if (should_write_header(fn_frozen, resume_mode)) {
		fout_frozen << "# d L W rho sample_idx N_1A N_1B N_0A N_0B N_starA N_starB\n";
	}
	if (write_WCC_sizes && should_write_header(fn_wcc, resume_mode)) {
		fout_wcc << "# d L W rho sample_idx wcc_size_1 wcc_size_2 ...\n";
	}
	if (write_absR_thermal && should_write_header(fn_absR, resume_mode)) {
		fout_absR << "# d L W rho <|R|>_sample_1 <|R|>_sample_2 ... <|R|>_sample_N_samples\n";
	}

	auto t_total_begin = std::chrono::high_resolution_clock::now();

	for (size_t li = 0; li < L_list.size(); ++li) {
		int L = L_list[li];
		int W_current = (W > 0) ? W : L;

		// 计算总节点数 N = L * W^(d-1)（含溢出防护与 int 索引上限检查）
		long long total_nodes = 1;
		try {
			if (L < 0 || W_current < 0) throw std::runtime_error("L and W must be non-negative");
			if constexpr (d == 1) {
				total_nodes = L;
			} else if constexpr (d == 2) {
				unsigned __int128 temp = static_cast<unsigned __int128>(L) * static_cast<unsigned __int128>(W_current);
				if (temp > static_cast<unsigned __int128>(std::numeric_limits<long long>::max())) {
					throw std::overflow_error("L * W too large for long long");
				}
				total_nodes = static_cast<long long>(temp);
			} else if constexpr (d == 3) {
				unsigned __int128 temp = static_cast<unsigned __int128>(L) *
					static_cast<unsigned __int128>(W_current) *
					static_cast<unsigned __int128>(W_current);
				if (temp > static_cast<unsigned __int128>(std::numeric_limits<long long>::max())) {
					throw std::overflow_error("L * W^2 too large for long long");
				}
				total_nodes = static_cast<long long>(temp);
			} else {
				if (d < 0) throw std::runtime_error("d must be non-negative");
				total_nodes = (d == 0) ? 1 : L;
				for (int i = 1; i < d; ++i) {
					unsigned __int128 temp = static_cast<unsigned __int128>(total_nodes) * W_current;
					if (temp > static_cast<unsigned __int128>(std::numeric_limits<long long>::max())) {
						throw std::overflow_error("L * W^(d-1) too large for long long");
					}
					total_nodes = static_cast<long long>(temp);
				}
			}
			if (total_nodes <= 0) throw std::overflow_error("Invalid total_nodes");
			if (total_nodes > static_cast<long long>(std::numeric_limits<int>::max())) {
				throw std::overflow_error("total_nodes exceeds INT_MAX while current implementation still uses int-based indexing");
			}
		} catch (const std::exception& e) {
			std::cerr << "Error for L=" << L << ": " << e.what() << std::endl;
			return 1;
		}

		// 预计算内部节点位图（仅依赖几何，按每个 L 构建一次）：
		// 标记“非边界”节点，供 get_neighbors 快路径直接常数偏移访问。
		std::vector<uint64_t> interior_mask;
		if constexpr (d >= 2) {
			long long total_blocks = (total_nodes + 63) / 64;
			interior_mask.assign(static_cast<size_t>(total_blocks), ~0ULL);

			if constexpr (d == 2) {
				for (long long i = 0; i < total_nodes; ++i) {
					int y = static_cast<int>(i / W_current);
					int x = static_cast<int>(i % W_current);
					if (x == 0 || x == W_current - 1 || y == 0 || y == L - 1) {
						interior_mask[static_cast<size_t>(i >> 6)] &= ~(1ULL << (i & 63));
					}
				}
			} else if constexpr (d == 3) {
				long long W2 = static_cast<long long>(W_current) * static_cast<long long>(W_current);
				for (long long i = 0; i < total_nodes; ++i) {
					int z = static_cast<int>(i / W2);
					int rem = static_cast<int>(i % W2);
					int y = rem / W_current;
					int x = rem % W_current;
					if (x == 0 || x == W_current - 1 ||
						y == 0 || y == W_current - 1 ||
						z == 0 || z == L - 1) {
						interior_mask[static_cast<size_t>(i >> 6)] &= ~(1ULL << (i & 63));
					}
				}
			}
		}

		// 预计算节点奇偶性：避免每个样本在构建二分图时重复做整除/取模。
		std::vector<uint8_t> parity_cache;

		// 构建基础参数模板：每个 rho/样本在此模板上覆盖少量字段。
		Params p_base;
		p_base.d = d;
		p_base.L = L;
		p_base.W = W_current;
		p_base.rho = 0.0;
		p_base.seed = base_seed;
		p_base.periodic_boundary = periodic_boundary;
		p_base.debug_output = false;
		p_base.mcmc_factor = mcmc_factor;
		p_base.use_acf_auto_sweeps = use_acf_auto_sweeps;
		p_base.acf_probe_sweeps = acf_probe_sweeps;
		p_base.acf_probe_burnin_sweeps = acf_probe_burnin_sweeps;
		p_base.acf_max_sweeps = acf_max_sweeps;
		p_base.acf_tau_multiplier = acf_tau_multiplier;
		p_base.acf_tau_estimator = acf_tau_estimator;
		p_base.cached_tau_int = 0.0;
		p_base.init_mode = init_mode;
		p_base.bp_iters = bp_iters;
		p_base.num_thermal_samples = num_thermal_samples;
		p_base.multi_start_mode = multi_start_mode;
		p_base.multi_spin_coding_mode = multi_spin_coding_mode;
		p_base.decorrelation_multiplier = decorrelation_multiplier;
		p_base.hk_use_greedy_init = hk_use_greedy_init;
		p_base.L_multipliers.assign(d, 0);
		p_base.interior_mask = &interior_mask;
		p_base.parity_cache = nullptr;
		if (d > 0) {
			if (d == 1) {
				p_base.L_multipliers[0] = 1;
			} else {
				p_base.L_multipliers[d - 1] = 1;
				for (int k = d - 2; k >= 1; --k) {
					unsigned __int128 temp = static_cast<unsigned __int128>(p_base.L_multipliers[k + 1]) * W_current;
					if (temp > static_cast<unsigned __int128>(std::numeric_limits<long long>::max())) {
						std::cerr << "Warning: multiplier overflow." << std::endl;
						p_base.L_multipliers[k] = std::numeric_limits<long long>::max();
					} else {
						p_base.L_multipliers[k] = static_cast<long long>(temp);
					}
				}
				unsigned __int128 temp = static_cast<unsigned __int128>(p_base.L_multipliers[1]) * W_current;
				if (temp > static_cast<unsigned __int128>(std::numeric_limits<long long>::max()))
					p_base.L_multipliers[0] = std::numeric_limits<long long>::max();
				else
					p_base.L_multipliers[0] = static_cast<long long>(temp);
			}
		}
		p_base.L_multipliers_ref = &p_base.L_multipliers;
		build_parity_cache(p_base, total_nodes, parity_cache);
		p_base.parity_cache = &parity_cache;

		std::cout << "\n=== L sweep " << (li + 1) << "/" << L_list.size()
				  << " : L=" << L << ", W=" << W_current
				  << ", total_nodes=" << total_nodes
				  << " ===" << std::endl;

		for (int ri = 0; ri < num_rho; ++ri) {
			double rho = rho_list[ri];
			CheckpointKey key{ d, L, W_current, rho_to_key(rho) };
			if (resume_mode && completed_checkpoints.find(key) != completed_checkpoints.end()) {
				std::cout << "[L " << (li + 1) << "/" << L_list.size() << "]"
						  << " [rho " << (ri + 1) << "/" << num_rho << "]"
						  << " rho=" << std::fixed << std::setprecision(6) << rho
						  << "  checkpoint hit -> skipped" << std::endl;
				continue;
			}

			std::vector<double> mean_R_samples(num_samples, 0.0);
			std::vector<double> mean_absR_samples(num_samples, 0.0);
			std::vector<double> mean_R2_samples(num_samples, 0.0);
			std::vector<double> mean_R3_samples(num_samples, 0.0);
			std::vector<double> mean_absR3_samples(num_samples, 0.0);
			std::vector<double> mean_R4_samples(num_samples, 0.0);
			std::vector<std::vector<double>> thermal_R_samples(num_samples);
			std::vector<double> sweeps_samples(num_samples, 0.0);
			int diag_max_wcc_size = 0;
			int diag_shared_sweeps = 0;
			std::vector<int> acf_probe_warn_samples(num_samples, 0);
			std::vector<int> N_1A_samples(num_samples, 0);
			std::vector<int> N_1B_samples(num_samples, 0);
			std::vector<int> N_0A_samples(num_samples, 0);
			std::vector<int> N_0B_samples(num_samples, 0);
			std::vector<int> N_starA_samples(num_samples, 0);
			std::vector<int> N_starB_samples(num_samples, 0);
			std::vector<std::vector<int>> wcc_sizes_samples(num_samples);

			auto t_rho_begin = std::chrono::high_resolution_clock::now();

			if (p_base.use_acf_auto_sweeps) {
				// 先尝试命中 tau cache，未命中则执行 pilot 子样本估计共享 sweeps。
				double cached_tau = 0.0;
				bool has_cached_tau = false;
				if (use_tau_cache) {
					auto it_tau = tau_cache_map.find(key);
					if (it_tau != tau_cache_map.end()) {
						has_cached_tau = true;
						cached_tau = it_tau->second;
					}
				}

				int shared_sweeps = p_base.mcmc_factor;
				int pilot_samples = 0;

				if (has_cached_tau) {
					shared_sweeps = static_cast<int>(std::ceil(p_base.acf_tau_multiplier * cached_tau));
					shared_sweeps = std::max(20, shared_sweeps);
					if (p_base.acf_max_sweeps > 0) shared_sweeps = std::min(shared_sweeps, p_base.acf_max_sweeps);

					std::cout << "[tau-cache-hit] "
						<< "[L " << (li + 1) << "/" << L_list.size() << "]"
						<< " [rho " << (ri + 1) << "/" << num_rho << "]"
						<< " rho=" << std::fixed << std::setprecision(6) << rho
						<< " cached_tau=" << std::fixed << std::setprecision(6) << cached_tau
						<< " shared_sweeps_applied=" << shared_sweeps
						<< std::endl;
				} else {
					// pilot 估计：对少量样本求 tau 与目标 sweeps，再取分位数稳健聚合。
					pilot_samples = std::min(num_samples, acf_estimation_samples);

					// pilot 阶段：只需 tau_est 和 shared_sweeps 来确定全局 sweeps
					std::vector<double> pilot_tau_est(pilot_samples, 0.0);
					std::vector<int> pilot_shared_sweeps(pilot_samples, 0);

					#pragma omp parallel for schedule(dynamic)
					for (int s = 0; s < pilot_samples; ++s) {
						uint64_t sample_seed = make_sample_seed(
							base_seed + 3141592653ULL,
							L,
							rho,
							s,
							use_common_random_numbers_across_rho);
						SingleSampleResult sr = run_single_sample(
							p_base,
							total_nodes,
							sample_seed,
							interior_mask,
							rho,
							1,
							false,
							0,
							0.0);
						pilot_tau_est[s] = sr.estimated_tau_int;
						pilot_shared_sweeps[s] = sr.shared_sweeps;
					}

					int pilot_max_target = 0;
					double robust_tau = 1.0;

					std::vector<double> valid_taus;
					std::vector<int> valid_targets;
					valid_taus.reserve(pilot_samples);
					valid_targets.reserve(pilot_samples);
					// 用分位数而非均值聚合，能减少异常样本对 sweeps 估计的影响。
					double q = std::clamp(pilot_tau_percentile, 0.0, 1.0);
					double q_pct = q * 100.0;

					for (int s = 0; s < pilot_samples; ++s) {
						if (pilot_tau_est[s] > 0.0 && std::isfinite(pilot_tau_est[s])) {
							valid_taus.push_back(pilot_tau_est[s]);
						}
						if (pilot_shared_sweeps[s] > 0) {
							valid_targets.push_back(pilot_shared_sweeps[s]);
						}
					}

					if (!valid_taus.empty()) {
						std::sort(valid_taus.begin(), valid_taus.end());
						int idx = static_cast<int>(std::floor(q * static_cast<double>(valid_taus.size() - 1)));
						if (idx < 0) idx = 0;
						if (idx >= static_cast<int>(valid_taus.size())) idx = static_cast<int>(valid_taus.size()) - 1;
						robust_tau = valid_taus[idx];
					}

					if (!valid_targets.empty()) {
						std::sort(valid_targets.begin(), valid_targets.end());
						int idx = static_cast<int>(std::floor(q * static_cast<double>(valid_targets.size() - 1)));
						if (idx < 0) idx = 0;
						if (idx >= static_cast<int>(valid_targets.size())) idx = static_cast<int>(valid_targets.size()) - 1;
						pilot_max_target = valid_targets[idx];
					}

					shared_sweeps = (pilot_samples > 0) ? pilot_max_target : p_base.mcmc_factor;

					if (write_tau_cache && robust_tau > 0.0 && std::isfinite(robust_tau)) {
						tau_cache_map[key] = robust_tau;
						if (!append_tau_cache_entry(tau_cache_file, d, L, W_current, rho, robust_tau)) {
							std::cerr << "Warning: failed to append tau cache entry to "
								<< tau_cache_file << std::endl;
						} else {
							std::cout << "[tau-cache-store] "
								<< "[L " << (li + 1) << "/" << L_list.size() << "]"
								<< " [rho " << (ri + 1) << "/" << num_rho << "]"
								<< " rho=" << std::fixed << std::setprecision(6) << rho
								<< " robust_tau(p" << std::fixed << std::setprecision(1) << q_pct << ")="
								<< std::fixed << std::setprecision(6) << robust_tau
								<< std::endl;
						}
					}

					std::cout << "[pilot-sweeps] "
						<< "[L " << (li + 1) << "/" << L_list.size() << "]"
						<< " [rho " << (ri + 1) << "/" << num_rho << "]"
						<< " rho=" << std::fixed << std::setprecision(6) << rho
						<< " pilot_samples=" << pilot_samples
						<< " pilot_p" << std::fixed << std::setprecision(1) << q_pct << "_target=" << pilot_max_target
						<< " robust_tau(p" << std::fixed << std::setprecision(1) << q_pct << ")="
						<< std::fixed << std::setprecision(6) << robust_tau
						<< " shared_sweeps_applied=" << shared_sweeps
						<< std::endl;
				}

				if (shared_sweeps > 0 && shared_sweeps < 10) shared_sweeps = 10; // 仅对需要 MCMC 的情形保底
				if (p_base.acf_max_sweeps > 0) shared_sweeps = std::min(shared_sweeps, p_base.acf_max_sweeps);

				// 正式阶段：固定 shared_sweeps 后并行跑全部无序样本。
				#pragma omp parallel for schedule(dynamic)
				for (int s = 0; s < num_samples; ++s) {
					uint64_t sample_seed = make_sample_seed(
						base_seed,
						L,
						rho,
						s,
						use_common_random_numbers_across_rho);
					SingleSampleResult sr = run_single_sample(
						p_base,
						total_nodes,
						sample_seed,
						interior_mask,
						rho,
						-1,
						true,
						shared_sweeps,
						0.0);
					mean_R_samples[s] = sr.mean_R_thermal;
					mean_absR_samples[s] = sr.mean_absR_thermal;
					mean_R2_samples[s] = sr.mean_R2_thermal;
					mean_R3_samples[s] = sr.mean_R3_thermal;
					mean_absR3_samples[s] = sr.mean_absR3_thermal;
					mean_R4_samples[s] = sr.mean_R4_thermal;
					thermal_R_samples[s] = std::move(sr.R_thermal_samples);
					sweeps_samples[s] = sr.sweeps_used;
					if (s == 0) { diag_max_wcc_size = sr.max_wcc_size; diag_shared_sweeps = sr.shared_sweeps; }
					acf_probe_warn_samples[s] = sr.acf_probe_too_short ? 1 : 0;
					N_1A_samples[s] = sr.N_1A;
					N_1B_samples[s] = sr.N_1B;
					N_0A_samples[s] = sr.N_0A;
					N_0B_samples[s] = sr.N_0B;
					N_starA_samples[s] = sr.N_starA;
					N_starB_samples[s] = sr.N_starB;
					// 保留每个无序样本的 WCC 尺寸分布，便于后续做结构尺度统计。
					wcc_sizes_samples[s] = std::move(sr.all_wcc_sizes);
				}
			} else {
				// 固定 sweeps 模式：直接按 p_base 参数并行采样。
				#pragma omp parallel for schedule(dynamic)
				for (int s = 0; s < num_samples; ++s) {
					uint64_t sample_seed = make_sample_seed(
						base_seed,
						L,
						rho,
						s,
						use_common_random_numbers_across_rho);
					SingleSampleResult sr = run_single_sample(
						p_base,
						total_nodes,
						sample_seed,
						interior_mask,
						rho,
						-1,
						false,
						0,
						0.0);
					mean_R_samples[s] = sr.mean_R_thermal;
					mean_absR_samples[s] = sr.mean_absR_thermal;
					mean_R2_samples[s] = sr.mean_R2_thermal;
					mean_R3_samples[s] = sr.mean_R3_thermal;
					mean_absR3_samples[s] = sr.mean_absR3_thermal;
					mean_R4_samples[s] = sr.mean_R4_thermal;
					thermal_R_samples[s] = std::move(sr.R_thermal_samples);
					sweeps_samples[s] = sr.sweeps_used;
					if (s == 0) { diag_max_wcc_size = sr.max_wcc_size; diag_shared_sweeps = sr.shared_sweeps; }
					acf_probe_warn_samples[s] = sr.acf_probe_too_short ? 1 : 0;
					N_1A_samples[s] = sr.N_1A;
					N_1B_samples[s] = sr.N_1B;
					N_0A_samples[s] = sr.N_0A;
					N_0B_samples[s] = sr.N_0B;
					N_starA_samples[s] = sr.N_starA;
					N_starB_samples[s] = sr.N_starB;
					// 固定 sweeps 分支同样写出 WCC 尺寸，保证两种模式输出字段一致。
					wcc_sizes_samples[s] = std::move(sr.all_wcc_sizes);
				}
			}

			auto t_rho_end = std::chrono::high_resolution_clock::now();
			double rho_time_sec = std::chrono::duration<double>(t_rho_end - t_rho_begin).count();

			// 计算统计量（双重平均：先热平均，再无序平均）。
			// 结果同时输出传统矩、磁化率分解与 Binder 累积量。
			double sum_mean_R = 0.0;
			double sum_mean_absR = 0.0;
			double sum_mean_R2 = 0.0;
			double sum_mean_R3 = 0.0;
			double sum_mean_absR3 = 0.0;
			double sum_mean_R4 = 0.0;
			double sum_mean_R_sq = 0.0;       // sum_i R_i^2 (for q_EA)
			double sum_mean_absR_sq = 0.0;     // sum_i A_i^2
			double sum_mean_R_p4 = 0.0;        // sum_i R_i^4
			double sum_mean_absR_p4 = 0.0;     // sum_i A_i^4
			double sum_mean_R2_sq = 0.0;       // sum_i (M_i^(2))^2
			double sum_thermal_var_absR = 0.0;
			double sum_sweeps = 0.0;
			double sum_sweeps_sq = 0.0;        // for sweeps_std
			int acf_probe_warn_count = 0;
			for (int s = 0; s < num_samples; ++s) {
				double mR = mean_R_samples[s];
				double mAbsR = mean_absR_samples[s];
				double mR2 = mean_R2_samples[s];
				double mR3 = mean_R3_samples[s];
				double mAbsR3 = mean_absR3_samples[s];
				double mR4 = mean_R4_samples[s];
				sum_mean_R += mR;
				sum_mean_absR += mAbsR;
				sum_mean_R2 += mR2;
				sum_mean_R3 += mR3;
				sum_mean_absR3 += mAbsR3;
				sum_mean_R4 += mR4;
				sum_mean_R_sq += mR * mR;
				sum_mean_absR_sq += mAbsR * mAbsR;
				sum_mean_R_p4 += mR * mR * mR * mR;
				sum_mean_absR_p4 += mAbsR * mAbsR * mAbsR * mAbsR;
				sum_mean_R2_sq += mR2 * mR2;
				sum_thermal_var_absR += (mR2 - mAbsR * mAbsR);
				sum_sweeps += sweeps_samples[s];
				sum_sweeps_sq += sweeps_samples[s] * sweeps_samples[s];
				acf_probe_warn_count += acf_probe_warn_samples[s];
			}
			// 基础矩（前缀 dis_ 表示 disorder average，即对无序样本做平均）
			double dis_mR = sum_mean_R / num_samples;
			double dis_mAbsR = sum_mean_absR / num_samples;
			double dis_mR2 = sum_mean_R2 / num_samples;
			double dis_mR3 = sum_mean_R3 / num_samples;
			double dis_mAbsR3 = sum_mean_absR3 / num_samples;
			double dis_mR4 = sum_mean_R4 / num_samples;
			// 热力学均值的无序高阶矩：
			// q_ea = [<R>^2]，常用于判别玻璃态相关性。
			double q_ea = sum_mean_R_sq / num_samples;
			double dis_mAbsR_sq = sum_mean_absR_sq / num_samples;
			double dis_mR_p4 = sum_mean_R_p4 / num_samples;
			double dis_mAbsR_p4 = sum_mean_absR_p4 / num_samples;
			double dis_mR2_sq = sum_mean_R2_sq / num_samples;
			// 基于 |R| 的磁化率
			double N_nodes = static_cast<double>(total_nodes);
			double thermal_var_absR_avg = sum_thermal_var_absR / num_samples;
			double chi_th_abs = N_nodes * thermal_var_absR_avg;  // N * ([<R^2>] - [<|R|>^2])
			double chi_dis_abs = N_nodes * (dis_mAbsR_sq - dis_mAbsR * dis_mAbsR); // N * ([<|R|>^2] - [<|R|>]^2)
			double chi_tot_abs = N_nodes * (dis_mR2 - dis_mAbsR * dis_mAbsR); // N * ([<R^2>] - [<|R|>]^2)
			double chi_tot_abs_sem_jk = 0.0;
			if (num_samples > 1) {
				const double inv_num_samples_minus_one = 1.0 / static_cast<double>(num_samples - 1);
				double chi_tot_abs_jk_mean = 0.0;
				for (int s = 0; s < num_samples; ++s) {
					const double loo_mAbsR = (sum_mean_absR - mean_absR_samples[s]) * inv_num_samples_minus_one;
					const double loo_mR2 = (sum_mean_R2 - mean_R2_samples[s]) * inv_num_samples_minus_one;
					chi_tot_abs_jk_mean += N_nodes * (loo_mR2 - loo_mAbsR * loo_mAbsR);
				}
				chi_tot_abs_jk_mean /= static_cast<double>(num_samples);

				double chi_tot_abs_jk_var_sum = 0.0;
				for (int s = 0; s < num_samples; ++s) {
					const double loo_mAbsR = (sum_mean_absR - mean_absR_samples[s]) * inv_num_samples_minus_one;
					const double loo_mR2 = (sum_mean_R2 - mean_R2_samples[s]) * inv_num_samples_minus_one;
					const double chi_tot_abs_loo = N_nodes * (loo_mR2 - loo_mAbsR * loo_mAbsR);
					const double delta = chi_tot_abs_loo - chi_tot_abs_jk_mean;
					chi_tot_abs_jk_var_sum += delta * delta;
				}
				chi_tot_abs_sem_jk = std::sqrt(std::max(
					0.0,
					(static_cast<double>(num_samples - 1) / static_cast<double>(num_samples)) * chi_tot_abs_jk_var_sum));
			}
			// 基于 R 的磁化率（自旋玻璃相分析用）
			double chi_th_R = N_nodes * (dis_mR2 - q_ea);         // N * ([<R^2>] - [<R>^2])
			double chi_dis_R = N_nodes * (q_ea - dis_mR * dis_mR); // N * ([<R>^2] - [<R>]^2)
			double chi_tot_R = chi_th_R + chi_dis_R;               // N * ([<R^2>] - [<R>]^2)
			// Binder 累积量：
			// 通过四阶矩/二阶矩组合刻画分布形状，常用于临界点定位与有限尺寸分析。
			double binder_U4 = 0.0;
			if (dis_mR2 > 1e-15) {
				binder_U4 = 1.0 - dis_mR4 / (3.0 * dis_mR2 * dis_mR2);
			}
			double U_EA = 0.0;
			if (q_ea > 1e-15) {
				U_EA = 1.0 - dis_mR_p4 / (3.0 * q_ea * q_ea);
			}
			double U_EA_abs = 0.0;
			if (dis_mAbsR_sq > 1e-15) {
				U_EA_abs = 1.0 - dis_mAbsR_p4 / (3.0 * dis_mAbsR_sq * dis_mAbsR_sq);
			}
			double U22 = 0.0;
			if (dis_mR2 > 1e-15) {
				U22 = 1.0 - dis_mR2_sq / (3.0 * dis_mR2 * dis_mR2);
			}
			// 元信息
			double mean_sweeps = sum_sweeps / num_samples;
			double sweeps_var = sum_sweeps_sq / num_samples - mean_sweeps * mean_sweeps;
			double sweeps_std = (sweeps_var > 0.0) ? std::sqrt(sweeps_var) : 0.0;
			double acf_probe_warn_frac = static_cast<double>(acf_probe_warn_count) / static_cast<double>(num_samples);

			// 写入原始数据：每行一个无序图，附带该图全部热样本 R
			if (write_R_mcmc_all) {
				for (int s = 0; s < num_samples; ++s) {
					fout_all << d << " " << L << " " << W_current << " "
							 << std::fixed << std::setprecision(10) << rho << " "
							 << s;
					for (double r_val : thermal_R_samples[s]) {
						fout_all << " " << std::fixed << std::setprecision(10) << r_val;
					}
					fout_all << "\n";
				}
			}

			for (int s = 0; s < num_samples; ++s) {
				fout_frozen << d << " " << L << " " << W_current << " "
							<< std::fixed << std::setprecision(10) << rho << " "
							<< s << " "
							<< N_1A_samples[s] << " "
							<< N_1B_samples[s] << " "
							<< N_0A_samples[s] << " "
							<< N_0B_samples[s] << " "
							<< N_starA_samples[s] << " "
							<< N_starB_samples[s] << "\n";

				if (write_WCC_sizes) {
					fout_wcc << d << " " << L << " " << W_current << " "
							 << std::fixed << std::setprecision(10) << rho << " "
							 << s;
					for (int wcc_sz : wcc_sizes_samples[s]) {
						fout_wcc << " " << wcc_sz;
					}
					fout_wcc << "\n";
				}
			}

			// 写入 <|R|> 热平均值：每行为一个 (L, rho) 点，含 N_samples 个 <|R|> 值
			if (write_absR_thermal) {
				fout_absR << d << " " << L << " " << W_current << " "
						  << std::fixed << std::setprecision(10) << rho;
				for (int s = 0; s < num_samples; ++s) {
					fout_absR << " " << std::fixed << std::setprecision(10) << mean_absR_samples[s];
				}
				fout_absR << "\n";
			}

			// 写入统计数据（33 列）：
			// d L W rho N_samples N_thermal mean_sweeps sweeps_std
			// dis_mR dis_mAbsR dis_mR2 dis_mR3 dis_mAbsR3 dis_mR4
			// q_EA dis_mAbsR_sq dis_mR_p4 dis_mAbsR_p4 dis_mR2_sq
			// chi_th_abs chi_dis_abs chi_tot_abs chi_tot_abs_sem_jk chi_th_R chi_dis_R chi_tot_R
			// U4 U_EA U_EA_abs U22
			// acf_probe_warn_count acf_probe_warn_frac
			fout_stat << d << " " << L << " " << W_current << " "
					  << std::fixed << std::setprecision(10) << rho << " "
					  << num_samples << " "
					  << num_thermal_samples << " "
					  << std::fixed << std::setprecision(10) << mean_sweeps << " "
					  << std::fixed << std::setprecision(10) << sweeps_std << " "
					  << std::fixed << std::setprecision(10) << dis_mR << " "
					  << std::fixed << std::setprecision(10) << dis_mAbsR << " "
					  << std::fixed << std::setprecision(10) << dis_mR2 << " "
					  << std::fixed << std::setprecision(10) << dis_mR3 << " "
					  << std::fixed << std::setprecision(10) << dis_mAbsR3 << " "
					  << std::fixed << std::setprecision(10) << dis_mR4 << " "
					  << std::fixed << std::setprecision(10) << q_ea << " "
					  << std::fixed << std::setprecision(10) << dis_mAbsR_sq << " "
					  << std::fixed << std::setprecision(10) << dis_mR_p4 << " "
					  << std::fixed << std::setprecision(10) << dis_mAbsR_p4 << " "
					  << std::fixed << std::setprecision(10) << dis_mR2_sq << " "
					  << std::fixed << std::setprecision(10) << chi_th_abs << " "
					  << std::fixed << std::setprecision(10) << chi_dis_abs << " "
					  << std::fixed << std::setprecision(10) << chi_tot_abs << " "
					  << std::fixed << std::setprecision(10) << chi_tot_abs_sem_jk << " "
					  << std::fixed << std::setprecision(10) << chi_th_R << " "
					  << std::fixed << std::setprecision(10) << chi_dis_R << " "
					  << std::fixed << std::setprecision(10) << chi_tot_R << " "
					  << std::fixed << std::setprecision(10) << binder_U4 << " "
					  << std::fixed << std::setprecision(10) << U_EA << " "
					  << std::fixed << std::setprecision(10) << U_EA_abs << " "
					  << std::fixed << std::setprecision(10) << U22 << " "
					  << acf_probe_warn_count << " "
					  << std::fixed << std::setprecision(10) << acf_probe_warn_frac << "\n";

			if (write_R_mcmc_all) fout_all.flush();
			fout_stat.flush();
			fout_frozen.flush();
			if (write_WCC_sizes) fout_wcc.flush();
			if (write_absR_thermal) fout_absR.flush();
			completed_checkpoints.insert(key);

			std::cout << "[L " << (li + 1) << "/" << L_list.size() << "]"
					  << " [rho " << (ri + 1) << "/" << num_rho << "]"
					  << " rho=" << std::fixed << std::setprecision(6) << rho
					  << "  [<|R|>]=" << std::fixed << std::setprecision(6) << dis_mAbsR
					  << "  qEA=[<R>^2]=" << std::fixed << std::setprecision(6) << q_ea
					  << "  chi_tot_abs=" << std::fixed << std::setprecision(6) << chi_tot_abs
					  << "  U4=" << std::fixed << std::setprecision(6) << binder_U4
					  << "  mean(sweeps)=" << std::fixed << std::setprecision(3) << mean_sweeps
					  << "  time=" << std::fixed << std::setprecision(2) << rho_time_sec << "s"
					  << std::endl;

			std::cout << "[rho-diagnostics] "
					  << "[L " << (li + 1) << "/" << L_list.size() << "]"
					  << " [rho " << (ri + 1) << "/" << num_rho << "]"
					  << " rho=" << std::fixed << std::setprecision(6) << rho
					  << "  sample0_max_wcc_size=" << diag_max_wcc_size
					  << "  sample0_shared_sweeps=" << diag_shared_sweeps
					  << std::endl;
		}
	}

	if (write_R_mcmc_all) fout_all.close();
	fout_stat.close();
	fout_frozen.close();
	if (write_WCC_sizes) fout_wcc.close();
	if (write_absR_thermal) fout_absR.close();

	auto t_total_end = std::chrono::high_resolution_clock::now();
	double total_time_sec = std::chrono::duration<double>(t_total_end - t_total_begin).count();
	std::cout << "\nAll done. Total time: " << std::fixed << std::setprecision(2) << total_time_sec << "s" << std::endl;
	if (write_R_mcmc_all) std::cout << "Raw data saved to: " << fn_all << std::endl;
	if (write_absR_thermal) std::cout << "absR thermal data saved to: " << fn_absR << std::endl;
	std::cout << "Statistics saved to: " << fn_stat << std::endl;

	return 0;
}
