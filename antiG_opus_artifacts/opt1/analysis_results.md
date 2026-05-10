# MC 粒子碰撞模拟 — 优化分析报告

本项目是一个 Monte Carlo 粒子碰撞模拟程序，使用网格加速邻居搜索、随机碰撞采样和 Lennard-Jones 力场。以下从 **7 个维度** 分析优化空间。

---

## 1. 🔴 算法复杂度 — 网格更新 `O(n)` → 可降至 `O(1)` per particle

### 问题
[main.rs L311-L331](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L311-L331)

当前每个时间步都会遍历 **所有网格单元** 的所有粒子来检测是否需要迁移，并使用 `Vec::remove(j)` 做删除（`O(n)` 移位操作）。

```rust
// 当前做法：遍历所有网格 + O(n) remove
for j in 0..grid[i_x][i_y][i_z].len() {
    let i = grid[i_x][i_y][i_z][j];
    // ... 检测是否需要迁移
    removes.push(j);
    grid[i_x_new][i_y_new][i_z_new].push(i);
}
for &j in removes.iter().rev() {
    grid[i_x][i_y][i_z].remove(j);  // O(n) 每次！
}
```

### 建议
1. **维护 particle→cell 的反向映射**，只更新需要移动的粒子，而非扫描全部网格。
2. 将 `Vec::remove()` 替换为 `swap_remove()`（`O(1)`），因为网格内粒子顺序不影响物理正确性。

```rust
// 优化方案
let mut particle_cell: Vec<[usize; 3]> = vec![[0; 3]; config.n];
// 初始化时记录每个粒子所在网格

// 更新时只遍历粒子
for i in 0..config.n {
    let new_cell = [r[i][0].floor() as usize, ...];
    if new_cell != particle_cell[i] {
        // 用 swap_remove 移出旧网格 (O(1))
        let old = particle_cell[i];
        let pos = grid[old[0]][old[1]][old[2]].iter().position(|&x| x == i).unwrap();
        grid[old[0]][old[1]][old[2]].swap_remove(pos);
        grid[new_cell[0]][new_cell[1]][new_cell[2]].push(i);
        particle_cell[i] = new_cell;
    }
}
```

**预期影响**：大幅降低网格更新开销，特别是在网格数 `L³` 远大于实际发生迁移的粒子数时。

---

## 2. 🔴 数据结构与内存布局

### 2a. 嵌套 Vec 导致缓存不友好

[main.rs L261-L262](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L261-L262)

```rust
let mut grid: Vec<Vec<Vec<Vec<usize>>>> = vec![vec![vec![vec![]; config.l]; config.l]; config.l];
```

四层嵌套 `Vec` 意味着 `L³` 次堆分配，且每个网格单元的数据散落在堆的不同位置，严重影响缓存命中率。

### 建议
使用**扁平化数组** + 索引宏：

```rust
// 扁平化网格：只有一次分配
let mut grid_data: Vec<Vec<usize>> = vec![vec![]; config.l * config.l * config.l];

#[inline]
fn grid_idx(x: usize, y: usize, z: usize, l: usize) -> usize {
    x * l * l + y * l + z
}

// 访问: grid_data[grid_idx(i_x, i_y, i_z, config.l)]
```

### 2b. 函数签名中 `&Vec<...>` 应改为 `&[...]`

[main.rs L93, L164](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L93-L164)

```rust
fn scatt_o1_optimized(grid: &Vec<Vec<Vec<Vec<usize>>>>, ...)
```

Clippy 会建议使用 `&[...]` 切片类型——虽然对性能影响微小，但可以改善 API 设计和泛用性。

---

## 3. 🟠 计算冗余 — 重复计算

### 3a. 力场计算中 `config.d.powi(2)` 重复计算

[main.rs L358-L360](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L358-L360)

```rust
if dr2 > config.d.powi(2) && dr2 < 10.0 * (config.d).powi(2) {
    force[k] = 24.0 * config.l_j_epsilon 
        * (2.0 * (config.d.powi(2) / dr2).powi(5) - (config.d.powi(2) / dr2).powi(2)) * dr[k];
}
```

`config.d.powi(2)` 在热循环中被反复计算。应在 `Config` 中预计算：

```rust
// 在 Config 中预计算
pub struct Config {
    // ...
    pub d_sq: f64,         // d²
    pub collision_coeff: f64, // π * d² * dt / n_test (用于碰撞概率)
}
```

### 3b. `(2.0 * config.e0 / config.mass).sqrt()` 重复计算

[main.rs L257-L259](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L257-L259)

初始化时每个粒子都重新计算 `(2.0 * config.e0 / config.mass).sqrt()`，应提前算好：

```rust
let v_magnitude = (2.0 * config.e0 / config.mass).sqrt();
```

### 3c. `vec_k` 的冗余除法

[calculate_collision_parameters](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L83-L86) 中 `vec_k[k] = dr[k] / dr2`，但后续在 `scatt_o1_optimized` 中使用 `vec_k[k] * dv_dr` 进行速度更新，这等价于 `dr[k] * dv_dr / dr2`。可以直接传递 `dr` 和 `dr2`，避免 `vec_k` 的构造。

---

## 4. 🟠 并行化 — 已注释的 Rayon

[main.rs L14](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L14)

```rust
// use rayon::prelude::*;
```

Rayon 已作为依赖但未使用。以下部分可以并行化：

| 部分 | 可并行性 | 说明 |
|------|---------|------|
| 粒子位置更新 (L273-L277) | ✅ 完全独立 | `par_chunks_mut` 直接并行 |
| 边界条件 & 压力 (L280-L307) | ⚠️ 需要 atomic sum | 压力是归约操作 |
| 网格更新 (L311-L331) | ❌ 需要互斥 | 多粒子可能写入同一网格 |
| 同网格碰撞 (L333-L370) | ⚠️ 需要着色 | 相邻网格不能同时写入 `v` |
| 跨网格碰撞 (L372-L386) | ⚠️ 需要着色 | 同上，可用红黑着色 |

**建议**：至少先将位置更新并行化，这是最容易的低垂果实。

---

## 5. 🟡 借用冲突 — 同网格碰撞循环

[main.rs L333-L370](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L333-L370)

这段代码使用 `grid.iter().for_each()` 遍历，但内部需要修改 `v`。在当前代码中，`r` 和 `v` 并没有被 `grid` 的闭包捕获为可变引用（它们是外部变量），这看起来可以工作，但在并行化时会遇到严重的借用冲突。

> [!TIP]
> 建议将碰撞对先收集到一个 `Vec<(usize, usize, [f64;3])>` 中，然后再统一应用速度变更。这也为并行化铺平道路。

---

## 6. 🟡 随机数生成器

[main.rs L250](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L250)

```rust
let mut rng = rand::thread_rng();
```

`ThreadRng` 是密码学安全的 RNG，对科学模拟来说太慢。

### 建议
使用更快的非密码学 RNG：

```rust
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

let mut rng = Xoshiro256StarStar::seed_from_u64(42);
// 速度约为 ThreadRng 的 2-3 倍
```

需要在 `Cargo.toml` 添加 `rand_xoshiro` 依赖，或使用 `rand` 内置的 `SmallRng`：

```rust
use rand::rngs::SmallRng;
let mut rng = SmallRng::from_entropy();
```

---

## 7. 🟢 I/O 与其他小优化

### 7a. 进度输出
[main.rs L388-L389](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L388-L389)

每个时间步都调用 `print!` + `flush()`，系统调用开销大。建议每 N 步输出一次：

```rust
if i_t % 100 == 0 {
    print!("{i_t}/{n_step}\r");
    io::stdout().flush().unwrap();
}
```

### 7b. 使用 `BufWriter` 写入结果

[main.rs L409-L423](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L409-L423)

```rust
// 当前：直接写 File（每次 writeln! 一次系统调用）
let mut file = File::create(...)?;

// 优化：使用 BufWriter
use std::io::BufWriter;
let mut file = BufWriter::new(File::create(...)?);
```

### 7c. 未使用的 `theta`/`phi` 向量

[main.rs L247-L248](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs#L247-L248)

`theta` 和 `phi` 在初始化后再也没有使用。可以改为临时局部变量，节省 `2 * N * 8` 字节内存：

```rust
for i in 0..config.n {
    let theta = (1.0 - 2.0 * rng.gen_range(0.0..1.0_f64)).acos();
    let phi = rng.gen_range(0.0..2.0 * PI);
    // ...
}
```

### 7d. 未使用的 import

```rust
use std::{io, vec};  // `vec` 宏不需要显式导入
```

---

## 优化优先级总结

| 优先级 | 优化项 | 预期收益 | 实现难度 |
|-------|--------|---------|---------|
| 🔴 P0 | `swap_remove` 替代 `remove` | 网格更新从 O(n²) → O(n) | ⭐ 简单 |
| 🔴 P0 | 扁平化网格数组 | 减少缓存未命中 | ⭐⭐ 中等 |
| 🟠 P1 | 预计算常量 (d², 碰撞系数等) | 减少热循环中的冗余计算 | ⭐ 简单 |
| 🟠 P1 | 使用 SmallRng 替代 ThreadRng | RNG 速度提升 2-3x | ⭐ 简单 |
| 🟠 P1 | 并行化位置更新 (Rayon) | 多核加速 | ⭐⭐ 中等 |
| 🟡 P2 | BufWriter 写文件 | I/O 性能提升 | ⭐ 简单 |
| 🟡 P2 | 减少进度输出频率 | 减少系统调用 | ⭐ 简单 |
| 🟡 P2 | 消除临时向量 (theta/phi) | 节省内存 | ⭐ 简单 |
| 🟢 P3 | 碰撞对缓存 + 批量更新 | 为并行化铺路 | ⭐⭐⭐ 复杂 |
| 🟢 P3 | 红黑网格着色并行碰撞 | 碰撞阶段多核加速 | ⭐⭐⭐ 复杂 |

---

> [!IMPORTANT]
> 最高性价比的优化是 **`swap_remove`**、**预计算常量** 和 **`SmallRng`**，三者都是代码改动极小但收益显著的"低垂果实"。

如需我实施其中任何一项优化，请告知！
