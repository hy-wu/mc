# MC 模拟优化 — 任务跟踪

## 基线
- [x] 编译并运行基线版本，记录时间和结果 (11.828s / 118ms per step)

## Step 1: swap_remove 替代 remove
- [x] 修改网格更新代码
- [x] 编译运行，对比时间和结果 (11.238s / 112ms)

## Step 2: 预计算常量 (d², 碰撞系数, v_magnitude)
- [x] 修改 Config 和 main.rs
- [x] 编译运行，对比时间和结果 (11.801s / 118ms)

## Step 3: SmallRng 替代 ThreadRng
- [x] 修改 RNG
- [x] 编译运行，对比时间和结果 (7.463s / 74ms) ⭐ 最大提升

## Step 4: 消除冗余 (theta/phi, 未使用 import)
- [x] 清理代码
- [x] 编译运行，对比时间和结果 (7.423s / 74ms)

## Step 5: BufWriter + 减少进度输出
- [x] 修改 I/O
- [x] 编译运行，对比时间和结果 (7.531s / 75ms)

## Step 6: 扁平化网格数组
- [x] 重构网格数据结构
- [x] 编译运行，对比时间和结果 (7.531s / 75ms)

## 清理
- [x] 移除 config.rs 中未使用字段 (e0, d_cubed, d_fourth)
- [x] 移除旧注释常量
- [x] 零 warnings 编译确认
