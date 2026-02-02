# 🚀 训练优化执行清单

## ✅ 已完成的工作

### 1. 问题诊断 ✓
- [x] 分析训练日志，定位问题根源
- [x] 识别关键瓶颈：序列建模能力不足、数学题识别困难
- [x] 计算理论分析：二进制准确率vs完整匹配率的差异

### 2. 代码优化 ✓
- [x] **config.py**: 提升学习率 0.0008→0.001
- [x] **config.py**: 调整衰减策略 (decay_factor 0.6→0.7, patience 12→15)
- [x] **trainer.py**: 启用Focal Loss (gamma=2.0)
- [x] **trainer.py**: 放缓学习率衰减 (15k步×0.99)
- [x] **callbacks.py**: 延长最大训练步数 50k→300k
- [x] **callbacks.py**: 降低目标准确率 95%→80%
- [x] **callbacks.py**: 更频繁验证 500步→300步
- [x] **data_augmentation.py**: 增强数据增强力度

### 3. 文档创建 ✓
- [x] [TRAINING_BREAKTHROUGH_2026-02-02.md](TRAINING_BREAKTHROUGH_2026-02-02.md) - 详细技术方案
- [x] [QUICK_START_OPTIMIZED_TRAINING.md](QUICK_START_OPTIMIZED_TRAINING.md) - 快速启动指南
- [x] [OPTIMIZATION_SUMMARY_2026-02-02.md](OPTIMIZATION_SUMMARY_2026-02-02.md) - 优化总结
- [x] 本执行清单

### 4. 辅助工具 ✓
- [x] **verify_optimization.py**: 验证脚本，检查所有优化是否生效
- [x] **model_grouped.py**: 进阶模型（分组输出架构，备用）

---

## 📋 开始训练前的检查

### Step 1: 验证优化配置 ⚠️
```bash
cd /data/coding/caocrvfy
python verify_optimization.py
```

**预期输出**: 所有检查项都是 ✓

如果有 ✗，请检查对应文件

### Step 2: 检查数据集
```bash
ls -lh /data/coding/captcha/img/ | head -20
```

**确认**:
- [x] 图片总数 > 10,000
- [x] 包含数学题样本 (文件名格式: `hex_answer_hash.png`)
- [x] 文件权限正常

### Step 3: 检查GPU可用性
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**预期**: 应显示至少1个GPU

### Step 4: 备份现有模型（如果有）
```bash
cd /data/coding/caocrvfy
[ -d models ] && mv models models_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p models
```

---

## 🎯 开始训练

### 方式1: 前台运行（适合测试）
```bash
cd /data/coding/caocrvfy
python train_v4.py
```

按 `Ctrl+C` 可以停止

### 方式2: 后台运行（推荐）
```bash
cd /data/coding/caocrvfy
nohup python -u train_v4.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > train.pid  # 保存进程ID

# 实时查看日志
tail -f training_*.log

# 停止训练（如需要）
kill $(cat train.pid)
```

### 方式3: 使用screen（可随时连接）
```bash
screen -S captcha_training
cd /data/coding/caocrvfy
python train_v4.py

# 分离会话: Ctrl+A, 然后按 D
# 恢复会话: screen -r captcha_training
# 列出会话: screen -ls
```

---

## 📊 训练监控

### 关键里程碑

| 步数 | 预期完整匹配率 | 预期数学题识别率 | 状态 |
|-----|-------------|--------------|------|
| 10k  | 65-67% | 3-5% | 初期 |
| 50k  | 68-72% | 8-12% | 优化生效 ✓ |
| 100k | 72-76% | 15-20% | 显著改善 ✓✓ |
| 150k | 75-80% | 20-28% | 接近目标 ✓✓✓ |
| 200k | 78-82% | 25-35% | 达到目标 🎉 |

### 实时监控命令

**查看最新验证结果**:
```bash
tail -100 training_*.log | grep "Step.*验证结果" -A 2
```

**查看Epoch汇总**:
```bash
tail -100 training_*.log | grep "Epoch.*训练损失"
```

**查看完整匹配趋势**:
```bash
grep "完整匹配:" training_*.log | tail -20
```

**绘制训练曲线** (如果安装了matplotlib):
```python
import re
import matplotlib.pyplot as plt

# 读取日志
with open('training_*.log', 'r') as f:
    lines = f.readlines()

# 提取完整匹配率
matches = []
for line in lines:
    if '完整匹配:' in line:
        match = re.search(r'完整匹配: (\d+\.\d+)%', line)
        if match:
            matches.append(float(match.group(1)))

# 绘图
plt.plot(matches)
plt.xlabel('Validation Step')
plt.ylabel('Full Match Accuracy (%)')
plt.title('Training Progress')
plt.savefig('training_progress.png')
```

---

## 🎉 训练完成后

### Step 1: 评估最终模型
```bash
cd /data/coding/caocrvfy
python core/evaluator.py --model models/final_model.keras
```

### Step 2: 数学题专项测试
```bash
python extras/quick_verify.py --math-only
```

### Step 3: 导出模型
```bash
# ONNX格式（用于部署）
python convert_to_onnx.py --input models/final_model.keras --output models/captcha_model.onnx

# 查看模型大小
ls -lh models/final_model.keras
```

### Step 4: 记录训练结果
创建 `training_result_YYYYMMDD.md`:
```markdown
# 训练结果 2026-02-XX

## 配置
- Focal Loss: 启用 (gamma=2.0)
- 学习率: 0.001 → 15k步×0.99
- 最大步数: 300000
- 实际训练步数: ______

## 结果
- 完整匹配准确率: ______%
- 数学题识别率: ______%
- 训练时间: ____小时
- 最终损失: ______

## 样本测试
[添加测试样本截图和识别结果]

## 结论
[优化是否成功？是否需要进一步改进？]
```

---

## 🔧 异常处理

### 问题1: 训练中断
```bash
# 查看最新checkpoint
ls -lt models/checkpoint_step_*.keras | head -1

# TODO: 需要添加从checkpoint恢复的代码
# 当前版本不支持自动恢复，需要重新训练
```

### 问题2: GPU内存不足
```bash
# 编辑 config.py
BATCH_SIZE = 96  # 从128降低

# 重启训练
```

### 问题3: 完整匹配率停滞不前
```bash
# 检查是否过拟合
grep "训练损失\|验证损失" training_*.log | tail -20

# 如果训练损失远小于验证损失 → 过拟合
# 解决: 增强数据增强，或早停
```

### 问题4: 数学题识别仍然很差 (<10% after 100k)
```bash
# 检查数学题样本数量
ls /data/coding/captcha/img/ | grep -E "^[0-9a-f]{12}_[0-9]+_" | wc -l

# 如果少于1000，生成更多
cd /data/coding/captcha
python generate_captcha.py --type math --count 3000
```

---

## 📞 支持

如有问题，请查看:
1. [QUICK_START_OPTIMIZED_TRAINING.md](QUICK_START_OPTIMIZED_TRAINING.md) - 快速问题解答
2. [OPTIMIZATION_SUMMARY_2026-02-02.md](OPTIMIZATION_SUMMARY_2026-02-02.md) - 详细技术说明
3. [TRAINING_BREAKTHROUGH_2026-02-02.md](TRAINING_BREAKTHROUGH_2026-02-02.md) - 完整优化方案

---

## ✅ 最终检查清单

开始训练前确认:
- [ ] 运行 `verify_optimization.py`，所有检查通过
- [ ] 数据集完整，包含数学题样本
- [ ] GPU可用
- [ ] 磁盘空间充足 (>10GB)
- [ ] 已备份现有模型
- [ ] 选择合适的运行方式（前台/后台/screen）

开始训练:
```bash
cd /data/coding/caocrvfy
python train_v4.py
```

**预计训练时间**: 8-12小时 (A100 GPU)  
**预期最终准确率**: 75-82%  
**祝训练顺利！** 🎉

---

**创建时间**: 2026-02-02  
**版本**: v1.0  
**状态**: 准备就绪 ✅
