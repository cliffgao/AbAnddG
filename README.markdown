# 抗体–抗原突变 ΔΔG 预测程序

## 1. 程序用途

本程序使用抗体–抗原复合物的野生型（WT）PDB 结构和突变信息，预测突变引起的结合自由能变化 ΔΔG。

程序不要求突变体 PDB。WT 和突变体分支使用相同的 WT 主链坐标，区别来自 WT/突变序列对应的 ESM-2 上下文嵌入；可选加入 typed persistent homology（TDA）特征。

主要程序如下：

| 文件 | 用途 |
|---|---|
| `code/generate_graphs.py` | 从 PDB、链信息和突变生成 WT/Mutant 配对图 |
| `code/trains.py` | 训练、验证并可选测试 ddG 模型 |
| `code/predict_ddg.py` | 使用训练好的模型预测未知突变 |
| `code/MultiTaskGeoBioKAN.py` | GNN 编码器、Siamese ddG 模型和反对称输出头 |
| `code/tda_features.py` | 计算可选的 40 维 typed-PH 特征 |
| `code/my_dataset_loader.py` | 加载和批处理配对图 |

## 2. 模型使用的特征

### 2.1 序列特征

每个残基使用 ESM-2 产生的 1280 维上下文嵌入。同一条链发生一个突变后，会重新计算该链的完整 ESM-2 嵌入，因此突变可以影响整条链的上下文表示。

### 2.2 结构和图特征

- 默认使用 WT PDB 中每个残基的 CA 坐标。
- 抗体重链–抗原、轻链–抗原之间按距离阈值构建界面图，当前阈值为 16.5 Å。
- 每条边包含：
  - 残基间距离；
  - 边来源的二元标记（重链–抗原或轻链–抗原）。
- 距离经过 16 维 RBF 展开后输入 GATv2Conv。
- 编码器包含两层 GATv2Conv、残差连接、GraphSizeNorm、attention pooling 和 global mean pooling。

每个节点最终输入维度为 1286：1280 维 ESM-2 嵌入、3 维链类型 one-hot（重链/轻链/抗原）和 3 维 WT 坐标。

### 2.3 可选 TDA 特征

TDA 默认关闭。启用时，程序按以下 5 个残基理化类型建立通道：

- 疏水；
- 芳香；
- 极性；
- 正电；
- 负电。

每个通道计算 H0 和 H1 persistent homology，并对每个同调维度提取：bar 数量、最大 persistence、总 persistence 和平均 persistence，最终得到：

```text
5 个通道 × 2 个同调维度 × 4 个统计量 = 40 维
```

WT 和 Mutant 共享坐标，但突变会使残基在理化通道之间移动，因此两者的 typed-PH 特征可以不同。TDA 特征与图 pooling 表示拼接后进入共享编码器输出。

### 2.4 Siamese 和反对称预测

同一个图编码器分别编码 WT 视角和 Mutant 视角：

```text
h_wt  = Encoder(graph_wt)
h_mut = Encoder(graph_mut)
delta = h_wt - h_mut
```

输出头采用：

```text
f(delta) = MLP(delta) - MLP(-delta)
```

因此交换两个输入时严格满足：

```text
f(WT, Mutant) = -f(Mutant, WT)
```

训练集默认同时加入交换顺序、标签取反的数据增强。

## 3. 输入数据

### 3.1 PDB 文件

每个复合物需要一个 WT PDB 文件：

```text
pdbs/<proid>.pdb
```

例如 CSV 中 `proid` 为 `1ABC`，程序会读取：

```text
pdbs/1ABC.pdb
```

PDB 必须包含 CSV 指定的重链、轻链和抗原链。

### 3.2 CSV 文件

训练和验证 CSV 必须包含：

```csv
proid,Hchain,Lchain,Achain,mutations,ddg
1ABC,H,L,A,H:53:R:Q,1.25
2XYZ,H,L,B,H:100A:V:L;-0.72
```

预测 CSV 不需要 `ddg`：

```csv
proid,Hchain,Lchain,Achain,mutations
1ABC,H,L,A,H:53:R:Q
```

列含义：

| 列 | 含义 |
|---|---|
| `proid` | PDB 文件名，不含 `.pdb` |
| `Hchain` | 抗体重链 ID |
| `Lchain` | 抗体轻链 ID |
| `Achain` | 抗原链 ID |
| `mutations` | 一个或多个突变 |
| `ddg` | 训练/验证标签；预测时不需要 |

多点突变使用分号连接：

```text
H:53:R:Q;H:97:L:W
```

每个突变格式为：

```text
链ID:位置:WT氨基酸:Mutant氨基酸
```

氨基酸使用单字母代码。

### 3.3 突变编号模式

程序支持两种编号方式：

- `--mutation-numbering seq0`：位置是结构中实际观测残基的 0-based 下标；这是兼容旧数据的默认值。
- `--mutation-numbering pdb`：位置是 PDB residue number，并支持 `100A` 这样的 insertion code。

推荐人工输入或通常的突变表使用 `pdb`。如果位置越界或指定的 WT 氨基酸与 PDB 不一致，程序会拒绝该样本，防止错误编号产生错误预测。

### 3.4 ddG 符号

本项目设计约定为：

```text
ddG = dG(WT) - dG(Mutant)
```

训练 CSV 的 `ddg` 必须统一采用同一符号。如果原始数据使用 `dG(Mutant) - dG(WT)`，应在训练前将标签乘以 `-1`。程序直接学习并输出原始 ddG 单位，不使用 `scaler_dg`、`scaler_kd` 或 ddG scaler。

## 4. 环境依赖

核心依赖包括：

```text
Python
PyTorch
PyTorch Geometric
transformers
Biopython
pandas
numpy
scikit-learn
scipy
matplotlib
```

启用 TDA 时额外需要：

```text
gudhi
```

还需要本地 ESM-2 模型目录。当前图节点维度按 1280 维 ESM-2 输出设计，因此所用 ESM-2 模型必须产生 1280 维残基嵌入。

## 5. 生成训练图

训练集：

```bash
python code/generate_graphs.py \
  --train \
  --ids data/train.csv \
  --pdb pdbs \
  --esm /path/to/esm2 \
  --out graphs_ddg \
  --atom CA \
  --mutation-numbering pdb
```

验证集使用相同的输出目录：

```bash
python code/generate_graphs.py \
  --train \
  --ids data/valid.csv \
  --pdb pdbs \
  --esm /path/to/esm2 \
  --out graphs_ddg \
  --atom CA \
  --mutation-numbering pdb
```

每行会生成：

```text
graphs_ddg/<proid>_<mutation>.pt
```

`.pt` 中保存 `graph_wt`、`graph_mut`、链掩码、ddG 标签及可选 TDA 特征。

### 启用 TDA

生成训练图和验证图时均增加：

```bash
--compute-tda
```

也可以用 `--tda-max-edge` 修改 Vietoris–Rips filtration 的最大边长，默认值为 16.0。

## 6. 训练模型

### 6.1 不使用 TDA

```bash
python code/trains.py \
  --train data/train.csv \
  --valid data/valid.csv \
  --graphs graphs_ddg \
  --from-csv \
  --flag models/ddg_model \
  --hidden-dim 16 \
  --heads 2 \
  --bsize 32 \
  --lr 0.001 \
  --seed 10
```

### 6.2 使用 TDA

图必须提前用 `--compute-tda` 生成，然后训练时增加：

```bash
--tda-dim 40
```

完整示例：

```bash
python code/trains.py \
  --train data/train.csv \
  --valid data/valid.csv \
  --graphs graphs_ddg_tda \
  --from-csv \
  --tda-dim 40 \
  --flag models/ddg_tda
```

### 6.3 迁移已有编码器

如果有兼容的绝对亲和力模型 checkpoint，可以增加：

```bash
--pretrained-encoder /path/to/pretrained_affinity.pt
```

该参数只初始化共享图编码器；ddG 输出头仍从随机参数开始训练。

### 6.4 训练输出

程序以验证集 MSE 作为最优模型标准。每当验证 MSE 下降，就覆盖保存：

```text
models/ddg_model.best.pt
```

同时输出：

```text
models/ddg_model.loss.png
models/ddg_model.valid_predictions.csv
```

`.best.pt` 是模型的 `state_dict`，预测时的 `--hidden-dim`、`--heads` 和 `--tda-dim` 必须与训练配置相同。

## 7. 训练后测试

可以在训练命令中加入测试集：

```bash
python code/trains.py \
  --train data/train.csv \
  --valid data/valid.csv \
  --test data/test.csv \
  --graphs graphs_ddg \
  --from-csv \
  --flag models/ddg_model
```

程序加载刚保存的最优 `.best.pt`，报告：

- MSE；
- RMSE；
- Pearson r；
- Spearman rho；
- R²。

测试预测保存到：

```text
models/ddg_model.test_predictions.csv
```

## 8. 预测未知突变

### 8.1 自动生成图并预测

```bash
python code/predict_ddg.py \
  --input data/mutations.csv \
  --output results/predictions.csv \
  --pdb pdbs \
  --graphs prediction_graphs \
  --esm /path/to/esm2 \
  --model models/ddg_model.best.pt \
  --hidden-dim 16 \
  --heads 2 \
  --mutation-numbering pdb
```

使用 TDA 模型时增加：

```bash
--tda-dim 40
```

此时程序会自动为预测样本计算 40 维 TDA 特征。

### 8.2 复用已经生成的图

```bash
python code/predict_ddg.py \
  --input data/mutations.csv \
  --output results/predictions.csv \
  --graphs prediction_graphs \
  --model models/ddg_model.best.pt \
  --skip-graph-generation
```

复用图时，`--tda-dim` 仍必须与模型和已生成图一致。

### 8.3 预测输出

输出 CSV 保留所有输入列，并增加：

```text
predicted_ddg
```

例如：

```csv
proid,Hchain,Lchain,Achain,mutations,predicted_ddg
1ABC,H,L,A,H:53:R:Q,0.84
```

## 9. 常见错误

### 找不到 PDB

确认文件名严格为 `<proid>.pdb`，并位于 `--pdb` 指定目录。

### WT 氨基酸不一致

通常由 `seq0`/`pdb` 编号模式选错、PDB 缺失残基或 insertion code 未写造成。不要绕过该检查，应先核实突变编号。

### 模型权重尺寸不匹配

预测参数 `--hidden-dim`、`--heads` 和 `--tda-dim` 必须与训练时完全一致。

### 模型要求 TDA，但图中没有 TDA

使用 `--compute-tda` 重新生成图，并在训练和预测时均使用 `--tda-dim 40`。

### 输出值没有物理意义

确认模型已经在真实 ddG 标签上训练，并核实数据的 ddG 符号和单位。随机初始化的 `.pt` 或绝对亲和力模型不能直接用于 ddG 预测。
