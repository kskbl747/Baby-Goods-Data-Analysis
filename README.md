# MotherBabyInsight 母婴电商消费行为分析

MotherBabyInsight 基于淘宝母婴购物数据，完成数据清洗、描述性统计、特征工程、商品大类识别模型、高购买量订单辅助模型、报告图表和 ECharts 可视化大屏生成。主体流程使用 `Python / Pandas / scikit-learn / Matplotlib / ECharts`，Spark 可作为后续迁移方案说明，不在默认流程中强制运行。

## 项目结构

```text
MotherBabyInsight/
├── mother_baby_insight.py
├── app.py
├── src/
│   └── mother_baby_insight.py
├── notebooks/
│   └── 01_run_project.ipynb
├── scripts/
│   └── run_all.sh
├── assets/
│   └── masks/
│       ├── mother_baby_wordcloud_mask_children.png
│       └── mother_baby_wordcloud_mask_stroller.png
├── data/
│   └── raw/
│       ├── sam_tianchi_mum_baby.csv
│       └── sam_tianchi_mum_baby_trade_history.csv
├── outputs/
│   ├── dashboard_data.json
│   ├── html/dashboard.html
│   ├── charts/
│   ├── report_figures/
│   ├── report_tables/
│   ├── model_results/
│   └── processed/
├── .env.example
├── requirements.txt
└── README.md
```

## 快速运行

Windows PowerShell：

```powershell
pip install -r requirements.txt
python src/mother_baby_insight.py
```

VMware Ubuntu：

```bash
cd ~/mother_baby_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/mother_baby_insight.py
```

或者使用脚本：

```bash
bash scripts/run_all.sh
```

## Jupyter Notebook 复现

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

打开：

```text
notebooks/01_run_project.ipynb
```

Notebook 包含环境检查、原始数据检查、完整主流程运行、输出文件检查和大屏嵌入。

## 可视化大屏

主程序会生成：

```text
outputs/html/dashboard.html
outputs/dashboard_data.json
```

大屏为双 Tab 结构：

- `业务总览`：KPI、购买趋势、商品大类 TOP6、高购买量订单、星期活跃、词云、结论总结。
- `模型验证`：模型摘要、cat1 主模型性能对比、特征重要性、模型结论总结、已匹配画像分析。

大屏优先读取 `outputs/dashboard_data.json` 渲染；考虑到双击本地 HTML 可能存在 `file://` 读取限制，程序也会向 HTML 注入同源真实 payload 作为 fallback。词云图片通过 `../charts/09_wordcloud.png` 引用，不写入 base64。

当前版本的大屏不依赖大模型 API。页面中的“结论总结”和“模型结论总结”均为基于项目统计结果、模型指标和特征重要性生成的固定分析内容，直接双击 `dashboard.html` 即可完整展示，不需要配置 API Key，也不需要启动后端服务。

如果本地双击路径不稳定，推荐启动静态服务：

```bash
cd outputs
python3 -m http.server 8000
```

访问：

```text
http://localhost:8000/html/dashboard.html
```

## AI 专家分析接口（可选扩展）

当前 `dashboard.html` 不依赖大模型接口；`app.py` 仅作为后续扩展文件保留。如果需要二次开发为动态 AI 分析，可启动该接口：

```bash
python app.py
```

接口：

```text
/api/expert-analysis?scene=business
/api/expert-analysis?scene=model
```

API Key 只从环境变量读取，不写入代码或 HTML。默认大屏运行不需要配置 API Key。

`.env.example`：

```text
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Windows PowerShell：

```powershell
$env:LLM_API_KEY="your_api_key_here"
$env:LLM_API_BASE="https://api.openai.com/v1"
$env:LLM_MODEL="gpt-4o-mini"
```

Linux / VMware Ubuntu：

```bash
export LLM_API_KEY="your_api_key_here"
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
```

没有配置 `LLM_API_KEY` 时，接口会返回本地规则分析；但当前大屏已经改为固定结论总结，不会请求该接口。

## AI 专家分析配置说明（仅二次开发需要）

如果后续需要重新启用动态 AI 分析，推荐使用 `.env` 或系统环境变量配置后端，由 `app.py` 统一调用大模型接口：

```text
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

启动后端：

```bash
python app.py
```

然后打开：

```text
outputs/html/dashboard.html
```

当前大屏已删除前端 AI 设置入口，不会在浏览器中保存 API Key。正式提交、上传虚拟机和答辩复现时，直接运行 `python src/mother_baby_insight.py` 后打开 `outputs/html/dashboard.html` 即可完整展示。

## 报告输出

报告表格：

```text
outputs/report_tables/
```

包括数据概览、cat1 主模型指标、high_buy 辅助模型指标、特征重要性、商品销量、月度趋势和星期趋势。

报告图表：

```text
outputs/report_figures/
```

包括高购买量订单分布、数值变量分布、相关系数热力图、特征重要性、模型对比、ROC、混淆矩阵、预测概率分布、词云和大屏索引图。`figure_index.csv` 记录所有图表文件。

## 数据口径

所有大屏指标统一由主程序计算并写入 `outputs/dashboard_data.json`，不要在 `dashboard.html` 中手写散落指标。当前口径包括：

- 原始交易记录数：来自交易历史原始 CSV。
- 清洗后有效交易记录数：清洗、去重和日期转换后的记录数。
- 交易用户数：清洗后 `user_id` 去重。
- 商品 ID 数：交易表中 `auction_id` 去重后的商品编号数量，当前为 28,141；该指标不是商品大类数量。
- 商品大类：由 `cat1` 数字编码映射为中文业务类别，共 6 类，用于商品结构分析、排行图和词云热点展示。
- 总购买数量：`buy_mount` 汇总。
- 高购买量订单：`buy_mount` 高于 75% 分位数。
- 销量贡献率和平均购买量提升：基于真实 `high_buy` 分组计算。
- 词云图：不是商品标题分词结果，而是基于商品大类中文映射、母婴业务词扩展和 `buy_mount` 购买数量加权生成的消费热点词云。

## 模型说明

主模型为 `cat1` 商品大类识别模型，使用商品细分类别、商品属性频次、商品热度、时间和画像特征。该任务属于商品层级分类识别，不是完全基于用户行为预测购买类别。

当前模型解释口径：

- 最高分模型：`Gradient Boosting`。
- 推荐展示模型：`Random Forest`，原因是指标较高且能输出特征重要性，适合报告和答辩解释。
- `high_buy` 是二分类辅助模型，AUC 只用于该任务，不混入 cat1 多分类图。
- `high_buy` 模型主动删除 `buy_mount`、`user_total_buy_mount`、`item_total_buy_mount`、`cat_total_buy_mount` 等泄露风险字段，结果较低但更真实可信。

## 词云 Mask

词云优先使用本次提供的儿童剪影 mask：

```text
assets/masks/mother_baby_wordcloud_mask_children.png
```

如果该 mask 缺失，程序会自动尝试使用婴儿车 mask `assets/masks/mother_baby_wordcloud_mask_stroller.png`；仍不存在时降级为普通矩形词云。
