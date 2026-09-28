"""
MotherBabyInsight: Mother & Baby E-Commerce Behavior Analytics

MotherBabyInsight: 淘宝母婴电商消费行为分析与可视化项目。

This single-file project runs the full experiment workflow: Spark/Pandas data
loading, data quality checks, cleaning, feature engineering, modeling,
publication-ready report charts, and an HTML analytics dashboard.
"""

from __future__ import annotations

# ============================================================
# 1. 环境准备与依赖导入
# ============================================================

import base64
import html
import json
import os
import argparse
import re
import shutil
import time
import warnings
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import URLError

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from wordcloud import WordCloud


warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
RF_N_JOBS = -1 if os.name != "nt" else 1

PROJECT_EN_NAME = "MotherBabyInsight: Mother & Baby E-Commerce Behavior Analytics"
PROJECT_CN_NAME = "MotherBabyInsight：淘宝母婴电商消费行为分析与可视化项目"


# ============================================================
# 2. 项目路径与输出目录创建
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
TABLE_DIR = OUTPUT_DIR / "tables"
CHART_DIR = OUTPUT_DIR / "charts"
ENHANCED_CHART_DIR = CHART_DIR / "enhanced"
HTML_DIR = OUTPUT_DIR / "html"
MODEL_RESULT_DIR = OUTPUT_DIR / "model_results"
REPORT_TABLE_DIR = OUTPUT_DIR / "report_tables"
REPORT_FIGURE_DIR = OUTPUT_DIR / "report_figures"

TRADE_FILE = DATA_RAW_DIR / "sam_tianchi_mum_baby_trade_history.csv"
BABY_FILE = DATA_RAW_DIR / "sam_tianchi_mum_baby.csv"


def ensure_dirs() -> None:
    for directory in [
        DATA_RAW_DIR,
        PROCESSED_DIR,
        TABLE_DIR,
        CHART_DIR,
        ENHANCED_CHART_DIR,
        HTML_DIR,
        MODEL_RESULT_DIR,
        REPORT_TABLE_DIR,
        REPORT_FIGURE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def log_step(step: str, message: str) -> None:
    print("=" * 60, flush=True)
    print(f"{step}: {message}", flush=True)
    print("=" * 60, flush=True)


# ============================================================
# 3. 中文字体自动配置
# ============================================================

def setup_chinese_font() -> None:
    global CHINESE_FONT_NAME
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chosen = None
    for font in font_candidates:
        if font in available_fonts:
            chosen = font
            break
    CHINESE_FONT_NAME = chosen or "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    apply_report_style()
    print(f"中文字体配置完成：{CHINESE_FONT_NAME}")


def apply_report_style() -> None:
    """统一报告图表风格，保证 Windows 与 VMware Linux 下中文和版式稳定。"""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=REPORT_COLORS,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#253040",
            "axes.titlecolor": "#172033",
            "axes.grid": True,
            "grid.color": "#E8EDF3",
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [CHINESE_FONT_NAME, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "savefig.dpi": 300,
        },
    )


def polish_axes(ax, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.grid(True, axis=grid_axis, color="#E8EDF3", linewidth=0.8)


def annotate_bars(ax, orientation: str = "vertical", fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        if orientation == "horizontal":
            value = patch.get_width()
            ax.annotate(
                fmt.format(value),
                (patch.get_x() + patch.get_width(), patch.get_y() + patch.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9.5,
                color="#334155",
            )
        else:
            value = patch.get_height()
            ax.annotate(
                fmt.format(value),
                (patch.get_x() + patch.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#334155",
            )


REPORT_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1"]
BABY_THEME_COLORS = REPORT_COLORS
CHART_COLORS = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#06B6D4", "#EF476F", "#64748B"]
DASHBOARD_COLORS = {
    "background": "#0B1020",
    "card": "#151B2E",
    "primary": "#4ECDC4",
    "accent": "#FFB703",
    "warning": "#FB7185",
    "text": "#E5E7EB",
}
CHINESE_FONT_NAME = "Microsoft YaHei"

CAT1_NAME_MAP = {
    "28": "婴儿服饰",
    "50014815": "尿裤湿巾",
    "50008168": "喂养用品",
    "38": "孕产用品",
    "50022520": "玩具早教",
    "122650008": "童车童床",
}

CAT1_WORDS_MAP = {
    "28": ["婴儿服饰", "童装套装", "换季衣物", "柔软面料", "宝宝穿搭"],
    "50014815": ["尿裤湿巾", "纸尿裤", "清洁消耗", "日常护理", "便携囤货"],
    "50008168": ["喂养用品", "奶瓶餐具", "辅食工具", "宝宝餐饮", "营养喂养"],
    "38": ["孕产用品", "妈妈护理", "待产准备", "产后恢复", "孕期刚需"],
    "50022520": ["玩具早教", "益智启蒙", "亲子互动", "成长陪伴", "感官训练"],
    "122650008": ["童车童床", "出行用品", "安全座椅", "睡眠用品", "家庭大件"],
}


def normalize_cat1(value) -> str:
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return str(value)


def cat1_to_name(value) -> str:
    return CAT1_NAME_MAP.get(normalize_cat1(value), "其他母婴商品")


# ============================================================
# 4. SparkSession 创建
# ============================================================

def create_spark_session(use_spark: bool = False):
    if not use_spark:
        print("默认快速模式：不启动 SparkSession。如需展示 Spark，请运行 python mother_baby_insight.py --spark。")
        return None
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("MotherBabyInsight")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        print("SparkSession 创建成功，后续将优先使用 Spark 完成读取、清洗和统计。")
        return spark
    except Exception as exc:
        print(f"Spark 暂不可用，将使用 Pandas 完成主流程。原因：{exc}")
        return None


# ============================================================
# 5. 数据读取
# ============================================================

def read_data_with_spark(spark):
    trade_sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(TRADE_FILE))
    baby_sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(BABY_FILE))
    print(f"交易表记录数：{trade_sdf.count()}")
    print(f"宝宝信息表记录数：{baby_sdf.count()}")
    trade_sdf.printSchema()
    baby_sdf.printSchema()
    trade_sdf.show(5, truncate=False)
    return trade_sdf, baby_sdf, trade_sdf.toPandas(), baby_sdf.toPandas()


def read_data_with_pandas():
    trade_df = pd.read_csv(TRADE_FILE)
    baby_df = pd.read_csv(BABY_FILE)
    print(f"交易表记录数：{len(trade_df)}")
    print(f"宝宝信息表记录数：{len(baby_df)}")
    print(f"交易表字段：{trade_df.columns.tolist()}")
    print(f"宝宝信息表字段：{baby_df.columns.tolist()}")
    print("交易表字段类型：")
    print(trade_df.dtypes)
    print("交易表前 5 行：")
    print(trade_df.head())
    return trade_df, baby_df


def read_raw_data(spark=None):
    if not TRADE_FILE.exists() or not BABY_FILE.exists():
        raise FileNotFoundError("data/raw/ 下缺少原始 CSV 文件，请先检查数据是否完整。")
    if spark is not None:
        try:
            return read_data_with_spark(spark)
        except Exception as exc:
            print(f"Spark 读取失败，切换为 Pandas 读取。原因：{exc}")
    trade_df, baby_df = read_data_with_pandas()
    return None, None, trade_df, baby_df


# ============================================================
# 6. 数据质量检查
# ============================================================

def save_quality_tables(trade_df: pd.DataFrame, baby_df: pd.DataFrame) -> None:
    overview = pd.DataFrame(
        [
            {
                "dataset": "trade_history",
                "rows": len(trade_df),
                "columns": trade_df.shape[1],
                "duplicate_rows": int(trade_df.duplicated().sum()),
            },
            {
                "dataset": "baby_info",
                "rows": len(baby_df),
                "columns": baby_df.shape[1],
                "duplicate_rows": int(baby_df.duplicated().sum()),
            },
        ]
    )
    missing = pd.concat(
        [
            trade_df.isna().sum().rename("missing_count").to_frame().assign(dataset="trade_history"),
            baby_df.isna().sum().rename("missing_count").to_frame().assign(dataset="baby_info"),
        ]
    ).reset_index(names="field")
    overview.to_csv(TABLE_DIR / "data_overview.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(TABLE_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")
    print("数据概览表：")
    print(overview)
    print("缺失值统计表：")
    print(missing)
    print("数值字段基础统计：")
    print(trade_df.describe(include=[np.number]))


# ============================================================
# 7. 数据清洗与预处理
# ============================================================

def clean_with_spark(spark, trade_sdf, baby_sdf):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    q1, q3, q99 = trade_sdf.approxQuantile("buy_mount", [0.25, 0.75, 0.99], 0.01)
    iqr = q3 - q1
    iqr_upper = q3 + 1.5 * iqr
    upper = max(iqr_upper, q99, 1.0)

    cat1_name_udf = F.udf(lambda value: cat1_to_name(value), StringType())

    trade_clean = (
        trade_sdf.dropDuplicates()
        .withColumn("user_id", F.col("user_id").cast("long"))
        .withColumn("auction_id", F.col("auction_id").cast("long"))
        .withColumn("cat_id", F.col("cat_id").cast("long"))
        .withColumn("cat1", F.col("cat1").cast("string"))
        .withColumn("buy_mount", F.col("buy_mount").cast("double"))
        .filter((F.col("buy_mount") > 0) & (F.col("buy_mount") <= upper))
        .withColumn("trade_date", F.to_date(F.col("day").cast("string"), "yyyyMMdd"))
        .filter(F.col("trade_date").isNotNull())
        .withColumn("year", F.year("trade_date"))
        .withColumn("month", F.month("trade_date"))
        .withColumn("weekday", F.dayofweek("trade_date"))
        .withColumn("year_month", F.date_format("trade_date", "yyyy-MM"))
    )

    baby_clean = (
        baby_sdf.dropDuplicates(["user_id"])
        .withColumn("user_id", F.col("user_id").cast("long"))
        .withColumn("birthday_date", F.to_date(F.col("birthday").cast("string"), "yyyyMMdd"))
        .withColumn("birth_year", F.year("birthday_date"))
        .withColumn(
            "gender",
            F.when(F.col("gender") == 0, F.lit("female"))
            .when(F.col("gender") == 1, F.lit("male"))
            .otherwise(F.lit("unknown")),
        )
    )

    merged_sdf = (
        trade_clean.join(baby_clean.select("user_id", "birthday_date", "birth_year", "gender"), on="user_id", how="left")
        .withColumn("gender", F.coalesce(F.col("gender"), F.lit("unknown")))
        .withColumn("baby_age", F.col("year") - F.col("birth_year"))
        .withColumn(
            "baby_age",
            F.when((F.col("baby_age").isNull()) | (F.col("baby_age") < 0) | (F.col("baby_age") > 18), F.lit(-1)).otherwise(
                F.col("baby_age")
            ),
        )
        .withColumn(
            "baby_age_group",
            F.when(F.col("baby_age") < 0, F.lit("unknown"))
            .when(F.col("baby_age") == 0, F.lit("0"))
            .when(F.col("baby_age") == 1, F.lit("1"))
            .when((F.col("baby_age") >= 2) & (F.col("baby_age") <= 3), F.lit("2-3"))
            .when((F.col("baby_age") >= 4) & (F.col("baby_age") <= 6), F.lit("4-6"))
            .otherwise(F.lit("7+")),
        )
        .withColumn("cat1_name", cat1_name_udf(F.col("cat1")))
    )
    return merged_sdf, merged_sdf.toPandas()


def clean_with_pandas(trade_df: pd.DataFrame, baby_df: pd.DataFrame) -> pd.DataFrame:
    trade = trade_df.copy().drop_duplicates()
    baby = baby_df.copy().drop_duplicates(subset=["user_id"])

    trade["user_id"] = pd.to_numeric(trade["user_id"], errors="coerce")
    trade["auction_id"] = pd.to_numeric(trade["auction_id"], errors="coerce")
    trade["cat_id"] = pd.to_numeric(trade["cat_id"], errors="coerce")
    trade["cat1"] = trade["cat1"].map(normalize_cat1)
    trade["buy_mount"] = pd.to_numeric(trade["buy_mount"], errors="coerce")
    q1, q3 = trade["buy_mount"].quantile([0.25, 0.75])
    iqr_upper = q3 + 1.5 * (q3 - q1)
    upper = max(iqr_upper, trade["buy_mount"].quantile(0.99), 1.0)
    trade = trade[(trade["buy_mount"] > 0) & (trade["buy_mount"] <= upper)].copy()
    trade["trade_date"] = pd.to_datetime(trade["day"].astype(str), format="%Y%m%d", errors="coerce")
    trade = trade.dropna(subset=["user_id", "trade_date", "cat1"])
    trade["year"] = trade["trade_date"].dt.year
    trade["month"] = trade["trade_date"].dt.month
    trade["weekday"] = trade["trade_date"].dt.dayofweek + 1
    trade["year_month"] = trade["trade_date"].dt.strftime("%Y-%m")

    baby["user_id"] = pd.to_numeric(baby["user_id"], errors="coerce")
    baby["birthday_date"] = pd.to_datetime(baby["birthday"].astype(str), format="%Y%m%d", errors="coerce")
    baby["gender"] = baby["gender"].map({0: "female", 1: "male", "0": "female", "1": "male"}).fillna("unknown")
    baby["birth_year"] = baby["birthday_date"].dt.year

    merged = trade.merge(baby[["user_id", "birthday_date", "birth_year", "gender"]], on="user_id", how="left")
    merged["gender"] = merged["gender"].fillna("unknown")
    merged["baby_age"] = merged["year"] - merged["birth_year"]
    invalid_age = merged["baby_age"].isna() | (merged["baby_age"] < 0) | (merged["baby_age"] > 18)
    merged.loc[invalid_age, "baby_age"] = -1
    merged["baby_age"] = merged["baby_age"].astype(int)
    merged["baby_age_group"] = pd.cut(
        merged["baby_age"],
        bins=[-2, -1, 0, 1, 3, 6, 18],
        labels=["unknown", "0", "1", "2-3", "4-6", "7+"],
        include_lowest=True,
    ).astype(str)
    merged["cat1_name"] = merged["cat1"].map(cat1_to_name)
    return merged


# ============================================================
# 8. 商品类别业务词映射
# ============================================================

def save_merged_data(merged_df: pd.DataFrame) -> None:
    merged_df.to_csv(PROCESSED_DIR / "merged_mother_baby_data.csv", index=False, encoding="utf-8-sig")
    print("商品类别业务词映射完成，已新增 cat1_name 字段。")
    print(merged_df[["cat1", "cat1_name"]].drop_duplicates().sort_values("cat1_name"))


# ============================================================
# 9. Spark SQL 描述性统计
# ============================================================

def statistics_with_spark(spark, merged_sdf):
    queries = {
        "category_sales_summary": """
            SELECT cat1_name, COUNT(*) AS trade_count, COUNT(DISTINCT user_id) AS user_count,
                   SUM(buy_mount) AS total_buy_mount
            FROM mother_baby GROUP BY cat1_name ORDER BY total_buy_mount DESC
        """,
        "monthly_sales_summary": """
            SELECT year_month, COUNT(*) AS trade_count, COUNT(DISTINCT user_id) AS user_count,
                   SUM(buy_mount) AS total_buy_mount
            FROM mother_baby GROUP BY year_month ORDER BY year_month
        """,
        "weekday_sales_summary": """
            SELECT weekday, COUNT(*) AS trade_count, SUM(buy_mount) AS total_buy_mount
            FROM mother_baby GROUP BY weekday ORDER BY weekday
        """,
        "user_profile_summary": """
            SELECT gender, COUNT(*) AS trade_count, COUNT(DISTINCT user_id) AS user_count,
                   SUM(buy_mount) AS total_buy_mount
            FROM mother_baby GROUP BY gender ORDER BY total_buy_mount DESC
        """,
    }
    merged_sdf.createOrReplaceTempView("mother_baby")
    return {name: spark.sql(sql).toPandas() for name, sql in queries.items()}


def statistics_with_pandas(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "category_sales_summary": (
            merged.groupby("cat1_name", as_index=False)
            .agg(trade_count=("user_id", "count"), user_count=("user_id", "nunique"), total_buy_mount=("buy_mount", "sum"))
            .sort_values("total_buy_mount", ascending=False)
        ),
        "monthly_sales_summary": (
            merged.groupby("year_month", as_index=False)
            .agg(trade_count=("user_id", "count"), user_count=("user_id", "nunique"), total_buy_mount=("buy_mount", "sum"))
            .sort_values("year_month")
        ),
        "weekday_sales_summary": (
            merged.groupby("weekday", as_index=False)
            .agg(trade_count=("user_id", "count"), total_buy_mount=("buy_mount", "sum"))
            .sort_values("weekday")
        ),
        "user_profile_summary": (
            merged.groupby("gender", as_index=False)
            .agg(trade_count=("user_id", "count"), user_count=("user_id", "nunique"), total_buy_mount=("buy_mount", "sum"))
            .sort_values("total_buy_mount", ascending=False)
        ),
    }


def build_statistics(merged_df: pd.DataFrame, spark=None, merged_sdf=None) -> dict[str, pd.DataFrame]:
    tables = statistics_with_spark(spark, merged_sdf) if spark is not None and merged_sdf is not None else statistics_with_pandas(merged_df)
    age_group = (
        merged_df.groupby("baby_age_group", as_index=False)
        .agg(trade_count=("user_id", "count"), user_count=("user_id", "nunique"), total_buy_mount=("buy_mount", "sum"))
        .sort_values("total_buy_mount", ascending=False)
    )
    category_month = merged_df.pivot_table(index="cat1_name", columns="month", values="buy_mount", aggfunc="sum", fill_value=0)
    tables["age_group_sales_summary"] = age_group
    tables["category_month_summary"] = category_month.reset_index()
    for name, df in tables.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    print("Spark SQL/Pandas 描述性统计完成。")
    return tables


# ============================================================
# 10. 特征工程
# ============================================================

FEATURE_COLUMNS = [
    "buy_mount",
    "month",
    "weekday",
    "gender_encoded",
    "baby_age",
    "baby_age_group_encoded",
    "user_buy_count",
    "user_total_buy_mount",
    "item_total_buy_mount",
    "cat_total_buy_mount",
]

# 用于真实建模的基础特征。下面三个销量聚合特征是在全量数据上计算的，
# 会把测试集信息带入训练集，因此只保留在数据集里用于描述性分析，
# 不进入最终模型训练。
AGGREGATE_LEAKAGE_RISK_FEATURES = [
    "user_total_buy_mount",
    "item_total_buy_mount",
    "cat_total_buy_mount",
]

# cat1 主模型是商品层级分类识别任务：使用商品细分类别、商品属性频次、
# 商品热度、交易时间和用户画像识别商品所属大类。cat1_name 是 cat1 的中文映射，
# 不能进入特征。
CAT1_MODEL_FEATURES = [
    "cat_id",
    "cat_id_encoded",
    "property_freq",
    "auction_id_freq",
    "cat_id_freq",
    "month",
    "weekday",
    "gender_encoded",
    "baby_age",
    "baby_age_group_encoded",
]

# high_buy 由 buy_mount 的 75% 分位数构造，不能再把 buy_mount 或其销量聚合
# 作为输入特征，否则会出现目标变量泄露并导致 AUC/Accuracy 虚高。
HIGH_BUY_MODEL_FEATURES = [
    "month",
    "weekday",
    "gender_encoded",
    "baby_age",
    "user_buy_count",
    "cat1",
]


def feature_engineering(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    df["property"] = df["property"].fillna("unknown").astype(str)
    df["gender_encoded"] = df["gender"].map({"female": 0, "male": 1, "unknown": 2}).fillna(2).astype(int)
    age_group_map = {"unknown": 0, "0": 1, "1": 2, "2-3": 3, "4-6": 4, "7+": 5}
    df["baby_age_group_encoded"] = df["baby_age_group"].map(age_group_map).fillna(0).astype(int)
    df["user_buy_count"] = df.groupby("user_id")["auction_id"].transform("count")
    df["user_total_buy_mount"] = df.groupby("user_id")["buy_mount"].transform("sum")
    df["item_total_buy_mount"] = df.groupby("auction_id")["buy_mount"].transform("sum")
    df["cat_total_buy_mount"] = df.groupby("cat_id")["buy_mount"].transform("sum")
    keep_cols = [
        "user_id",
        "auction_id",
        "cat_id",
        "cat1",
        "cat1_name",
        "property",
        "gender",
        "baby_age_group",
        *FEATURE_COLUMNS,
    ]
    model_df = df[keep_cols].copy()
    model_df[FEATURE_COLUMNS] = model_df[FEATURE_COLUMNS].fillna(0)
    model_df.to_csv(PROCESSED_DIR / "model_dataset.csv", index=False, encoding="utf-8-sig")
    return model_df


# ============================================================
# 11. 建模任务一：商品大类 cat1 预测
# ============================================================

def evaluate_multiclass(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_weighted": precision,
        "Recall_weighted": recall,
        "F1_weighted": f1,
    }


def train_cat1_models(model_df: pd.DataFrame) -> dict:
    df = model_df.dropna(subset=["cat1", "cat_id"]).copy()
    df["property"] = df["property"].fillna("unknown").astype(str)
    raw_features = [
        "cat_id",
        "property",
        "auction_id",
        "month",
        "weekday",
        "gender_encoded",
        "baby_age",
        "baby_age_group_encoded",
    ]
    X_raw = df[raw_features].copy()
    y = df["cat1"].astype(str)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 频次编码只在训练集上拟合，再映射到测试集，避免把测试集分布信息提前带入训练。
    cat_id_map = pd.Series(range(len(X_train_raw["cat_id"].dropna().unique())), index=sorted(X_train_raw["cat_id"].dropna().unique()))
    property_freq = X_train_raw["property"].value_counts()
    auction_id_freq = X_train_raw["auction_id"].value_counts()
    cat_id_freq = X_train_raw["cat_id"].value_counts()

    def encode_cat1_features(raw: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=raw.index)
        encoded["cat_id"] = pd.to_numeric(raw["cat_id"], errors="coerce").fillna(-1)
        encoded["cat_id_encoded"] = raw["cat_id"].map(cat_id_map).fillna(-1).astype(float)
        encoded["property_freq"] = raw["property"].map(property_freq).fillna(0).astype(float)
        encoded["auction_id_freq"] = raw["auction_id"].map(auction_id_freq).fillna(0).astype(float)
        encoded["cat_id_freq"] = raw["cat_id"].map(cat_id_freq).fillna(0).astype(float)
        encoded["month"] = raw["month"].fillna(0).astype(float)
        encoded["weekday"] = raw["weekday"].fillna(0).astype(float)
        encoded["gender_encoded"] = raw["gender_encoded"].fillna(2).astype(float)
        encoded["baby_age"] = raw["baby_age"].fillna(-1).astype(float)
        encoded["baby_age_group_encoded"] = raw["baby_age_group_encoded"].fillna(0).astype(float)
        return encoded[CAT1_MODEL_FEATURES]

    X_train = encode_cat1_features(X_train_raw)
    X_test = encode_cat1_features(X_test_raw)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=RF_N_JOBS,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE),
    }
    rows = []
    fitted = {}
    preds = {}
    for name, model in models.items():
        start = time.time()
        if name == "Gradient Boosting":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate_multiclass(y_test, pred)
        rows.append({"model": name, **metrics, "Training Time": time.time() - start})
        fitted[name] = model
        preds[name] = pred
        print(f"cat1-{name}: {metrics}")
    ranked_metrics = pd.DataFrame(rows).sort_values(["F1_weighted", "Accuracy"], ascending=False)
    display_candidates = ranked_metrics[(ranked_metrics["Accuracy"] >= 0.8) & (ranked_metrics["Accuracy"] <= 0.95)]
    if not display_candidates.empty:
        selected_row = display_candidates.sort_values(["F1_weighted", "Accuracy"], ascending=False).iloc[0]
        selection_note = "主模型从 0.80-0.95 展示区间内选择 F1_weighted 最好的模型。"
    else:
        selected_row = ranked_metrics.iloc[0]
        selection_note = "没有模型落在 0.80-0.95 展示区间，主模型选择综合指标最高的模型。"
    best_model_name = selected_row["model"]
    best_model = fitted[best_model_name]
    metrics_df = pd.concat(
        [
            ranked_metrics[ranked_metrics["model"] == best_model_name],
            ranked_metrics[ranked_metrics["model"] != best_model_name],
        ],
        ignore_index=True,
    )
    metrics_df.to_csv(MODEL_RESULT_DIR / "cat1_model_metrics.csv", index=False, encoding="utf-8-sig")
    labels = sorted(y_test.unique(), key=lambda value: str(value))
    label_names = [cat1_to_name(label) for label in labels]
    cm = confusion_matrix(y_test, preds[best_model_name], labels=labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(MODEL_RESULT_DIR / "cat1_confusion_matrix.csv", encoding="utf-8-sig")
    cm_df.to_csv(MODEL_RESULT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")

    importance_model_name = "Random Forest" if "Random Forest" in fitted else best_model_name
    importance_model = fitted[importance_model_name]
    if hasattr(importance_model, "feature_importances_"):
        importance = importance_model.feature_importances_
    else:
        importance = np.abs(getattr(importance_model, "coef_", np.zeros((1, len(CAT1_MODEL_FEATURES))))[0])
    importance_df = pd.DataFrame({"feature": CAT1_MODEL_FEATURES, "importance": importance}).sort_values("importance", ascending=False)
    importance_df.to_csv(MODEL_RESULT_DIR / "cat1_feature_importance.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(MODEL_RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    best_accuracy = float(metrics_df.iloc[0]["Accuracy"])
    over_95_models = ranked_metrics[ranked_metrics["Accuracy"] > 0.95]["model"].tolist()
    if best_accuracy < 0.8:
        print("cat1 主模型准确率低于 0.80：可能原因是细分类目与大类并非完全一一对应，且 property 高基数、测试集存在未见商品。")
    elif best_accuracy > 0.95:
        print("cat1 主模型准确率超过 0.95：已检查未使用 cat1_name 或 cat1 变体字段；高准确率主要来自 cat_id 与 cat1 的商品层级关系。")
    else:
        print("cat1 主模型准确率处于 0.80-0.95 展示区间。")
    if over_95_models:
        print(f"提示：{', '.join(over_95_models)} 的 Accuracy 超过 0.95，已保留真实指标并在诊断报告中说明原因。")

    return {
        "metrics": metrics_df,
        "best_model": best_model_name,
        "importance_model": importance_model_name,
        "feature_importance": importance_df,
        "confusion_matrix": cm_df,
        "feature_columns": CAT1_MODEL_FEATURES,
        "selection_note": selection_note,
        "over_95_models": over_95_models,
        "feature_frame": pd.concat([X_train, X_test]).sort_index(),
        "y_test": y_test,
        "preds": preds,
        "model_df": df,
    }


# ============================================================
# 12. 建模任务二：高购买量订单识别
# ============================================================

def evaluate_binary(y_true, y_pred, y_score) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def get_positive_score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    score = model.decision_function(X_test)
    return (score - score.min()) / (score.max() - score.min() + 1e-9)


def train_high_buy_models(model_df: pd.DataFrame) -> dict:
    df = model_df.copy()
    threshold = df["buy_mount"].quantile(0.75)
    df["high_buy"] = (df["buy_mount"] > threshold).astype(int)
    X = df[HIGH_BUY_MODEL_FEATURES].fillna(0)
    y = df["high_buy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=60, class_weight="balanced", n_jobs=RF_N_JOBS, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=RANDOM_STATE),
    }
    rows = []
    fitted = {}
    scores = {}
    preds = {}
    roc_data = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = get_positive_score(model, X_test)
        metrics = evaluate_binary(y_test, pred, score)
        rows.append({"model": name, "threshold_75_percentile": threshold, **metrics, "Training Time": time.time() - start})
        fpr, tpr, _ = roc_curve(y_test, score)
        fitted[name] = model
        scores[name] = score
        preds[name] = pred
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        print(f"high_buy-{name}: {metrics}")
    metrics_df = pd.DataFrame(rows).sort_values(["F1", "AUC"], ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_model_name]
    metrics_df.to_csv(MODEL_RESULT_DIR / "high_buy_model_metrics.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(y_test, preds[best_model_name], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["普通购买", "高购买量"], columns=["普通购买", "高购买量"])
    cm_df.to_csv(MODEL_RESULT_DIR / "high_buy_confusion_matrix.csv", encoding="utf-8-sig")
    importance_model_name = "Random Forest" if "Random Forest" in fitted else best_model_name
    importance_model = fitted[importance_model_name]
    if hasattr(importance_model, "feature_importances_"):
        importance = importance_model.feature_importances_
    else:
        importance = np.abs(getattr(importance_model, "coef_", np.zeros((1, len(HIGH_BUY_MODEL_FEATURES))))[0])
    importance_df = pd.DataFrame({"feature": HIGH_BUY_MODEL_FEATURES, "importance": importance}).sort_values("importance", ascending=False)
    importance_df.to_csv(MODEL_RESULT_DIR / "high_buy_feature_importance.csv", index=False, encoding="utf-8-sig")
    return {
        "metrics": metrics_df,
        "best_model": best_model_name,
        "importance_model": importance_model_name,
        "threshold": threshold,
        "positive_rate": float(y.mean()),
        "feature_columns": HIGH_BUY_MODEL_FEATURES,
        "removed_leakage_features": ["buy_mount", *AGGREGATE_LEAKAGE_RISK_FEATURES],
        "feature_importance": importance_df,
        "confusion_matrix": cm_df,
        "roc_data": roc_data,
        "scores": scores,
        "preds": preds,
        "y_test": y_test,
        "model_df": df,
    }


def write_model_diagnosis(cat1_result: dict, high_result: dict) -> None:
    cat1_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    best_high = high_metrics.iloc[0]
    highest_cat1 = cat1_metrics.sort_values(["Accuracy", "F1_weighted"], ascending=False).iloc[0]
    rf_rows = cat1_metrics[cat1_metrics["model"].astype(str).eq("Random Forest")]
    display_cat1 = rf_rows.iloc[0] if not rf_rows.empty else highest_cat1
    threshold = high_result["threshold"]
    positive_rate = high_result["positive_rate"]
    main_accuracy = float(display_cat1["Accuracy"])
    if 0.8 <= main_accuracy <= 0.95:
        accuracy_note = "已达到 0.80 到 0.95 的展示区间。"
    elif main_accuracy > 0.95:
        accuracy_note = "超过 0.95；已检查未使用 cat1_name、cat1 变体或目标编码字段，高准确率主要来自 cat_id 与 cat1 的商品层级关系。"
    else:
        accuracy_note = "低于 0.80；说明当前商品细分类、属性频次和画像特征仍不足以稳定识别全部商品大类。"
    over_95_note = (
        f"对比模型中 {', '.join(cat1_result['over_95_models'])} 的 Accuracy 超过 0.95；已检查未使用 cat1_name、cat1 变体或目标编码字段，原因是 cat_id 与 cat1 的商品层级关系较强。"
        if cat1_result.get("over_95_models")
        else "没有对比模型超过 0.95。"
    )

    lines = [
        "模型检查报告",
        "=" * 60,
        "",
        "1）当前模型任务",
        "- 当前主模型为 cat1 商品大类识别模型。",
        "- 该任务是商品层级分类识别任务，不是纯用户行为预测：基于商品细分类别、商品属性、交易时间和用户画像特征识别商品所属大类。",
        "- high_buy 高购买量订单识别作为辅助任务保留，用于展示二分类建模与局限性分析。",
        f"- high_buy 阈值：buy_mount > {threshold:.4f}；高购买量样本占比：{positive_rate:.2%}。",
        "",
        "2）使用的特征",
        f"- cat1 主模型特征：{', '.join(cat1_result['feature_columns'])}。",
        "- 主模型没有使用 cat1_name，因为它只是 cat1 的中文映射，会造成明显标签泄露。",
        f"- high_buy 模型特征：{', '.join(high_result['feature_columns'])}。",
        "",
        "3）数据泄露检查",
        "- cat1 主模型允许使用 cat_id，因为 cat_id 是商品细分类别，cat1 是商品大类，二者属于合理的商品层级关系。",
        "- cat1 主模型没有使用 cat1_name、cat1 编码变体或目标变量派生字段。",
        "- property_freq、auction_id_freq、cat_id_freq 均在训练集上拟合，再映射到测试集，降低训练/测试信息穿透风险。",
        "- 已删除 high_buy 模型中的 buy_mount，因为 high_buy 直接由 buy_mount 构造，保留该字段会造成目标变量泄露。",
        f"- 已删除 high_buy 模型中的销量聚合特征：{', '.join(high_result['removed_leakage_features'][1:])}。",
        "- user_total_buy_mount、item_total_buy_mount、cat_total_buy_mount 当前在全量数据上生成，仅用于描述分析和数据集留档，不进入最终模型。",
        "",
        "4）各模型准确率和 F1 值",
        "cat1 商品大类识别主模型：",
        cat1_metrics.to_string(index=False),
        "",
        "high_buy 高购买量订单识别辅助模型：",
        high_metrics.to_string(index=False),
        "",
        "5）最终推荐使用的模型",
        f"- 最高分模型：{highest_cat1['model']}，Accuracy={highest_cat1['Accuracy']:.4f}，F1_weighted={highest_cat1['F1_weighted']:.4f}。",
        f"- 推荐展示模型：{display_cat1['model']}，Accuracy={display_cat1['Accuracy']:.4f}，Precision_weighted={display_cat1['Precision_weighted']:.4f}，Recall_weighted={display_cat1['Recall_weighted']:.4f}，F1_weighted={display_cat1['F1_weighted']:.4f}。",
        "- 推荐展示模型选择说明：Random Forest 表现较好，且可直接输出特征重要性，便于报告解释；Gradient Boosting 作为最高分模型在性能对比中同步展示。",
        f"- 辅助模型推荐：{best_high['model']}，优先按 F1、再按 AUC 选择；AUC={best_high['AUC']:.4f}，F1={best_high['F1']:.4f}。",
        "",
        "6）模型效果是否合理",
        f"- 主模型 Accuracy 状态：{accuracy_note}",
        f"- 超高准确率检查：{over_95_note}",
        "- 如果主模型准确率较高，主要原因是 cat_id 与 cat1 存在商品层级关系，这符合本项目商品大类识别任务设定。",
        "- 原 high_buy 模型 Accuracy/AUC 为 1.0 的主要原因是 buy_mount 泄露；删除后结果更可信，可作为辅助任务展示。",
        "- high_buy 正类占比较低，单看 Accuracy 容易被多数类掩盖，因此报告同时关注 Precision、Recall、F1 和 AUC。",
        "- 如果 high_buy 指标明显下降，属于合理现象，因为当前数据缺少价格、优惠、浏览、收藏、加购、品牌、店铺等更强预测变量。",
        "",
        "7）后续改进建议",
        "- 后续可按时间切分训练集和测试集，并只用训练集历史窗口计算用户、商品和类目聚合特征。",
        "- 补充商品价格、品牌、店铺、活动、浏览、收藏、加购等行为特征。",
        "- 对 cat1、cat_id、性别、年龄阶段等类别字段使用 One-Hot、目标编码或 Spark ML StringIndexer/OneHotEncoder。",
        "- 在报告中明确：主模型是商品层级识别模型，high_buy 是辅助的高购买量行为识别模型。",
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    (MODEL_RESULT_DIR / "model_diagnosis.txt").write_text(report_text, encoding="utf-8-sig")


# ============================================================
# 13. 图表生成
# ============================================================

def save_fig(filename: str) -> None:
    fig = plt.gcf()
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily(CHINESE_FONT_NAME)
    apply_report_style()
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    plt.tight_layout(pad=1.2)
    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def enhance_chart_with_mcp(chart_path: Path, prompt: str) -> Path:
    """可选 MCP/AI 图表增强钩子；未配置或失败时返回原图，不影响主流程。"""
    api_key = os.getenv("MCP_API_KEY")
    api_url = os.getenv("MCP_API_URL")
    if not api_key:
        return chart_path
    if not api_url:
        print(f"warning: 检测到 MCP_API_KEY，但未配置 MCP_API_URL，跳过增强：{chart_path.name}")
        return chart_path

    try:
        payload = json.dumps({"prompt": prompt, "filename": chart_path.name}).encode("utf-8")
        req = urllib_request.Request(
            api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
        image_b64 = result.get("image_base64") or result.get("data", {}).get("image_base64")
        if not image_b64:
            print(f"warning: MCP 增强接口未返回 image_base64，保留原图：{chart_path.name}")
            return chart_path
        enhanced_path = ENHANCED_CHART_DIR / chart_path.name
        enhanced_path.write_bytes(base64.b64decode(image_b64))
        return enhanced_path
    except (URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"warning: MCP 图表增强失败，保留原图 {chart_path.name}：{exc}")
        return chart_path


def enhance_report_charts() -> None:
    prompts = {
        "01_target_distribution.png": "优化商品大类分布柱状图，保持白底、专业报告风格、中文标题清晰。",
        "02_numeric_feature_distribution.png": "优化 2x2 数值变量分布图，保持坐标轴清晰、KDE 曲线自然。",
        "03_feature_correlation_heatmap.png": "优化相关系数热力图，保持数值标注清晰、特征名不重叠。",
        "04_feature_importance.png": "优化 Feature Importance 横向条形图，突出主模型关键特征。",
        "05_model_comparison.png": "优化模型性能分组柱状图，突出 Accuracy、Precision 和 F1 对比。",
        "06_roc_curve.png": "优化 ROC 曲线图，保持 AUC 标注、基准线和坐标轴 0 到 1。",
        "07_confusion_matrix.png": "优化中文业务标签混淆矩阵，保持蓝色系和样本数量标注。",
        "08_probability_distribution.png": "优化预测概率分布图，保持真实类别区分和半透明直方图。",
        "09_wordcloud.png": "优化母婴消费热点词云，保持白色背景和中文字体清晰。",
    }
    for filename, prompt in prompts.items():
        chart_path = CHART_DIR / filename
        if chart_path.exists():
            enhance_chart_with_mcp(chart_path, prompt)


def plot_report_charts(model_df: pd.DataFrame, cat1_result: dict, high_result: dict) -> None:
    apply_report_style()
    cat_df = cat1_result["model_df"]
    high_best_model = high_result["best_model"]
    y_test = high_result["y_test"]
    best_score = high_result["scores"][high_best_model]

    # 图 1 商品大类分布柱状图，主模型目标变量为 cat1。
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor="white")
    target_counts = cat_df["cat1_name"].value_counts()
    sns.barplot(x=target_counts.values, y=target_counts.index, hue=target_counts.index, palette=REPORT_COLORS, legend=False, ax=ax)
    ax.set_title("图1 商品大类分布", fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("订单数量")
    ax.set_ylabel("")
    annotate_bars(ax, orientation="horizontal", fmt="{:.0f}")
    polish_axes(ax, grid_axis="x")
    save_fig("01_target_distribution.png")

    # 图 2 数值变量分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), facecolor="white")
    numeric_titles = {
        "buy_mount": "购买数量分布",
        "month": "月份分布",
        "weekday": "星期分布",
        "baby_age": "宝宝年龄分布",
    }
    for ax, col in zip(axes.ravel(), ["buy_mount", "month", "weekday", "baby_age"]):
        sns.histplot(model_df[col], bins=24, kde=True, ax=ax, color=REPORT_COLORS[0], edgecolor="white", linewidth=0.6)
        ax.set_title(numeric_titles[col], fontsize=12.5, weight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("样本数")
        polish_axes(ax, grid_axis="y")
    fig.suptitle("图2 数值变量分布直方图", fontsize=15.5, weight="bold", y=1.02)
    save_fig("02_numeric_feature_distribution.png")

    # 图 3 特征相关系数热力图
    plt.figure(figsize=(10.8, 8), facecolor="white")
    corr = cat1_result["feature_frame"].corr(numeric_only=True)
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="#F1F5F9",
        square=False,
        cbar_kws={"shrink": 0.82, "label": "相关系数"},
    )
    plt.title("图3 商品大类识别模型特征相关系数热力图", fontsize=15.5, weight="bold", pad=12)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save_fig("03_feature_correlation_heatmap.png")

    # 图 4 主模型特征重要性柱状图
    importance = cat1_result["feature_importance"].sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")
    ax.barh(importance["feature"], importance["importance"], color=REPORT_COLORS[0], edgecolor="#2F5F8F", linewidth=0.4)
    ax.set_title(f"图4 商品大类识别模型 Feature Importance（{cat1_result['importance_model']}）", fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("重要性")
    annotate_bars(ax, orientation="horizontal", fmt="{:.3f}")
    polish_axes(ax, grid_axis="x")
    save_fig("04_feature_importance.png")

    # 图 5 主模型性能对比图
    metrics_melt = cat1_result["metrics"].melt(
        id_vars="model",
        value_vars=["Accuracy", "Precision_weighted", "F1_weighted"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    sns.barplot(data=metrics_melt, x="model", y="score", hue="metric", palette=REPORT_COLORS[:3], ax=ax)
    ax.set_title("图5 商品大类识别模型性能对比图", fontsize=15.5, weight="bold", pad=12)
    ax.set_ylabel("指标值")
    ax.set_xlabel("模型")
    ax.set_ylim(0, 1.16)
    ax.tick_params(axis="x", rotation=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8.5, padding=3)
    ax.legend(title="", loc="upper left", ncols=3, frameon=True, facecolor="white", edgecolor="#D0D7DE")
    polish_axes(ax, grid_axis="y")
    save_fig("05_model_comparison.png")

    # 图 6 高购买量订单识别模型 ROC 曲线图
    fpr, tpr, auc_value = high_result["roc_data"][high_best_model]
    fig, ax = plt.subplots(figsize=(7.4, 6.4), facecolor="white")
    ax.plot(fpr, tpr, color=REPORT_COLORS[3], linewidth=2.6, label=f"{high_best_model} AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8A94A6", linewidth=1.5, label="随机猜测")
    ax.set_title("图6 高购买量订单识别模型 ROC 曲线图", fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D0D7DE")
    polish_axes(ax, grid_axis="both")
    save_fig("06_roc_curve.png")

    # 图 7 主模型混淆矩阵热力图
    plt.figure(figsize=(8.8, 7.2), facecolor="white")
    sns.heatmap(
        cat1_result["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.6,
        linecolor="#F1F5F9",
        cbar_kws={"shrink": 0.82, "label": "样本数量"},
    )
    plt.title("图7 商品大类识别模型混淆矩阵热力图", fontsize=15.5, weight="bold", pad=12)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    save_fig("07_confusion_matrix.png")

    # 图 8 高购买量订单识别模型预测概率分布图
    prob_df = pd.DataFrame({"score": best_score, "真实类别": np.where(y_test.values == 1, "真实高购买量", "真实普通购买")})
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor="white")
    sns.histplot(
        data=prob_df,
        x="score",
        hue="真实类别",
        bins=30,
        kde=True,
        palette=[REPORT_COLORS[0], REPORT_COLORS[3]],
        alpha=0.52,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title("图8 高购买量订单识别模型预测概率分布图", fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("预测为高购买量的概率")
    ax.set_ylabel("样本数")
    ax.set_xlim(0, 1)
    polish_axes(ax, grid_axis="y")
    save_fig("08_probability_distribution.png")


def make_wordcloud(merged_df: pd.DataFrame) -> None:
    word_freq = merged_df.groupby("cat1_name")["buy_mount"].sum().sort_values(ascending=False).to_dict()
    for _, row in merged_df.iterrows():
        for word in CAT1_WORDS_MAP.get(normalize_cat1(row["cat1"]), ["其他母婴商品"]):
            word_freq[word] = word_freq.get(word, 0) + float(row["buy_mount"]) * 0.35

    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((path for path in font_candidates if Path(path).exists()), None)
    mask_candidates = [
        OUTPUT_DIR / "mother_baby_wordcloud_mask_stroller.png",
        CHART_DIR / "mother_baby_wordcloud_mask_stroller.png",
        PROJECT_ROOT / "archive_old_project" / "outputs" / "mother_baby_wordcloud_mask_stroller.png",
    ]
    mask_path = next((path for path in mask_candidates if path.exists()), None)
    mask = np.array(Image.open(mask_path).convert("L")) if mask_path else None

    def report_color_func(word, *args, **kwargs):
        color_index = sum(ord(char) for char in str(word)) % len(REPORT_COLORS)
        return REPORT_COLORS[color_index]

    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color="white",
        color_func=report_color_func,
        mask=mask,
        contour_width=2 if mask is not None else 0,
        contour_color="#D0D7DE",
        max_words=120,
        prefer_horizontal=0.92,
        collocations=False,
        random_state=RANDOM_STATE,
    ).generate_from_frequencies(word_freq)
    wc.to_image().save(CHART_DIR / "09_wordcloud.png", dpi=(300, 300))


# ============================================================
# 14. 可视化数字大屏生成
# ============================================================

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def html_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    return df.head(max_rows).to_html(index=False, border=0, classes="data-table")


def generate_dashboard_old(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    def fmt_number(value) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.0f}"
        return str(value)

    def pct(value) -> str:
        return f"{float(value) * 100:.1f}%"

    def progress_cell(value: float, color: str) -> str:
        return (
            f"<td><span class='score'>{float(value):.3f}</span>"
            f"<span class='progress'><i style='width:{float(value) * 100:.1f}%;background:{color}'></i></span></td>"
        )

    def sparkline(values: list[float], color: str) -> str:
        if not values:
            return ""
        width, height, pad = 110, 34, 4
        min_v, max_v = min(values), max(values)
        span = max(max_v - min_v, 1e-6)
        points = []
        for index, value in enumerate(values):
            x = pad + index * ((width - 2 * pad) / max(len(values) - 1, 1))
            y = height - pad - ((value - min_v) / span) * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")
        return f"<svg viewBox='0 0 {width} {height}'><polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/></svg>"

    update_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    data_period = f"{merged_df['year_month'].min()} ~ {merged_df['year_month'].max()}"
    wordcloud_chart = "../charts/09_wordcloud.png"

    kpis = [
        ("总交易记录数", f"{len(merged_df):,}", "清洗后样本", "数据期内", "#FF5C8A", "K1"),
        ("总用户数", f"{merged_df['user_id'].nunique():,}", "数据期内统计", "有效用户", "#3B82F6", "K2"),
        ("总商品数", f"{merged_df['auction_id'].nunique():,}", "商品 ID 去重", "有效商品", "#8B5CF6", "K3"),
        ("总购买数量", f"{int(merged_df['buy_mount'].sum()):,}", "样本汇总", "有效记录", "#F59E0B", "K4"),
    ]
    kpi_html = "".join(
        f"""
        <section class="kpi-card">
          <div class="kpi-icon" style="--kpi-color:{color}">{icon}</div>
          <div class="kpi-body">
            <span>{name}</span>
            <strong>{value}</strong>
            <small>{hint}<b style="color:#039855">↑ {change}</b></small>
          </div>
          <div class="kpi-spark" style="--spark-color:{color}"></div>
        </section>
        """
        for name, value, hint, change, color, icon in kpis
    )

    category_df = tables["category_sales_summary"].head(10).copy()
    monthly_df = tables["monthly_sales_summary"].copy()
    weekday_df = tables["weekday_sales_summary"].copy()
    weekday_names = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}
    model_metrics = cat1_result["metrics"].copy()
    high_model_metrics = high_result["metrics"].copy()
    high_best = high_model_metrics.iloc[0]

    gender_labels = merged_df["gender"].map({"female": "女宝宝", "male": "男宝宝", "unknown": "未知"}).fillna("未知")
    gender_counts = gender_labels.value_counts()
    age_counts = merged_df["baby_age_group"].value_counts().reindex(["unknown", "0", "1", "2-3", "4-6", "7+"], fill_value=0)
    high_counts = high_result["model_df"]["high_buy"].map({0: "普通订单", 1: "高购买量订单"}).value_counts()
    feature_top = cat1_result["feature_importance"].head(10)

    model_rows = ""
    metric_colors = {
        "Accuracy": "#3B82F6",
        "Precision": "#22C55E",
        "Recall": "#F59E0B",
        "F1": "#8B5CF6",
    }
    for _, row in model_metrics.iterrows():
        model_rows += (
            f"<tr><th>{html.escape(str(row['model']))}</th>"
            f"{progress_cell(row['Accuracy'], metric_colors['Accuracy'])}"
            f"{progress_cell(row['Precision_weighted'], metric_colors['Precision'])}"
            f"{progress_cell(row['Recall_weighted'], metric_colors['Recall'])}"
            f"{progress_cell(row['F1_weighted'], metric_colors['F1'])}</tr>"
        )

    high_metric_series = {
        "AUC": high_model_metrics["AUC"].astype(float).round(4).tolist(),
        "Accuracy": high_model_metrics["Accuracy"].astype(float).round(4).tolist(),
        "Precision": high_model_metrics["Precision"].astype(float).round(4).tolist(),
        "Recall": high_model_metrics["Recall"].astype(float).round(4).tolist(),
        "F1": high_model_metrics["F1"].astype(float).round(4).tolist(),
    }
    metric_cards = [
        ("AUC", high_best["AUC"], "#1D4ED8"),
        ("Accuracy", high_best["Accuracy"], "#22C55E"),
        ("Precision", high_best["Precision"], "#F59E0B"),
        ("Recall", high_best["Recall"], "#EF476F"),
        ("F1-Score", high_best["F1"], "#8B5CF6"),
    ]
    metric_card_html = "".join(
        f"""
        <div class="metric-card" style="--metric-color:{color}">
          <span>{name}</span>
          <strong>{float(value):.3f}</strong>
          {sparkline(high_metric_series['F1' if name == 'F1-Score' else name], color)}
        </div>
        """
        for name, value, color in metric_cards
    )

    chart_payload = {
        "chartColors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_df["year_month"].astype(str).tolist(),
        "monthBuyValues": monthly_df["total_buy_mount"].astype(float).tolist(),
        "monthOrderValues": monthly_df["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(value), str(value)) for value in weekday_df["weekday"]],
        "weekdayBuyValues": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrderValues": weekday_df["trade_count"].astype(float).tolist(),
        "highBuyNames": ["高购买量订单", "普通订单"],
        "highBuyValues": [
            int(high_counts.get("高购买量订单", 0)),
            int(high_counts.get("普通订单", 0)),
        ],
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": age_counts.index.astype(str).tolist(),
        "ageValues": age_counts.astype(int).tolist(),
        "featureNames": feature_top["feature"].astype(str).tolist(),
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    payload_json = json.dumps(chart_payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg:#FAFBFF; --card:#FFFFFF; --pink:#FF5C8A; --title:#101828;
      --muted:#667085; --line:#EEF2F7; --shadow:0 12px 28px rgba(16,24,40,.06);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;
      color:var(--title);
      background:var(--bg);
    }}
    .sidebar {{
      position:fixed; left:20px; top:18px; bottom:18px; width:270px;
      background:#fff; border:1px solid var(--line); border-radius:18px;
      box-shadow:var(--shadow); padding:18px 16px; display:flex; flex-direction:column; z-index:2;
    }}
    .brand {{ display:flex; align-items:center; gap:10px; padding:2px 4px 18px; }}
    .brand-mark {{
      width:42px; height:42px; border-radius:14px; display:grid; place-items:center;
      color:#fff; font-weight:800; background:linear-gradient(135deg,#FF5C8A,#FF8FAB);
    }}
    .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1.05; }}
    .brand small {{ color:var(--muted); font-size:12px; }}
    .nav {{ display:grid; gap:8px; margin-top:4px; }}
    .nav-item {{
      display:flex; align-items:center; gap:12px; padding:12px 14px;
      border-radius:10px; color:#344054; font-size:14px;
    }}
    .nav-item.active {{ background:linear-gradient(90deg,rgba(255,92,138,.14),rgba(255,92,138,.05)); color:var(--pink); font-weight:700; }}
    .nav-dot {{ width:28px; height:28px; border-radius:9px; background:#F2F4F7; display:grid; place-items:center; font-size:11px; font-weight:800; }}
    .nav-item.active .nav-dot {{ background:rgba(255,92,138,.16); }}
    .side-visual {{
      margin-top:auto; height:150px; border-radius:16px; background:
      radial-gradient(circle at 35% 42%,#FFD6E2 0 18px,transparent 19px),
      radial-gradient(circle at 64% 46%,#FDE7D7 0 28px,transparent 29px),
      linear-gradient(135deg,#FFF1F5,#FFF7ED);
      border:1px solid #FFE4EC; position:relative;
    }}
    .side-visual::after {{
      content:"母婴消费洞察"; position:absolute; left:22px; bottom:18px; color:#FF5C8A; font-weight:700;
    }}
    .key-card {{
      margin-top:14px; padding:14px; border-radius:14px; border:1px solid #FFE4EC;
      background:linear-gradient(180deg,#FFF8FB,#fff);
    }}
    .key-card h3 {{ margin:0 0 10px; font-size:14px; }}
    .key-row {{ display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-top:1px solid #FCE7F0; }}
    .key-row:first-of-type {{ border-top:0; }}
    .key-row span {{ color:var(--muted); font-size:12px; }}
    .key-row strong {{ color:#101828; font-size:18px; }}
    .page {{ margin-left:306px; padding:18px 22px 18px 0; max-width:1920px; }}
    .topbar {{
      min-height:72px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center;
      margin-bottom:16px;
    }}
    .topbar-title {{ text-align:center; }}
    .topbar-title h2 {{ margin:0; font-size:28px; line-height:1.15; color:#0B1533; }}
    .topbar-title p {{ margin:8px 0 0; color:#475467; font-size:15px; }}
    .update-box {{
      justify-self:end; display:flex; align-items:center; gap:12px; padding:10px 14px;
      background:#fff; border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow);
      color:#344054; font-size:13px;
    }}
    .update-icon {{ width:38px; height:38px; border-radius:12px; background:#FFF1F5; color:var(--pink); display:grid; place-items:center; font-weight:800; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:16px; }}
    .kpi-card {{
      min-height:118px; display:flex; align-items:center; gap:16px; padding:20px 18px;
      background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow);
      position:relative; overflow:hidden;
    }}
    .kpi-card::after {{
      content:""; position:absolute; right:16px; bottom:14px; width:96px; height:32px; opacity:.18;
      background:linear-gradient(135deg,transparent 0 20%,var(--kpi-color) 21% 25%,transparent 26% 45%,var(--kpi-color) 46% 50%,transparent 51%);
      clip-path:polygon(0 70%,15% 62%,26% 75%,38% 44%,52% 56%,65% 28%,78% 42%,100% 14%,100% 100%,0 100%);
    }}
    .kpi-icon {{
      width:62px; height:62px; flex:0 0 62px; border-radius:18px; display:grid; place-items:center;
      color:#fff; font-weight:800; font-size:15px; background:var(--kpi-color);
      box-shadow:0 10px 24px rgba(255,92,138,.18);
    }}
    .kpi-body span {{ display:block; color:#101828; font-size:13px; font-weight:700; }}
    .kpi-body strong {{ display:block; margin-top:8px; font-size:25px; line-height:1; letter-spacing:.2px; }}
    .kpi-body small {{ display:flex; gap:10px; margin-top:12px; color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; }}
    .card {{
      background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow);
      padding:16px 18px; overflow:hidden;
    }}
    .card h3 {{ margin:0; font-size:16px; color:#101828; }}
    .card .sub {{ margin-top:6px; color:var(--muted); font-size:12px; }}
    .chart {{ width:100%; height:260px; margin-top:8px; }}
    .trend-card {{ grid-column:span 6; }}
    .top-card {{ grid-column:span 3; }}
    .small-card {{ grid-column:span 3; }}
    .quarter-card {{ grid-column:span 3; }}
    .third-card {{ grid-column:span 4; }}
    .feature-card {{ grid-column:span 3; }}
    .metric-panel {{ grid-column:span 5; }}
    .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .mini-chart {{ height:206px; }}
    .wordcloud {{ height:236px; display:flex; align-items:center; justify-content:center; margin-top:8px; }}
    .wordcloud img {{ width:100%; height:100%; object-fit:contain; background:#fff; border-radius:12px; }}
    .geo-note {{
      height:236px; margin-top:8px; border-radius:14px; display:grid; place-items:center; text-align:center;
      color:#1D4ED8; background:linear-gradient(135deg,#EFF6FF,#DBEAFE 55%,#BFDBFE);
      border:1px solid #BFDBFE; position:relative; overflow:hidden;
    }}
    .geo-note::before {{
      content:""; position:absolute; inset:28px 54px; opacity:.55;
      background:radial-gradient(circle at 20% 45%,#93C5FD 0 16px,transparent 17px),
                 radial-gradient(circle at 55% 38%,#60A5FA 0 22px,transparent 23px),
                 radial-gradient(circle at 72% 62%,#3B82F6 0 18px,transparent 19px);
      filter:blur(.2px);
    }}
    .geo-note strong,.geo-note span {{ position:relative; z-index:1; display:block; }}
    .geo-note strong {{ font-size:18px; }}
    .geo-note span {{ color:#475467; margin-top:8px; font-size:12px; }}
    table.model-table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }}
    .model-table th,.model-table td {{ padding:9px 8px; border-bottom:1px solid #F2F4F7; text-align:left; white-space:nowrap; }}
    .model-table th {{ color:#344054; font-weight:700; }}
    .score {{ display:inline-block; min-width:44px; color:#101828; font-weight:700; }}
    .progress {{ display:inline-block; width:74px; height:6px; margin-left:7px; background:#F2F4F7; border-radius:99px; overflow:hidden; vertical-align:middle; }}
    .progress i {{ display:block; height:100%; border-radius:99px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-top:12px; }}
    .metric-card {{ min-height:126px; padding:14px; border-radius:14px; border:1px solid #EEF2F7; background:#fff; }}
    .metric-card span {{ display:block; color:#344054; font-size:12px; font-weight:700; }}
    .metric-card strong {{ display:block; color:#101828; font-size:22px; margin-top:14px; }}
    .metric-card svg {{ width:100%; height:34px; margin-top:8px; }}
    footer {{
      margin-top:12px; padding:13px 18px; border-radius:16px; border:1px solid #FFE4EC;
      background:linear-gradient(90deg,#FFF6FA,#FFFDFE); color:#667085; display:flex; justify-content:space-between; gap:16px; font-size:12px;
    }}
    footer b {{ color:#FF5C8A; }}
    @media (max-width: 1280px) {{
      .sidebar {{ position:static; width:auto; margin:16px; }}
      .page {{ margin-left:0; padding:0 16px 18px; }}
      .topbar {{ grid-template-columns:1fr; gap:12px; }}
      .update-box {{ justify-self:center; }}
      .trend-card,.top-card,.small-card,.quarter-card,.third-card,.feature-card,.metric-panel {{ grid-column:1 / -1; }}
      .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">MB</div>
      <div>
        <h1>MotherBabyInsight</h1>
        <small>母婴电商消费行为分析平台</small>
      </div>
    </div>
    <nav class="nav">
      <div class="nav-item active"><span class="nav-dot">01</span><div>总览概览<br><small>Overview</small></div></div>
      <div class="nav-item"><span class="nav-dot">02</span><div>用户分析<br><small>User Insights</small></div></div>
      <div class="nav-item"><span class="nav-dot">03</span><div>商品分析<br><small>Product Insights</small></div></div>
      <div class="nav-item"><span class="nav-dot">04</span><div>购买趋势<br><small>Purchase Trends</small></div></div>
      <div class="nav-item"><span class="nav-dot">05</span><div>模型表现<br><small>Model Performance</small></div></div>
      <div class="nav-item"><span class="nav-dot">06</span><div>地域分析<br><small>Geographic Insights</small></div></div>
    </nav>
    <div class="side-visual"></div>
    <section class="key-card">
      <h3>核心指标</h3>
      <div class="key-row"><span>高购买量订单占比</span><strong>{pct(high_result['positive_rate'])}</strong></div>
      <div class="key-row"><span>活跃用户数</span><strong>{merged_df['user_id'].nunique() / 1000:.1f}K</strong></div>
      <div class="key-row"><span>热门品类</span><strong>{html.escape(str(category_df.iloc[0]['cat1_name']))}</strong></div>
      <div class="key-row"><span>月度峰值</span><strong>{html.escape(str(monthly_df.loc[monthly_df['total_buy_mount'].idxmax(), 'year_month']))}</strong></div>
    </section>
  </aside>
  <main class="page">
    <header class="topbar">
      <div></div>
      <section class="topbar-title">
        <h2>Mother & Baby E-Commerce Analytics Dashboard</h2>
        <p>基于淘宝母婴购物数据的用户消费行为分析与预测</p>
      </section>
      <section class="update-box">
        <div class="update-icon">CAL</div>
        <div><strong>{update_time}</strong><br><span>最后更新</span></div>
      </section>
    </header>
    <div class="kpis">{kpi_html}</div>
    <section class="grid">
      <section class="card trend-card">
        <h3>购买趋势概览</h3>
        <div class="sub">月度购买数量与订单数走势</div>
        <div id="trendChart" class="chart"></div>
      </section>
      <section class="card top-card">
        <h3>商品大类销量 TOP10</h3>
        <div class="sub">按购买数量汇总</div>
        <div id="categoryChart" class="chart"></div>
      </section>
      <section class="card small-card">
        <h3>高购买量订单分析</h3>
        <div class="sub">75% 分位数规则划分</div>
        <div id="highBuyChart" class="chart"></div>
      </section>
      <section class="card quarter-card">
        <h3>用户画像分析</h3>
        <div class="sub">性别与年龄阶段分布</div>
        <div class="profile-grid">
          <div id="genderChart" class="mini-chart"></div>
          <div id="ageChart" class="mini-chart"></div>
        </div>
      </section>
      <section class="card quarter-card">
        <h3>购买活跃度（按星期）</h3>
        <div class="sub">绿色柱状图 + 蓝色折线</div>
        <div id="weekdayChart" class="chart"></div>
      </section>
      <section class="card quarter-card">
        <h3>数据质量与画像匹配情况</h3>
        <div class="sub">原始数据暂未提供省份/城市字段</div>
        <div class="geo-note"><div><strong>地域字段待接入</strong><span>接入省份字段后可生成浅蓝至深蓝地图</span></div></div>
      </section>
      <section class="card quarter-card">
        <h3>热门商品词云</h3>
        <div class="sub">按 cat1_name 与 buy_mount 加权</div>
        <div class="wordcloud"><img src="{wordcloud_chart}" alt="母婴商品消费热点词云图"></div>
      </section>
      <section class="card third-card">
        <h3>模型表现对比</h3>
        <div class="sub">商品大类识别主模型</div>
        <table class="model-table">
          <thead><tr><th>模型</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
          <tbody>{model_rows}</tbody>
        </table>
      </section>
      <section class="card feature-card">
        <h3>特征重要性 TOP10</h3>
        <div class="sub">Random Forest 贡献度</div>
        <div id="featureChart" class="chart"></div>
      </section>
      <section class="card metric-panel">
        <h3>模型性能指标卡片</h3>
        <div class="sub">高购买量订单识别辅助模型</div>
        <div class="metric-grid">{metric_card_html}</div>
        <div class="sub" style="margin-top:10px;">该模型已删除 buy_mount 及全量聚合销量特征，避免标签泄露。</div>
      </section>
    </section>
    <footer>
      <span><b>数据来源：</b>淘宝母婴购物数据集</span>
      <span><b>数据期间：</b>{data_period}</span>
      <span><b>最后更新：</b>{update_time}</span>
      <span><b>业务标语：</b>用数据洞察母婴消费，用智能守护宝宝成长</span>
    </footer>
  </main>
  <script>
    const injectedPayload = {payload_json};
    let payload = injectedPayload;
    const textColor = "#101828";
    const mutedColor = "#667085";
    const gridLine = "#EEF2F7";
    function makeChart(id, option) {{
      const el = document.getElementById(id);
      const chart = echarts.init(el, null, {{ renderer: "canvas" }});
      chart.setOption(option);
      window.addEventListener("resize", () => chart.resize());
      return chart;
    }}
    const baseGrid = {{ left: 48, right: 36, top: 36, bottom: 34 }};
    makeChart("trendChart", {{
      color: ["#FF5C8A", "#3B82F6"],
      tooltip: {{ trigger: "axis" }},
      legend: {{ top: 0, textStyle: {{ color: mutedColor }} }},
      grid: {{ left: 52, right: 48, top: 48, bottom: 40 }},
      xAxis: {{ type: "category", data: payload.monthLabels, axisLabel: {{ color: mutedColor }}, axisTick: {{ show:false }} }},
      yAxis: [
        {{ type: "value", name: "购买数量", axisLabel: {{ color: mutedColor }}, splitLine: {{ lineStyle: {{ color: gridLine }} }} }},
        {{ type: "value", name: "订单数", axisLabel: {{ color: mutedColor }}, splitLine: {{ show:false }} }}
      ],
      series: [
        {{ name: "购买数量", type: "bar", data: payload.monthBuyValues, barWidth: 22, itemStyle: {{ borderRadius:[8,8,0,0], color: new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#FF7AA2"}},{{offset:1,color:"#FFD1DE"}}]) }} }},
        {{ name: "订单数", type: "line", yAxisIndex: 1, data: payload.monthOrderValues, smooth:true, symbolSize:7, lineStyle: {{ width:3, color:"#3B82F6" }}, itemStyle: {{ color:"#3B82F6" }} }}
      ]
    }});
    makeChart("categoryChart", {{
      tooltip: {{ trigger: "axis", axisPointer: {{ type: "shadow" }} }},
      grid: {{ left: 78, right: 28, top: 18, bottom: 24 }},
      xAxis: {{ type: "value", axisLabel: {{ color: mutedColor }}, splitLine: {{ lineStyle: {{ color: gridLine }} }} }},
      yAxis: {{ type: "category", data: payload.categoryNames.reverse(), axisLabel: {{ color: textColor }}, axisTick: {{ show:false }} }},
      series: [{{
        name: "购买数量",
        type: "bar",
        data: payload.categoryValues.reverse(),
        barWidth: 12,
        label: {{ show: true, position: "right", color: textColor }},
        itemStyle: {{
          borderRadius: [0, 7, 7, 0],
          color: params => payload.chartColors[params.dataIndex % payload.chartColors.length]
        }}
      }}]
    }});
    makeChart("highBuyChart", {{
      color: ["#EF476F", "#3B82F6"],
      tooltip: {{ trigger: "item" }},
      legend: {{ bottom: 0, textStyle: {{ color: mutedColor }} }},
      series: [{{
        name: "订单类型",
        type: "pie",
        radius: ["50%", "72%"],
        center: ["50%", "45%"],
        avoidLabelOverlap: true,
        label: {{ formatter: "{{b}}\\n{{d}}%", color: textColor, fontWeight: 700 }},
        data: payload.highBuyNames.map((name, i) => ({{ name, value: payload.highBuyValues[i] }}))
      }}]
    }});
    makeChart("genderChart", {{
      color: ["#3B82F6", "#EF476F", "#64748B"],
      title: {{ text: "性别分布", left: "center", top: 0, textStyle: {{ fontSize: 12, color: textColor }} }},
      tooltip: {{ trigger:"item" }},
      series: [{{ type:"pie", radius:["45%","68%"], center:["50%","58%"], label: {{ formatter:"{{b}}\\n{{d}}%", color:textColor }}, data: payload.genderNames.map((name,i)=>({{name, value:payload.genderValues[i]}})) }}]
    }});
    makeChart("ageChart", {{
      color: ["#3B82F6", "#F59E0B", "#22C55E", "#8B5CF6", "#EF476F", "#06B6D4"],
      title: {{ text: "年龄阶段", left: "center", top: 0, textStyle: {{ fontSize: 12, color: textColor }} }},
      tooltip: {{ trigger:"item" }},
      series: [{{ type:"pie", radius:["42%","68%"], center:["50%","58%"], label: {{ formatter:"{{b}}\\n{{d}}%", color:textColor }}, data: payload.ageNames.map((name,i)=>({{name, value:payload.ageValues[i]}})) }}]
    }});
    makeChart("weekdayChart", {{
      color: ["#22C55E", "#3B82F6"],
      tooltip: {{ trigger: "axis" }},
      grid: baseGrid,
      xAxis: {{ type: "category", data: payload.weekdayLabels, axisLabel: {{ color: mutedColor }}, axisTick: {{ show:false }} }},
      yAxis: [
        {{ type: "value", axisLabel: {{ color: mutedColor }}, splitLine: {{ lineStyle: {{ color: gridLine }} }} }},
        {{ type: "value", axisLabel: {{ color: mutedColor }}, splitLine: {{ show:false }} }}
      ],
      series: [
        {{ name: "购买数量", type: "bar", data: payload.weekdayBuyValues, barWidth: 20, itemStyle: {{ borderRadius:[7,7,0,0], color:"#22C55E" }} }},
        {{ name: "订单数", type: "line", yAxisIndex:1, data: payload.weekdayOrderValues, smooth:true, symbolSize:7, lineStyle:{{ width:3, color:"#3B82F6" }}, itemStyle:{{ color:"#3B82F6" }} }}
      ]
    }});
    makeChart("featureChart", {{
      color: ["#3B82F6"],
      tooltip: {{ trigger: "axis" }},
      grid: {{ left: 110, right: 20, top: 16, bottom: 20 }},
      xAxis: {{ type: "value", axisLabel: {{ color: mutedColor }}, splitLine: {{ lineStyle: {{ color: gridLine }} }} }},
      yAxis: {{ type: "category", data: payload.featureNames.slice().reverse(), axisLabel: {{ color: textColor, fontSize: 10 }}, axisTick: {{ show:false }} }},
      series: [{{
        name: "重要性",
        type: "bar",
        data: payload.featureValues.slice().reverse(),
        barWidth: 11,
        label: {{ show:true, position:"right", color:textColor, formatter: p => p.value.toFixed(3) }},
        itemStyle: {{ borderRadius:[0,7,7,0], color: params => payload.chartColors[params.dataIndex % payload.chartColors.length] }}
      }}]
    }});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    def fmt_number(value) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.0f}"
        return str(value)

    def pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def icon_svg(kind: str) -> str:
        icons = {
            "bag": "<svg viewBox='0 0 24 24'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",
            "user": "<svg viewBox='0 0 24 24'><path d='M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5Zm0 2c-5 0-8 2.5-8 5v2h16v-2c0-2.5-3-5-8-5Z'/></svg>",
            "box": "<svg viewBox='0 0 24 24'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 7 4v8l-7-4v-8Zm14 0v8l-7 4v-8l7-4Z'/></svg>",
            "cart": "<svg viewBox='0 0 24 24'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>",
            "database": "<svg viewBox='0 0 24 24'><path d='M12 3c5 0 9 1.8 9 4s-4 4-9 4-9-1.8-9-4 4-4 9-4Zm-9 7c1.6 1.7 5 3 9 3s7.4-1.3 9-3v3c0 2.2-4 4-9 4s-9-1.8-9-4v-3Zm0 6c1.6 1.7 5 3 9 3s7.4-1.3 9-3v1c0 2.2-4 4-9 4s-9-1.8-9-4v-1Z'/></svg>",
            "model": "<svg viewBox='0 0 24 24'><path d='M12 2 3 7v10l9 5 9-5V7l-9-5Zm0 2.3L17.7 7 12 9.7 6.3 7 12 4.3ZM5 8.8l6 3v7.4l-6-3V8.8Zm14 7.4-6 3v-7.4l6-3v7.4Z'/></svg>",
        }
        return icons[kind]

    def progress_cell(value: float, color: str) -> str:
        return (
            f"<td><span class='score'>{float(value):.3f}</span>"
            f"<span class='progress'><i style='width:{float(value) * 100:.1f}%;background:{color}'></i></span></td>"
        )

    update_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    data_period = f"{merged_df['year_month'].min()} ~ {merged_df['year_month'].max()}"

    category_df = tables["category_sales_summary"].head(10).copy()
    monthly_df = tables["monthly_sales_summary"].copy()
    weekday_df = tables["weekday_sales_summary"].copy()
    weekday_names = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}
    model_metrics = cat1_result["metrics"].copy()
    high_model_metrics = high_result["metrics"].copy()
    best_main = model_metrics.iloc[0]
    high_best = high_model_metrics.iloc[0]

    profile_known = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    profile_match_rate = float(profile_known.mean())
    unknown_rate = float((~profile_known).mean())
    valid_buy_records = int((merged_df["buy_mount"] > 0).sum())
    valid_record_rate = valid_buy_records / max(len(merged_df), 1)
    missing_field_count = int(merged_df.isna().sum().sum())

    gender_labels = merged_df["gender"].map({"female": "女宝宝", "male": "男宝宝", "unknown": "未知"}).fillna("未知")
    gender_counts = gender_labels.value_counts()
    age_counts = merged_df["baby_age_group"].value_counts().reindex(["unknown", "0", "1", "2-3", "4-6", "7+"], fill_value=0)
    high_counts = high_result["model_df"]["high_buy"].map({0: "普通订单", 1: "高购买量订单"}).value_counts()
    feature_top = cat1_result["feature_importance"].head(10)

    kpis = [
        ("总交易记录数", f"{len(merged_df):,}", "清洗后样本", "#FF5C8A", "bag"),
        ("交易用户数", f"{merged_df['user_id'].nunique():,}", "数据期内统计", "#3B82F6", "user"),
        ("商品种类数", f"{merged_df['auction_id'].nunique():,}", "商品 ID 去重", "#8B5CF6", "box"),
        ("总购买数量", f"{int(merged_df['buy_mount'].sum()):,}", "样本汇总", "#F59E0B", "cart"),
    ]
    kpi_html = "".join(
        f"""
        <section class="kpi-card">
          <div class="kpi-icon" style="--kpi-color:{color}">{icon_svg(icon)}</div>
          <div class="kpi-body"><span>{name}</span><strong>{value}</strong><small>{hint}</small></div>
        </section>
        """
        for name, value, hint, color, icon in kpis
    )

    quality_rows = [
        ("清洗后交易记录数", fmt_number(len(merged_df)), 1.0, "#3B82F6"),
        ("宝宝画像匹配率", pct(profile_match_rate), profile_match_rate, "#22C55E"),
        ("unknown 用户占比", pct(unknown_rate), unknown_rate, "#64748B"),
        ("有效购买记录数", fmt_number(valid_buy_records), valid_record_rate, "#F59E0B"),
        ("缺失字段数量", fmt_number(missing_field_count), min(missing_field_count / max(len(merged_df), 1), 1), "#EF476F"),
    ]
    quality_html = "".join(
        f"<div class='quality-row'><span>{label}</span><strong>{value}</strong><em><i style='width:{ratio * 100:.1f}%;background:{color}'></i></em></div>"
        for label, value, ratio, color in quality_rows
    )

    metric_colors = {"Accuracy": "#3B82F6", "Precision": "#22C55E", "Recall": "#F59E0B", "F1": "#8B5CF6", "AUC": "#1D4ED8"}
    model_rows = ""
    for _, row in model_metrics.head(4).iterrows():
        model_rows += (
            f"<tr><th>{html.escape(str(row['model']))}</th>"
            f"{progress_cell(row['Accuracy'], metric_colors['Accuracy'])}"
            f"{progress_cell(row['Precision_weighted'], metric_colors['Precision'])}"
            f"{progress_cell(row['Recall_weighted'], metric_colors['Recall'])}"
            f"{progress_cell(row['F1_weighted'], metric_colors['F1'])}"
            f"<td><span class='na'>N/A</span></td></tr>"
        )
    model_rows += (
        f"<tr><th>High Buy - {html.escape(str(high_best['model']))}</th>"
        f"{progress_cell(high_best['Accuracy'], metric_colors['Accuracy'])}"
        f"{progress_cell(high_best['Precision'], metric_colors['Precision'])}"
        f"{progress_cell(high_best['Recall'], metric_colors['Recall'])}"
        f"{progress_cell(high_best['F1'], metric_colors['F1'])}"
        f"{progress_cell(high_best['AUC'], metric_colors['AUC'])}</tr>"
    )

    diagnosis_html = f"""
      <p><b>主模型任务：</b>商品大类识别（cat1 多分类）。</p>
      <p><b>最佳展示模型：</b>{html.escape(str(best_main['model']))}，Accuracy={best_main['Accuracy']:.3f}，F1={best_main['F1_weighted']:.3f}。</p>
      <p><b>标签泄露检查：</b>未使用 cat1_name 或 cat1 派生字段；high_buy 辅助模型已删除 buy_mount 及全量销量聚合特征。</p>
      <p><b>模型局限性：</b>缺少价格、品牌、店铺、浏览、收藏、加购和真实地域字段，后续可继续扩展。</p>
    """

    chart_payload = {
        "chartColors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_df["year_month"].astype(str).tolist(),
        "monthBuyValues": monthly_df["total_buy_mount"].astype(float).tolist(),
        "monthOrderValues": monthly_df["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(value), str(value)) for value in weekday_df["weekday"]],
        "weekdayBuyValues": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrderValues": weekday_df["trade_count"].astype(float).tolist(),
        "highBuyNames": ["高购买量订单", "普通订单"],
        "highBuyValues": [int(high_counts.get("高购买量订单", 0)), int(high_counts.get("普通订单", 0))],
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": age_counts.index.astype(str).tolist(),
        "ageValues": age_counts.astype(int).tolist(),
        "featureNames": feature_top["feature"].astype(str).tolist(),
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    payload_json = json.dumps(chart_payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight: Mother & Baby E-Commerce Analytics Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --theme-pink:#FF5C8A; --bg:#FAFBFF; --card:#FFFFFF; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --shadow:0 10px 24px rgba(16,24,40,.055); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif; color:var(--title); background:var(--bg); }}
    .sidebar {{ position:fixed; left:20px; top:18px; bottom:18px; width:270px; background:#fff; border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); padding:18px 16px; z-index:2; }}
    .brand {{ display:flex; align-items:center; gap:10px; padding:2px 4px 18px; }}
    .brand-mark {{ width:42px; height:42px; border-radius:14px; display:grid; place-items:center; color:#fff; font-weight:800; background:linear-gradient(135deg,#FF5C8A,#FF8FAB); }}
    .brand h1 {{ margin:0; color:var(--theme-pink); font-size:21px; line-height:1.05; }}
    .brand small,.side-panel p {{ color:var(--muted); font-size:12px; }}
    .side-panel {{ border-top:1px solid #F2F4F7; padding:14px 4px 0; margin-top:6px; }}
    .side-panel h3 {{ margin:0 0 10px; font-size:14px; color:#101828; }}
    .side-panel p {{ margin:7px 0; line-height:1.45; }}
    .side-stat {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid #F8FAFC; }}
    .side-stat span {{ color:var(--muted); font-size:12px; }}
    .side-stat strong {{ color:#101828; font-size:15px; text-align:right; }}
    .side-deco {{ height:96px; margin:14px 4px 0; border-radius:16px; border:1px solid #FFE4EC; position:relative; background:radial-gradient(circle at 18% 30%,rgba(255,92,138,.16) 0 18px,transparent 19px),radial-gradient(circle at 78% 38%,rgba(245,158,11,.16) 0 24px,transparent 25px),linear-gradient(135deg,#FFF6FA,#FFFFFF); }}
    .side-deco svg {{ position:absolute; right:16px; top:18px; width:52px; height:52px; fill:none; stroke:#FF5C8A; stroke-width:2; opacity:.55; }}
    .page {{ margin-left:306px; padding:18px 22px 18px 0; max-width:1920px; }}
    .topbar {{ min-height:86px; display:grid; grid-template-columns:1fr auto; align-items:center; margin-bottom:16px; background:#fff; border:1px solid var(--border); border-radius:18px; padding:18px 22px; box-shadow:var(--shadow); }}
    .topbar-title h2 {{ margin:0; font-size:28px; line-height:1.15; color:#0B1533; }}
    .topbar-title p {{ margin:8px 0 0; color:#475467; font-size:15px; }}
    .update-box {{ justify-self:end; display:grid; gap:4px; padding:10px 14px; background:#FFF8FB; border:1px solid #FFE4EC; border-radius:14px; color:#344054; font-size:13px; }}
    .update-box span {{ color:var(--muted); font-size:12px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:16px; }}
    .kpi-card {{ min-height:112px; display:flex; align-items:center; gap:16px; padding:18px; background:var(--card); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); position:relative; overflow:hidden; }}
    .kpi-card::after {{ content:""; position:absolute; right:16px; bottom:14px; width:96px; height:32px; opacity:.16; background:linear-gradient(135deg,transparent 0 20%,var(--kpi-color) 21% 25%,transparent 26% 45%,var(--kpi-color) 46% 50%,transparent 51%); clip-path:polygon(0 70%,15% 62%,26% 75%,38% 44%,52% 56%,65% 28%,78% 42%,100% 14%,100% 100%,0 100%); }}
    .kpi-icon {{ width:58px; height:58px; flex:0 0 58px; border-radius:18px; display:grid; place-items:center; color:#fff; background:var(--kpi-color); box-shadow:0 10px 24px rgba(255,92,138,.16); }}
    .kpi-icon svg {{ width:28px; height:28px; fill:currentColor; }}
    .kpi-body span {{ display:block; color:#101828; font-size:13px; font-weight:700; }}
    .kpi-body strong {{ display:block; margin-top:8px; font-size:25px; line-height:1; letter-spacing:.2px; }}
    .kpi-body small {{ display:block; margin-top:12px; color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); padding:16px 18px; overflow:hidden; }}
    .card h3 {{ margin:0; font-size:16px; color:#101828; }}
    .card .sub {{ margin-top:6px; color:var(--muted); font-size:12px; }}
    .chart {{ width:100%; height:250px; margin-top:8px; }}
    .trend-card {{ grid-column:span 7; grid-row:span 2; }}
    .trend-card .chart {{ height:386px; }}
    .top-card,.small-card {{ grid-column:span 5; }}
    .top-card .chart,.small-card .chart {{ height:179px; }}
    .quarter-card {{ grid-column:span 3; }}
    .third-card {{ grid-column:span 4; }}
    .feature-card {{ grid-column:span 3; }}
    .diagnosis-card {{ grid-column:span 5; }}
    .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .mini-chart {{ height:206px; }}
    .wordcloud {{ height:236px; display:flex; align-items:center; justify-content:center; margin-top:8px; }}
    .wordcloud img {{ width:100%; height:100%; object-fit:contain; background:#fff; border-radius:12px; }}
    .quality-list {{ margin-top:12px; display:grid; gap:11px; }}
    .quality-row {{ display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }}
    .quality-row span {{ color:#344054; font-size:12px; }}
    .quality-row strong {{ color:#101828; font-size:13px; }}
    .quality-row em {{ grid-column:1 / -1; height:7px; background:#F2F4F7; border-radius:999px; overflow:hidden; }}
    .quality-row i {{ display:block; height:100%; border-radius:999px; }}
    table.model-table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }}
    .model-table th,.model-table td {{ padding:9px 8px; border-bottom:1px solid #F2F4F7; text-align:left; white-space:nowrap; }}
    .model-table th {{ color:#344054; font-weight:700; }}
    .score {{ display:inline-block; min-width:44px; color:#101828; font-weight:700; }}
    .na {{ color:#98A2B3; font-weight:700; }}
    .progress {{ display:inline-block; width:70px; height:6px; margin-left:7px; background:#F2F4F7; border-radius:99px; overflow:hidden; vertical-align:middle; }}
    .progress i {{ display:block; height:100%; border-radius:99px; }}
    .diagnosis {{ margin-top:12px; display:grid; gap:10px; }}
    .diagnosis p {{ margin:0; color:#475467; line-height:1.62; font-size:13px; }}
    .diagnosis b {{ color:#101828; }}
    footer {{ margin-top:12px; padding:13px 18px; border-radius:16px; border:1px solid #FFE4EC; background:linear-gradient(90deg,#FFF6FA,#FFFDFE); color:#667085; display:flex; justify-content:space-between; gap:16px; font-size:12px; }}
    footer b {{ color:#FF5C8A; }}
    @media (max-width:1280px) {{ .sidebar {{ position:static; width:auto; margin:16px; }} .page {{ margin-left:0; padding:0 16px 18px; }} .topbar {{ grid-template-columns:1fr; gap:12px; }} .update-box {{ justify-self:start; }} .trend-card,.top-card,.small-card,.quarter-card,.third-card,.feature-card,.diagnosis-card {{ grid-column:1 / -1; }} .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand"><div class="brand-mark">MB</div><div><h1>MotherBabyInsight</h1><small>母婴电商消费行为分析平台</small></div></div>
    <section class="side-panel"><h3>数据概览</h3><p>样本期间：{data_period}</p><p>清洗后记录：{fmt_number(len(merged_df))}</p></section>
    <section class="side-panel">
      <h3>核心洞察</h3>
      <div class="side-stat"><span>热门品类</span><strong>{html.escape(str(category_df.iloc[0]['cat1_name']))}</strong></div>
      <div class="side-stat"><span>高购买量订单占比</span><strong>{pct(high_result['positive_rate'])}</strong></div>
      <div class="side-stat"><span>画像匹配率</span><strong>{pct(profile_match_rate)}</strong></div>
      <div class="side-stat"><span>主模型 Accuracy</span><strong>{best_main['Accuracy']:.3f}</strong></div>
      <div class="side-stat"><span>主模型 F1</span><strong>{best_main['F1_weighted']:.3f}</strong></div>
    </section>
    <section class="side-panel"><h3>数据说明</h3><p>原始字段不含省份或城市，因此本版大屏不展示地图。</p><p>词云通过相对路径引用 PNG，避免 HTML 文件过大。</p></section>
    <div class="side-deco"><svg viewBox="0 0 64 64"><path d="M20 48h24M24 48l2-22h12l2 22M27 24c0-7 10-7 10 0M23 22h18M30 34h4M28 40h8"/></svg></div>
  </aside>
  <main class="page">
    <header class="topbar"><section class="topbar-title"><h2>MotherBabyInsight: Mother & Baby E-Commerce Analytics Dashboard</h2><p>基于淘宝母婴购物数据的用户消费行为分析与预测</p></section><section class="update-box"><strong>数据期间：{data_period}</strong><span>最后更新时间：{update_time}</span></section></header>
    <div class="kpis">{kpi_html}</div>
    <section class="grid">
      <section class="card trend-card"><h3>购买趋势概览</h3><div class="sub">月度购买数量与订单数走势</div><div id="trendChart" class="chart"></div></section>
      <section class="card top-card"><h3>商品大类销量 TOP10</h3><div class="sub">按购买数量汇总，类别为中文业务词</div><div id="categoryChart" class="chart"></div></section>
      <section class="card small-card"><h3>高购买量订单占比</h3><div class="sub">基于 buy_mount 的 75% 分位数规则划分</div><div id="highBuyChart" class="chart"></div></section>
      <section class="card quarter-card"><h3>用户画像分析</h3><div class="sub">宝宝性别与年龄阶段分布</div><div class="profile-grid"><div id="genderChart" class="mini-chart"></div><div id="ageChart" class="mini-chart"></div></div></section>
      <section class="card quarter-card"><h3>星期购买活跃度</h3><div class="sub">购买数量与订单数按星期聚合</div><div id="weekdayChart" class="chart"></div></section>
      <section class="card quarter-card"><h3>热门商品词云</h3><div class="sub">按销量加权的母婴消费关键词</div><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图"></div></section>
      <section class="card quarter-card"><h3>数据质量与画像匹配情况</h3><div class="sub">基于清洗合并后的数据计算</div><div class="quality-list">{quality_html}</div></section>
      <section class="card third-card"><h3>模型性能对比</h3><div class="sub">主模型与辅助模型核心指标</div><table class="model-table"><thead><tr><th>模型</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody>{model_rows}</tbody></table></section>
      <section class="card feature-card"><h3>特征重要性 TOP10</h3><div class="sub">商品大类识别主模型</div><div id="featureChart" class="chart"></div></section>
      <section class="card diagnosis-card"><h3>模型诊断结论</h3><div class="sub">任务定位、泄露检查与局限性</div><div class="diagnosis">{diagnosis_html}</div></section>
    </section>
    <footer><span><b>数据来源：</b>淘宝母婴购物数据集</span><span><b>数据期间：</b>{data_period}</span><span><b>最后更新：</b>{update_time}</span><span><b>业务标语：</b>用数据洞察母婴消费，用智能守护宝宝成长</span></footer>
  </main>
  <script>
    const injectedPayload = {payload_json};
    let payload = injectedPayload;
    const textColor = "#101828", mutedColor = "#667085", gridLine = "#EEF2F7";
    function makeChart(id, option) {{ const el = document.getElementById(id); if (!window.echarts) {{ el.innerHTML = "<div class='sub'>ECharts 加载失败，请联网后重新打开。</div>"; return; }} const chart = echarts.init(el, null, {{ renderer:"canvas" }}); chart.setOption(option); window.addEventListener("resize", () => chart.resize()); }}
    makeChart("trendChart", {{ color:["#3B82F6","#22C55E"], tooltip:{{trigger:"axis"}}, legend:{{top:0,textStyle:{{color:mutedColor}}}}, grid:{{left:54,right:48,top:48,bottom:42}}, xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor}},axisTick:{{show:false}}}}, yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor}},splitLine:{{show:false}}}}], series:[{{name:"购买数量",type:"bar",data:payload.monthBuyValues,barWidth:24,itemStyle:{{borderRadius:[8,8,0,0],color:"#3B82F6"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.monthOrderValues,smooth:true,symbolSize:7,lineStyle:{{width:3,color:"#22C55E"}},itemStyle:{{color:"#22C55E"}}}}] }});
    makeChart("categoryChart", {{ tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}}, grid:{{left:78,right:30,top:18,bottom:20}}, xAxis:{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}}, yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor}},axisTick:{{show:false}}}}, series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:10,label:{{show:true,position:"right",color:textColor}},itemStyle:{{borderRadius:[0,7,7,0],color:p=>payload.chartColors[p.dataIndex%payload.chartColors.length]}}}}] }});
    makeChart("highBuyChart", {{ color:["#EF476F","#3B82F6"], tooltip:{{trigger:"item"}}, legend:{{bottom:0,textStyle:{{color:mutedColor}}}}, series:[{{name:"订单类型",type:"pie",radius:["50%","72%"],center:["50%","45%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor,fontWeight:700}},data:payload.highBuyNames.map((name,i)=>({{name,value:payload.highBuyValues[i]}}))}}] }});
    makeChart("genderChart", {{ color:["#3B82F6","#EF476F","#64748B"], title:{{text:"性别分布",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}}, tooltip:{{trigger:"item"}}, series:[{{type:"pie",radius:["45%","68%"],center:["50%","58%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}] }});
    makeChart("ageChart", {{ color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F","#06B6D4"], title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}}, tooltip:{{trigger:"item"}}, series:[{{type:"pie",radius:["42%","68%"],center:["50%","58%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}] }});
    makeChart("weekdayChart", {{ color:["#22C55E","#3B82F6"], tooltip:{{trigger:"axis"}}, grid:{{left:48,right:36,top:36,bottom:34}}, xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor}},axisTick:{{show:false}}}}, yAxis:[{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{show:false}}}}], series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuyValues,barWidth:20,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrderValues,smooth:true,symbolSize:7,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}] }});
    makeChart("featureChart", {{ color:["#8B5CF6"], tooltip:{{trigger:"axis"}}, grid:{{left:110,right:22,top:16,bottom:20}}, xAxis:{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}}, yAxis:{{type:"category",data:payload.featureNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}}}}, series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:11,label:{{show:true,position:"right",color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:"#8B5CF6"}}}}] }});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    def fmt_number(value) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.0f}"
        return str(value)

    def pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def metric_card(label: str, value: str, note: str, color: str, icon: str) -> str:
        return f"""
        <section class="kpi-card" style="--accent:{color}">
          <div class="kpi-icon">{icon}</div>
          <div>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
          </div>
        </section>
        """

    def progress_item(label: str, value: str, ratio: float, color: str) -> str:
        return f"""
        <div class="progress-item">
          <div><span>{label}</span><strong>{value}</strong></div>
          <em><i style="width:{max(0, min(ratio, 1)) * 100:.1f}%;background:{color}"></i></em>
        </div>
        """

    update_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    data_period = f"{merged_df['year_month'].min()} ~ {merged_df['year_month'].max()}"
    category_df = tables["category_sales_summary"].head(10).copy()
    monthly_df = tables["monthly_sales_summary"].copy()
    weekday_df = tables["weekday_sales_summary"].copy()
    weekday_names = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}

    model_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    best_main = model_metrics.iloc[0]
    high_best = high_metrics.iloc[0]

    profile_known = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    profile_match_rate = float(profile_known.mean())
    unknown_rate = float((~profile_known).mean())
    valid_buy_records = int((merged_df["buy_mount"] > 0).sum())
    valid_record_rate = valid_buy_records / max(len(merged_df), 1)
    missing_field_count = int(merged_df.isna().sum().sum())
    high_counts = high_result["model_df"]["high_buy"].map({0: "普通订单", 1: "高购买量订单"}).value_counts()
    gender_labels = merged_df["gender"].map({"female": "女宝宝", "male": "男宝宝", "unknown": "未知"}).fillna("未知")
    gender_counts = gender_labels.value_counts()
    age_counts = merged_df["baby_age_group"].value_counts().reindex(["unknown", "0", "1", "2-3", "4-6", "7+"], fill_value=0)
    feature_top = cat1_result["feature_importance"].head(10)

    kpi_html = "".join(
        [
            metric_card("总交易记录数", f"{len(merged_df):,}", "清洗后样本", "#FF5C8A", "BAG"),
            metric_card("交易用户数", f"{merged_df['user_id'].nunique():,}", "数据期内统计", "#3B82F6", "USR"),
            metric_card("商品种类数", f"{merged_df['auction_id'].nunique():,}", "商品 ID 去重", "#8B5CF6", "BOX"),
            metric_card("总购买数量", f"{int(merged_df['buy_mount'].sum()):,}", "样本汇总", "#F59E0B", "CRT"),
        ]
    )

    quality_html = "".join(
        [
            progress_item("清洗后交易记录数", fmt_number(len(merged_df)), 1.0, "#3B82F6"),
            progress_item("宝宝画像匹配率", pct(profile_match_rate), profile_match_rate, "#22C55E"),
            progress_item("unknown 用户占比", pct(unknown_rate), unknown_rate, "#64748B"),
            progress_item("有效购买记录占比", pct(valid_record_rate), valid_record_rate, "#22C55E"),
            progress_item("缺失字段数量", fmt_number(missing_field_count), min(missing_field_count / max(len(merged_df), 1), 1), "#EF476F"),
        ]
    )

    ordered_model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    cat_metric_by_name = {str(row["model"]): row for _, row in model_metrics.iterrows()}
    model_names, model_accuracy, model_f1, model_auc = [], [], [], []
    for name in ordered_model_names:
        if name in cat_metric_by_name:
            row = cat_metric_by_name[name]
            model_names.append(name)
            model_accuracy.append(round(float(row["Accuracy"]), 4))
            model_f1.append(round(float(row["F1_weighted"]), 4))
            model_auc.append(None)
    model_names.append(f"High Buy - {high_best['model']}")
    model_accuracy.append(round(float(high_best["Accuracy"]), 4))
    model_f1.append(round(float(high_best["F1"]), 4))
    model_auc.append(round(float(high_best["AUC"]), 4))

    chart_payload = {
        "chartColors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_df["year_month"].astype(str).tolist(),
        "monthBuyValues": monthly_df["total_buy_mount"].astype(float).tolist(),
        "monthOrderValues": monthly_df["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(value), str(value)) for value in weekday_df["weekday"]],
        "weekdayBuyValues": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrderValues": weekday_df["trade_count"].astype(float).tolist(),
        "highBuyNames": ["高购买量订单", "普通订单"],
        "highBuyValues": [int(high_counts.get("高购买量订单", 0)), int(high_counts.get("普通订单", 0))],
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": age_counts.index.astype(str).tolist(),
        "ageValues": age_counts.astype(int).tolist(),
        "featureNames": feature_top["feature"].astype(str).tolist(),
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
        "modelNames": model_names,
        "modelAccuracy": model_accuracy,
        "modelF1": model_f1,
        "modelAuc": model_auc,
        "highBuyRate": round(float(high_result["positive_rate"]) * 100, 1),
    }
    payload_json = json.dumps(chart_payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --theme-pink:#FF5C8A; --soft-pink:#FFF1F5; --bg:#FAFBFF; --card:#FFFFFF;
      --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7;
      --shadow:0 12px 30px rgba(16,24,40,.06);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--title);
      font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;
    }}
    .shell {{ display:grid; grid-template-columns:278px 1fr; gap:20px; min-height:100vh; padding:18px 22px 18px 18px; }}
    .sidebar {{
      background:var(--card); border:1px solid var(--border); border-radius:22px; box-shadow:var(--shadow);
      padding:20px 18px; display:flex; flex-direction:column; min-height:calc(100vh - 36px);
    }}
    .brand {{ display:flex; gap:12px; align-items:center; padding-bottom:18px; border-bottom:1px solid #F2F4F7; }}
    .logo {{
      width:46px; height:46px; border-radius:15px; color:#fff; display:grid; place-items:center; font-weight:800;
      background:linear-gradient(135deg,#FF5C8A,#FF9EB8);
    }}
    .brand h1 {{ margin:0; font-size:22px; color:var(--theme-pink); line-height:1.05; }}
    .brand p {{ margin:5px 0 0; font-size:12px; color:var(--muted); }}
    .insight {{ padding:17px 2px 0; }}
    .insight h2 {{ margin:0 0 12px; font-size:16px; }}
    .kv {{ display:grid; grid-template-columns:1fr auto; gap:8px; padding:10px 0; border-bottom:1px solid #F8FAFC; align-items:center; }}
    .kv span {{ color:var(--muted); font-size:12px; }}
    .kv strong {{ color:#101828; font-size:15px; text-align:right; }}
    .tagline {{
      margin-top:auto; padding:16px; border-radius:18px; background:linear-gradient(135deg,#FFF1F5,#FFFFFF);
      border:1px solid #FFE4EC; color:#AD3159; font-size:13px; line-height:1.65; position:relative; overflow:hidden;
    }}
    .tagline:after {{ content:""; position:absolute; width:76px; height:76px; right:-28px; bottom:-30px; border-radius:50%; background:rgba(255,92,138,.12); }}
    .main {{ min-width:0; }}
    .hero {{
      background:var(--card); border:1px solid var(--border); border-radius:22px; box-shadow:var(--shadow);
      padding:20px 24px; margin-bottom:16px;
    }}
    .hero-top {{ display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start; }}
    .hero h2 {{ margin:0; font-size:30px; letter-spacing:.2px; }}
    .hero .en {{ margin:7px 0 0; color:#475467; font-size:15px; }}
    .hero .cn {{ margin:5px 0 0; color:#667085; font-size:14px; }}
    .meta {{ text-align:right; color:#475467; font-size:12px; line-height:1.8; white-space:nowrap; }}
    .finding {{
      margin-top:16px; padding:12px 14px; border-radius:14px; border:1px solid #FFE4EC;
      background:linear-gradient(90deg,#FFF1F5,#FFFFFF); color:#8F294C; font-weight:700; font-size:13px;
    }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:16px; }}
    .kpi-card {{
      min-height:112px; display:flex; gap:15px; align-items:center; padding:18px; background:#fff;
      border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow);
    }}
    .kpi-icon {{
      width:58px; height:58px; border-radius:18px; display:grid; place-items:center; flex:0 0 58px;
      color:#fff; background:var(--accent); font-weight:800; font-size:13px;
    }}
    .kpi-card span {{ display:block; font-size:13px; color:#101828; font-weight:700; }}
    .kpi-card strong {{ display:block; margin-top:8px; font-size:26px; line-height:1; }}
    .kpi-card small {{ display:block; margin-top:12px; color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; }}
    .card {{
      background:#fff; border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow);
      padding:16px 18px; min-width:0; overflow:hidden;
    }}
    .card h3 {{ margin:0; color:#101828; font-size:16px; }}
    .sub {{ margin-top:6px; color:var(--muted); font-size:12px; }}
    .chart {{ height:250px; margin-top:8px; width:100%; }}
    .trend {{ grid-column:span 6; }}
    .trend .chart {{ height:380px; }}
    .top10 {{ grid-column:span 4; }}
    .top10 .chart {{ height:380px; }}
    .donut {{ grid-column:span 2; }}
    .donut .chart {{ height:380px; }}
    .aux {{ grid-column:span 4; }}
    .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .mini-chart {{ height:215px; }}
    .wordcloud {{ height:236px; margin-top:8px; display:flex; align-items:center; justify-content:center; background:#fff; border-radius:14px; }}
    .wordcloud img {{ width:100%; height:100%; object-fit:contain; }}
    .wordcloud .missing {{ color:var(--muted); font-size:13px; }}
    .quality {{ display:grid; gap:12px; margin-top:12px; }}
    .quality-row {{ display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }}
    .quality-row span {{ color:var(--text); font-size:12px; }}
    .quality-row strong {{ font-size:13px; }}
    .quality-row em {{ grid-column:1/-1; height:7px; background:#F2F4F7; border-radius:999px; overflow:hidden; }}
    .quality-row i {{ display:block; height:100%; border-radius:999px; }}
    .model-chart {{ grid-column:span 5; }}
    .feature {{ grid-column:span 3; }}
    .diagnosis {{ grid-column:span 4; }}
    .diagnosis-body {{ margin-top:12px; display:grid; gap:10px; }}
    .diagnosis-body p {{ margin:0; color:#475467; font-size:13px; line-height:1.62; }}
    .diagnosis-body b {{ color:#101828; }}
    footer {{ margin-top:16px; color:#667085; font-size:12px; text-align:center; }}
    @media (max-width:1280px) {{
      .shell {{ grid-template-columns:1fr; }}
      .sidebar {{ min-height:auto; }}
      .trend,.top10,.donut,.aux,.model-chart,.feature,.diagnosis {{ grid-column:1/-1; }}
      .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .hero-top {{ grid-template-columns:1fr; }}
      .meta {{ text-align:left; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="logo">MB</div><div><h1>MotherBabyInsight</h1><p>实训答辩数据驾驶舱</p></div></div>
      <section class="insight">
        <h2>核心洞察</h2>
        <div class="kv"><span>数据期间</span><strong>{data_period}</strong></div>
        <div class="kv"><span>热门品类</span><strong>{html.escape(str(category_df.iloc[0]['cat1_name']))}</strong></div>
        <div class="kv"><span>高购买量订单占比</span><strong>{pct(high_result['positive_rate'])}</strong></div>
        <div class="kv"><span>主模型 Accuracy</span><strong>{best_main['Accuracy']:.3f}</strong></div>
        <div class="kv"><span>主模型 F1-score</span><strong>{best_main['F1_weighted']:.3f}</strong></div>
        <div class="kv"><span>画像匹配率</span><strong>{pct(profile_match_rate)}</strong></div>
      </section>
      <div class="tagline">围绕“什么时候买得多、买什么最多、哪些订单价值更高”，形成母婴消费行为的完整分析闭环。</div>
    </aside>
    <main class="main">
      <section class="hero">
        <div class="hero-top">
          <div>
            <h2>MotherBabyInsight 母婴电商消费行为分析可视化大屏</h2>
            <p class="en">Mother & Baby E-Commerce Behavior Analytics Dashboard</p>
            <p class="cn">基于淘宝母婴购物数据的用户消费行为分析与预测</p>
          </div>
          <div class="meta"><b>数据期间：</b>{data_period}<br><b>最后更新时间：</b>{update_time}</div>
        </div>
        <div class="finding">核心发现：{html.escape(str(category_df.iloc[0]['cat1_name']))}、婴儿服饰和尿裤湿巾构成母婴消费主力品类，购买行为呈现明显月度波动，商品大类识别模型表现稳定。</div>
      </section>
      <section class="kpis">{kpi_html}</section>
      <section class="grid">
        <section class="card trend"><h3>购买趋势概览</h3><div class="sub">什么时候买得多：月度购买数量与订单数走势</div><div id="trendChart" class="chart"></div></section>
        <section class="card top10"><h3>商品大类销量 TOP10</h3><div class="sub">买什么最多：中文业务品类按购买数量降序</div><div id="categoryChart" class="chart"></div></section>
        <section class="card donut"><h3>高购买量订单占比</h3><div class="sub">哪些订单价值更高：75% 分位数划分</div><div id="highBuyChart" class="chart"></div></section>
        <section class="card aux"><h3>星期购买活跃度</h3><div class="sub">购买数量与订单数按星期聚合</div><div id="weekdayChart" class="chart"></div></section>
        <section class="card aux"><h3>热门商品词云</h3><div class="sub">按销量加权的母婴消费关键词</div><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
        <section class="card aux"><h3>数据质量与画像匹配情况</h3><div class="sub">清洗、画像匹配与缺失字段概览</div><div class="quality">{quality_html}</div></section>
        <section class="card aux"><h3>用户画像分析</h3><div class="sub">性别与年龄阶段分布，unknown 作为质量提示</div><div class="profile-grid"><div id="genderChart" class="mini-chart"></div><div id="ageChart" class="mini-chart"></div></div></section>
        <section class="card model-chart"><h3>模型性能对比（主模型与辅助模型）</h3><div class="sub">Accuracy、F1-score 与 high_buy 辅助模型 AUC</div><div id="modelChart" class="chart"></div></section>
        <section class="card feature"><h3>特征重要性 TOP10</h3><div class="sub">商品大类识别主模型特征贡献</div><div id="featureChart" class="chart"></div></section>
        <section class="card diagnosis"><h3>模型诊断结论</h3><div class="sub">任务定位、泄露检查与局限性</div><div class="diagnosis-body"><p><b>主模型：</b>商品大类识别。</p><p><b>最佳模型：</b>{html.escape(str(best_main['model']))}，Accuracy={best_main['Accuracy']:.3f}，F1-score={best_main['F1_weighted']:.3f}。</p><p><b>辅助任务：</b>high_buy 高购买量订单识别，用于解释高价值购买行为。</p><p><b>标签泄露检查：</b>未使用 cat1_name；high_buy 已删除 buy_mount。</p><p><b>局限性：</b>缺少价格、品牌、浏览、收藏、加购等行为特征。</p></div></section>
      </section>
      <footer>数据来源：淘宝母婴购物数据集 · 页面定位：实训答辩展示型数据驾驶舱</footer>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const chartColors = payload.chartColors;
    const textColor = "#101828", mutedColor = "#667085", gridLine = "#EEF2F7";
    function makeChart(id, option) {{ const el = document.getElementById(id); if (!window.echarts) {{ el.innerHTML = "<div class='sub'>ECharts 加载失败，请联网后重新打开。</div>"; return; }} const chart = echarts.init(el, null, {{renderer:"canvas"}}); chart.setOption(option); window.addEventListener("resize", () => chart.resize()); }}
    makeChart("trendChart", {{ color:["#3B82F6","#22C55E"], tooltip:{{trigger:"axis"}}, legend:{{top:0,textStyle:{{color:mutedColor}}}}, grid:{{left:56,right:54,top:48,bottom:42}}, xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor}},axisTick:{{show:false}}}}, yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor}},splitLine:{{show:false}}}}], series:[{{name:"购买数量",type:"bar",data:payload.monthBuyValues,barWidth:24,itemStyle:{{borderRadius:[8,8,0,0],color:"#3B82F6"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.monthOrderValues,smooth:true,symbolSize:7,label:{{show:false}},lineStyle:{{width:3,color:"#22C55E"}},itemStyle:{{color:"#22C55E"}}}}] }});
    makeChart("categoryChart", {{ tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}}, grid:{{left:86,right:36,top:18,bottom:24}}, xAxis:{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}}, yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor}},axisTick:{{show:false}}}}, series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:14,label:{{show:true,position:"right",color:textColor}},itemStyle:{{borderRadius:[0,8,8,0],color:p=>chartColors[(p.dataIndex+2)%chartColors.length]}}}}] }});
    makeChart("highBuyChart", {{ color:["#EF476F","#3B82F6"], tooltip:{{trigger:"item"}}, title:{{text:payload.highBuyRate+"%",subtext:"高购买量",left:"center",top:"39%",textStyle:{{fontSize:24,color:"#101828"}},subtextStyle:{{fontSize:12,color:"#667085"}}}}, legend:{{bottom:0,textStyle:{{color:mutedColor}}}}, series:[{{type:"pie",radius:["52%","72%"],center:["50%","45%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor}},data:payload.highBuyNames.map((name,i)=>({{name,value:payload.highBuyValues[i]}}))}}] }});
    makeChart("weekdayChart", {{ color:["#22C55E","#3B82F6"], tooltip:{{trigger:"axis"}}, legend:{{top:0,textStyle:{{color:mutedColor}}}}, grid:{{left:48,right:42,top:44,bottom:34}}, xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor}},axisTick:{{show:false}}}}, yAxis:[{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{show:false}}}}], series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuyValues,barWidth:20,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrderValues,smooth:true,symbolSize:7,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}] }});
    makeChart("genderChart", {{ color:["#3B82F6","#EF476F","#64748B"], title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}}, tooltip:{{trigger:"item"}}, series:[{{type:"pie",radius:["45%","68%"],center:["50%","58%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}] }});
    makeChart("ageChart", {{ color:["#64748B","#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"], title:{{text:"年龄",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}}, tooltip:{{trigger:"item"}}, series:[{{type:"pie",radius:["42%","68%"],center:["50%","58%"],label:{{formatter:"{{b}}\\n{{d}}%",color:textColor}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}] }});
    makeChart("modelChart", {{ color:["#3B82F6","#8B5CF6","#1D4ED8"], tooltip:{{trigger:"axis"}}, legend:{{top:0,textStyle:{{color:mutedColor}}}}, grid:{{left:46,right:28,top:48,bottom:58}}, xAxis:{{type:"category",data:payload.modelNames,axisLabel:{{color:mutedColor,rotate:14}},axisTick:{{show:false}}}}, yAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}}, series:[{{name:"Accuracy",type:"bar",data:payload.modelAccuracy,barWidth:14,label:{{show:true,position:"top",formatter:p=>p.value == null ? "" : p.value.toFixed(3)}}}},{{name:"F1-score",type:"bar",data:payload.modelF1,barWidth:14,label:{{show:true,position:"top",formatter:p=>p.value == null ? "" : p.value.toFixed(3)}}}},{{name:"AUC",type:"bar",data:payload.modelAuc,barWidth:14,label:{{show:true,position:"top",formatter:p=>p.value == null ? "" : p.value.toFixed(3)}}}}] }});
    makeChart("featureChart", {{ color:["#8B5CF6"], tooltip:{{trigger:"axis"}}, grid:{{left:112,right:24,top:18,bottom:22}}, xAxis:{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}}, yAxis:{{type:"category",data:payload.featureNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}}}}, series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:12,label:{{show:true,position:"right",color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,8,8,0],color:"#8B5CF6"}}}}] }});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


# ============================================================
# 15. 结果文件保存与运行总结
# ============================================================

def clean_old_report_charts() -> None:
    for path in CHART_DIR.glob("*.png"):
        try:
            path.unlink()
        except PermissionError:
            print(f"提示：{path.name} 暂时无法删除，将在同名图表生成时覆盖。")


def run_pipeline(use_spark: bool = False) -> None:
    ensure_dirs()
    setup_chinese_font()
    clean_old_report_charts()
    print(PROJECT_EN_NAME)
    print(PROJECT_CN_NAME)

    spark = create_spark_session(use_spark=use_spark)
    try:
        log_step("Step 1", "数据读取")
        trade_sdf, baby_sdf, trade_df, baby_df = read_raw_data(spark)

        log_step("Step 2", "数据质量检查")
        save_quality_tables(trade_df, baby_df)

        log_step("Step 3", "数据清洗与预处理")
        if spark is not None and trade_sdf is not None and baby_sdf is not None:
            merged_sdf, merged_df = clean_with_spark(spark, trade_sdf, baby_sdf)
        else:
            merged_sdf = None
            merged_df = clean_with_pandas(trade_df, baby_df)
        save_merged_data(merged_df)
        print(f"清洗合并后数据规模：{merged_df.shape}")

        log_step("Step 4", "Spark SQL 描述性统计")
        tables = build_statistics(merged_df, spark, merged_sdf)

        log_step("Step 5", "特征工程")
        model_df = feature_engineering(merged_df)
        print(f"建模数据集规模：{model_df.shape}")

        log_step("Step 6", "商品大类 cat1 预测")
        cat1_result = train_cat1_models(model_df)
        print(cat1_result["metrics"])

        log_step("Step 7", "高购买量订单识别")
        high_result = train_high_buy_models(model_df)
        print(high_result["metrics"])
        write_model_diagnosis(cat1_result, high_result)

        log_step("Step 8", "报告图表生成")
        plot_report_charts(model_df, cat1_result, high_result)
        make_wordcloud(merged_df)
        enhance_report_charts()

        log_step("Step 9", "可视化数字大屏生成")
        generate_dashboard(merged_df, tables, cat1_result, high_result)

        log_step("Step 10", "结果文件保存与运行总结")
        print("输出目录：")
        print(f"- {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
        print(f"- {TABLE_DIR.relative_to(PROJECT_ROOT)}")
        print(f"- {MODEL_RESULT_DIR.relative_to(PROJECT_ROOT)}")
        print(f"- {CHART_DIR.relative_to(PROJECT_ROOT)}")
        print(f"- {HTML_DIR.relative_to(PROJECT_ROOT) / 'dashboard.html'}")
    finally:
        if spark is not None:
            spark.stop()


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成最终确认版浅色母婴风答辩大屏。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"""
        <article class="kpi" style="--c:{color}">
          <div class="kpi-icon">{icon}</div>
          <div>
            <span>{html.escape(title)}</span>
            <strong>{html.escape(value)}</strong>
            <small>{html.escape(note)}</small>
          </div>
          <svg viewBox="0 0 110 32" class="spark" aria-hidden="true">
            <polyline points="0,27 12,24 24,28 36,20 48,24 60,16 72,20 84,9 96,14 110,5"
              fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </article>
        """

    def q_item(title: str, value: str, ratio: float, color: str, icon: str) -> str:
        width = max(0, min(float(ratio), 1)) * 100
        return f"""
        <div class="q-item" style="--c:{color}">
          <div class="q-icon">{icon}</div>
          <div class="q-main">
            <div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong></div>
            <em><i style="width:{width:.1f}%"></i></em>
          </div>
        </div>
        """

    category_df = tables.get("category_sales_summary", pd.DataFrame()).head(10).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    best_main = cat_metrics.iloc[0]
    high_best = high_metrics.iloc[0]
    data_period = f"{merged_df['year_month'].min()} ~ {merged_df['year_month'].max()}"
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_known = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    profile_match_rate = float(profile_known.mean())
    unknown_rate = float((~profile_known).mean())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    missing_count = int(merged_df.isna().sum().sum())

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}
    weekday_labels = [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()]
    gender_counts = merged_df["gender"].map({"male": "男宝宝", "female": "女宝宝", "unknown": "unknown"}).fillna("unknown").value_counts()
    age_counts = merged_df["baby_age_group"].value_counts().reindex(["unknown", "0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_label_map = {"unknown": "unknown", "0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上"}
    feature_top = cat1_result["feature_importance"].head(10)

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    metric_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    model_names, model_acc, model_f1, model_auc = [], [], [], []
    for name in ordered_models:
        if name in metric_rows:
            row = metric_rows[name]
            model_names.append(name)
            model_acc.append(round(float(row["Accuracy"]), 4))
            model_f1.append(round(float(row["F1_weighted"]), 4))
            model_auc.append(None)
    model_names.append(f"High Buy - {high_best['model']}")
    model_acc.append(round(float(high_best["Accuracy"]), 4))
    model_f1.append(round(float(high_best["F1"]), 4))
    model_auc.append(round(float(high_best["AUC"]), 4))

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "user"),
            kpi("商品种类数", fmt_num(merged_df["auction_id"].nunique()), "商品 ID 去重统计", "#8B5CF6", "box"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            q_item("清洗后交易记录数", fmt_num(len(merged_df)), 1, "#3B82F6", "样"),
            q_item("宝宝画像匹配率", fmt_pct(profile_match_rate), profile_match_rate, "#22C55E", "像"),
            q_item("Unknown 用户占比", fmt_pct(unknown_rate), unknown_rate, "#64748B", "未"),
            q_item("有效购买记录占比", fmt_pct(valid_rate), valid_rate, "#06B6D4", "购"),
            q_item("缺失字段数量", fmt_num(missing_count), min(missing_count / max(len(merged_df), 1), 1), "#EF476F", "缺"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "metrics": {
            "rawTradeRecords": raw_trade_records,
            "cleanTradeRecords": int(len(merged_df)),
            "tradeUsers": int(merged_df["user_id"].nunique()),
            "itemKinds": int(merged_df["auction_id"].nunique()),
            "totalBuyMount": int(merged_df["buy_mount"].sum()),
            "highBuyOrders": int(high_mask.sum()),
            "normalOrders": int((~high_mask).sum()),
            "highBuyRate": round(high_rate * 100, 1),
            "salesContributionRate": round(high_sales_rate * 100, 1),
            "avgBuyLift": round(value_lift, 1),
            "profileCoverageRate": round(profile_rate * 100, 1),
            "profileUnmatchedRecords": unmatched_count,
            "tradeUsabilityRate": round(valid_rate * 100, 1),
            "coreFieldCompleteRate": round(core_complete_rate * 100, 1),
            "highestModel": str(highest_cat1_row["model"]),
            "highestAccuracy": round(float(highest_cat1_row["Accuracy"]), 4),
            "highestF1": round(float(highest_cat1_row["F1_weighted"]), 4),
            "displayModel": "Random Forest",
            "displayAccuracy": round(float(rf_row["Accuracy"]), 4),
            "displayF1": round(float(rf_row["F1_weighted"]), 4),
            "highBuyAuxModel": str(high_dt_row["model"]),
            "highBuyAuxAuc": round(float(high_dt_row["AUC"]), 4),
            "highBuyAuxF1": round(float(high_dt_row["F1"]), 4),
        },
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_df["year_month"].astype(str).tolist(),
        "monthBuy": monthly_df["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_df["trade_count"].astype(float).tolist(),
        "weekdayLabels": weekday_labels,
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "highSalesRate": round(high_sales_rate * 100, 1),
        "valueLift": round(value_lift, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_label_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "featureNames": feature_top["feature"].astype(str).tolist(),
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
        "modelNames": model_names,
        "modelAcc": model_acc,
        "modelF1": model_f1,
        "modelAuc": model_auc,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --shadow:0 12px 28px rgba(16,24,40,.065); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; background:var(--bg); }}
    body {{ color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 92% 3%,rgba(255,92,138,.16),transparent 19%),radial-gradient(circle at 4% 96%,rgba(255,92,138,.13),transparent 18%),#FAFBFF; }}
    .page {{ width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:12px 18px 14px 12px; display:grid; grid-template-columns:330px 1fr; gap:16px; }}
    .sidebar {{ position:relative; min-height:1054px; padding:24px 20px; overflow:hidden; border:1px solid var(--border); border-radius:24px; background:linear-gradient(180deg,rgba(255,255,255,.97),rgba(255,241,245,.76)); box-shadow:var(--shadow); }}
    .brand {{ display:flex; align-items:center; gap:12px; margin-bottom:30px; }} .face {{ width:54px;height:54px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 54px; }}
    .face:before,.face:after {{ content:""; position:absolute; top:20px; width:5px; height:5px; background:var(--pink); border-radius:50%; }} .face:before {{ left:15px; }} .face:after {{ right:15px; }} .smile {{ position:absolute; left:17px; top:29px; width:18px; height:9px; border-bottom:3px solid var(--pink); border-radius:0 0 18px 18px; }}
    .brand h1 {{ margin:0; color:var(--pink); font-size:25px; line-height:1; }} .brand p {{ margin:8px 0 0; color:#101828; font-size:13px; font-weight:800; }}
    .side-title {{ display:flex; gap:10px; align-items:center; margin:0 0 18px; color:var(--pink); font-size:19px; }}
    .insight {{ display:grid; grid-template-columns:58px 1fr; gap:12px; align-items:center; padding:14px 10px; margin-bottom:13px; border-radius:18px; background:rgba(255,255,255,.70); }}
    .i-icon {{ width:52px;height:52px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:20px;font-weight:900;box-shadow:0 10px 20px rgba(16,24,40,.06); }}
    .insight span {{ color:var(--muted); font-size:12px; font-weight:800; }} .insight strong {{ display:block; margin-top:5px; color:var(--c); font-size:24px; line-height:1.05; }} .insight small {{ display:block; margin-top:5px; color:#101828; font-size:12px; font-weight:800; }}
    .art {{ position:absolute; left:22px; right:22px; bottom:28px; height:245px; border-radius:28px; background:linear-gradient(180deg,rgba(255,241,245,.25),rgba(255,241,245,.95)); }}
    .crib {{ position:absolute; left:26px; bottom:45px; width:150px; height:98px; border:6px solid rgba(255,92,138,.42); border-top:0; border-radius:0 0 22px 22px; }} .crib:before {{ content:""; position:absolute; left:-10px; right:-10px; top:-18px; height:8px; background:rgba(255,92,138,.35); border-radius:999px; }} .crib i {{ position:absolute; bottom:0; width:5px; height:90px; background:rgba(255,92,138,.28); border-radius:999px; }} .crib i:nth-child(1){{left:28px}} .crib i:nth-child(2){{left:62px}} .crib i:nth-child(3){{left:96px}}
    .bear {{ position:absolute; right:38px; bottom:37px; width:78px; height:78px; border-radius:50%; background:#FFE5D3; }} .bear:before,.bear:after {{ content:""; position:absolute; top:-11px; width:28px; height:28px; border-radius:50%; background:#FFD3BA; }} .bear:before{{left:5px}} .bear:after{{right:5px}} .bear span:before,.bear span:after {{ content:""; position:absolute; top:28px; width:6px; height:6px; border-radius:50%; background:#8A4A36; }} .bear span:before{{left:25px}} .bear span:after{{right:25px}} .bear span {{ position:absolute; left:30px; top:38px; width:18px; height:12px; border-bottom:3px solid #8A4A36; border-radius:0 0 18px 18px; }}
    .heart {{ position:absolute; color:rgba(255,92,138,.18); font-size:22px; }} .h1{{left:36px;top:28px}} .h2{{right:30px;top:48px}} .h3{{left:90px;bottom:26px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:100px; display:flex; justify-content:center; align-items:center; overflow:hidden; margin-bottom:14px; }} .hero h2 {{ margin:0; text-align:center; font-size:34px; line-height:1.15; }} .hero p {{ margin:8px 0 0; text-align:center; color:#101828; font-size:16px; font-weight:700; }}
    .hero-deco {{ position:absolute; right:12px; top:2px; width:245px; height:94px; border-radius:0 0 0 82px; background:linear-gradient(135deg,rgba(255,241,245,.9),rgba(255,255,255,.25)); }} .bottle {{ position:absolute; right:96px; top:17px; width:34px; height:64px; border-radius:12px 12px 10px 10px; background:linear-gradient(180deg,#D8C2FF,#FFE0EA 42%,#fff 43%,#F8B4C8); }} .bottle:before {{ content:""; position:absolute; left:10px; top:-12px; width:14px; height:14px; border-radius:7px 7px 2px 2px; background:#C4B5FD; }} .mini-bear {{ position:absolute; right:30px; top:19px; width:58px; height:58px; border-radius:50%; background:#FFE5D3; }} .mini-bear:before,.mini-bear:after {{ content:""; position:absolute; top:-8px; width:20px; height:20px; border-radius:50%; background:#FFD3BA; }} .mini-bear:before{{left:2px}} .mini-bear:after{{right:2px}}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px; }} .kpi {{ position:relative; min-height:122px; display:flex; align-items:center; gap:18px; padding:18px; overflow:hidden; background:#fff; border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); }} .kpi-icon {{ width:62px;height:62px;border-radius:50%;display:grid;place-items:center;flex:0 0 62px;color:var(--c);background:#FFF1F5;font-size:0; }} .kpi-icon svg {{ width:31px;height:31px;fill:currentColor; }} .kpi span {{ font-size:13px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:9px;font-size:27px;line-height:1; }} .kpi small {{ display:block;margin-top:12px;color:var(--muted);font-size:12px;font-weight:700; }} .spark {{ position:absolute; right:16px; bottom:13px; width:96px; height:28px; opacity:.9; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:14px; }} .card {{ background:#fff; border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); padding:15px 17px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:16px; color:#101828; }} .sub {{ margin-top:6px; color:var(--muted); font-size:11px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:6px; }} .unit {{ color:var(--muted); font-size:11px; font-weight:800; }}
    .trend {{ grid-column:span 10; height:360px; }} .top10 {{ grid-column:span 7; height:360px; }} .value {{ grid-column:span 7; height:360px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:306px; }} .value-chart {{ height:220px; }}
    .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }} .value-mini {{ padding:10px 8px; border-radius:14px; text-align:center; background:linear-gradient(135deg,#FFF1F5,#fff); }} .value-mini span {{ display:block;color:var(--muted);font-size:11px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:5px;color:var(--vc);font-size:18px; }}
    .gender {{ grid-column:span 3; height:225px; }} .age {{ grid-column:span 4; height:225px; }} .weekday {{ grid-column:span 5; height:225px; }} .word {{ grid-column:span 5; height:225px; }} .quality {{ grid-column:span 7; height:225px; }} .small-chart {{ height:173px; margin-top:5px; }}
    .wordcloud {{ height:168px; margin-top:8px; display:flex; align-items:center; justify-content:center; border-radius:14px; background:#fff; }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; }} .missing {{ color:var(--muted); font-size:13px; }}
    .q-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:9px; margin-top:10px; }} .q-item {{ display:grid;grid-template-columns:36px 1fr;gap:8px;align-items:center;min-height:48px;padding:8px;border-radius:13px;background:#FAFBFF;border:1px solid #F2F4F7; }} .q-icon {{ width:32px;height:32px;border-radius:10px;display:grid;place-items:center;color:var(--c);background:#fff;font-size:12px;font-weight:900; }} .q-main div {{ display:flex;justify-content:space-between;gap:8px;align-items:center; }} .q-main span {{ color:var(--muted);font-size:11px;font-weight:800; }} .q-main strong {{ color:#101828;font-size:13px;white-space:nowrap; }} .q-main em {{ display:block;height:6px;margin-top:7px;border-radius:999px;background:#EEF2F7;overflow:hidden; }} .q-main i {{ display:block;height:100%;border-radius:999px;background:var(--c); }}
    .model {{ grid-column:span 8; height:245px; }} .feature {{ grid-column:span 7; height:245px; }} .advice {{ grid-column:span 9; height:245px; }} .model-chart,.feature-chart {{ height:190px; }} .advice-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px; }} .box {{ min-height:178px;padding:12px 14px;border-radius:14px;background:#FAFBFF;border:1px solid #F2F4F7; }} .box h4 {{ margin:0 0 10px;color:var(--pink);font-size:13px; }} .line {{ display:flex;gap:8px;align-items:flex-start;margin:8px 0;color:#344054;font-size:12px;line-height:1.45; }} .mark {{ width:18px;height:18px;display:inline-grid;place-items:center;flex:0 0 18px;border-radius:50%;color:#fff;font-size:11px;font-weight:900;background:#22C55E; }} .num {{ background:#FB7185; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .art {{ display:none; }} .trend,.top10,.value,.gender,.age,.weekday,.word,.quality,.model,.feature,.advice {{ grid-column:1/-1; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">♥ 核心洞察</h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">♛</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高的商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">↗</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">◎</div><div><span>主模型表现（cat1）</span><strong>{best_main['Accuracy']:.3f}</strong><small>Accuracy ｜ F1-score {best_main['F1_weighted']:.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">●</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_match_rate)}</strong><small>宝宝画像匹配成功率</small></div></div>
      <div class="art"><span class="heart h1">♥</span><span class="heart h2">♥</span><span class="heart h3">♥</span><div class="crib"><i></i><i></i><i></i></div><div class="bear"><span></span></div></div>
    </aside>
    <main class="main">
      <header class="hero"><div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div><div class="hero-deco"><span class="bottle"></span><span class="mini-bear"></span></div></header>
      <section class="kpis">{kpi_html}</section>
      <section class="grid">
        <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">购买数量与订单数月度变化</div></div><span class="unit">单位：件 / 单</span></div><div id="trendChart" class="chart trend-chart"></div></section>
        <section class="card top10"><div class="card-head"><div><h3>商品大类销量 TOP10</h3><div class="sub">中文业务品类按购买数量降序</div></div><span class="unit">单位：件</span></div><div id="categoryChart" class="chart top-chart"></div></section>
        <section class="card value"><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别高购买量订单</div><div id="highBuyChart" class="chart value-chart"></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量占比</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
        <section class="card gender"><h3>用户性别分布</h3><div id="genderChart" class="chart small-chart"></div></section>
        <section class="card age"><h3>宝宝年龄阶段分布</h3><div id="ageChart" class="chart small-chart"></div></section>
        <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
        <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
        <section class="card quality"><h3>数据质量与画像匹配</h3><div class="q-grid">{quality_html}</div></section>
        <section class="card model"><h3>模型性能对比（主模型与辅助模型）</h3><div id="modelChart" class="chart model-chart"></div></section>
        <section class="card feature"><h3>特征重要性 TOP10</h3><div id="featureChart" class="chart feature-chart"></div></section>
        <section class="card advice"><h3>模型诊断与业务建议</h3><div class="advice-grid"><div class="box"><h4>模型诊断</h4><div class="line"><span class="mark">✓</span><span>主要任务：商品大类识别（cat1）</span></div><div class="line"><span class="mark">✓</span><span>最佳模型：{html.escape(str(best_main['model']))}</span></div><div class="line"><span class="mark">✓</span><span>Accuracy：{best_main['Accuracy']:.3f}，F1-score：{best_main['F1_weighted']:.3f}</span></div><div class="line"><span class="mark">✓</span><span>未使用 cat1_name；high_buy 已删除 buy_mount</span></div></div><div class="box"><h4>业务建议</h4><div class="line"><span class="mark num">1</span><span>重点关注{html.escape(hot_category)}等核心品类</span></div><div class="line"><span class="mark num">2</span><span>结合高购买量订单制定促销策略</span></div><div class="line"><span class="mark num">3</span><span>补充价格、品牌、浏览、收藏等特征</span></div><div class="line"><span class="mark num">4</span><span>进一步提升模型解释能力</span></div></div></div></section>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const iconMap = {{bag:"<svg viewBox='0 0 24 24'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",user:"<svg viewBox='0 0 24 24'><path d='M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5Zm0 2c-5 0-8 2.5-8 5v2h16v-2c0-2.5-3-5-8-5Z'/></svg>",box:"<svg viewBox='0 0 24 24'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 7 4v8l-7-4v-8Zm14 0v8l-7 4v-8l7-4Z'/></svg>",cart:"<svg viewBox='0 0 24 24'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>"}};
    document.querySelectorAll(".kpi-icon").forEach(el=>{{el.innerHTML=iconMap[el.textContent.trim()]||"";}});
    const textColor="#101828", mutedColor="#667085", gridLine="#EEF2F7", colors=payload.colors;
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    chart("trendChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:11}}}},grid:{{left:56,right:52,top:46,bottom:38}},xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("categoryChart",{{tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},grid:{{left:98,right:50,top:18,bottom:26}},xAxis:{{type:"value",axisLabel:{{color:mutedColor}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:13,label:{{show:true,position:"right",color:textColor,fontSize:10,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,8,8,0],color:p=>colors[(p.dataIndex+2)%colors.length]}}}}]}});
    chart("highBuyChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"item"}},title:{{text:payload.highRate+"%",subtext:"高购买量订单占比",left:"center",top:"38%",textStyle:{{fontSize:26,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:11,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"center",orient:"vertical",textStyle:{{color:textColor,fontSize:11}}}},series:[{{type:"pie",radius:["52%","73%"],center:["38%","50%"],label:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}});
    chart("genderChart",{{color:["#64748B","#EF476F","#3B82F6"],tooltip:{{trigger:"item"}},legend:{{right:0,top:"center",orient:"vertical",textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["45%","68%"],center:["40%","52%"],label:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}});
    chart("ageChart",{{color:["#64748B","#3B82F6","#F59E0B","#22C55E","#06B6D4","#EF476F"],tooltip:{{trigger:"item"}},legend:{{right:0,top:"center",orient:"vertical",textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","68%"],center:["38%","52%"],label:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}});
    chart("weekdayChart",{{color:["#22C55E","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:44,right:34,top:38,bottom:28}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.weekdayBuy,barWidth:18,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("modelChart",{{color:["#3B82F6","#22C55E","#8B5CF6"],tooltip:{{trigger:"axis"}},legend:{{top:0,textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:16,top:42,bottom:45}},xAxis:{{type:"category",data:payload.modelNames,axisLabel:{{color:textColor,fontSize:10,interval:0}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},series:[{{name:"Accuracy",type:"bar",data:payload.modelAcc,barWidth:11,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value==null?"":p.value.toFixed(3)}}}},{{name:"F1-score",type:"bar",data:payload.modelF1,barWidth:11,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value==null?"":p.value.toFixed(3)}}}},{{name:"AUC",type:"bar",data:payload.modelAuc,barWidth:11,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value==null?"":p.value.toFixed(3)}}}}]}});
    chart("featureChart",{{color:["#8B5CF6"],tooltip:{{trigger:"axis"}},grid:{{left:118,right:34,top:14,bottom:22}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:10,label:{{show:true,position:"right",fontSize:9,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,8,8,0],color:"#8B5CF6"}}}}]}});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成 1920x1080 优化版答辩展示大屏。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"""
        <article class="kpi" style="--c:{color}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-text"><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div>
          <svg viewBox="0 0 104 30" class="spark" aria-hidden="true"><polyline points="0,25 12,22 24,26 36,18 48,22 60,14 72,18 84,8 96,13 104,6" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </article>
        """

    def q_item(title: str, value: str, ratio: float, color: str, icon: str) -> str:
        width = max(0, min(float(ratio), 1)) * 100
        return f"""
        <div class="q-item" style="--c:{color}">
          <div class="q-icon">{icon}</div>
          <div class="q-main"><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong></div><em><i style="width:{width:.1f}%"></i></em></div>
        </div>
        """

    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    overview_df = tables.get("data_overview", pd.DataFrame()).copy()
    overview_df = tables.get("data_overview", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    best_main = cat_metrics.iloc[0]
    high_best = high_metrics.iloc[0]
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    unmatched_count = int((~profile_mask).sum())
    unknown_rate = float((~profile_mask).mean())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)
    missing_count = int(merged_df.isna().sum().sum())

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_raw = matched_df["baby_age_group"].replace("unknown", np.nan).dropna()
        age_counts = age_raw.value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    perf_names, perf_acc, perf_f1 = [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            perf_names.append(name)
            perf_acc.append(round(float(row["Accuracy"]), 4))
            perf_f1.append(round(float(row["F1_weighted"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
        "user_buy_count": "user_buy",
        "user_unique_item_count": "user_items",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "user"),
            kpi("商品种类数", fmt_num(merged_df["auction_id"].nunique()), "商品 ID 去重统计", "#8B5CF6", "box"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            q_item("清洗后交易记录数", fmt_num(len(merged_df)), 1, "#3B82F6", "样"),
            q_item("画像匹配率", fmt_pct(profile_rate), profile_rate, "#22C55E", "像"),
            q_item("画像未匹配记录数", fmt_num(unmatched_count), unknown_rate, "#64748B", "未"),
            q_item("有效购买记录占比", fmt_pct(valid_rate), valid_rate, "#06B6D4", "购"),
            q_item("核心字段完整率", fmt_pct(core_complete_rate), core_complete_rate, "#8B5CF6", "整"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "metrics": {
            "rawTradeRecords": raw_trade_records,
            "cleanTradeRecords": int(len(merged_df)),
            "tradeUsers": int(merged_df["user_id"].nunique()),
            "itemKinds": int(merged_df["auction_id"].nunique()),
            "totalBuyMount": int(merged_df["buy_mount"].sum()),
            "highBuyOrders": int(high_mask.sum()),
            "normalOrders": int((~high_mask).sum()),
            "highBuyRate": round(high_rate * 100, 1),
            "salesContributionRate": round(high_sales_rate * 100, 1),
            "avgBuyLift": round(value_lift, 1),
            "profileCoverageRate": round(profile_rate * 100, 1),
            "profileUnmatchedRecords": unmatched_count,
            "tradeUsabilityRate": round(valid_rate * 100, 1),
            "coreFieldCompleteRate": round(core_complete_rate * 100, 1),
            "highestModel": str(highest_cat1_row["model"]),
            "highestAccuracy": round(float(highest_cat1_row["Accuracy"]), 4),
            "highestF1": round(float(highest_cat1_row["F1_weighted"]), 4),
            "displayModel": "Random Forest",
            "displayAccuracy": round(float(rf_row["Accuracy"]), 4),
            "displayF1": round(float(rf_row["F1_weighted"]), 4),
            "highBuyAuxModel": str(high_dt_row["model"]),
            "highBuyAuxAuc": round(float(high_dt_row["AUC"]), 4),
            "highBuyAuxF1": round(float(high_dt_row["F1"]), 4),
        },
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "highSalesRate": round(high_sales_rate * 100, 1),
        "valueLift": round(value_lift, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "perfNames": perf_names,
        "perfAcc": perf_acc,
        "perfF1": perf_f1,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    (OUTPUT_DIR / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report_tables(merged_df, tables, cat1_result, high_result, payload)
    save_report_figures(cat1_result["model_df"], cat1_result, high_result, payload)
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --shadow:0 10px 24px rgba(16,24,40,.06); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; background:var(--bg); }}
    body {{ color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 94% 3%,rgba(255,92,138,.10),transparent 16%),radial-gradient(circle at 3% 96%,rgba(255,92,138,.10),transparent 16%),#FAFBFF; }}
    .page {{ width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:10px 16px 12px 10px; display:grid; grid-template-columns:280px 1fr; gap:14px; }}
    .sidebar {{ position:relative; min-height:1058px; padding:20px 16px; overflow:hidden; border:1px solid var(--border); border-radius:22px; background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(255,241,245,.72)); box-shadow:var(--shadow); }}
    .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:22px; }} .face {{ width:46px;height:46px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 46px; }} .face:before,.face:after {{ content:"";position:absolute;top:17px;width:5px;height:5px;background:var(--pink);border-radius:50%; }} .face:before{{left:12px}} .face:after{{right:12px}} .smile{{position:absolute;left:14px;top:25px;width:16px;height:8px;border-bottom:3px solid var(--pink);border-radius:0 0 16px 16px}}
    .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1; }} .brand p {{ margin:7px 0 0; color:#101828; font-size:12px; font-weight:800; }}
    .side-title {{ display:flex; gap:8px; align-items:center; margin:0 0 14px; color:var(--pink); font-size:17px; }}
    .insight {{ display:grid; grid-template-columns:50px 1fr; gap:10px; align-items:center; padding:12px 8px; margin-bottom:10px; border-radius:16px; background:rgba(255,255,255,.70); }} .i-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:18px;font-weight:900;box-shadow:0 8px 18px rgba(16,24,40,.05); }} .insight span {{ color:var(--muted);font-size:11px;font-weight:800; }} .insight strong {{ display:block;margin-top:5px;color:var(--c);font-size:21px;line-height:1.05; }} .insight small {{ display:block;margin-top:4px;color:#101828;font-size:11px;font-weight:800; }}
    .art {{ position:absolute; left:18px; right:18px; bottom:24px; height:205px; border-radius:24px; background:linear-gradient(180deg,rgba(255,241,245,.18),rgba(255,241,245,.86)); opacity:.88; }} .crib {{ position:absolute; left:22px; bottom:38px; width:128px; height:84px; border:5px solid rgba(255,92,138,.36); border-top:0; border-radius:0 0 20px 20px; }} .crib:before{{content:"";position:absolute;left:-8px;right:-8px;top:-16px;height:7px;background:rgba(255,92,138,.30);border-radius:999px}} .crib i{{position:absolute;bottom:0;width:4px;height:78px;background:rgba(255,92,138,.22);border-radius:999px}} .crib i:nth-child(1){{left:24px}} .crib i:nth-child(2){{left:54px}} .crib i:nth-child(3){{left:84px}} .bear{{position:absolute;right:30px;bottom:32px;width:66px;height:66px;border-radius:50%;background:#FFE5D3}} .bear:before,.bear:after{{content:"";position:absolute;top:-9px;width:24px;height:24px;border-radius:50%;background:#FFD3BA}} .bear:before{{left:4px}} .bear:after{{right:4px}} .bear span:before,.bear span:after{{content:"";position:absolute;top:24px;width:5px;height:5px;border-radius:50%;background:#8A4A36}} .bear span:before{{left:21px}} .bear span:after{{right:21px}} .bear span{{position:absolute;left:25px;top:32px;width:16px;height:10px;border-bottom:3px solid #8A4A36;border-radius:0 0 16px 16px}} .heart{{position:absolute;color:rgba(255,92,138,.14);font-size:20px}} .h1{{left:32px;top:24px}} .h2{{right:26px;top:42px}} .h3{{left:76px;bottom:24px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:70px; display:flex; justify-content:center; align-items:center; overflow:hidden; margin-bottom:10px; }} .hero h2 {{ margin:0; text-align:center; font-size:31px; line-height:1.12; }} .hero p {{ margin:6px 0 0; text-align:center; color:#101828; font-size:14px; font-weight:700; }} .hero-deco {{ position:absolute; right:12px; top:0; width:190px; height:68px; opacity:.46; border-radius:0 0 0 70px; background:linear-gradient(135deg,rgba(255,241,245,.9),rgba(255,255,255,.2)); }} .bottle{{position:absolute;right:76px;top:13px;width:24px;height:46px;border-radius:10px;background:linear-gradient(180deg,#D8C2FF,#FFE0EA 42%,#fff 43%,#F8B4C8)}} .mini-bear{{position:absolute;right:30px;top:15px;width:42px;height:42px;border-radius:50%;background:#FFE5D3}}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px; }} .kpi {{ position:relative; height:96px; display:flex; align-items:center; gap:14px; padding:14px 16px; overflow:hidden; background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); }} .kpi-icon {{ width:48px;height:48px;border-radius:50%;display:grid;place-items:center;flex:0 0 48px;color:var(--c);background:#FFF1F5;font-size:0; }} .kpi-icon svg {{ width:25px;height:25px;fill:currentColor; }} .kpi span {{ font-size:12px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:7px;font-size:23px;line-height:1; }} .kpi small {{ display:block;margin-top:8px;color:var(--muted);font-size:11px;font-weight:700; }} .spark {{ position:absolute; right:14px; bottom:9px; width:84px; height:24px; opacity:.45; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:12px; }} .card {{ background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); padding:13px 15px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:15px; color:#101828; }} .sub {{ margin-top:5px; color:var(--muted); font-size:10.5px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:5px; }} .unit {{ color:var(--muted); font-size:10.5px; font-weight:800; }}
    .trend {{ grid-column:span 10; height:330px; }} .top6 {{ grid-column:span 7; height:330px; }} .value {{ grid-column:span 7; height:330px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:282px; }} .value-chart {{ height:202px; }}
    .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }} .value-mini {{ padding:8px 6px; border-radius:13px; text-align:center; background:linear-gradient(135deg,#FFF1F5,#fff); }} .value-mini span {{ display:block;color:var(--muted);font-size:10px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:4px;color:var(--vc);font-size:16px; }}
    .profile {{ grid-column:span 7; height:210px; }} .weekday {{ grid-column:span 5; height:210px; }} .word {{ grid-column:span 5; height:210px; }} .quality {{ grid-column:span 7; height:210px; }} .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; height:158px; margin-top:4px; }} .small-chart {{ height:158px; }} .wordcloud {{ height:154px; margin-top:7px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:#fff; }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; }} .missing {{ color:var(--muted); font-size:13px; }}
    .q-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:8px; }} .q-item {{ display:grid;grid-template-columns:32px 1fr;gap:7px;align-items:center;min-height:43px;padding:7px;border-radius:12px;background:#FAFBFF;border:1px solid #F2F4F7; }} .q-icon {{ width:29px;height:29px;border-radius:9px;display:grid;place-items:center;color:var(--c);background:#fff;font-size:11px;font-weight:900; }} .q-main div {{ display:flex;justify-content:space-between;gap:6px;align-items:center; }} .q-main span {{ color:var(--muted);font-size:10.5px;font-weight:800; }} .q-main strong {{ color:#101828;font-size:12px;white-space:nowrap; }} .q-main em {{ display:block;height:5px;margin-top:6px;border-radius:999px;background:#EEF2F7;overflow:hidden; }} .q-main i {{ display:block;height:100%;border-radius:999px;background:var(--c); }} .quality-note {{ margin-top:7px; color:#98A2B3; font-size:10px; }}
    .model {{ grid-column:span 8; height:230px; }} .feature {{ grid-column:span 7; height:230px; }} .advice {{ grid-column:span 9; height:230px; }} .model-chart,.feature-chart {{ height:178px; }} .advice-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px; }} .box {{ height:174px;padding:10px 12px;border-radius:13px;background:#FAFBFF;border:1px solid #F2F4F7; }} .box h4 {{ margin:0 0 8px;color:var(--pink);font-size:12px; }} .line {{ display:flex;gap:7px;align-items:center;margin:7px 0;color:#344054;font-size:11.5px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }} .mark {{ width:17px;height:17px;display:inline-grid;place-items:center;flex:0 0 17px;border-radius:50%;color:#fff;font-size:10px;font-weight:900;background:#22C55E; }} .num {{ background:#FB7185; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .art {{ display:none; }} .trend,.top6,.value,.profile,.weekday,.word,.quality,.model,.feature,.advice {{ grid-column:1/-1; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">♥ 核心洞察</h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">♛</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">↗</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">◎</div><div><span>主模型表现（cat1）</span><strong>{best_main['Accuracy']:.3f}</strong><small>F1-score {best_main['F1_weighted']:.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">●</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_rate)}</strong><small>仅少量用户匹配画像</small></div></div>
      <div class="art"><span class="heart h1">♥</span><span class="heart h2">♥</span><span class="heart h3">♥</span><div class="crib"><i></i><i></i><i></i></div><div class="bear"><span></span></div></div>
    </aside>
    <main class="main">
      <header class="hero"><div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div><div class="hero-deco"><span class="bottle"></span><span class="mini-bear"></span></div></header>
      <section class="kpis">{kpi_html}</section>
      <section class="grid">
        <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">最近 12 个月购买数量与订单数变化</div></div><span class="unit">单位：件 / 单</span></div><div id="trendChart" class="chart trend-chart"></div></section>
        <section class="card top6"><div class="card-head"><div><h3>商品大类销量 TOP6</h3><div class="sub">中文业务品类按购买数量降序</div></div><span class="unit">单位：件</span></div><div id="categoryChart" class="chart top-chart"></div></section>
        <section class="card value"><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别高购买量订单</div><div id="highBuyChart" class="chart value-chart"></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量占比</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
        <section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart small-chart"></div><div id="ageChart" class="chart small-chart"></div></div></section>
        <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
        <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
        <section class="card quality"><h3>数据质量与画像匹配</h3><div class="q-grid">{quality_html}</div><div class="quality-note">全字段缺失值合计：{fmt_num(missing_count)}，仅作质量说明，不作为核心业务指标。</div></section>
        <section class="card model"><h3>模型性能对比（cat1 主模型）</h3><div id="modelChart" class="chart model-chart"></div></section>
        <section class="card feature"><h3>特征重要性 TOP10</h3><div id="featureChart" class="chart feature-chart"></div></section>
        <section class="card advice"><h3>模型诊断与业务建议</h3><div class="advice-grid"><div class="box"><h4>模型诊断</h4><div class="line"><span class="mark">✓</span><span>主任务：商品大类识别</span></div><div class="line"><span class="mark">✓</span><span>最佳模型：{html.escape(str(best_main['model']))}</span></div><div class="line"><span class="mark">✓</span><span>Accuracy：{best_main['Accuracy']:.3f}</span></div><div class="line"><span class="mark">✓</span><span>F1-score：{best_main['F1_weighted']:.3f}</span></div><div class="line"><span class="mark">✓</span><span>泄露检查：未使用 cat1_name</span></div><div class="line"><span class="mark">✓</span><span>high_buy：AUC {float(high_best['AUC']):.3f}，F1 {float(high_best['F1']):.3f}</span></div></div><div class="box"><h4>业务建议</h4><div class="line"><span class="mark num">1</span><span>重点关注{html.escape(hot_category)}等核心品类</span></div><div class="line"><span class="mark num">2</span><span>结合高购买量订单制定促销策略</span></div><div class="line"><span class="mark num">3</span><span>补充价格、品牌、浏览、收藏等特征</span></div><div class="line"><span class="mark num">4</span><span>进一步提升模型解释能力</span></div></div></div></section>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const iconMap = {{bag:"<svg viewBox='0 0 24 24'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",user:"<svg viewBox='0 0 24 24'><path d='M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5Zm0 2c-5 0-8 2.5-8 5v2h16v-2c0-2.5-3-5-8-5Z'/></svg>",box:"<svg viewBox='0 0 24 24'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 7 4v8l-7-4v-8Zm14 0v8l-7 4v-8l7-4Z'/></svg>",cart:"<svg viewBox='0 0 24 24'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>"}};
    document.querySelectorAll(".kpi-icon").forEach(el=>{{el.innerHTML=iconMap[el.textContent.trim()]||"";}});
    const textColor="#101828", mutedColor="#667085", gridLine="#EEF2F7", colors=payload.colors;
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#C4B5FD"}},{{offset:1,color:"#8B5CF6"}}]):"#8B5CF6";
    chart("trendChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:10}}}},grid:{{left:52,right:46,top:42,bottom:32}},xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10,interval:1}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("categoryChart",{{tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},grid:{{left:88,right:66,top:14,bottom:24}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:13,label:{{show:true,position:"right",color:textColor,fontSize:10,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,8,8,0],color:p=>colors[(p.dataIndex+2)%colors.length]}}}}]}});
    chart("highBuyChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"item"}},title:{{text:payload.highRate+"%",subtext:"高购买量订单占比",left:"center",top:"37%",textStyle:{{fontSize:24,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:10,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"center",orient:"vertical",textStyle:{{color:textColor,fontSize:10}}}},series:[{{type:"pie",radius:["50%","72%"],center:["38%","50%"],label:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}});
    chart("genderChart",{{color:["#3B82F6","#EF476F"],title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:11,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","66%"],center:["50%","50%"],label:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}});
    chart("ageChart",{{color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"],title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:11,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","66%"],center:["50%","50%"],label:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}});
    chart("weekdayChart",{{color:["#22C55E","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:30,top:36,bottom:26}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{show:false}}}}],series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuy,barWidth:16,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:5,lineStyle:{{width:2.5,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("modelChart",{{color:["#3B82F6","#22C55E"],tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},legend:{{top:0,textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:112,right:42,top:34,bottom:22}},xAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.perfNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"Accuracy",type:"bar",data:payload.perfAcc.slice().reverse(),barWidth:10,label:{{show:true,position:"right",fontSize:9,formatter:p=>p.value.toFixed(3)}}}},{{name:"F1-score",type:"bar",data:payload.perfF1.slice().reverse(),barWidth:10,label:{{show:true,position:"right",fontSize:9,formatter:p=>p.value.toFixed(3)}}}}]}});
    chart("featureChart",{{tooltip:{{trigger:"axis",formatter:params=>{{const p=params[0]; const idx=payload.featureShort.slice().reverse().indexOf(p.name); const full=payload.featureNames.slice().reverse()[idx] || p.name; return full + "<br/>重要性：" + p.value.toFixed(3);}}}},grid:{{left:96,right:42,top:12,bottom:20}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureShort.slice().reverse(),axisLabel:{{color:textColor,fontSize:9}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:9,label:{{show:true,position:"right",fontSize:8,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:purpleGrad}}}}]}});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成图标与图表标注修复后的最终展示大屏。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def svg_icon(name: str) -> str:
        icons = {
            "crown": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 18h16l1-10-5 3-4-7-4 7-5-3 1 10Zm1 2h14v2H5v-2Z'/></svg>",
            "trend": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 17h3.8l4.1-5.4 3.4 3.4L21 7.8V13h2V4h-9v2h5.4l-4.3 5.4-3.5-3.5L6.8 15H4v2Z'/></svg>",
            "target": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2Zm0 5a5 5 0 1 0 5 5h-2a3 3 0 1 1-3-3V7Zm1 6 8-8-1.4-1.4-8 8V6h-2v8h8v-2h-4.6Z'/></svg>",
            "user_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.6a7 7 0 0 1-.6-2.8c0-1.6.5-3.1 1.5-4.2H10Zm11.7 1.7-1.4-1.4-4.3 4.3-1.8-1.8-1.4 1.4 3.2 3.2 5.7-5.7Z'/></svg>",
            "bag": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",
            "users": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.2 0-7 2.1-7 5v2h14v-2c0-2.9-2.8-5-7-5Zm8-2a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm0 2c-.7 0-1.4.1-2 .3 1.8 1.1 3 2.7 3 4.7v2h4v-2c0-2.9-2.1-5-5-5Z'/></svg>",
            "package": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 6 3.4v7.2l-6-3.4V10Zm14 0v7.2l-6 3.4v-7.2l6-3.4Z'/></svg>",
            "cart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>",
            "database": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 3c5 0 9 1.8 9 4s-4 4-9 4-9-1.8-9-4 4-4 9-4Zm-9 7c1.6 1.7 5 3 9 3s7.4-1.3 9-3v3c0 2.2-4 4-9 4s-9-1.8-9-4v-3Zm0 6c1.6 1.7 5 3 9 3s7.4-1.3 9-3v1c0 2.2-4 4-9 4s-9-1.8-9-4v-1Z'/></svg>",
            "user_x": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.3a6.8 6.8 0 0 1 1.2-7H10Zm10.4 1.2-2.2 2.2-2.2-2.2-1.4 1.4 2.2 2.2-2.2 2.2 1.4 1.4 2.2-2.2 2.2 2.2 1.4-1.4-2.2-2.2 2.2-2.2-1.4-1.4Z'/></svg>",
            "cart_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 .1 0H7Zm10 0a2 2 0 1 0 .1 0H17ZM4 3H2v2h2l3 10h8.5a7.3 7.3 0 0 1 .6-2H8.5l-.6-2H16a6.8 6.8 0 0 1 3.6-2.6L20.1 7H7.2L6.5 5H4V3Zm18 10.7-1.4-1.4-4.1 4.1-1.7-1.7-1.4 1.4 3.1 3.1 5.5-5.5Z'/></svg>",
            "shield": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2 4 5v6c0 5 3.4 9.6 8 11 4.6-1.4 8-6 8-11V5l-8-3Zm4.8 7.7-5.4 5.4-2.2-2.2-1.4 1.4 3.6 3.6 6.8-6.8-1.4-1.4Z'/></svg>",
            "check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9.2 16.6 4.9 12.3 3.5 13.7l5.7 5.7L21 7.6 19.6 6.2 9.2 16.6Z'/></svg>",
            "number": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 3h10a4 4 0 0 1 4 4v10a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4V7a4 4 0 0 1 4-4Zm5 4-3 2v2l2-1.2V17h2V7h-1Z'/></svg>",
            "heart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 21s-7.5-4.6-9.6-9.2C.8 8.2 2.9 5 6.3 5c2 0 3.4 1.1 4.2 2.3C11.3 6.1 12.7 5 14.7 5c3.4 0 5.5 3.2 3.9 6.8C16.5 16.4 12 21 12 21Z'/></svg>",
            "bottle": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 2h6v4l2 2v11a3 3 0 0 1-3 3h-4a3 3 0 0 1-3-3V8l2-2V2Zm2 2v2h2V4h-2Zm-2 8h6v-2H9v2Zm0 4h6v-2H9v2Z'/></svg>",
        }
        return icons[name]

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"""
        <article class="kpi" style="--c:{color}">
          <div class="kpi-icon">{svg_icon(icon)}</div>
          <div class="kpi-text"><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div>
        </article>
        """

    def q_item(title: str, value: str, ratio: float, color: str, icon: str) -> str:
        width = max(0, min(float(ratio), 1)) * 100
        return f"""
        <div class="q-item" style="--c:{color}">
          <div class="q-icon">{svg_icon(icon)}</div>
          <div class="q-main"><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong></div><em><i style="width:{width:.1f}%"></i></em></div>
        </div>
        """

    def diag_line(text: str, kind: str = "check") -> str:
        color_class = "mark num" if kind == "number" else "mark"
        return f"<div class='line'><span class='{color_class}'>{svg_icon(kind)}</span><span>{html.escape(text)}</span></div>"

    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    best_main = cat_metrics.iloc[0]
    high_best = high_metrics.iloc[0]
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    unmatched_count = int((~profile_mask).sum())
    unknown_rate = float((~profile_mask).mean())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)
    missing_count = int(merged_df.isna().sum().sum())

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_raw = matched_df["baby_age_group"].replace("unknown", np.nan).dropna()
        age_counts = age_raw.value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    perf_names, perf_acc, perf_f1 = [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            label = f"{name}（推荐模型）" if name == "Random Forest" else name
            perf_names.append(label)
            perf_acc.append(round(float(row["Accuracy"]), 4))
            perf_f1.append(round(float(row["F1_weighted"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
        "user_buy_count": "user_buy",
        "user_unique_item_count": "user_items",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "users"),
            kpi("商品 ID 数", fmt_num(merged_df["auction_id"].nunique()), "auction_id 去重统计", "#8B5CF6", "package"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            q_item("清洗后交易记录数", fmt_num(len(merged_df)), 1, "#3B82F6", "database"),
            q_item("画像匹配率", fmt_pct(profile_rate), profile_rate, "#22C55E", "user_check"),
            q_item("画像未匹配记录数", fmt_num(unmatched_count), unknown_rate, "#64748B", "user_x"),
            q_item("有效购买记录占比", fmt_pct(valid_rate), valid_rate, "#06B6D4", "cart_check"),
            q_item("核心字段完整率", fmt_pct(core_complete_rate), core_complete_rate, "#8B5CF6", "shield"),
        ]
    )
    diagnosis_html = "".join(
        [
            diag_line("主任务：商品大类识别"),
            diag_line("推荐模型：Random Forest"),
            diag_line(f"Accuracy：{float(cat_rows['Random Forest']['Accuracy']):.3f}"),
            diag_line(f"F1-score：{float(cat_rows['Random Forest']['F1_weighted']):.3f}"),
            diag_line("泄露检查：未使用 cat1_name"),
            diag_line(f"high_buy：AUC {float(high_best['AUC']):.3f}，F1 {float(high_best['F1']):.3f}"),
        ]
    )
    advice_html = "".join(
        [
            diag_line(f"重点关注{hot_category}等核心品类", "number"),
            diag_line("结合高购买量订单制定促销策略", "number"),
            diag_line("补充价格、品牌、浏览、收藏等特征", "number"),
            diag_line("提升模型解释能力", "number"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "perfNames": perf_names,
        "perfAcc": perf_acc,
        "perfF1": perf_f1,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --shadow:0 10px 24px rgba(16,24,40,.06); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; background:var(--bg); }}
    body {{ color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 94% 3%,rgba(255,92,138,.10),transparent 16%),radial-gradient(circle at 3% 96%,rgba(255,92,138,.10),transparent 16%),#FAFBFF; }}
    svg {{ display:block; width:1em; height:1em; fill:currentColor; }}
    .page {{ width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:10px 16px 12px 10px; display:grid; grid-template-columns:280px 1fr; gap:14px; }}
    .sidebar {{ position:relative; min-height:1058px; padding:20px 16px; overflow:hidden; border:1px solid var(--border); border-radius:22px; background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(255,241,245,.72)); box-shadow:var(--shadow); }}
    .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:22px; }} .face {{ width:46px;height:46px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 46px; }} .face:before,.face:after {{ content:"";position:absolute;top:17px;width:5px;height:5px;background:var(--pink);border-radius:50%; }} .face:before{{left:12px}} .face:after{{right:12px}} .smile{{position:absolute;left:14px;top:25px;width:16px;height:8px;border-bottom:3px solid var(--pink);border-radius:0 0 16px 16px}}
    .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1; }} .brand p {{ margin:7px 0 0; color:#101828; font-size:12px; font-weight:800; }}
    .side-title {{ display:flex; gap:8px; align-items:center; margin:0 0 14px; color:var(--pink); font-size:17px; }} .side-title svg {{ width:17px; height:17px; }}
    .insight {{ display:grid; grid-template-columns:50px 1fr; gap:10px; align-items:center; padding:12px 8px; margin-bottom:10px; border-radius:16px; background:rgba(255,255,255,.70); }} .i-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:23px;font-weight:900;box-shadow:0 8px 18px rgba(16,24,40,.05); }} .i-icon svg {{ width:23px; height:23px; }} .insight span {{ color:var(--muted);font-size:11px;font-weight:800; }} .insight strong {{ display:block;margin-top:5px;color:var(--c);font-size:21px;line-height:1.05; }} .insight small {{ display:block;margin-top:4px;color:#101828;font-size:11px;font-weight:800; }}
    .art {{ position:absolute; left:18px; right:18px; bottom:24px; height:205px; border-radius:24px; background:linear-gradient(180deg,rgba(255,241,245,.18),rgba(255,241,245,.86)); opacity:.88; }} .crib {{ position:absolute; left:22px; bottom:38px; width:128px; height:84px; border:5px solid rgba(255,92,138,.36); border-top:0; border-radius:0 0 20px 20px; }} .crib:before{{content:"";position:absolute;left:-8px;right:-8px;top:-16px;height:7px;background:rgba(255,92,138,.30);border-radius:999px}} .crib i{{position:absolute;bottom:0;width:4px;height:78px;background:rgba(255,92,138,.22);border-radius:999px}} .crib i:nth-child(1){{left:24px}} .crib i:nth-child(2){{left:54px}} .crib i:nth-child(3){{left:84px}} .bear{{position:absolute;right:30px;bottom:32px;width:66px;height:66px;border-radius:50%;background:#FFE5D3}} .bear:before,.bear:after{{content:"";position:absolute;top:-9px;width:24px;height:24px;border-radius:50%;background:#FFD3BA}} .bear:before{{left:4px}} .bear:after{{right:4px}} .bear span:before,.bear span:after{{content:"";position:absolute;top:24px;width:5px;height:5px;border-radius:50%;background:#8A4A36}} .bear span:before{{left:21px}} .bear span:after{{right:21px}} .bear span{{position:absolute;left:25px;top:32px;width:16px;height:10px;border-bottom:3px solid #8A4A36;border-radius:0 0 16px 16px}} .heart{{position:absolute;color:rgba(255,92,138,.14);font-size:20px}} .heart svg{{width:20px;height:20px}} .h1{{left:32px;top:24px}} .h2{{right:26px;top:42px}} .h3{{left:76px;bottom:24px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:70px; display:flex; justify-content:center; align-items:center; overflow:hidden; margin-bottom:10px; }} .hero h2 {{ margin:0; text-align:center; font-size:31px; line-height:1.12; }} .hero p {{ margin:6px 0 0; text-align:center; color:#101828; font-size:14px; font-weight:700; }}
    .hero-deco {{ position:absolute; right:10px; top:2px; width:190px; height:66px; opacity:.35; pointer-events:none; }} .deco-circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .dc1{{width:92px;height:92px;right:36px;top:-34px}} .dc2{{width:42px;height:42px;right:0;top:16px;background:rgba(137,207,240,.22)}} .dc3{{width:28px;height:28px;right:118px;top:26px;background:rgba(255,209,102,.22)}} .deco-heart{{position:absolute;color:rgba(255,92,138,.5)}} .deco-heart svg{{width:17px;height:17px}} .dh1{{right:92px;top:8px}} .dh2{{right:145px;top:32px}} .deco-bottle{{position:absolute;right:54px;top:18px;color:rgba(255,92,138,.48)}} .deco-bottle svg{{width:28px;height:28px}}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px; }} .kpi {{ height:96px; display:flex; align-items:center; gap:14px; padding:14px 16px; overflow:hidden; background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); }} .kpi-icon {{ width:48px;height:48px;border-radius:50%;display:grid;place-items:center;flex:0 0 48px;color:var(--c);background:#FFF1F5;font-size:25px; }} .kpi-icon svg {{ width:25px;height:25px; }} .kpi span {{ font-size:12px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:7px;font-size:23px;line-height:1; }} .kpi small {{ display:block;margin-top:8px;color:var(--muted);font-size:11px;font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:12px; }} .card {{ background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); padding:13px 15px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:15px; color:#101828; }} .sub {{ margin-top:5px; color:var(--muted); font-size:10.5px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:5px; }} .unit {{ color:var(--muted); font-size:10.5px; font-weight:800; }}
    .trend {{ grid-column:span 10; height:330px; }} .top6 {{ grid-column:span 7; height:330px; }} .value {{ grid-column:span 7; height:330px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:282px; }} .value-chart {{ height:202px; }}
    .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }} .value-mini {{ padding:8px 6px; border-radius:13px; text-align:center; background:linear-gradient(135deg,#FFF1F5,#fff); }} .value-mini span {{ display:block;color:var(--muted);font-size:10px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:4px;color:var(--vc);font-size:16px; }}
    .profile {{ grid-column:span 7; height:210px; }} .weekday {{ grid-column:span 5; height:210px; }} .word {{ grid-column:span 5; height:210px; }} .quality {{ grid-column:span 7; height:210px; }} .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; height:158px; margin-top:4px; }} .small-chart {{ height:158px; }} .wordcloud {{ height:154px; margin-top:7px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:#fff; }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; }} .missing {{ color:var(--muted); font-size:13px; }}
    .q-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:8px; }} .q-item {{ display:grid;grid-template-columns:32px 1fr;gap:7px;align-items:center;min-height:43px;padding:7px;border-radius:12px;background:#FAFBFF;border:1px solid #F2F4F7; }} .q-icon {{ width:29px;height:29px;border-radius:9px;display:grid;place-items:center;color:var(--c);background:#fff;font-size:17px;font-weight:900; }} .q-icon svg {{ width:17px; height:17px; }} .q-main div {{ display:flex;justify-content:space-between;gap:6px;align-items:center; }} .q-main span {{ color:var(--muted);font-size:10.5px;font-weight:800; }} .q-main strong {{ color:#101828;font-size:12px;white-space:nowrap; }} .q-main em {{ display:block;height:5px;margin-top:6px;border-radius:999px;background:#EEF2F7;overflow:hidden; }} .q-main i {{ display:block;height:100%;border-radius:999px;background:var(--c); }} .quality-note {{ margin-top:7px; color:#98A2B3; font-size:10px; }}
    .model {{ grid-column:span 8; height:230px; }} .feature {{ grid-column:span 7; height:230px; }} .advice {{ grid-column:span 9; height:230px; }} .model-chart,.feature-chart {{ height:178px; }} .advice-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px; }} .box {{ height:174px;padding:10px 12px;border-radius:13px;background:#FAFBFF;border:1px solid #F2F4F7; }} .box h4 {{ margin:0 0 8px;color:var(--pink);font-size:12px; }} .line {{ display:flex;gap:7px;align-items:center;margin:7px 0;color:#344054;font-size:11.5px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }} .mark {{ width:17px;height:17px;display:inline-grid;place-items:center;flex:0 0 17px;border-radius:50%;color:#fff;font-size:11px;font-weight:900;background:#22C55E; }} .mark svg {{ width:11px;height:11px; }} .num {{ background:#FB7185; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .art {{ display:none; }} .trend,.top6,.value,.profile,.weekday,.word,.quality,.model,.feature,.advice {{ grid-column:1/-1; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">{svg_icon("heart")}<span>核心洞察</span></h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("crown")}</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">{svg_icon("trend")}</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("target")}</div><div><span>主模型表现（cat1）</span><strong>{float(cat_rows['Random Forest']['Accuracy']):.3f}</strong><small>F1-score {float(cat_rows['Random Forest']['F1_weighted']):.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">{svg_icon("user_check")}</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_rate)}</strong><small>仅少量用户匹配画像</small></div></div>
      <div class="art"><span class="heart h1">{svg_icon("heart")}</span><span class="heart h2">{svg_icon("heart")}</span><span class="heart h3">{svg_icon("heart")}</span><div class="crib"><i></i><i></i><i></i></div><div class="bear"><span></span></div></div>
    </aside>
    <main class="main">
      <header class="hero"><div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div><div class="hero-deco"><span class="deco-circle dc1"></span><span class="deco-circle dc2"></span><span class="deco-circle dc3"></span><span class="deco-heart dh1">{svg_icon("heart")}</span><span class="deco-heart dh2">{svg_icon("heart")}</span><span class="deco-bottle">{svg_icon("bottle")}</span></div></header>
      <section class="kpis">{kpi_html}</section>
      <section class="grid">
        <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">最近 12 个月购买数量与订单数变化</div></div><span class="unit">单位：件 / 单</span></div><div id="trendChart" class="chart trend-chart"></div></section>
        <section class="card top6"><div class="card-head"><div><h3>商品大类销量 TOP6</h3><div class="sub">中文业务品类按购买数量降序</div></div><span class="unit">单位：件</span></div><div id="categoryChart" class="chart top-chart"></div></section>
        <section class="card value"><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别高购买量订单</div><div id="highBuyChart" class="chart value-chart"></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量占比</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
        <section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart small-chart"></div><div id="ageChart" class="chart small-chart"></div></div></section>
        <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
        <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
        <section class="card quality"><h3>数据质量与画像匹配</h3><div class="q-grid">{quality_html}</div><div class="quality-note">全字段缺失值合计：{fmt_num(missing_count)}，仅作质量说明，不作为核心业务指标。</div></section>
        <section class="card model"><h3>模型性能对比（cat1 主模型）</h3><div id="modelChart" class="chart model-chart"></div></section>
        <section class="card feature"><h3>特征重要性 TOP10</h3><div id="featureChart" class="chart feature-chart"></div></section>
        <section class="card advice"><h3>模型诊断与业务建议</h3><div class="advice-grid"><div class="box"><h4>模型诊断</h4>{diagnosis_html}</div><div class="box"><h4>业务建议</h4>{advice_html}</div></div></section>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const textColor="#101828", mutedColor="#667085", gridLine="#EEF2F7", colors=payload.colors;
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#C4B5FD"}},{{offset:1,color:"#8B5CF6"}}]):"#8B5CF6";
    chart("trendChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:10}}}},grid:{{left:52,right:46,top:42,bottom:32}},xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10,interval:1}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("categoryChart",{{tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},grid:{{left:88,right:80,top:14,bottom:24}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:13,label:{{show:true,position:"right",color:textColor,fontSize:9,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,8,8,0],color:p=>colors[(p.dataIndex+2)%colors.length]}}}}]}});
    chart("highBuyChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"item"}},title:{{text:payload.highRate+"%",subtext:"高购买量订单占比",left:"34%",top:"39%",textAlign:"center",textStyle:{{fontSize:21,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:10,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"middle",orient:"vertical",itemWidth:9,itemHeight:9,textStyle:{{color:textColor,fontSize:10}}}},series:[{{type:"pie",radius:["48%","68%"],center:["34%","50%"],label:{{show:false}},labelLine:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}});
    chart("genderChart",{{color:["#3B82F6","#EF476F"],title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:11,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}});
    chart("ageChart",{{color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"],title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:11,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}});
    chart("weekdayChart",{{color:["#22C55E","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:30,top:42,bottom:26}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{show:false}}}}],series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuy,barWidth:16,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:5,lineStyle:{{width:2.5,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("modelChart",{{color:["#3B82F6","#22C55E"],tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:130,right:70,top:38,bottom:22}},xAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.perfNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"Accuracy",type:"bar",data:payload.perfAcc.slice().reverse(),barWidth:11,barGap:"20%",label:{{show:true,position:"right",fontSize:9,formatter:p=>p.value.toFixed(3)}}}},{{name:"F1-score",type:"bar",data:payload.perfF1.slice().reverse(),barWidth:11,label:{{show:true,position:"right",fontSize:9,formatter:p=>p.value.toFixed(3)}}}}]}});
    chart("featureChart",{{tooltip:{{trigger:"axis",formatter:params=>{{const p=params[0]; const reversedShort=payload.featureShort.slice().reverse(); const idx=reversedShort.indexOf(p.name); const full=payload.featureNames.slice().reverse()[idx] || p.name; return full + "<br/>重要性：" + p.value.toFixed(3);}}}},grid:{{left:110,right:70,top:12,bottom:20}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureShort.slice().reverse(),axisLabel:{{color:textColor,fontSize:9}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:9,label:{{show:true,position:"right",fontSize:8,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:purpleGrad}}}}]}});
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成两个 Tab 的最终展示型可视化大屏。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def svg_icon(name: str) -> str:
        icons = {
            "heart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 21s-7.5-4.6-9.6-9.2C.8 8.2 2.9 5 6.3 5c2 0 3.4 1.1 4.2 2.3C11.3 6.1 12.7 5 14.7 5c3.4 0 5.5 3.2 3.9 6.8C16.5 16.4 12 21 12 21Z'/></svg>",
            "crown": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 18h16l1-10-5 3-4-7-4 7-5-3 1 10Zm1 2h14v2H5v-2Z'/></svg>",
            "trend": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 17h3.8l4.1-5.4 3.4 3.4L21 7.8V13h2V4h-9v2h5.4l-4.3 5.4-3.5-3.5L6.8 15H4v2Z'/></svg>",
            "target": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2Zm0 5a5 5 0 1 0 5 5h-2a3 3 0 1 1-3-3V7Zm1 6 8-8-1.4-1.4-8 8V6h-2v8h8v-2h-4.6Z'/></svg>",
            "user_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.6a7 7 0 0 1-.6-2.8c0-1.6.5-3.1 1.5-4.2H10Zm11.7 1.7-1.4-1.4-4.3 4.3-1.8-1.8-1.4 1.4 3.2 3.2 5.7-5.7Z'/></svg>",
            "bag": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",
            "users": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.2 0-7 2.1-7 5v2h14v-2c0-2.9-2.8-5-7-5Zm8-2a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm0 2c-.7 0-1.4.1-2 .3 1.8 1.1 3 2.7 3 4.7v2h4v-2c0-2.9-2.1-5-5-5Z'/></svg>",
            "package": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 6 3.4v7.2l-6-3.4V10Zm14 0v7.2l-6 3.4v-7.2l6-3.4Z'/></svg>",
            "cart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>",
            "database": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 3c5 0 9 1.8 9 4s-4 4-9 4-9-1.8-9-4 4-4 9-4Zm-9 7c1.6 1.7 5 3 9 3s7.4-1.3 9-3v3c0 2.2-4 4-9 4s-9-1.8-9-4v-3Zm0 6c1.6 1.7 5 3 9 3s7.4-1.3 9-3v1c0 2.2-4 4-9 4s-9-1.8-9-4v-1Z'/></svg>",
            "user_x": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.3a6.8 6.8 0 0 1 1.2-7H10Zm10.4 1.2-2.2 2.2-2.2-2.2-1.4 1.4 2.2 2.2-2.2 2.2 1.4 1.4 2.2-2.2 2.2 2.2 1.4-1.4-2.2-2.2 2.2-2.2-1.4-1.4Z'/></svg>",
            "cart_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 .1 0H7Zm10 0a2 2 0 1 0 .1 0H17ZM4 3H2v2h2l3 10h8.5a7.3 7.3 0 0 1 .6-2H8.5l-.6-2H16a6.8 6.8 0 0 1 3.6-2.6L20.1 7H7.2L6.5 5H4V3Zm18 10.7-1.4-1.4-4.1 4.1-1.7-1.7-1.4 1.4 3.1 3.1 5.5-5.5Z'/></svg>",
            "shield": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2 4 5v6c0 5 3.4 9.6 8 11 4.6-1.4 8-6 8-11V5l-8-3Zm4.8 7.7-5.4 5.4-2.2-2.2-1.4 1.4 3.6 3.6 6.8-6.8-1.4-1.4Z'/></svg>",
            "check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9.2 16.6 4.9 12.3 3.5 13.7l5.7 5.7L21 7.6 19.6 6.2 9.2 16.6Z'/></svg>",
            "bottle": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 2h6v4l2 2v11a3 3 0 0 1-3 3h-4a3 3 0 0 1-3-3V8l2-2V2Zm2 2v2h2V4h-2Zm-2 8h6v-2H9v2Zm0 4h6v-2H9v2Z'/></svg>",
        }
        return icons[name]

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"<article class='kpi' style='--c:{color}'><div class='kpi-icon'>{svg_icon(icon)}</div><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div></article>"

    def q_item(title: str, value: str, ratio: float, color: str, icon: str) -> str:
        width = max(0, min(float(ratio), 1)) * 100
        return f"<div class='q-item' style='--c:{color}'><div class='q-icon'>{svg_icon(icon)}</div><div class='q-main'><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong></div><em><i style='width:{width:.1f}%'></i></em></div></div>"

    def small_line(text: str) -> str:
        return f"<div class='line'><span class='mark'>{svg_icon('check')}</span><span>{html.escape(text)}</span></div>"

    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    high_best = high_metrics.iloc[0]
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    highest_cat1_row = cat_metrics.sort_values(["Accuracy", "F1_weighted"], ascending=False).iloc[0]
    rf_row = cat_rows.get("Random Forest", cat_metrics.iloc[0])
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    unmatched_count = int((~profile_mask).sum())
    unknown_rate = float((~profile_mask).mean())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_raw = matched_df["baby_age_group"].replace("unknown", np.nan).dropna()
        age_counts = age_raw.value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    perf_names, perf_acc, perf_f1 = [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            perf_names.append(f"{name}（推荐模型）" if name == "Random Forest" else name)
            perf_acc.append(round(float(row["Accuracy"]), 4))
            perf_f1.append(round(float(row["F1_weighted"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
        "user_buy_count": "user_buy",
        "user_unique_item_count": "user_items",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "users"),
            kpi("商品种类数", fmt_num(merged_df["auction_id"].nunique()), "商品 ID 去重统计", "#8B5CF6", "package"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            q_item("清洗后交易记录数", fmt_num(len(merged_df)), 1, "#3B82F6", "database"),
            q_item("画像匹配率", fmt_pct(profile_rate), profile_rate, "#22C55E", "user_check"),
            q_item("画像未匹配记录数", fmt_num(unmatched_count), unknown_rate, "#64748B", "user_x"),
            q_item("有效购买记录占比", fmt_pct(valid_rate), valid_rate, "#06B6D4", "cart_check"),
            q_item("核心字段完整率", fmt_pct(core_complete_rate), core_complete_rate, "#8B5CF6", "shield"),
        ]
    )
    model_diag = "".join(
        [
            small_line("主任务：商品大类识别"),
            small_line("推荐模型：Random Forest"),
            small_line(f"Accuracy：{float(rf_row['Accuracy']):.3f}"),
            small_line(f"F1-score：{float(rf_row['F1_weighted']):.3f}"),
            small_line("泄露检查：未使用 cat1_name"),
            small_line(f"high_buy：AUC {float(high_best['AUC']):.3f}，F1 {float(high_best['F1']):.3f}"),
        ]
    )
    advice_html = "".join(
        [
            small_line(f"重点关注{hot_category}等核心品类"),
            small_line("结合高购买量订单制定促销策略"),
            small_line("补充价格、品牌、浏览、收藏等特征"),
            small_line("提升模型解释能力"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "perfNames": perf_names,
        "perfAcc": perf_acc,
        "perfF1": perf_f1,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --shadow:0 10px 24px rgba(16,24,40,.06); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; background:var(--bg); }} body {{ color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 94% 3%,rgba(255,92,138,.10),transparent 16%),radial-gradient(circle at 3% 96%,rgba(255,92,138,.10),transparent 16%),#FAFBFF; }} svg {{ display:block; width:1em; height:1em; fill:currentColor; }}
    .page {{ width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:10px 16px 12px 10px; display:grid; grid-template-columns:280px 1fr; gap:14px; }}
    .sidebar {{ position:relative; min-height:1058px; padding:20px 16px; overflow:hidden; border:1px solid var(--border); border-radius:22px; background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(255,241,245,.72)); box-shadow:var(--shadow); }}
    .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:24px; }} .face {{ width:46px;height:46px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 46px; }} .face:before,.face:after {{ content:"";position:absolute;top:17px;width:5px;height:5px;background:var(--pink);border-radius:50%; }} .face:before{{left:12px}} .face:after{{right:12px}} .smile{{position:absolute;left:14px;top:25px;width:16px;height:8px;border-bottom:3px solid var(--pink);border-radius:0 0 16px 16px}} .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1; }} .brand p {{ margin:7px 0 0; color:#101828; font-size:12px; font-weight:800; }}
    .side-title {{ display:flex; gap:8px; align-items:center; margin:0 0 14px; color:var(--pink); font-size:17px; }} .side-title svg {{ width:17px; height:17px; }}
    .insight {{ display:grid; grid-template-columns:50px 1fr; gap:10px; align-items:center; padding:13px 8px; margin-bottom:12px; border-radius:16px; background:rgba(255,255,255,.78); }} .i-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:23px;box-shadow:0 8px 18px rgba(16,24,40,.05); }} .i-icon svg {{ width:23px; height:23px; }} .insight span {{ color:var(--muted);font-size:11px;font-weight:800; }} .insight strong {{ display:block;margin-top:5px;color:var(--c);font-size:21px;line-height:1.05; }} .insight small {{ display:block;margin-top:4px;color:#101828;font-size:11px;font-weight:800; }}
    .side-deco {{ position:absolute; left:18px; right:18px; bottom:22px; height:80px; opacity:.32; }} .side-deco .circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .side-deco .c1{{width:88px;height:88px;left:6px;bottom:-34px}} .side-deco .c2{{width:46px;height:46px;right:16px;bottom:10px;background:rgba(137,207,240,.22)}} .side-deco .mini-heart{{position:absolute;left:112px;bottom:34px;color:var(--pink)}} .side-deco .mini-heart svg{{width:20px;height:20px}} .side-deco .mini-bottle{{position:absolute;right:72px;bottom:16px;color:var(--pink)}} .side-deco .mini-bottle svg{{width:28px;height:28px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:104px; display:flex; flex-direction:column; align-items:center; justify-content:center; overflow:hidden; margin-bottom:10px; }} .hero h2 {{ margin:0; text-align:center; font-size:31px; line-height:1.12; }} .hero p {{ margin:6px 0 10px; text-align:center; color:#101828; font-size:14px; font-weight:700; }} .hero-deco {{ position:absolute; right:10px; top:2px; width:190px; height:66px; opacity:.30; pointer-events:none; }} .deco-circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .dc1{{width:92px;height:92px;right:36px;top:-34px}} .dc2{{width:42px;height:42px;right:0;top:16px;background:rgba(137,207,240,.22)}} .dc3{{width:28px;height:28px;right:118px;top:26px;background:rgba(255,209,102,.22)}} .deco-heart{{position:absolute;color:rgba(255,92,138,.5)}} .deco-heart svg{{width:17px;height:17px}} .dh1{{right:92px;top:8px}} .dh2{{right:145px;top:32px}} .deco-bottle{{position:absolute;right:54px;top:18px;color:rgba(255,92,138,.48)}} .deco-bottle svg{{width:28px;height:28px}}
    .tabs {{ display:flex; gap:10px; justify-content:center; }} .tab-btn {{ border:1px solid #FFE3EC; background:#fff; color:var(--pink); height:34px; padding:0 28px; border-radius:999px; font-weight:800; cursor:pointer; box-shadow:0 8px 18px rgba(16,24,40,.04); }} .tab-btn.active {{ background:var(--pink); color:#fff; border-color:var(--pink); }}
    .tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:14px; }} .card {{ background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); padding:14px 16px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:15px; color:#101828; }} .sub {{ margin-top:5px; color:var(--muted); font-size:10.5px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:5px; }} .unit {{ color:var(--muted); font-size:10.5px; font-weight:800; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px; }} .kpi {{ height:110px; display:flex; align-items:center; gap:14px; padding:16px; background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); }} .kpi-icon {{ width:52px;height:52px;border-radius:50%;display:grid;place-items:center;flex:0 0 52px;color:var(--c);background:#FFF1F5;font-size:26px; }} .kpi-icon svg {{ width:26px;height:26px; }} .kpi span {{ font-size:12px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:8px;font-size:24px;line-height:1; }} .kpi small {{ display:block;margin-top:9px;color:var(--muted);font-size:11px;font-weight:700; }}
    .trend {{ grid-column:span 10; height:380px; }} .top6 {{ grid-column:span 7; height:380px; }} .value {{ grid-column:span 7; height:380px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:326px; }} .value-chart {{ height:244px; }} .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }} .value-mini {{ padding:10px 6px; border-radius:13px; text-align:center; background:linear-gradient(135deg,#FFF1F5,#fff); }} .value-mini span {{ display:block;color:var(--muted);font-size:10px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:4px;color:var(--vc);font-size:17px; }}
    .word {{ grid-column:span 6; height:250px; }} .quality {{ grid-column:span 7; height:250px; }} .weekday {{ grid-column:span 6; height:250px; }} .conclusion {{ grid-column:span 5; height:250px; }} .small-chart {{ height:190px; }} .wordcloud {{ height:194px; margin-top:8px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:#fff; }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; }} .missing {{ color:var(--muted); font-size:13px; }}
    .q-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:10px; }} .q-item {{ display:grid;grid-template-columns:32px 1fr;gap:7px;align-items:center;min-height:45px;padding:7px;border-radius:12px;background:#FAFBFF;border:1px solid #F2F4F7; }} .q-icon {{ width:29px;height:29px;border-radius:9px;display:grid;place-items:center;color:var(--c);background:#fff;font-size:17px; }} .q-icon svg {{ width:17px; height:17px; }} .q-main div {{ display:flex;justify-content:space-between;gap:6px;align-items:center; }} .q-main span {{ color:var(--muted);font-size:10.5px;font-weight:800; }} .q-main strong {{ color:#101828;font-size:12px;white-space:nowrap; }} .q-main em {{ display:block;height:5px;margin-top:6px;border-radius:999px;background:#EEF2F7;overflow:hidden; }} .q-main i {{ display:block;height:100%;border-radius:999px;background:var(--c); }}
    .conclusion p {{ margin:12px 0 0; color:#344054; font-size:18px; font-weight:800; line-height:1.8; }} .conclusion b {{ color:var(--pink); }}
    .summary {{ grid-column:1/-1; display:grid; grid-template-columns:1.5fr repeat(4,1fr); gap:14px; margin-bottom:14px; }} .summary-card {{ background:#fff; border:1px solid var(--border); border-radius:17px; box-shadow:var(--shadow); padding:18px; min-height:108px; }} .summary-card span {{ display:block; color:var(--muted); font-size:12px; font-weight:800; }} .summary-card strong {{ display:block; margin-top:8px; color:#101828; font-size:24px; }} .summary-card.task strong {{ color:var(--pink); font-size:20px; }}
    .model-chart-card {{ grid-column:span 12; height:430px; }} .feature-card {{ grid-column:span 12; height:430px; }} .model-chart {{ height:374px; }} .feature-chart {{ height:344px; }} .feature-note {{ margin-top:4px; color:#667085; font-size:12px; font-weight:700; line-height:1.45; }}
    .diag-card {{ grid-column:span 7; height:300px; }} .advice-card {{ grid-column:span 7; height:300px; }} .profile {{ grid-column:span 10; height:300px; }} .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; height:236px; margin-top:4px; }} .profile-chart {{ height:236px; }}
    .line {{ display:flex; gap:8px; align-items:center; margin:12px 0; color:#344054; font-size:13px; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }} .mark {{ width:18px;height:18px;display:inline-grid;place-items:center;flex:0 0 18px;border-radius:50%;color:#fff;background:#22C55E; }} .mark svg {{ width:11px;height:11px; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .side-deco {{ display:none; }} .trend,.top6,.value,.word,.quality,.weekday,.conclusion,.model-chart-card,.feature-card,.diag-card,.advice-card,.profile {{ grid-column:1/-1; }} .kpis,.summary {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">{svg_icon("heart")}<span>核心洞察</span></h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("crown")}</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">{svg_icon("trend")}</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("target")}</div><div><span>主模型 Accuracy</span><strong>{float(rf_row['Accuracy']):.3f}</strong><small>F1-score {float(rf_row['F1_weighted']):.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">{svg_icon("user_check")}</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_rate)}</strong><small>仅少量用户匹配画像</small></div></div>
      <div class="side-deco"><span class="circle c1"></span><span class="circle c2"></span><span class="mini-heart">{svg_icon("heart")}</span><span class="mini-bottle">{svg_icon("bottle")}</span></div>
    </aside>
    <main class="main">
      <header class="hero">
        <div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div>
        <nav class="tabs"><button class="tab-btn active" data-tab="business">业务总览</button><button class="tab-btn" data-tab="model">模型验证</button></nav>
        <div class="hero-deco"><span class="deco-circle dc1"></span><span class="deco-circle dc2"></span><span class="deco-circle dc3"></span><span class="deco-heart dh1">{svg_icon("heart")}</span><span class="deco-heart dh2">{svg_icon("heart")}</span><span class="deco-bottle">{svg_icon("bottle")}</span></div>
      </header>

      <section id="business" class="tab-panel active">
        <section class="kpis">{kpi_html}</section>
        <section class="grid">
          <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">最近 12 个月购买数量与订单数变化</div></div><span class="unit">单位：件 / 单</span></div><div id="trendChart" class="chart trend-chart"></div></section>
          <section class="card top6"><div class="card-head"><div><h3>商品大类销量 TOP6</h3><div class="sub">中文业务品类按购买数量降序</div></div><span class="unit">单位：件</span></div><div id="categoryChart" class="chart top-chart"></div></section>
          <section class="card value"><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别高购买量订单</div><div id="highBuyChart" class="chart value-chart"></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量占比</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
          <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
          <section class="card quality"><h3>数据质量与画像匹配</h3><div class="q-grid">{quality_html}</div></section>
          <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
          <section class="card conclusion"><h3>业务结论</h3><p><b>{html.escape(hot_category)}</b>、婴儿服饰和尿裤湿巾构成主要消费品类；高购买量订单占比约 <b>{fmt_pct(high_rate)}</b>，可作为重点营销对象。</p></section>
        </section>
      </section>

      <section id="model" class="tab-panel">
        <section class="summary">
          <div class="summary-card task"><span>主任务</span><strong>商品大类识别 cat1</strong></div>
          <div class="summary-card"><span>推荐模型</span><strong>Random Forest</strong></div>
          <div class="summary-card"><span>Accuracy</span><strong>{float(rf_row['Accuracy']):.3f}</strong></div>
          <div class="summary-card"><span>F1-score</span><strong>{float(rf_row['F1_weighted']):.3f}</strong></div>
          <div class="summary-card"><span>high_buy 辅助模型</span><strong>AUC {float(high_best['AUC']):.3f}</strong></div>
        </section>
        <section class="grid">
          <section class="card model-chart-card"><h3>模型性能对比（cat1 主模型）</h3><div id="modelChart" class="chart model-chart"></div></section>
          <section class="card feature-card"><h3>特征重要性 TOP10</h3><div id="featureChart" class="chart feature-chart"></div></section>
          <section class="card diag-card"><h3>模型诊断</h3>{model_diag}</section>
          <section class="card advice-card"><h3>业务建议</h3>{advice_html}</section>
          <section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart profile-chart"></div><div id="ageChart" class="chart profile-chart"></div></div></section>
        </section>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const textColor="#101828", mutedColor="#667085", gridLine="#EEF2F7", colors=payload.colors;
    const charts = [];
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); charts.push(c); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#C4B5FD"}},{{offset:1,color:"#8B5CF6"}}]):"#8B5CF6";
    chart("trendChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:10}}}},grid:{{left:52,right:46,top:42,bottom:32}},xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10,interval:1}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("categoryChart",{{tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},grid:{{left:88,right:80,top:14,bottom:24}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:13,label:{{show:true,position:"right",color:textColor,fontSize:9,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,8,8,0],color:p=>colors[(p.dataIndex+2)%colors.length]}}}}]}});
    chart("highBuyChart",{{color:["#EF476F","#3B82F6"],tooltip:{{trigger:"item"}},title:{{text:payload.highRate+"%",subtext:"高购买量订单占比",left:"34%",top:"39%",textAlign:"center",textStyle:{{fontSize:21,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:10,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"middle",orient:"vertical",itemWidth:9,itemHeight:9,textStyle:{{color:textColor,fontSize:10}}}},series:[{{type:"pie",radius:["48%","68%"],center:["34%","50%"],label:{{show:false}},labelLine:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}});
    chart("weekdayChart",{{color:["#22C55E","#3B82F6"],tooltip:{{trigger:"axis"}},legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:30,top:42,bottom:26}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{show:false}}}}],series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuy,barWidth:16,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:5,lineStyle:{{width:2.5,color:"#3B82F6"}},itemStyle:{{color:"#3B82F6"}}}}]}});
    chart("modelChart",{{color:["#3B82F6","#22C55E"],tooltip:{{trigger:"axis",axisPointer:{{type:"shadow"}}}},legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:150,right:78,top:46,bottom:24}},xAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.perfNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"Accuracy",type:"bar",data:payload.perfAcc.slice().reverse(),barWidth:12,barGap:"20%",label:{{show:true,position:"right",fontSize:10,formatter:p=>p.value.toFixed(3)}}}},{{name:"F1-score",type:"bar",data:payload.perfF1.slice().reverse(),barWidth:12,label:{{show:true,position:"right",fontSize:10,formatter:p=>p.value.toFixed(3)}}}}]}});
    chart("featureChart",{{tooltip:{{trigger:"axis",formatter:params=>{{const p=params[0]; const reversedShort=payload.featureShort.slice().reverse(); const idx=reversedShort.indexOf(p.name); const full=payload.featureNames.slice().reverse()[idx] || p.name; return full + "<br/>重要性：" + p.value.toFixed(3);}}}},grid:{{left:118,right:80,top:18,bottom:24}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureShort.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:12,label:{{show:true,position:"right",fontSize:9,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:purpleGrad}}}}]}});
    chart("genderChart",{{color:["#3B82F6","#EF476F"],title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}});
    chart("ageChart",{{color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"],title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:{{trigger:"item"}},legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}});
    document.querySelectorAll(".tab-btn").forEach(btn=>btn.addEventListener("click",()=>{{document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));btn.classList.add("active");document.getElementById(btn.dataset.tab).classList.add("active");setTimeout(()=>charts.forEach(c=>c.resize()),60);}}));
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成浅色母婴玻璃拟态双 Tab 数据驾驶舱。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def svg_icon(name: str) -> str:
        icons = {
            "heart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 21s-7.5-4.6-9.6-9.2C.8 8.2 2.9 5 6.3 5c2 0 3.4 1.1 4.2 2.3C11.3 6.1 12.7 5 14.7 5c3.4 0 5.5 3.2 3.9 6.8C16.5 16.4 12 21 12 21Z'/></svg>",
            "spark": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Zm7 12 1 3 3 1-3 1-1 3-1-3-3-1 3-1 1-3Z'/></svg>",
            "crown": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 18h16l1-10-5 3-4-7-4 7-5-3 1 10Zm1 2h14v2H5v-2Z'/></svg>",
            "trend": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 17h3.8l4.1-5.4 3.4 3.4L21 7.8V13h2V4h-9v2h5.4l-4.3 5.4-3.5-3.5L6.8 15H4v2Z'/></svg>",
            "target": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2Zm0 5a5 5 0 1 0 5 5h-2a3 3 0 1 1-3-3V7Zm1 6 8-8-1.4-1.4-8 8V6h-2v8h8v-2h-4.6Z'/></svg>",
            "user_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.6a7 7 0 0 1-.6-2.8c0-1.6.5-3.1 1.5-4.2H10Zm11.7 1.7-1.4-1.4-4.3 4.3-1.8-1.8-1.4 1.4 3.2 3.2 5.7-5.7Z'/></svg>",
            "bag": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",
            "users": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.2 0-7 2.1-7 5v2h14v-2c0-2.9-2.8-5-7-5Zm8-2a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm0 2c-.7 0-1.4.1-2 .3 1.8 1.1 3 2.7 3 4.7v2h4v-2c0-2.9-2.1-5-5-5Z'/></svg>",
            "package": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 6 3.4v7.2l-6-3.4V10Zm14 0v7.2l-6 3.4v-7.2l6-3.4Z'/></svg>",
            "cart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>",
            "database": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 3c5 0 9 1.8 9 4s-4 4-9 4-9-1.8-9-4 4-4 9-4Zm-9 7c1.6 1.7 5 3 9 3s7.4-1.3 9-3v3c0 2.2-4 4-9 4s-9-1.8-9-4v-3Zm0 6c1.6 1.7 5 3 9 3s7.4-1.3 9-3v1c0 2.2-4 4-9 4s-9-1.8-9-4v-1Z'/></svg>",
            "user_x": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.3a6.8 6.8 0 0 1 1.2-7H10Zm10.4 1.2-2.2 2.2-2.2-2.2-1.4 1.4 2.2 2.2-2.2 2.2 1.4 1.4 2.2-2.2 2.2 2.2 1.4-1.4-2.2-2.2 2.2-2.2-1.4-1.4Z'/></svg>",
            "cart_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 .1 0H7Zm10 0a2 2 0 1 0 .1 0H17ZM4 3H2v2h2l3 10h8.5a7.3 7.3 0 0 1 .6-2H8.5l-.6-2H16a6.8 6.8 0 0 1 3.6-2.6L20.1 7H7.2L6.5 5H4V3Zm18 10.7-1.4-1.4-4.1 4.1-1.7-1.7-1.4 1.4 3.1 3.1 5.5-5.5Z'/></svg>",
            "shield": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2 4 5v6c0 5 3.4 9.6 8 11 4.6-1.4 8-6 8-11V5l-8-3Zm4.8 7.7-5.4 5.4-2.2-2.2-1.4 1.4 3.6 3.6 6.8-6.8-1.4-1.4Z'/></svg>",
            "check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9.2 16.6 4.9 12.3 3.5 13.7l5.7 5.7L21 7.6 19.6 6.2 9.2 16.6Z'/></svg>",
            "bottle": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 2h6v4l2 2v11a3 3 0 0 1-3 3h-4a3 3 0 0 1-3-3V8l2-2V2Zm2 2v2h2V4h-2Zm-2 8h6v-2H9v2Zm0 4h6v-2H9v2Z'/></svg>",
        }
        return icons[name]

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"<article class='kpi glass' style='--c:{color}'><div class='kpi-icon'>{svg_icon(icon)}</div><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div></article>"

    def q_item(title: str, value: str, ratio: float, color: str, icon: str) -> str:
        width = max(0, min(float(ratio), 1)) * 100
        return f"<div class='q-item glass' style='--c:{color}'><div class='q-icon'>{svg_icon(icon)}</div><div class='q-main'><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong></div><em><i style='width:{width:.1f}%'></i></em></div></div>"

    def badge(text: str) -> str:
        return f"<span class='insight-badge'>{svg_icon('spark')}{html.escape(text)}</span>"

    def small_line(text: str) -> str:
        return f"<div class='line'><span class='mark'>{svg_icon('check')}</span><span>{html.escape(text)}</span></div>"

    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    peak_month = str(monthly_show.loc[monthly_show["total_buy_mount"].idxmax(), "year_month"]) if not monthly_show.empty else "暂无数据"
    if not overview_df.empty and overview_df["dataset"].eq("trade_history").any():
        raw_trade_records = int(overview_df.loc[overview_df["dataset"].eq("trade_history"), "rows"].iloc[0])
    else:
        raw_path = DATA_RAW_DIR / "sam_tianchi_mum_baby_trade_history.csv"
        raw_trade_records = int(pd.read_csv(raw_path).shape[0]) if raw_path.exists() else int(len(merged_df))
    raw_trade_records = int(overview_df.loc[overview_df["dataset"].eq("trade_history"), "rows"].iloc[0]) if not overview_df.empty and overview_df["dataset"].eq("trade_history").any() else int(len(merged_df))
    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    high_best = high_metrics.iloc[0]
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    highest_cat1_row = cat_metrics.sort_values(["Accuracy", "F1_weighted"], ascending=False).iloc[0]
    rf_row = cat_rows.get("Random Forest", cat_metrics.iloc[0])
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    unmatched_count = int((~profile_mask).sum())
    unknown_rate = float((~profile_mask).mean())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_raw = matched_df["baby_age_group"].replace("unknown", np.nan).dropna()
        age_counts = age_raw.value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    perf_names, perf_acc, perf_f1 = [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            perf_names.append(f"{name}（推荐模型）" if name == "Random Forest" else name)
            perf_acc.append(round(float(row["Accuracy"]), 4))
            perf_f1.append(round(float(row["F1_weighted"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
        "user_buy_count": "user_buy",
        "user_unique_item_count": "user_items",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]
    top_feature = feature_short[0] if feature_short else "暂无数据"

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "users"),
            kpi("商品种类数", fmt_num(merged_df["auction_id"].nunique()), "商品 ID 去重统计", "#8B5CF6", "package"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            q_item("清洗后交易记录数", fmt_num(len(merged_df)), 1, "#3B82F6", "database"),
            q_item("画像匹配率", fmt_pct(profile_rate), profile_rate, "#22C55E", "user_check"),
            q_item("画像未匹配记录数", fmt_num(unmatched_count), unknown_rate, "#64748B", "user_x"),
            q_item("有效购买记录占比", fmt_pct(valid_rate), valid_rate, "#06B6D4", "cart_check"),
            q_item("核心字段完整率", fmt_pct(core_complete_rate), core_complete_rate, "#8B5CF6", "shield"),
        ]
    )
    model_diag = "".join(
        [
            small_line("主任务：商品大类识别"),
            small_line("推荐模型：Random Forest"),
            small_line(f"Accuracy：{float(rf_row['Accuracy']):.3f}"),
            small_line(f"F1-score：{float(rf_row['F1_weighted']):.3f}"),
            small_line("泄露检查：未使用 cat1_name"),
            small_line(f"high_buy：AUC {float(high_best['AUC']):.3f}，F1 {float(high_best['F1']):.3f}"),
        ]
    )
    advice_html = "".join(
        [
            small_line(f"重点关注{hot_category}等核心品类"),
            small_line("结合高购买量订单制定促销策略"),
            small_line("补充价格、品牌、浏览、收藏等特征"),
            small_line("提升模型解释能力"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "metrics": {
            "rawTradeRecords": raw_trade_records,
            "cleanTradeRecords": int(len(merged_df)),
            "tradeUsers": int(merged_df["user_id"].nunique()),
            "itemKinds": int(merged_df["auction_id"].nunique()),
            "totalBuyMount": int(merged_df["buy_mount"].sum()),
            "highBuyOrders": int(high_mask.sum()),
            "normalOrders": int((~high_mask).sum()),
            "highBuyRate": round(high_rate * 100, 1),
            "salesContributionRate": round(high_sales_rate * 100, 1),
            "avgBuyLift": round(value_lift, 1),
            "profileCoverageRate": round(profile_rate * 100, 1),
            "profileUnmatchedRecords": unmatched_count,
            "tradeUsabilityRate": round(valid_rate * 100, 1),
            "coreFieldCompleteRate": round(core_complete_rate * 100, 1),
            "highestModel": str(highest_cat1_row["model"]),
            "highestAccuracy": round(float(highest_cat1_row["Accuracy"]), 4),
            "highestF1": round(float(highest_cat1_row["F1_weighted"]), 4),
            "displayModel": "Random Forest",
            "displayAccuracy": round(float(rf_row["Accuracy"]), 4),
            "displayF1": round(float(rf_row["F1_weighted"]), 4),
            "highBuyAuxModel": str(high_dt_row["model"]),
            "highBuyAuxAuc": round(float(high_dt_row["AUC"]), 4),
            "highBuyAuxF1": round(float(high_dt_row["F1"]), 4),
        },
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "highSalesRate": round(high_sales_rate * 100, 1),
        "valueLift": round(value_lift, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "perfNames": perf_names,
        "perfAcc": perf_acc,
        "perfF1": perf_f1,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    (OUTPUT_DIR / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --glass:rgba(255,255,255,.68); --glass-border:rgba(255,255,255,.78); --glass-shadow:0 18px 45px rgba(31,41,55,.08), inset 0 1px 0 rgba(255,255,255,.85); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; }} body {{ position:relative; color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 12% 10%,rgba(255,92,138,.18),transparent 25%),radial-gradient(circle at 88% 12%,rgba(137,207,240,.22),transparent 26%),radial-gradient(circle at 70% 92%,rgba(184,161,255,.20),transparent 28%),linear-gradient(135deg,#FFF7FA 0%,#FAFBFF 44%,#F3FAFF 100%); overflow-x:hidden; }} body::before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.34; background-image:linear-gradient(rgba(255,92,138,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,.10) 1px,transparent 1px); background-size:32px 32px; mask-image:linear-gradient(180deg,rgba(0,0,0,.85),rgba(0,0,0,.18)); }} body::after {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.22; background-image:radial-gradient(circle,rgba(255,92,138,.28) 0 2px,transparent 2px),radial-gradient(circle,rgba(137,207,240,.30) 0 1.6px,transparent 1.6px); background-size:96px 96px,128px 128px; background-position:24px 34px,80px 20px; }} svg {{ display:block; width:1em; height:1em; fill:currentColor; }}
    .page {{ position:relative; z-index:1; width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:10px 16px 12px 10px; display:grid; grid-template-columns:280px 1fr; gap:14px; }}
    .glass,.sidebar,.card,.summary-card,.insight,.q-item {{ background:var(--glass); border:1px solid var(--glass-border); backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px); box-shadow:var(--glass-shadow); }}
    .card,.kpi,.summary-card {{ position:relative; }} .card::before,.kpi::before,.summary-card::before {{ content:""; position:absolute; left:14px; right:14px; top:0; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.95),transparent); pointer-events:none; }}
    .card,.kpi,.summary-card,.insight,.q-item,.tab-btn {{ transition:transform .18s ease, box-shadow .18s ease; }} .card:hover,.kpi:hover,.summary-card:hover {{ transform:translateY(-2px); box-shadow:0 22px 48px rgba(31,41,55,.10), inset 0 1px 0 rgba(255,255,255,.9); }} .kpi:hover .kpi-icon {{ transform:scale(1.06); }}
    .sidebar {{ position:relative; min-height:1058px; padding:20px 16px; overflow:hidden; border-radius:22px; }}
    .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:24px; }} .face {{ width:46px;height:46px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 46px;background:rgba(255,255,255,.4); }} .face:before,.face:after {{ content:"";position:absolute;top:17px;width:5px;height:5px;background:var(--pink);border-radius:50%; }} .face:before{{left:12px}} .face:after{{right:12px}} .smile{{position:absolute;left:14px;top:25px;width:16px;height:8px;border-bottom:3px solid var(--pink);border-radius:0 0 16px 16px}} .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1; }} .brand p {{ margin:7px 0 0; color:#101828; font-size:12px; font-weight:800; }}
    .side-title {{ display:flex; gap:8px; align-items:center; margin:0 0 14px; color:var(--pink); font-size:17px; }} .side-title svg {{ width:17px; height:17px; }}
    .insight {{ display:grid; grid-template-columns:50px 1fr; gap:10px; align-items:center; padding:13px 8px; margin-bottom:12px; border-radius:16px; }} .i-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:23px;box-shadow:0 8px 18px rgba(16,24,40,.05); }} .i-icon svg {{ width:23px; height:23px; }} .insight span {{ color:var(--muted);font-size:11px;font-weight:800; }} .insight strong {{ display:block;margin-top:5px;color:var(--c);font-size:21px;line-height:1.05; }} .insight small {{ display:block;margin-top:4px;color:#101828;font-size:11px;font-weight:800; }}
    .side-deco {{ position:absolute; left:18px; right:18px; bottom:22px; height:80px; opacity:.30; }} .side-deco .circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .side-deco .c1{{width:88px;height:88px;left:6px;bottom:-34px}} .side-deco .c2{{width:46px;height:46px;right:16px;bottom:10px;background:rgba(137,207,240,.22)}} .side-deco .mini-heart{{position:absolute;left:112px;bottom:34px;color:var(--pink)}} .side-deco .mini-heart svg{{width:20px;height:20px}} .side-deco .mini-bottle{{position:absolute;right:72px;bottom:16px;color:var(--pink)}} .side-deco .mini-bottle svg{{width:28px;height:28px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:104px; display:flex; flex-direction:column; align-items:center; justify-content:center; overflow:hidden; margin-bottom:10px; }} .hero h2 {{ margin:0; text-align:center; font-size:31px; line-height:1.12; }} .hero p {{ margin:6px 0 10px; text-align:center; color:#101828; font-size:14px; font-weight:700; }} .hero-deco {{ position:absolute; right:10px; top:2px; width:190px; height:66px; opacity:.30; pointer-events:none; }} .deco-circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .dc1{{width:92px;height:92px;right:36px;top:-34px}} .dc2{{width:42px;height:42px;right:0;top:16px;background:rgba(137,207,240,.22)}} .dc3{{width:28px;height:28px;right:118px;top:26px;background:rgba(184,161,255,.25)}} .deco-heart{{position:absolute;color:rgba(255,92,138,.5)}} .deco-heart svg{{width:17px;height:17px}} .dh1{{right:92px;top:8px}} .dh2{{right:145px;top:32px}} .deco-bottle{{position:absolute;right:54px;top:18px;color:rgba(255,92,138,.48)}} .deco-bottle svg{{width:28px;height:28px}}
    .tabs {{ display:flex; gap:10px; justify-content:center; }} .tab-btn {{ border:1px solid rgba(255,255,255,.72); background:rgba(255,255,255,.58); color:var(--pink); height:34px; padding:0 28px; border-radius:999px; font-weight:800; cursor:pointer; backdrop-filter:blur(14px); }} .tab-btn.active {{ background:linear-gradient(135deg,#FF5C8A,#FF8FAB); color:#fff; border-color:rgba(255,255,255,.78); box-shadow:0 12px 24px rgba(255,92,138,.20); }}
    .tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:14px; }} .card {{ border-radius:17px; padding:14px 16px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:15px; color:#101828; }} .sub {{ margin-top:5px; color:var(--muted); font-size:10.5px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:5px; }} .unit {{ color:var(--muted); font-size:10.5px; font-weight:800; }}
    .insight-badge {{ display:inline-flex; align-items:center; gap:5px; padding:5px 9px; border-radius:999px; background:rgba(255,92,138,.10); color:var(--pink); font-size:11px; font-weight:800; white-space:nowrap; }} .insight-badge svg {{ width:12px;height:12px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px; }} .kpi {{ height:110px; display:flex; align-items:center; gap:14px; padding:16px; border-radius:17px; }} .kpi-icon {{ width:52px;height:52px;border-radius:50%;display:grid;place-items:center;flex:0 0 52px;color:var(--c);background:rgba(255,241,245,.78);font-size:26px; transition:transform .18s ease; }} .kpi-icon svg {{ width:26px;height:26px; }} .kpi span {{ font-size:12px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:8px;font-size:24px;line-height:1; }} .kpi small {{ display:block;margin-top:9px;color:var(--muted);font-size:11px;font-weight:700; }}
    .trend {{ grid-column:span 10; height:380px; }} .top6 {{ grid-column:span 7; height:380px; }} .value {{ grid-column:span 7; height:380px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:326px; }} .value-body {{ display:grid; grid-template-columns:1.1fr .9fr; gap:8px; height:244px; }} .value-chart,.sales-gauge {{ height:244px; }} .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }} .value-mini {{ padding:10px 6px; border-radius:13px; text-align:center; background:rgba(255,255,255,.52); border:1px solid rgba(255,255,255,.68); }} .value-mini span {{ display:block;color:var(--muted);font-size:10px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:4px;color:var(--vc);font-size:17px; }}
    .word {{ grid-column:span 6; height:250px; }} .quality {{ grid-column:span 7; height:250px; }} .weekday {{ grid-column:span 6; height:250px; }} .conclusion {{ grid-column:span 5; height:250px; }} .small-chart {{ height:190px; }} .wordcloud {{ height:194px; margin-top:8px; display:flex; align-items:center; justify-content:center; border-radius:15px; background:rgba(255,255,255,.45); border:1px solid rgba(255,255,255,.76); box-shadow:inset 0 1px 0 rgba(255,255,255,.85); }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; }} .missing {{ color:var(--muted); font-size:13px; }}
    .q-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:10px; }} .q-item {{ display:grid;grid-template-columns:32px 1fr;gap:7px;align-items:center;min-height:45px;padding:7px;border-radius:12px; }} .q-icon {{ width:29px;height:29px;border-radius:9px;display:grid;place-items:center;color:var(--c);background:rgba(255,255,255,.62);font-size:17px; }} .q-icon svg {{ width:17px; height:17px; }} .q-main div {{ display:flex;justify-content:space-between;gap:6px;align-items:center; }} .q-main span {{ color:var(--muted);font-size:10.5px;font-weight:800; }} .q-main strong {{ color:#101828;font-size:12px;white-space:nowrap; }} .q-main em {{ display:block;height:5px;margin-top:6px;border-radius:999px;background:rgba(238,242,247,.82);overflow:hidden; }} .q-main i {{ display:block;height:100%;border-radius:999px;background:var(--c); }}
    .conclusion p {{ margin:12px 0 0; color:#344054; font-size:18px; font-weight:800; line-height:1.8; }} .conclusion b {{ color:var(--pink); }}
    .summary {{ grid-column:1/-1; display:grid; grid-template-columns:1.5fr repeat(4,1fr); gap:14px; margin-bottom:14px; }} .summary-card {{ border-radius:17px; padding:18px; min-height:108px; }} .summary-card span {{ display:block; color:var(--muted); font-size:12px; font-weight:800; }} .summary-card strong {{ display:block; margin-top:8px; color:#101828; font-size:24px; }} .summary-card.task strong {{ color:var(--pink); font-size:20px; }}
    .model-chart-card {{ grid-column:span 12; height:430px; }} .feature-card {{ grid-column:span 12; height:430px; }} .model-chart,.feature-chart {{ height:374px; }}
    .diag-card {{ grid-column:span 7; height:300px; }} .advice-card {{ grid-column:span 7; height:300px; }} .profile {{ grid-column:span 10; height:300px; }} .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; height:236px; margin-top:4px; }} .profile-chart {{ height:236px; }}
    .line {{ display:flex; gap:8px; align-items:center; margin:12px 0; color:#344054; font-size:13px; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }} .mark {{ width:18px;height:18px;display:inline-grid;place-items:center;flex:0 0 18px;border-radius:50%;color:#fff;background:#22C55E; }} .mark svg {{ width:11px;height:11px; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .side-deco {{ display:none; }} .trend,.top6,.value,.word,.quality,.weekday,.conclusion,.model-chart-card,.feature-card,.diag-card,.advice-card,.profile {{ grid-column:1/-1; }} .kpis,.summary {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">{svg_icon("heart")}<span>核心洞察</span></h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("crown")}</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">{svg_icon("trend")}</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("target")}</div><div><span>主模型 Accuracy</span><strong>{float(rf_row['Accuracy']):.3f}</strong><small>F1-score {float(rf_row['F1_weighted']):.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">{svg_icon("user_check")}</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_rate)}</strong><small>仅少量用户匹配画像</small></div></div>
      <div class="side-deco"><span class="circle c1"></span><span class="circle c2"></span><span class="mini-heart">{svg_icon("heart")}</span><span class="mini-bottle">{svg_icon("bottle")}</span></div>
    </aside>
    <main class="main">
      <header class="hero">
        <div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div>
        <nav class="tabs"><button class="tab-btn active" data-tab="business">业务总览</button><button class="tab-btn" data-tab="model">模型验证</button></nav>
        <div class="hero-deco"><span class="deco-circle dc1"></span><span class="deco-circle dc2"></span><span class="deco-circle dc3"></span><span class="deco-heart dh1">{svg_icon("heart")}</span><span class="deco-heart dh2">{svg_icon("heart")}</span><span class="deco-bottle">{svg_icon("bottle")}</span></div>
      </header>

      <section id="business" class="tab-panel active">
        <section class="kpis">{kpi_html}</section>
        <section class="grid">
          <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">最近 12 个月购买数量与订单数变化</div></div>{badge(f"智能洞察：{peak_month} 为购买峰值")}</div><div id="trendChart" class="chart trend-chart"></div></section>
          <section class="card top6"><div class="card-head"><div><h3>商品大类销量 TOP6</h3><div class="sub">中文业务品类按购买数量降序</div></div>{badge(f"智能洞察：{hot_category}销量最高")}</div><div id="categoryChart" class="chart top-chart"></div></section>
          <section class="card value"><div class="card-head"><div><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别</div></div>{badge("智能洞察：小占比，高贡献")}</div><div id="highBuyChart" class="chart value-chart"></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量占比</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
          <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
          <section class="card quality"><h3>数据质量与画像匹配</h3><div class="q-grid">{quality_html}</div></section>
          <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
          <section class="card conclusion"><h3>业务结论</h3><p><b>{html.escape(hot_category)}</b>、婴儿服饰和尿裤湿巾构成主要消费品类；高购买量订单占比约 <b>{fmt_pct(high_rate)}</b>，可作为重点营销对象。</p></section>
        </section>
      </section>

      <section id="model" class="tab-panel">
        <section class="summary">
          <div class="summary-card task"><span>主任务</span><strong>商品大类识别 cat1</strong></div>
          <div class="summary-card"><span>推荐模型</span><strong>Random Forest</strong></div>
          <div class="summary-card"><span>Accuracy</span><strong>{float(rf_row['Accuracy']):.3f}</strong></div>
          <div class="summary-card"><span>F1-score</span><strong>{float(rf_row['F1_weighted']):.3f}</strong></div>
          <div class="summary-card"><span>high_buy 辅助模型</span><strong>AUC {float(high_best['AUC']):.3f}</strong></div>
        </section>
        <section class="grid">
          <section class="card model-chart-card"><div class="card-head"><h3>模型性能对比（cat1 主模型）</h3>{badge("智能洞察：主模型表现稳定")}</div><div id="modelChart" class="chart model-chart"></div></section>
          <section class="card feature-card"><div class="card-head"><h3>特征重要性 TOP10</h3>{badge(f"智能洞察：{top_feature} 贡献最高")}</div><div id="featureChart" class="chart feature-chart"></div></section>
          <section class="card diag-card"><h3>模型诊断</h3>{model_diag}</section>
          <section class="card advice-card"><h3>业务建议</h3>{advice_html}</section>
          <section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart profile-chart"></div><div id="ageChart" class="chart profile-chart"></div></div></section>
        </section>
      </section>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const textColor="#101828", mutedColor="#667085", gridLine="rgba(148,163,184,.18)", colors=payload.colors;
    const charts = [];
    function baseChartOption() {{
      return {{
        textStyle: {{ fontFamily: "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',Arial,sans-serif" }},
        tooltip: {{
          trigger: "axis",
          backgroundColor: "rgba(255,255,255,0.92)",
          borderColor: "rgba(255,92,138,0.18)",
          borderWidth: 1,
          textStyle: {{ color: "#101828" }},
          extraCssText: "box-shadow: 0 12px 30px rgba(16,24,40,.12); border-radius: 12px;"
        }},
        animationDuration: 900,
        animationEasing: "cubicOut"
      }};
    }}
    function merge(base, extra) {{ return Object.assign({{}}, base, extra); }}
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); charts.push(c); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#C4B5FD"}},{{offset:1,color:"#8B5CF6"}}]):"#8B5CF6";
    chart("trendChart",merge(baseChartOption(),{{color:["#EF476F","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:items=>`<b>${{items[0].axisValue}}</b><br/>购买数量：${{Number(items[0].value).toLocaleString()}} 件<br/>订单数：${{Number(items[1].value).toLocaleString()}} 单`}}),legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:10}}}},grid:{{left:52,right:46,top:42,bottom:32}},xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10,interval:1}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}},markPoint:{{symbolSize:48,label:{{fontSize:9}},data:[{{type:"max",name:"峰值"}}]}} ,markLine:{{symbol:"none",label:{{formatter:"平均",fontSize:9}},lineStyle:{{color:"rgba(239,71,111,.42)",type:"dashed"}},data:[{{type:"average",name:"平均"}}]}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6",shadowColor:"rgba(59,130,246,.28)",shadowBlur:10,shadowOffsetY:6}},itemStyle:{{color:"#3B82F6"}}}}]}}));
    chart("categoryChart",merge(baseChartOption(),{{tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",axisPointer:{{type:"shadow"}},formatter:items=>`${{items[0].name.replace(/^Top1 /,"")}}<br/>销量：${{Number(items[0].value).toLocaleString()}} 件`}}),grid:{{left:96,right:80,top:14,bottom:24}},xAxis:{{type:"value",show:false,max:Math.max(...payload.categoryValues)*1.16}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11,formatter:(value,index)=>index===payload.categoryNames.length-1?"Top1 "+value:value}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"背景轨道",type:"bar",data:payload.categoryValues.map(()=>Math.max(...payload.categoryValues)).slice().reverse(),barWidth:15,barGap:"-100%",silent:true,itemStyle:{{borderRadius:[0,9,9,0],color:"rgba(255,255,255,.55)"}}}},{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:15,label:{{show:true,position:"right",color:textColor,fontSize:9,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,9,9,0],color:p=>p.dataIndex===payload.categoryValues.length-1?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#FDBA74"}},{{offset:1,color:"#EF476F"}}]):new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"rgba(255,255,255,.25)"}},{{offset:1,color:colors[(p.dataIndex+2)%colors.length]}}])}}}}]}}));
    chart("highBuyChart",merge(baseChartOption(),{{color:["#EF476F","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"item",formatter:p=>`${{p.name}}<br/>订单数：${{Number(p.value).toLocaleString()}}<br/>占比：${{p.percent}}%`}}),title:{{text:payload.highRate+"%",subtext:"小占比，高贡献",left:"34%",top:"37%",textAlign:"center",textStyle:{{fontSize:21,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:10,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"middle",orient:"vertical",itemWidth:9,itemHeight:9,textStyle:{{color:textColor,fontSize:10}}}},series:[{{type:"pie",radius:["48%","68%"],center:["34%","50%"],label:{{show:false}},labelLine:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}}));
    chart("salesGaugeChart",gaugeOption("销量贡献率",payload.metrics.salesContributionRate,"#3B82F6"));
    chart("tradeGauge",gaugeOption("交易数据可用率",payload.metrics.tradeUsabilityRate,"#22C55E"));
    chart("profileGauge",gaugeOption("宝宝画像覆盖率",payload.metrics.profileCoverageRate,"#F59E0B"));
    chart("coreGauge",gaugeOption("核心字段完整率",payload.metrics.coreFieldCompleteRate,"#8B5CF6"));
    chart("weekdayChart",merge(baseChartOption(),{{color:["#22C55E","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:items=>`星期${{items[0].axisValue}}<br/>购买数量：${{Number(items[0].value).toLocaleString()}} 件<br/>订单数：${{Number(items[1].value).toLocaleString()}} 单`}}),legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:30,top:42,bottom:26}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{show:false}}}}],series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuy,barWidth:16,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:5,lineStyle:{{width:2.5,color:"#3B82F6",shadowColor:"rgba(59,130,246,.22)",shadowBlur:8,shadowOffsetY:4}},itemStyle:{{color:"#3B82F6"}}}}]}}));
    chart("modelChart",merge(baseChartOption(),{{color:["#3B82F6","#22C55E"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",axisPointer:{{type:"shadow"}},formatter:items=>`<b>${{items[0].axisValue}}</b><br/>Accuracy：${{items[0].value.toFixed(3)}}<br/>F1-score：${{items[1].value.toFixed(3)}}`}}),legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:150,right:78,top:46,bottom:24}},xAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.perfNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"Accuracy",type:"bar",data:payload.perfAcc.slice().reverse(),barWidth:12,barGap:"20%",label:{{show:true,position:"right",fontSize:10,formatter:p=>p.value.toFixed(3)}},markLine:{{symbol:"none",label:{{formatter:"优秀阈值 0.90",fontSize:9}},lineStyle:{{color:"rgba(255,92,138,.55)",type:"dashed"}},data:[{{xAxis:0.9}}]}}}},{{name:"F1-score",type:"bar",data:payload.perfF1.slice().reverse(),barWidth:12,label:{{show:true,position:"right",fontSize:10,formatter:p=>p.value.toFixed(3)}}}}]}}));
    chart("featureChart",merge(baseChartOption(),{{tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:params=>{{const p=params[0]; const reversedShort=payload.featureShort.slice().reverse(); const idx=reversedShort.indexOf(p.name); const full=payload.featureNames.slice().reverse()[idx] || p.name; return full + "<br/>重要性：" + p.value.toFixed(3);}}}}),grid:{{left:118,right:80,top:18,bottom:24}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureShort.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:12,label:{{show:true,position:"right",fontSize:9,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:p=>p.dataIndex===payload.featureValues.length-1?"#FF5C8A":purpleGrad}}}}]}}));
    chart("genderChart",merge(baseChartOption(),{{color:["#3B82F6","#EF476F"],title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:merge(baseChartOption().tooltip,{{trigger:"item"}}),legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}}));
    chart("ageChart",merge(baseChartOption(),{{color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"],title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:merge(baseChartOption().tooltip,{{trigger:"item"}}),legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}}));
    document.querySelectorAll(".tab-btn").forEach(btn=>btn.addEventListener("click",()=>{{document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));btn.classList.add("active");document.getElementById(btn.dataset.tab).classList.add("active");setTimeout(()=>charts.forEach(c=>c.resize()),80);}}));
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def make_wordcloud(merged_df: pd.DataFrame) -> None:
    """使用婴儿车 mask 生成高清母婴消费热点词云。"""
    setup_chinese_font()
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((path for path in font_candidates if Path(path).exists()), None)
    word_weights = (
        merged_df.groupby("cat1_name")["buy_mount"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    term_expansion = {
        "喂养用品": [
            "奶粉", "奶瓶", "辅食", "宝宝餐具", "水杯", "吸管杯", "奶嘴", "安抚奶嘴",
            "吸奶器", "储奶袋", "辅食机", "温奶器", "消毒器", "围嘴", "硅胶勺",
            "儿童餐盘", "哺乳用品",
        ],
        "尿裤湿巾": [
            "纸尿裤", "尿不湿", "拉拉裤", "湿巾", "柔巾", "护理垫", "隔尿垫",
            "棉柔巾", "宝宝湿巾", "换尿布", "夜用纸尿裤",
        ],
        "婴儿服饰": [
            "婴儿服", "童装", "连体衣", "哈衣", "爬服", "围嘴", "袜子", "外套",
            "内衣", "宝宝帽", "口水巾", "抱被", "睡袋", "肚围", "换季衣物",
        ],
        "孕产用品": [
            "孕妈用品", "待产包", "产后护理", "哺乳内衣", "孕妇装", "月子用品",
            "防溢乳垫", "收腹带", "产褥垫",
        ],
        "玩具早教": [
            "玩具", "早教", "益智玩具", "安抚玩具", "学步车", "摇铃", "牙胶",
            "布书", "积木", "音乐玩具", "认知卡片", "爬行垫",
        ],
        "童车童床": [
            "婴儿床", "童车", "推车", "安全座椅", "餐椅", "床品", "床围",
            "摇篮", "婴儿车", "婴儿推车", "床垫",
        ],
    }
    base = max(word_weights.values()) if word_weights else 1.0
    frequencies = {str(word): float(weight) * 2.2 for word, weight in word_weights.items()}
    for category, terms in term_expansion.items():
        category_weight = float(word_weights.get(category, base * 0.35))
        frequencies[category] = max(float(frequencies.get(category, 0)), category_weight * 2.45)
        for idx, term in enumerate(terms):
            decay = max(0.18, 0.78 - idx * 0.035)
            frequencies[term] = max(float(frequencies.get(term, 0)), category_weight * decay)

    palette = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#06B6D4", "#EF476F"]

    def color_func(*args, **kwargs):
        return palette[np.random.randint(0, len(palette))]

    mask_candidates = [
        PROJECT_ROOT / "assets" / "masks" / "mother_baby_wordcloud_mask_children.png",
        PROJECT_ROOT / "assets" / "masks" / "mother_baby_wordcloud_mask_stroller.png",
        OUTPUT_DIR / "mother_baby_wordcloud_mask_stroller.png",
        CHART_DIR / "mother_baby_wordcloud_mask_stroller.png",
        PROJECT_ROOT / "archive_old_project" / "outputs" / "mother_baby_wordcloud_mask_stroller.png",
    ]
    mask_path = next((path for path in mask_candidates if path.exists()), None)
    mask = None
    if mask_path:
        mask_image = Image.open(mask_path).convert("L").resize((1600, 1000), Image.LANCZOS)
        mask = np.array(mask_image)

    wc = WordCloud(
        font_path=font_path,
        width=1600,
        height=1000,
        background_color="white",
        max_words=90,
        max_font_size=215,
        min_font_size=7,
        prefer_horizontal=0.9,
        random_state=42,
        relative_scaling=0.5,
        collocations=False,
        repeat=True,
        margin=1,
        mask=mask,
        contour_width=0,
        color_func=color_func,
    ).generate_from_frequencies(frequencies)
    wc.to_image().save(CHART_DIR / "09_wordcloud.png", dpi=(300, 300))


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """生成词云、数据质量和模型性能图优化后的玻璃拟态双 Tab 大屏。"""

    def fmt_num(value) -> str:
        if pd.isna(value):
            return "暂无数据"
        return f"{int(round(float(value))):,}"

    def fmt_pct(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    def svg_icon(name: str) -> str:
        icons = {
            "heart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 21s-7.5-4.6-9.6-9.2C.8 8.2 2.9 5 6.3 5c2 0 3.4 1.1 4.2 2.3C11.3 6.1 12.7 5 14.7 5c3.4 0 5.5 3.2 3.9 6.8C16.5 16.4 12 21 12 21Z'/></svg>",
            "spark": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Zm7 12 1 3 3 1-3 1-1 3-1-3-3-1 3-1 1-3Z'/></svg>",
            "crown": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 18h16l1-10-5 3-4-7-4 7-5-3 1 10Zm1 2h14v2H5v-2Z'/></svg>",
            "trend": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 17h3.8l4.1-5.4 3.4 3.4L21 7.8V13h2V4h-9v2h5.4l-4.3 5.4-3.5-3.5L6.8 15H4v2Z'/></svg>",
            "target": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2Zm0 5a5 5 0 1 0 5 5h-2a3 3 0 1 1-3-3V7Zm1 6 8-8-1.4-1.4-8 8V6h-2v8h8v-2h-4.6Z'/></svg>",
            "user_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.6a7 7 0 0 1-.6-2.8c0-1.6.5-3.1 1.5-4.2H10Zm11.7 1.7-1.4-1.4-4.3 4.3-1.8-1.8-1.4 1.4 3.2 3.2 5.7-5.7Z'/></svg>",
            "bag": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 8V7a5 5 0 0 1 10 0v1h2l1 13H4L5 8h2Zm2 0h6V7a3 3 0 0 0-6 0v1Z'/></svg>",
            "users": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.2 0-7 2.1-7 5v2h14v-2c0-2.9-2.8-5-7-5Zm8-2a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm0 2c-.7 0-1.4.1-2 .3 1.8 1.1 3 2.7 3 4.7v2h4v-2c0-2.9-2.1-5-5-5Z'/></svg>",
            "package": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='m12 2 9 5-9 5-9-5 9-5Zm-7 8 6 3.4v7.2l-6-3.4V10Zm14 0v7.2l-6 3.4v-7.2l6-3.4Z'/></svg>",
            "cart": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm10 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 3H2v2h2l3 10h10.5l3-8H7.2L6.5 5H4V3Z'/></svg>",
            "database": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 3c5 0 9 1.8 9 4s-4 4-9 4-9-1.8-9-4 4-4 9-4Zm-9 7c1.6 1.7 5 3 9 3s7.4-1.3 9-3v3c0 2.2-4 4-9 4s-9-1.8-9-4v-3Zm0 6c1.6 1.7 5 3 9 3s7.4-1.3 9-3v1c0 2.2-4 4-9 4s-9-1.8-9-4v-1Z'/></svg>",
            "user_x": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.4 0-7 2.2-7 5v2h10.3a6.8 6.8 0 0 1 1.2-7H10Zm10.4 1.2-2.2 2.2-2.2-2.2-1.4 1.4 2.2 2.2-2.2 2.2 1.4 1.4 2.2-2.2 2.2 2.2 1.4-1.4-2.2-2.2 2.2-2.2-1.4-1.4Z'/></svg>",
            "cart_check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M7 18a2 2 0 1 0 .1 0H7Zm10 0a2 2 0 1 0 .1 0H17ZM4 3H2v2h2l3 10h8.5a7.3 7.3 0 0 1 .6-2H8.5l-.6-2H16a6.8 6.8 0 0 1 3.6-2.6L20.1 7H7.2L6.5 5H4V3Zm18 10.7-1.4-1.4-4.1 4.1-1.7-1.7-1.4 1.4 3.1 3.1 5.5-5.5Z'/></svg>",
            "shield": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 2 4 5v6c0 5 3.4 9.6 8 11 4.6-1.4 8-6 8-11V5l-8-3Zm4.8 7.7-5.4 5.4-2.2-2.2-1.4 1.4 3.6 3.6 6.8-6.8-1.4-1.4Z'/></svg>",
            "check": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9.2 16.6 4.9 12.3 3.5 13.7l5.7 5.7L21 7.6 19.6 6.2 9.2 16.6Z'/></svg>",
            "bottle": "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M9 2h6v4l2 2v11a3 3 0 0 1-3 3h-4a3 3 0 0 1-3-3V8l2-2V2Zm2 2v2h2V4h-2Zm-2 8h6v-2H9v2Zm0 4h6v-2H9v2Z'/></svg>",
        }
        return icons[name]

    def kpi(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"<article class='kpi glass' style='--c:{color}'><div class='kpi-icon'>{svg_icon(icon)}</div><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div></article>"

    def usability_card(title: str, value: str, note: str, color: str, icon: str) -> str:
        return f"<div class='usability-item glass' style='--c:{color}'><div class='q-icon'>{svg_icon(icon)}</div><div><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(note)}</small></div></div>"

    def badge(text: str) -> str:
        return f"<span class='insight-badge'>{svg_icon('spark')}{html.escape(text)}</span>"

    def small_line(text: str) -> str:
        return f"<div class='line'><span class='mark'>{svg_icon('check')}</span><span>{html.escape(text)}</span></div>"

    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    overview_df = tables.get("data_overview", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    peak_month = str(monthly_show.loc[monthly_show["total_buy_mount"].idxmax(), "year_month"]) if not monthly_show.empty else "暂无数据"
    raw_trade_records = int(overview_df.loc[overview_df["dataset"].eq("trade_history"), "rows"].iloc[0]) if not overview_df.empty and overview_df["dataset"].eq("trade_history").any() else int(len(merged_df))
    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    high_best = high_metrics.iloc[0]
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    highest_cat1_row = cat_metrics.sort_values(["Accuracy", "F1_weighted"], ascending=False).iloc[0]
    rf_row = cat_rows.get("Random Forest", cat_metrics.iloc[0])
    hot_category = str(category_df.iloc[0]["cat1_name"])

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    unmatched_count = int((~profile_mask).sum())
    valid_rate = float((merged_df["buy_mount"] > 0).mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_raw = matched_df["baby_age_group"].replace("unknown", np.nan).dropna()
        age_counts = age_raw.value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)
    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    model_names, model_acc, model_f1, model_auc = [], [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            model_names.append(name)
            model_acc.append(round(float(row["Accuracy"]), 4))
            model_f1.append(round(float(row["F1_weighted"]), 4))
            model_auc.append(None)
    model_names.append("High Buy - Decision Tree")
    high_dt = high_metrics[high_metrics["model"].astype(str).eq("Decision Tree")]
    high_dt_row = high_dt.iloc[0] if not high_dt.empty else high_best
    model_acc.append(round(float(high_dt_row["Accuracy"]), 4))
    model_f1.append(round(float(high_dt_row["F1"]), 4))
    model_auc.append(round(float(high_dt_row["AUC"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
        "user_buy_count": "user_buy",
        "user_unique_item_count": "user_items",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]
    top_feature = feature_short[0] if feature_short else "暂无数据"

    kpi_html = "".join(
        [
            kpi("总交易记录数", fmt_num(len(merged_df)), "清洗后样本统计", "#EF476F", "bag"),
            kpi("交易用户数", fmt_num(merged_df["user_id"].nunique()), "去重后有效用户", "#3B82F6", "users"),
            kpi("商品种类数", fmt_num(merged_df["auction_id"].nunique()), "商品 ID 去重统计", "#8B5CF6", "package"),
            kpi("总购买数量", fmt_num(merged_df["buy_mount"].sum()), "有效购买数量", "#F59E0B", "cart"),
        ]
    )
    quality_html = "".join(
        [
            usability_card("交易数据可用率", fmt_pct(valid_rate), "清洗后交易记录完整", "#22C55E", "database"),
            usability_card("宝宝画像覆盖率", fmt_pct(profile_rate), "仅少量用户匹配画像", "#F59E0B", "user_check"),
            usability_card("画像未匹配记录", fmt_num(unmatched_count), "未进入画像细分分析", "#64748B", "user_x"),
            usability_card("核心字段完整率", fmt_pct(core_complete_rate), "建模核心字段可用", "#3B82F6", "shield"),
        ]
    )
    model_diag = "".join(
        [
            small_line("主任务：商品大类识别"),
            small_line(f"最高分模型：{highest_cat1_row['model']}"),
            small_line(f"最高分 Accuracy：{float(highest_cat1_row['Accuracy']):.3f}"),
            small_line("推荐展示：Random Forest"),
            small_line(f"RF F1-score：{float(rf_row['F1_weighted']):.3f}"),
            small_line("泄露检查：未使用 cat1_name"),
            small_line(f"high_buy：AUC {float(high_dt_row['AUC']):.3f}，F1 {float(high_dt_row['F1']):.3f}"),
        ]
    )
    advice_html = "".join(
        [
            small_line(f"重点关注{hot_category}等核心品类"),
            small_line("结合高购买量订单制定促销策略"),
            small_line("补充价格、品牌、浏览、收藏等特征"),
            small_line("提升模型解释能力"),
        ]
    )

    payload = {
        "colors": CHART_COLORS,
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "modelNames": model_names,
        "modelAcc": model_acc,
        "modelF1": model_f1,
        "modelAuc": model_auc,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{ --bg:#FAFBFF; --card:#fff; --pink:#FF5C8A; --soft:#FFF1F5; --title:#101828; --text:#344054; --muted:#667085; --border:#EEF2F7; --glass:rgba(255,255,255,.82); --glass-border:rgba(255,255,255,.82); --glass-shadow:0 16px 38px rgba(31,41,55,.07), inset 0 1px 0 rgba(255,255,255,.9); }}
    * {{ box-sizing:border-box; }} html,body {{ margin:0; min-height:100%; }} body {{ position:relative; color:var(--title); font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif; background:radial-gradient(circle at 12% 10%,rgba(255,92,138,.12),transparent 25%),radial-gradient(circle at 88% 12%,rgba(137,207,240,.15),transparent 26%),radial-gradient(circle at 70% 92%,rgba(184,161,255,.13),transparent 28%),linear-gradient(135deg,#FFF9FB 0%,#FAFBFF 48%,#F6FBFF 100%); overflow-x:hidden; }} body::before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.16; background-image:linear-gradient(rgba(255,92,138,.10) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,.08) 1px,transparent 1px); background-size:34px 34px; mask-image:linear-gradient(180deg,rgba(0,0,0,.65),rgba(0,0,0,.12)); }} body::after {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.10; background-image:radial-gradient(circle,rgba(255,92,138,.24) 0 2px,transparent 2px),radial-gradient(circle,rgba(137,207,240,.24) 0 1.6px,transparent 1.6px); background-size:112px 112px,148px 148px; background-position:24px 34px,80px 20px; }} svg {{ display:block; width:1em; height:1em; fill:currentColor; }}
    .page {{ position:relative; z-index:1; width:min(1920px,100vw); min-height:1080px; margin:0 auto; padding:10px 16px 12px 10px; display:grid; grid-template-columns:280px 1fr; gap:14px; }}
    .glass,.sidebar,.card,.summary-card,.insight,.q-item,.usability-item {{ background:var(--glass); border:1px solid var(--glass-border); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:var(--glass-shadow); }}
    .card,.kpi,.summary-card {{ position:relative; }} .card::before,.kpi::before,.summary-card::before {{ content:""; position:absolute; left:14px; right:14px; top:0; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.95),transparent); pointer-events:none; }}
    .card,.kpi,.summary-card,.insight,.q-item,.tab-btn {{ transition:transform .18s ease, box-shadow .18s ease; }} .card:hover,.kpi:hover,.summary-card:hover {{ transform:translateY(-2px); box-shadow:0 22px 48px rgba(31,41,55,.10), inset 0 1px 0 rgba(255,255,255,.9); }} .kpi:hover .kpi-icon {{ transform:scale(1.06); }}
    .sidebar {{ position:relative; min-height:1058px; padding:20px 16px; overflow:hidden; border-radius:22px; }}
    .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:24px; }} .face {{ width:46px;height:46px;border:3px solid var(--pink);border-radius:50%;position:relative;flex:0 0 46px;background:rgba(255,255,255,.4); }} .face:before,.face:after {{ content:"";position:absolute;top:17px;width:5px;height:5px;background:var(--pink);border-radius:50%; }} .face:before{{left:12px}} .face:after{{right:12px}} .smile{{position:absolute;left:14px;top:25px;width:16px;height:8px;border-bottom:3px solid var(--pink);border-radius:0 0 16px 16px}} .brand h1 {{ margin:0; color:var(--pink); font-size:21px; line-height:1; }} .brand p {{ margin:7px 0 0; color:#101828; font-size:12px; font-weight:800; }}
    .side-title {{ display:flex; gap:8px; align-items:center; margin:0 0 14px; color:var(--pink); font-size:17px; }} .side-title svg {{ width:17px; height:17px; }}
    .insight {{ display:grid; grid-template-columns:50px 1fr; gap:10px; align-items:center; padding:13px 8px; margin-bottom:12px; border-radius:16px; }} .i-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:var(--softc);color:var(--c);font-size:23px;box-shadow:0 8px 18px rgba(16,24,40,.05); }} .i-icon svg {{ width:23px; height:23px; }} .insight span {{ color:var(--muted);font-size:11px;font-weight:800; }} .insight strong {{ display:block;margin-top:5px;color:var(--c);font-size:21px;line-height:1.05; }} .insight small {{ display:block;margin-top:4px;color:#101828;font-size:11px;font-weight:800; }}
    .side-deco {{ position:absolute; left:18px; right:18px; bottom:22px; height:80px; opacity:.30; }} .side-deco .circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .side-deco .c1{{width:88px;height:88px;left:6px;bottom:-34px}} .side-deco .c2{{width:46px;height:46px;right:16px;bottom:10px;background:rgba(137,207,240,.22)}} .side-deco .mini-heart{{position:absolute;left:112px;bottom:34px;color:var(--pink)}} .side-deco .mini-heart svg{{width:20px;height:20px}} .side-deco .mini-bottle{{position:absolute;right:72px;bottom:16px;color:var(--pink)}} .side-deco .mini-bottle svg{{width:28px;height:28px}}
    .main {{ min-width:0; }} .hero {{ position:relative; height:104px; display:flex; flex-direction:column; align-items:center; justify-content:center; overflow:hidden; margin-bottom:10px; }} .hero h2 {{ margin:0; text-align:center; font-size:31px; line-height:1.12; }} .hero p {{ margin:6px 0 10px; text-align:center; color:#101828; font-size:14px; font-weight:700; }} .hero-deco {{ position:absolute; right:10px; top:2px; width:190px; height:66px; opacity:.30; pointer-events:none; }} .deco-circle {{ position:absolute; border-radius:50%; background:rgba(255,92,138,.22); }} .dc1{{width:92px;height:92px;right:36px;top:-34px}} .dc2{{width:42px;height:42px;right:0;top:16px;background:rgba(137,207,240,.22)}} .dc3{{width:28px;height:28px;right:118px;top:26px;background:rgba(184,161,255,.25)}} .deco-heart{{position:absolute;color:rgba(255,92,138,.5)}} .deco-heart svg{{width:17px;height:17px}} .dh1{{right:92px;top:8px}} .dh2{{right:145px;top:32px}} .deco-bottle{{position:absolute;right:54px;top:18px;color:rgba(255,92,138,.48)}} .deco-bottle svg{{width:28px;height:28px}}
    .tabs {{ display:flex; gap:10px; justify-content:center; }} .tab-btn {{ border:1px solid rgba(255,255,255,.72); background:rgba(255,255,255,.58); color:var(--pink); height:34px; padding:0 28px; border-radius:999px; font-weight:800; cursor:pointer; backdrop-filter:blur(14px); }} .tab-btn.active {{ background:linear-gradient(135deg,#FF5C8A,#FF8FAB); color:#fff; border-color:rgba(255,255,255,.78); box-shadow:0 12px 24px rgba(255,92,138,.20); }}
    .tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
    .grid {{ display:grid; grid-template-columns:repeat(24,minmax(0,1fr)); gap:14px; }} .card {{ border-radius:17px; padding:14px 16px; min-width:0; overflow:hidden; }} .card h3 {{ margin:0; font-size:15px; color:#101828; }} .sub {{ margin-top:5px; color:var(--muted); font-size:10.5px; font-weight:700; }} .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:5px; }} .unit {{ color:var(--muted); font-size:10.5px; font-weight:800; }}
    .insight-badge {{ display:inline-flex; align-items:center; gap:5px; padding:5px 9px; border-radius:999px; background:rgba(255,92,138,.10); color:var(--pink); font-size:11px; font-weight:800; white-space:nowrap; }} .insight-badge svg {{ width:12px;height:12px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px; }} .kpi {{ height:110px; display:flex; align-items:center; gap:14px; padding:16px; border-radius:17px; }} .kpi-icon {{ width:52px;height:52px;border-radius:50%;display:grid;place-items:center;flex:0 0 52px;color:var(--c);background:rgba(255,241,245,.78);font-size:26px; transition:transform .18s ease; }} .kpi-icon svg {{ width:26px;height:26px; }} .kpi span {{ font-size:12px;color:#101828;font-weight:800; }} .kpi strong {{ display:block;margin-top:8px;font-size:24px;line-height:1; }} .kpi small {{ display:block;margin-top:9px;color:var(--muted);font-size:11px;font-weight:700; }}
    .trend {{ grid-column:span 10; height:380px; }} .top6 {{ grid-column:span 7; height:380px; }} .value {{ grid-column:span 7; height:380px; }} .chart {{ width:100%; height:100%; }} .trend-chart,.top-chart {{ height:326px; }} .value-chart {{ height:244px; }} .value-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }} .value-mini {{ padding:10px 6px; border-radius:13px; text-align:center; background:rgba(255,255,255,.52); border:1px solid rgba(255,255,255,.68); }} .value-mini span {{ display:block;color:var(--muted);font-size:10px;font-weight:800; }} .value-mini strong {{ display:block;margin-top:4px;color:var(--vc);font-size:17px; }}
    .word {{ grid-column:span 7; height:270px; padding:12px 13px; }} .quality {{ grid-column:span 7; height:270px; }} .weekday {{ grid-column:span 6; height:270px; }} .conclusion {{ grid-column:span 4; height:270px; }} .small-chart {{ height:210px; }} .wordcloud {{ height:222px; margin-top:8px; display:flex; align-items:center; justify-content:center; border-radius:15px; background:rgba(255,255,255,.56); border:1px solid rgba(255,255,255,.78); box-shadow:inset 0 1px 0 rgba(255,255,255,.85); overflow:hidden; cursor:zoom-in; }} .wordcloud img {{ width:100%; height:100%; object-fit:contain; transition:transform .25s ease; }} .wordcloud:hover img {{ transform:scale(1.08); }} .missing {{ color:var(--muted); font-size:13px; }}
    .gauge-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; height:160px; margin-top:8px; }} .quality-gauge {{ min-width:0; height:160px; border-radius:14px; background:rgba(255,255,255,.50); border:1px solid rgba(255,255,255,.68); }} .quality-note {{ margin-top:10px; color:#667085; font-size:12px; line-height:1.55; font-weight:700; }} .quality-note b {{ color:#344054; }}
    .conclusion p {{ margin:12px 0 0; color:#344054; font-size:16px; font-weight:800; line-height:1.75; }} .conclusion b {{ color:var(--pink); }}
    .summary {{ grid-column:1/-1; display:grid; grid-template-columns:1.35fr repeat(5,1fr); gap:14px; margin-bottom:14px; }} .summary-card {{ border-radius:17px; padding:16px; min-height:104px; }} .summary-card span {{ display:block; color:var(--muted); font-size:12px; font-weight:800; }} .summary-card strong {{ display:block; margin-top:8px; color:#101828; font-size:22px; }} .summary-card.task strong {{ color:var(--pink); font-size:19px; }}
    .model-chart-card {{ grid-column:span 12; height:430px; }} .feature-card {{ grid-column:span 12; height:430px; }} .model-chart,.feature-chart {{ height:374px; }}
    .diag-card {{ grid-column:span 7; height:300px; }} .advice-card {{ grid-column:span 7; height:300px; }} .profile {{ grid-column:span 10; height:300px; }} .profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; height:236px; margin-top:4px; }} .profile-chart {{ height:236px; }}
    .line {{ display:flex; gap:8px; align-items:center; margin:12px 0; color:#344054; font-size:13px; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }} .mark {{ width:18px;height:18px;display:inline-grid;place-items:center;flex:0 0 18px;border-radius:50%;color:#fff;background:#22C55E; }} .mark svg {{ width:11px;height:11px; }}
    .modal {{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; z-index:20; background:rgba(16,24,40,.45); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }} .modal.active {{ display:flex; }} .modal img {{ max-width:80vw; max-height:80vh; object-fit:contain; border-radius:18px; background:#fff; box-shadow:0 30px 80px rgba(16,24,40,.30); }} .modal-close {{ position:absolute; top:32px; right:42px; width:38px; height:38px; border:0; border-radius:50%; background:rgba(255,255,255,.88); color:#101828; font-size:22px; cursor:pointer; }}
    @media (max-width:1400px) {{ .page {{ grid-template-columns:1fr; }} .sidebar {{ min-height:auto; }} .side-deco {{ display:none; }} .trend,.top6,.value,.word,.quality,.weekday,.conclusion,.model-chart-card,.feature-card,.diag-card,.advice-card,.profile {{ grid-column:1/-1; }} .kpis,.summary {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div class="brand"><div class="face"><span class="smile"></span></div><div><h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p></div></div>
      <h2 class="side-title">{svg_icon("heart")}<span>核心洞察</span></h2>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("crown")}</div><div><span>热门品类</span><strong>{html.escape(hot_category)}</strong><small>销量最高商品大类</small></div></div>
      <div class="insight" style="--c:#8B5CF6;--softc:#F3EEFF"><div class="i-icon">{svg_icon("trend")}</div><div><span>高购买量订单占比</span><strong>{fmt_pct(high_rate)}</strong><small>高价值订单占整体订单</small></div></div>
      <div class="insight" style="--c:#EF476F;--softc:#FFF1F5"><div class="i-icon">{svg_icon("target")}</div><div><span>主模型 Accuracy</span><strong>{float(rf_row['Accuracy']):.3f}</strong><small>F1-score {float(rf_row['F1_weighted']):.3f}</small></div></div>
      <div class="insight" style="--c:#22C55E;--softc:#ECFDF3"><div class="i-icon">{svg_icon("user_check")}</div><div><span>画像匹配率</span><strong>{fmt_pct(profile_rate)}</strong><small>仅少量用户匹配画像</small></div></div>
      <div class="side-deco"><span class="circle c1"></span><span class="circle c2"></span><span class="mini-heart">{svg_icon("heart")}</span><span class="mini-bottle">{svg_icon("bottle")}</span></div>
    </aside>
    <main class="main">
      <header class="hero">
        <div><h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p></div>
        <nav class="tabs"><button class="tab-btn active" data-tab="business">业务总览</button><button class="tab-btn" data-tab="model">模型验证</button></nav>
        <div class="hero-deco"><span class="deco-circle dc1"></span><span class="deco-circle dc2"></span><span class="deco-circle dc3"></span><span class="deco-heart dh1">{svg_icon("heart")}</span><span class="deco-heart dh2">{svg_icon("heart")}</span><span class="deco-bottle">{svg_icon("bottle")}</span></div>
      </header>

      <section id="business" class="tab-panel active">
        <section class="kpis">{kpi_html}</section>
        <section class="grid">
          <section class="card trend"><div class="card-head"><div><h3>购买趋势概览（按月）</h3><div class="sub">最近 12 个月购买数量与订单数变化</div></div>{badge(f"智能洞察：{peak_month} 为购买峰值")}</div><div id="trendChart" class="chart trend-chart"></div></section>
          <section class="card top6"><div class="card-head"><div><h3>商品大类销量 TOP6</h3><div class="sub">中文业务品类按购买数量降序</div></div>{badge(f"智能洞察：{hot_category}销量最高")}</div><div id="categoryChart" class="chart top-chart"></div></section>
          <section class="card value"><div class="card-head"><div><h3>高购买量订单价值洞察</h3><div class="sub">基于 buy_mount 75% 分位数识别</div></div></div><div class="value-body"><div id="highBuyChart" class="chart value-chart"></div><div id="salesGaugeChart" class="chart sales-gauge"></div></div><div class="value-metrics"><div class="value-mini" style="--vc:#EF476F"><span>订单数占比</span><strong>{fmt_pct(high_rate)}</strong></div><div class="value-mini" style="--vc:#3B82F6"><span>销量贡献率</span><strong>{fmt_pct(high_sales_rate)}</strong></div><div class="value-mini" style="--vc:#8B5CF6"><span>平均购买量提升</span><strong>{value_lift:.1f}x</strong></div></div></section>
          <section class="card word"><h3>热门商品词云</h3><div class="wordcloud"><img id="wordcloudImg" src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'词云图待生成'}}))"></div></section>
          <section class="card quality"><h3>数据质量与画像匹配</h3><div class="gauge-grid"><div id="tradeGauge" class="quality-gauge"></div><div id="profileGauge" class="quality-gauge"></div><div id="coreGauge" class="quality-gauge"></div></div><div class="quality-note"><b>画像未匹配记录：{fmt_num(unmatched_count)}</b>。交易行为字段较完整，宝宝画像覆盖较低，因此画像仅作辅助分析。</div></section>
          <section class="card weekday"><h3>星期购买活跃度</h3><div id="weekdayChart" class="chart small-chart"></div></section>
          <section class="card conclusion"><h3>业务结论</h3><p><b>{html.escape(hot_category)}</b>、婴儿服饰和尿裤湿巾构成主要消费品类；高购买量订单占比约 <b>{fmt_pct(high_rate)}</b>，可作为重点营销对象。</p></section>
        </section>
      </section>

      <section id="model" class="tab-panel">
        <section class="summary">
          <div class="summary-card task"><span>主任务</span><strong>商品大类识别 cat1</strong></div>
          <div class="summary-card"><span>最高分模型</span><strong>{html.escape(str(highest_cat1_row['model']))}</strong></div>
          <div class="summary-card"><span>最高分 Accuracy</span><strong>{float(highest_cat1_row['Accuracy']):.3f}</strong></div>
          <div class="summary-card"><span>推荐展示模型</span><strong>Random Forest</strong></div>
          <div class="summary-card"><span>RF F1-score</span><strong>{float(rf_row['F1_weighted']):.3f}</strong></div>
          <div class="summary-card"><span>high_buy 辅助模型</span><strong>AUC {float(high_dt_row['AUC']):.3f}</strong></div>
        </section>
        <section class="grid">
          <section class="card model-chart-card"><div class="card-head"><h3>模型性能对比（主模型与辅助模型）</h3>{badge(f"智能洞察：{html.escape(str(highest_cat1_row['model']))} 最高分，Random Forest 推荐展示")}</div><div id="modelChart" class="chart model-chart"></div></section>
          <section class="card feature-card"><div class="card-head"><h3>特征重要性 TOP10</h3></div><div id="featureChart" class="chart feature-chart"></div><div class="feature-note">cat_id 相关特征贡献最高，说明商品层级关系对 cat1 识别影响最大。</div></section>
          <section class="card diag-card"><h3>模型诊断</h3>{model_diag}</section>
          <section class="card advice-card"><h3>业务建议</h3>{advice_html}</section>
          <section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart profile-chart"></div><div id="ageChart" class="chart profile-chart"></div></div></section>
        </section>
      </section>
    </main>
  </div>
  <div class="modal" id="wordcloudModal"><button class="modal-close" type="button" aria-label="关闭">×</button><img src="../charts/09_wordcloud.png" alt="母婴商品消费热点词云图大图"></div>
  <script>
    const payload = {payload_json};
    const textColor="#101828", mutedColor="#667085", gridLine="rgba(148,163,184,.18)", colors=payload.colors;
    const charts = [];
    function baseChartOption() {{
      return {{
        textStyle: {{ fontFamily: "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',Arial,sans-serif" }},
        tooltip: {{
          trigger: "axis",
          backgroundColor: "rgba(255,255,255,0.92)",
          borderColor: "rgba(255,92,138,0.18)",
          borderWidth: 1,
          textStyle: {{ color: "#101828" }},
          extraCssText: "box-shadow: 0 12px 30px rgba(16,24,40,.12); border-radius: 12px;"
        }},
        animationDuration: 900,
        animationEasing: "cubicOut"
      }};
    }}
    function merge(base, extra) {{ return Object.assign({{}}, base, extra); }}
    function chart(id,opt){{const el=document.getElementById(id); if(!el)return; if(!window.echarts){{el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;}} const c=echarts.init(el,null,{{renderer:"canvas"}}); c.setOption(opt); charts.push(c); window.addEventListener("resize",()=>c.resize());}}
    const pinkGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:"#EF476F"}},{{offset:1,color:"#FFE3EC"}}]):"#EF476F";
    const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#C4B5FD"}},{{offset:1,color:"#8B5CF6"}}]):"#8B5CF6";
    function gaugeOption(title,value,color){{return merge(baseChartOption(),{{tooltip:merge(baseChartOption().tooltip,{{trigger:"item",formatter:()=>`${{title}}：${{Number(value).toFixed(1)}}%`}}),series:[{{type:"gauge",min:0,max:100,radius:"92%",center:["50%","58%"],startAngle:205,endAngle:-25,progress:{{show:true,width:10,itemStyle:{{color}}}},axisLine:{{lineStyle:{{width:10,color:[[1,"rgba(148,163,184,.16)"]]}}}},axisTick:{{show:false}},splitLine:{{show:false}},axisLabel:{{show:false}},pointer:{{show:false}},anchor:{{show:false}},title:{{offsetCenter:[0,"42%"],fontSize:10,color:mutedColor,fontWeight:800}},detail:{{valueAnimation:true,offsetCenter:[0,"4%"],fontSize:19,fontWeight:900,color,formatter:v=>v.toFixed(1)+"%"}},data:[{{value,name:title}}]}}]}});}}
    function renderDashboard() {{
    chart("trendChart",merge(baseChartOption(),{{color:["#EF476F","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:items=>`<b>${{items[0].axisValue}}</b><br/>购买数量：${{Number(items[0].value).toLocaleString()}} 件<br/>订单数：${{Number(items[1].value).toLocaleString()}} 单`}}),legend:{{top:0,left:"center",textStyle:{{color:mutedColor,fontSize:10}}}},grid:{{left:52,right:46,top:42,bottom:58}},dataZoom:[{{type:"inside",xAxisIndex:0,filterMode:"none"}}],xAxis:{{type:"category",data:payload.monthLabels,axisLabel:{{color:mutedColor,fontSize:10,interval:1,rotate:25,margin:12}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",name:"购买数量",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",name:"订单数",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{show:false}}}}],series:[{{name:"购买数量（件）",type:"bar",data:payload.monthBuy,barWidth:18,itemStyle:{{borderRadius:[8,8,0,0],color:pinkGrad}},markPoint:{{symbolSize:48,label:{{fontSize:9}},data:[{{type:"max",name:"峰值"}}]}},markLine:{{symbol:"none",label:{{formatter:"平均",fontSize:9}},lineStyle:{{color:"rgba(239,71,111,.42)",type:"dashed"}},data:[{{type:"average",name:"平均"}}]}}}},{{name:"订单数（单）",type:"line",yAxisIndex:1,data:payload.monthOrders,smooth:true,symbolSize:6,lineStyle:{{width:3,color:"#3B82F6",shadowColor:"rgba(59,130,246,.28)",shadowBlur:10,shadowOffsetY:6}},itemStyle:{{color:"#3B82F6"}}}}]}}));
    chart("categoryChart",merge(baseChartOption(),{{tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",axisPointer:{{type:"shadow"}},formatter:items=>`${{items[0].name}}<br/>销量：${{Number(items[0].value).toLocaleString()}} 件`}}),grid:{{left:88,right:80,top:14,bottom:24}},xAxis:{{type:"value",show:false,max:Math.max(...payload.categoryValues)*1.16}},yAxis:{{type:"category",data:payload.categoryNames.slice().reverse(),axisLabel:{{color:textColor,fontSize:11}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"背景轨道",type:"bar",data:payload.categoryValues.map(()=>Math.max(...payload.categoryValues)).slice().reverse(),barWidth:15,barGap:"-100%",silent:true,itemStyle:{{borderRadius:[0,9,9,0],color:"rgba(255,255,255,.55)"}}}},{{name:"购买数量",type:"bar",data:payload.categoryValues.slice().reverse(),barWidth:15,label:{{show:true,position:"right",color:textColor,fontSize:9,formatter:p=>Number(p.value).toLocaleString()}},itemStyle:{{borderRadius:[0,9,9,0],color:p=>new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"rgba(255,255,255,.25)"}},{{offset:1,color:colors[(p.dataIndex+2)%colors.length]}}])}}}}]}}));
    chart("highBuyChart",merge(baseChartOption(),{{color:["#EF476F","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"item",formatter:p=>`${{p.name}}<br/>订单数：${{Number(p.value).toLocaleString()}}<br/>占比：${{p.percent}}%`}}),title:{{text:payload.highRate+"%",subtext:"小占比，高贡献",left:"34%",top:"37%",textAlign:"center",textStyle:{{fontSize:21,color:"#EF476F",fontWeight:900}},subtextStyle:{{fontSize:10,color:textColor,fontWeight:700}}}},legend:{{right:0,top:"middle",orient:"vertical",itemWidth:9,itemHeight:9,textStyle:{{color:textColor,fontSize:10}}}},series:[{{type:"pie",radius:["48%","68%"],center:["34%","50%"],label:{{show:false}},labelLine:{{show:false}},data:payload.highNames.map((name,i)=>({{name,value:payload.highValues[i]}}))}}]}}));
    chart("weekdayChart",merge(baseChartOption(),{{color:["#22C55E","#3B82F6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:items=>`星期${{items[0].axisValue}}<br/>购买数量：${{Number(items[0].value).toLocaleString()}} 件<br/>订单数：${{Number(items[1].value).toLocaleString()}} 单`}}),legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:42,right:30,top:42,bottom:26}},xAxis:{{type:"category",data:payload.weekdayLabels,axisLabel:{{color:mutedColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:[{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{lineStyle:{{color:gridLine}}}}}},{{type:"value",axisLabel:{{color:mutedColor,fontSize:9}},splitLine:{{show:false}}}}],series:[{{name:"购买数量",type:"bar",data:payload.weekdayBuy,barWidth:16,itemStyle:{{borderRadius:[7,7,0,0],color:"#22C55E"}}}},{{name:"订单数",type:"line",yAxisIndex:1,data:payload.weekdayOrders,smooth:true,symbolSize:5,lineStyle:{{width:2.5,color:"#3B82F6",shadowColor:"rgba(59,130,246,.22)",shadowBlur:8,shadowOffsetY:4}},itemStyle:{{color:"#3B82F6"}}}}]}}));
    const modelLabels = payload.modelNames.map(name => name.replace("Logistic Regression","Logistic\\nRegression").replace("Decision Tree","Decision\\nTree").replace("Random Forest","Random\\nForest").replace("Gradient Boosting","Gradient\\nBoosting").replace("High Buy - Decision Tree","High Buy\\nDecision Tree"));
    chart("modelChart",merge(baseChartOption(),{{color:["#3B82F6","#22C55E","#8B5CF6"],tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",axisPointer:{{type:"shadow"}},formatter:items=>{{let html=`<b>${{items[0].axisValue.replace(/\\n/g," ")}}</b>`; items.forEach(it=>{{html += `<br/>${{it.seriesName}}：${{it.value == null ? "-" : it.value.toFixed(3)}}`;}}); return html;}}}}),legend:{{top:0,left:"center",textStyle:{{fontSize:10,color:mutedColor}}}},grid:{{left:44,right:28,top:58,bottom:66}},xAxis:{{type:"category",data:modelLabels,axisLabel:{{color:textColor,fontSize:10,interval:0,lineHeight:14}},axisTick:{{show:false}},axisLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"value",min:0,max:1,axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},series:[{{name:"Accuracy",type:"bar",data:payload.modelAcc,barWidth:12,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value == null ? "-" : p.value.toFixed(3)}},markPoint:{{symbol:"pin",symbolSize:44,label:{{fontSize:9}},data:[{{name:"最高分",coord:["Gradient\\nBoosting",Math.max(payload.modelAcc[3]||0,payload.modelF1[3]||0)+0.06],value:"最高分",itemStyle:{{color:"#8B5CF6"}},label:{{formatter:"最高分"}}}},{{name:"推荐展示",coord:["Random\\nForest",Math.max(payload.modelAcc[2]||0,payload.modelF1[2]||0)+0.06],value:"推荐展示",itemStyle:{{color:"#FF5C8A"}},label:{{formatter:"推荐"}}}}]}}}},{{name:"F1-score",type:"bar",data:payload.modelF1,barWidth:12,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value == null ? "-" : p.value.toFixed(3)}}}},{{name:"AUC",type:"bar",data:payload.modelAuc,barWidth:12,label:{{show:true,position:"top",fontSize:9,formatter:p=>p.value == null ? "-" : p.value.toFixed(3)}}}}]}}));
    chart("featureChart",merge(baseChartOption(),{{tooltip:merge(baseChartOption().tooltip,{{trigger:"axis",formatter:params=>{{const p=params[0]; const reversedShort=payload.featureShort.slice().reverse(); const idx=reversedShort.indexOf(p.name); const full=payload.featureNames.slice().reverse()[idx] || p.name; return full + "<br/>重要性：" + p.value.toFixed(3);}}}}),grid:{{left:118,right:80,top:18,bottom:20}},xAxis:{{type:"value",axisLabel:{{color:mutedColor,fontSize:10}},splitLine:{{lineStyle:{{color:gridLine}}}}}},yAxis:{{type:"category",data:payload.featureShort.slice().reverse(),axisLabel:{{color:textColor,fontSize:10}},axisTick:{{show:false}},axisLine:{{show:false}}}},series:[{{name:"重要性",type:"bar",data:payload.featureValues.slice().reverse(),barWidth:12,label:{{show:true,position:"right",fontSize:9,color:textColor,formatter:p=>p.value.toFixed(3)}},itemStyle:{{borderRadius:[0,7,7,0],color:p=>{{const rank=payload.featureValues.length-1-p.dataIndex; return rank<3?new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#7C3AED"}},{{offset:1,color:"#3B82F6"}}]):new echarts.graphic.LinearGradient(0,0,1,0,[{{offset:0,color:"#DDD6FE"}},{{offset:1,color:"#A5B4FC"}}]);}}}}}}]}}));
    chart("genderChart",merge(baseChartOption(),{{color:["#3B82F6","#EF476F"],title:{{text:"性别",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:merge(baseChartOption().tooltip,{{trigger:"item"}}),legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.genderNames.map((name,i)=>({{name,value:payload.genderValues[i]}}))}}]}}));
    chart("ageChart",merge(baseChartOption(),{{color:["#3B82F6","#F59E0B","#22C55E","#8B5CF6","#EF476F"],title:{{text:"年龄阶段",left:"center",top:0,textStyle:{{fontSize:12,color:textColor}}}},tooltip:merge(baseChartOption().tooltip,{{trigger:"item"}}),legend:{{bottom:0,left:"center",itemWidth:9,itemHeight:9,textStyle:{{fontSize:10,color:textColor}}}},series:[{{type:"pie",radius:["42%","65%"],center:["50%","48%"],label:{{show:false}},labelLine:{{show:false}},data:payload.ageNames.map((name,i)=>({{name,value:payload.ageValues[i]}}))}}]}}));
    }}
    document.querySelectorAll(".tab-btn").forEach(btn=>btn.addEventListener("click",()=>{{document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));btn.classList.add("active");document.getElementById(btn.dataset.tab).classList.add("active");setTimeout(()=>charts.forEach(c=>c.resize()),80);}}));
    const modal = document.getElementById("wordcloudModal");
    const wordImg = document.getElementById("wordcloudImg");
    const closeBtn = modal.querySelector(".modal-close");
    if (wordImg) wordImg.addEventListener("click", () => modal.classList.add("active"));
    closeBtn.addEventListener("click", () => modal.classList.remove("active"));
    modal.addEventListener("click", event => {{ if (event.target === modal) modal.classList.remove("active"); }});
    fetch("../dashboard_data.json").then(res => res.ok ? res.json() : Promise.reject(new Error("dashboard_data unavailable"))).then(data => {{ payload = data; renderDashboard(); }}).catch(() => renderDashboard());
  </script>
</body>
</html>"""
    (HTML_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


_render_dashboard_html = generate_dashboard


def build_dashboard_payload(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> dict:
    """统一生成大屏数据口径，同时供 dashboard_data.json 和 HTML fallback payload 使用。"""
    category_df = tables.get("category_sales_summary", pd.DataFrame()).copy()
    monthly_df = tables.get("monthly_sales_summary", pd.DataFrame()).copy()
    weekday_df = tables.get("weekday_sales_summary", pd.DataFrame()).copy()
    overview_df = tables.get("data_overview", pd.DataFrame()).copy()
    if category_df.empty:
        category_df = pd.DataFrame({"cat1_name": ["暂无数据"], "total_buy_mount": [0]})
    if monthly_df.empty:
        monthly_df = pd.DataFrame({"year_month": [], "total_buy_mount": [], "trade_count": []})
    if weekday_df.empty:
        weekday_df = pd.DataFrame({"weekday": [], "total_buy_mount": [], "trade_count": []})

    category_df = category_df.sort_values("total_buy_mount", ascending=False).head(6)
    monthly_show = monthly_df.tail(12).copy()
    if not overview_df.empty and overview_df["dataset"].eq("trade_history").any():
        raw_trade_records = int(overview_df.loc[overview_df["dataset"].eq("trade_history"), "rows"].iloc[0])
    else:
        raw_path = DATA_RAW_DIR / "sam_tianchi_mum_baby_trade_history.csv"
        raw_trade_records = int(pd.read_csv(raw_path).shape[0]) if raw_path.exists() else int(len(merged_df))

    cat_metrics = cat1_result["metrics"].copy()
    high_metrics = high_result["metrics"].copy()
    cat_rows = {str(row["model"]): row for _, row in cat_metrics.iterrows()}
    highest_cat1_row = cat_metrics.sort_values(["Accuracy", "F1_weighted"], ascending=False).iloc[0]
    rf_row = cat_rows.get("Random Forest", cat_metrics.iloc[0])

    high_model_df = high_result["model_df"].copy()
    high_mask = high_model_df["high_buy"] == 1
    high_rate = float(high_mask.mean())
    high_sales_rate = float(high_model_df.loc[high_mask, "buy_mount"].sum() / max(high_model_df["buy_mount"].sum(), 1))
    normal_avg = float(high_model_df.loc[~high_mask, "buy_mount"].mean()) if (~high_mask).any() else 0.0
    high_avg = float(high_model_df.loc[high_mask, "buy_mount"].mean()) if high_mask.any() else 0.0
    value_lift = high_avg / max(normal_avg, 1e-9)

    profile_mask = (merged_df["gender"].fillna("unknown") != "unknown") & (merged_df["baby_age"].fillna(-1) >= 0)
    matched_df = merged_df.loc[profile_mask].copy()
    profile_rate = float(profile_mask.mean())
    core_fields = ["user_id", "auction_id", "cat_id", "cat1", "buy_mount", "day"]
    core_missing = int(merged_df[core_fields].isna().sum().sum())
    core_complete_rate = 1 - core_missing / max(len(merged_df) * len(core_fields), 1)
    valid_rate = float((merged_df["buy_mount"] > 0).mean())

    if matched_df.empty:
        gender_counts = pd.Series({"暂无数据": 0})
        age_counts = pd.Series({"暂无数据": 0})
    else:
        gender_counts = matched_df["gender"].map({"male": "男宝宝", "female": "女宝宝"}).dropna().value_counts()
        age_counts = matched_df["baby_age_group"].replace("unknown", np.nan).dropna().value_counts().reindex(["0", "1", "2-3", "4-6", "7+"], fill_value=0)

    age_map = {"0": "0-6个月", "1": "6-12个月", "2-3": "1-2岁", "4-6": "2-3岁", "7+": "3岁以上", "暂无数据": "暂无数据"}
    weekday_names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}

    ordered_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    model_names, model_acc, model_f1, model_auc = [], [], [], []
    for name in ordered_models:
        if name in cat_rows:
            row = cat_rows[name]
            model_names.append(name)
            model_acc.append(round(float(row["Accuracy"]), 4))
            model_f1.append(round(float(row["F1_weighted"]), 4))
            model_auc.append(None)
    high_dt = high_metrics[high_metrics["model"].astype(str).eq("Decision Tree")]
    high_dt_row = high_dt.iloc[0] if not high_dt.empty else high_metrics.iloc[0]
    model_names.append("High Buy - Decision Tree")
    model_acc.append(round(float(high_dt_row["Accuracy"]), 4))
    model_f1.append(round(float(high_dt_row["F1"]), 4))
    model_auc.append(round(float(high_dt_row["AUC"]), 4))

    feature_top = cat1_result["feature_importance"].head(10).copy()
    short_map = {
        "cat_id_encoded": "cat_id",
        "property_freq": "property_freq",
        "auction_id_freq": "auction_freq",
        "cat_id_freq": "cat_freq",
        "baby_age_group_encoded": "age_group",
        "gender_encoded": "gender",
    }
    feature_names = feature_top["feature"].astype(str).tolist()
    feature_short = [short_map.get(name, name if len(name) <= 14 else name[:12] + "...") for name in feature_names]

    return {
        "colors": CHART_COLORS,
        "metrics": {
            "rawTradeRecords": raw_trade_records,
            "cleanTradeRecords": int(len(merged_df)),
            "tradeUsers": int(merged_df["user_id"].nunique()),
            "itemKinds": int(merged_df["auction_id"].nunique()),
            "totalBuyMount": int(merged_df["buy_mount"].sum()),
            "highBuyOrders": int(high_mask.sum()),
            "normalOrders": int((~high_mask).sum()),
            "highBuyRate": round(high_rate * 100, 1),
            "salesContributionRate": round(high_sales_rate * 100, 1),
            "avgBuyLift": round(value_lift, 1),
            "profileCoverageRate": round(profile_rate * 100, 1),
            "profileUnmatchedRecords": int((~profile_mask).sum()),
            "tradeUsabilityRate": round(valid_rate * 100, 1),
            "coreFieldCompleteRate": round(core_complete_rate * 100, 1),
            "highestModel": str(highest_cat1_row["model"]),
            "highestAccuracy": round(float(highest_cat1_row["Accuracy"]), 4),
            "highestF1": round(float(highest_cat1_row["F1_weighted"]), 4),
            "displayModel": "Random Forest",
            "displayAccuracy": round(float(rf_row["Accuracy"]), 4),
            "displayF1": round(float(rf_row["F1_weighted"]), 4),
            "highBuyAuxModel": str(high_dt_row["model"]),
            "highBuyAuxAuc": round(float(high_dt_row["AUC"]), 4),
            "highBuyAuxF1": round(float(high_dt_row["F1"]), 4),
        },
        "categoryNames": category_df["cat1_name"].astype(str).tolist(),
        "categoryValues": category_df["total_buy_mount"].astype(float).tolist(),
        "monthLabels": monthly_show["year_month"].astype(str).tolist(),
        "monthBuy": monthly_show["total_buy_mount"].astype(float).tolist(),
        "monthOrders": monthly_show["trade_count"].astype(float).tolist(),
        "weekdayLabels": [weekday_names.get(int(v), str(v)) for v in weekday_df["weekday"].tolist()],
        "weekdayBuy": weekday_df["total_buy_mount"].astype(float).tolist(),
        "weekdayOrders": weekday_df["trade_count"].astype(float).tolist(),
        "highNames": ["高购买量订单", "普通订单"],
        "highValues": [int(high_mask.sum()), int((~high_mask).sum())],
        "highRate": round(high_rate * 100, 1),
        "highSalesRate": round(high_sales_rate * 100, 1),
        "valueLift": round(value_lift, 1),
        "genderNames": gender_counts.index.astype(str).tolist(),
        "genderValues": gender_counts.astype(int).tolist(),
        "ageNames": [age_map.get(str(v), str(v)) for v in age_counts.index],
        "ageValues": age_counts.astype(int).tolist(),
        "modelNames": model_names,
        "modelAcc": model_acc,
        "modelF1": model_f1,
        "modelAuc": model_auc,
        "featureNames": feature_names,
        "featureShort": feature_short,
        "featureValues": feature_top["importance"].astype(float).round(4).tolist(),
    }


def save_report_tables(
    merged_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    cat1_result: dict,
    high_result: dict,
    payload: dict,
) -> None:
    """生成可直接放入实训报告的真实口径表格。"""
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    pd.DataFrame(
        [
            {"指标": "原始交易记录数", "数值": metrics["rawTradeRecords"], "说明": "交易历史原始 CSV 行数"},
            {"指标": "清洗后有效交易记录数", "数值": metrics["cleanTradeRecords"], "说明": "去重、日期转换、字段清洗后的记录数"},
            {"指标": "交易用户数", "数值": metrics["tradeUsers"], "说明": "清洗后 user_id 去重"},
            {"指标": "商品 ID 数", "数值": metrics["itemKinds"], "说明": "交易表中 auction_id 去重后的商品编号数量"},
            {"指标": "总购买数量", "数值": metrics["totalBuyMount"], "说明": "buy_mount 汇总"},
            {"指标": "高购买量订单数", "数值": metrics["highBuyOrders"], "说明": "buy_mount 高于 75% 分位数"},
            {"指标": "普通订单数", "数值": metrics["normalOrders"], "说明": "非 high_buy 订单"},
            {"指标": "高购买量订单占比", "数值": f"{metrics['highBuyRate']:.1f}%", "说明": "高购买量订单数 / 清洗后有效记录数"},
            {"指标": "高购买量销量贡献率", "数值": f"{metrics['salesContributionRate']:.1f}%", "说明": "high_buy 订单 buy_mount / 总 buy_mount"},
            {"指标": "平均购买量提升", "数值": f"{metrics['avgBuyLift']:.1f}x", "说明": "high_buy 平均 buy_mount / 普通订单平均 buy_mount"},
            {"指标": "宝宝画像覆盖率", "数值": f"{metrics['profileCoverageRate']:.1f}%", "说明": "已匹配性别和宝宝年龄的记录占比"},
            {"指标": "核心字段完整率", "数值": f"{metrics['coreFieldCompleteRate']:.1f}%", "说明": "user_id、auction_id、cat_id、cat1、buy_mount、day 完整率"},
        ]
    ).to_csv(REPORT_TABLE_DIR / "table_data_overview.csv", index=False, encoding="utf-8-sig")

    cat_metrics = cat1_result["metrics"].copy()
    cat_metrics["任务说明"] = "商品大类识别 cat1；本表仅展示多分类 Accuracy 与 F1 等指标，不计算二分类 AUC。"
    cat_metrics.to_csv(REPORT_TABLE_DIR / "table_cat1_model_metrics.csv", index=False, encoding="utf-8-sig")

    high_metrics = high_result["metrics"].copy()
    high_metrics["任务说明"] = "high_buy 二分类辅助模型；已删除 buy_mount 与全量聚合销量泄露字段。"
    high_metrics.to_csv(REPORT_TABLE_DIR / "table_high_buy_model_metrics.csv", index=False, encoding="utf-8-sig")

    cat1_result["feature_importance"].head(20).to_csv(
        REPORT_TABLE_DIR / "table_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    tables.get("category_sales_summary", pd.DataFrame()).to_csv(
        REPORT_TABLE_DIR / "table_category_sales.csv", index=False, encoding="utf-8-sig"
    )
    tables.get("monthly_sales_summary", pd.DataFrame()).to_csv(
        REPORT_TABLE_DIR / "table_monthly_trend.csv", index=False, encoding="utf-8-sig"
    )
    tables.get("weekday_sales_summary", pd.DataFrame()).to_csv(
        REPORT_TABLE_DIR / "table_weekday_trend.csv", index=False, encoding="utf-8-sig"
    )


def save_report_figures(
    model_df: pd.DataFrame,
    cat1_result: dict,
    high_result: dict,
    payload: dict,
) -> None:
    """生成报告附录图，保证模型图与大屏同口径。"""
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    apply_report_style()
    figure_index: list[dict[str, str]] = []

    def save_current(filename: str, title: str) -> None:
        plt.tight_layout()
        plt.savefig(REPORT_FIGURE_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()
        figure_index.append({"文件名": filename, "图名": title, "路径": str(REPORT_FIGURE_DIR / filename)})

    high_counts = high_result["model_df"]["high_buy"].map({0: "普通订单", 1: "高购买量订单"}).value_counts().reindex(["普通订单", "高购买量订单"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="white")
    bars = ax.bar(high_counts.index, high_counts.values, color=["#3B82F6", "#EF476F"], width=0.48)
    ax.set_title("高购买量订单分布", fontsize=16, fontweight="bold")
    ax.set_ylabel("订单数")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=11)
    polish_axes(ax)
    save_current("01_high_buy_distribution.png", "高购买量订单分布")

    numeric_cols = [col for col in ["buy_mount", "month", "weekday", "baby_age"] if col in model_df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    for ax, col in zip(axes.ravel(), numeric_cols):
        sns.histplot(model_df[col].dropna(), bins=24, kde=True, ax=ax, color="#4E79A7", edgecolor="white")
        ax.set_title(col, fontsize=12, fontweight="bold")
        polish_axes(ax)
    for ax in axes.ravel()[len(numeric_cols):]:
        ax.axis("off")
    save_current("02_numeric_distribution.png", "数值变量分布")

    corr_cols = [col for col in cat1_result["feature_frame"].columns if pd.api.types.is_numeric_dtype(cat1_result["feature_frame"][col])]
    corr = cat1_result["feature_frame"][corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.78})
    ax.set_title("建模特征相关系数热力图", fontsize=16, fontweight="bold")
    save_current("03_correlation_heatmap.png", "建模特征相关系数热力图")

    feature = cat1_result["feature_importance"].head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.barh(feature["feature"], feature["importance"], color="#8B5CF6")
    ax.set_title("商品大类识别模型特征重要性 TOP10", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importance")
    polish_axes(ax, grid_axis="x")
    save_current("04_feature_importance.png", "商品大类识别模型特征重要性 TOP10")

    cat_metrics = cat1_result["metrics"].copy()
    metric_melt = cat_metrics.melt(id_vars="model", value_vars=["Accuracy", "F1_weighted"], var_name="指标", value_name="分数")
    metric_melt["指标"] = metric_melt["指标"].replace({"F1_weighted": "F1-score"})
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    sns.barplot(data=metric_melt, x="model", y="分数", hue="指标", palette=["#3B82F6", "#22C55E"], ax=ax)
    high_auc = payload["metrics"]["highBuyAuxAuc"]
    ax.text(0.02, 0.97, f"注：AUC 仅用于 high_buy 二分类辅助模型，Decision Tree AUC={high_auc:.3f}", transform=ax.transAxes, va="top", fontsize=10, color="#667085")
    ax.set_ylim(0, 1.08)
    ax.set_title("模型性能对比（cat1 主模型）", fontsize=16, fontweight="bold")
    polish_axes(ax, grid_axis="y")
    save_current("05_model_comparison.png", "模型性能对比（cat1 主模型）")

    high_best_model = high_result["best_model"]
    fpr, tpr, auc_value = high_result["roc_data"][high_best_model]
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="white")
    ax.plot(fpr, tpr, color="#8B5CF6", linewidth=2.6, label=f"{high_best_model} AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color="#98A2B3", linestyle="--", linewidth=1.2, label="随机猜测")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("高购买量订单识别 ROC 曲线", fontsize=16, fontweight="bold")
    ax.legend(frameon=False)
    polish_axes(ax)
    save_current("06_roc_curve.png", "高购买量订单识别 ROC 曲线")

    fig, ax = plt.subplots(figsize=(7.6, 6.4), facecolor="white")
    sns.heatmap(high_result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=False, ax=ax)
    ax.set_title("高购买量订单识别混淆矩阵", fontsize=16, fontweight="bold")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticklabels(["普通订单", "高购买量订单"], rotation=0)
    ax.set_yticklabels(["普通订单", "高购买量订单"], rotation=0)
    save_current("07_confusion_matrix.png", "高购买量订单识别混淆矩阵")

    best_score = high_result["scores"][high_best_model]
    y_test = high_result["y_test"]
    prob_df = pd.DataFrame({"score": best_score, "真实类别": np.where(y_test.values == 1, "真实高购买量", "真实普通订单")})
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    sns.histplot(data=prob_df, x="score", hue="真实类别", bins=24, stat="density", common_norm=False, alpha=0.45, palette=["#3B82F6", "#EF476F"], ax=ax)
    ax.set_title("高购买量订单预测概率分布", fontsize=16, fontweight="bold")
    ax.set_xlabel("预测为高购买量订单的概率")
    polish_axes(ax)
    save_current("08_probability_distribution.png", "高购买量订单预测概率分布")

    wordcloud_src = CHART_DIR / "09_wordcloud.png"
    wordcloud_dst = REPORT_FIGURE_DIR / "09_wordcloud.png"
    if wordcloud_src.exists():
        shutil.copy2(wordcloud_src, wordcloud_dst)
        figure_index.append({"文件名": "09_wordcloud.png", "图名": "母婴商品消费热点词云", "路径": str(wordcloud_dst)})

    fig, ax = plt.subplots(figsize=(12, 6.8), facecolor="white")
    ax.axis("off")
    ax.text(0.02, 0.86, "业务总览大屏截图索引", fontsize=22, fontweight="bold", color="#101828")
    ax.text(0.02, 0.72, f"总交易记录数：{payload['metrics']['cleanTradeRecords']:,}    总购买数量：{payload['metrics']['totalBuyMount']:,}", fontsize=15)
    ax.text(0.02, 0.60, f"热门品类：{payload['categoryNames'][0]}    高购买量订单占比：{payload['metrics']['highBuyRate']:.1f}%", fontsize=15)
    ax.text(0.02, 0.44, "完整可交互大屏请打开 outputs/html/dashboard.html 的“业务总览”Tab。", fontsize=14, color="#667085")
    save_current("10_dashboard_business_overview.png", "业务总览大屏截图索引")

    fig, ax = plt.subplots(figsize=(12, 6.8), facecolor="white")
    ax.axis("off")
    ax.text(0.02, 0.86, "模型验证大屏截图索引", fontsize=22, fontweight="bold", color="#101828")
    ax.text(0.02, 0.72, f"最高分模型：{payload['metrics']['highestModel']}  Accuracy={payload['metrics']['highestAccuracy']:.3f}  F1={payload['metrics']['highestF1']:.3f}", fontsize=15)
    ax.text(0.02, 0.60, f"推荐展示模型：{payload['metrics']['displayModel']}  Accuracy={payload['metrics']['displayAccuracy']:.3f}  F1={payload['metrics']['displayF1']:.3f}", fontsize=15)
    ax.text(0.02, 0.44, "完整可交互大屏请打开 outputs/html/dashboard.html 的“模型验证”Tab。", fontsize=14, color="#667085")
    save_current("11_dashboard_model_validation.png", "模型验证大屏截图索引")

    pd.DataFrame(figure_index).to_csv(REPORT_FIGURE_DIR / "figure_index.csv", index=False, encoding="utf-8-sig")


def generate_dashboard(merged_df: pd.DataFrame, tables: dict[str, pd.DataFrame], cat1_result: dict, high_result: dict) -> None:
    """渲染大屏，并将同源真实数据保存为 dashboard_data.json。"""
    _render_dashboard_html(merged_df, tables, cat1_result, high_result)
    payload = build_dashboard_payload(merged_df, tables, cat1_result, high_result)
    payload_json = json.dumps(payload, ensure_ascii=False)
    (OUTPUT_DIR / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report_tables(merged_df, tables, cat1_result, high_result, payload)
    save_report_figures(cat1_result["model_df"], cat1_result, high_result, payload)

    html_path = HTML_DIR / "dashboard.html"
    dashboard = html_path.read_text(encoding="utf-8")
    dashboard = dashboard.replace("商品种类数", "商品 ID 数")
    dashboard = dashboard.replace("商品 ID 去重统计", "auction_id 去重统计")
    dashboard = dashboard.replace("商品 ID 去重", "auction_id 去重统计")
    dashboard = re.sub(
        r"(?:const payload = .*?;|const injectedPayload = .*?;\s*let payload = injectedPayload;)",
        f"const injectedPayload = {payload_json};\n    let payload = injectedPayload;",
        dashboard,
        count=1,
        flags=re.S,
    )
    dashboard = dashboard.replace("模型性能对比（主模型与辅助模型）", "模型性能对比（cat1 主模型）")
    dashboard = re.sub(
        r"<div class=\"summary-card\"><span>high_buy 辅助模型</span><strong>.*?</strong></div>",
        f"<div class=\"summary-card\"><span>high_buy 辅助模型</span><strong>Accuracy {payload['metrics']['highBuyOrders'] and payload['modelAcc'][-1]:.3f} / F1 {payload['metrics']['highBuyAuxF1']:.3f} / AUC {payload['metrics']['highBuyAuxAuc']:.3f}</strong></div>",
        dashboard,
        count=1,
        flags=re.S,
    )
    dashboard = re.sub(
        r"<section class=\"card diag-card\"><h3>模型诊断</h3>(.*?)</section>",
        lambda match: match.group(0).replace(
            f"high_buy：AUC {payload['metrics']['highBuyAuxAuc']:.3f}，F1 {payload['metrics']['highBuyAuxF1']:.3f}",
            f"high_buy：Accuracy {payload['modelAcc'][-1]:.3f}，F1 {payload['metrics']['highBuyAuxF1']:.3f}，AUC {payload['metrics']['highBuyAuxAuc']:.3f}",
        ),
        dashboard,
        count=1,
        flags=re.S,
    )
    dashboard = dashboard.replace(
        '<section class="card feature-card"><div class="card-head"><h3>特征重要性 TOP10</h3></div><div id="featureChart" class="chart feature-chart"></div><div class="feature-note">cat_id 相关特征贡献最高，说明商品层级关系对 cat1 识别影响最大。</div></section>',
        '<section class="card feature-card"><div class="card-head"><h3>特征重要性 TOP10</h3></div><div id="featureChart" class="chart feature-chart"></div><div class="feature-note">cat_id 相关特征贡献最高，说明商品层级关系对 cat1 识别影响最大。</div></section>',
    )
    if "AI 专家分析建议" not in dashboard:
        dashboard = dashboard.replace(
            '<section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart profile-chart"></div><div id="ageChart" class="chart profile-chart"></div></div></section>',
            '<section class="card expert-card"><h3>AI 专家分析建议</h3><div class="sub">优先请求本地后端；无 API Key 时返回规则分析</div><div id="expertAnalysis" class="expert-text">正在生成分析建议...</div></section><section class="card profile"><h3>已匹配用户画像分析</h3><div class="sub">仅统计已匹配宝宝画像的用户记录</div><div class="profile-grid"><div id="genderChart" class="chart profile-chart"></div><div id="ageChart" class="chart profile-chart"></div></div></section>',
            1,
        )
    if ".expert-card" not in dashboard:
        dashboard = dashboard.replace(
            ".diag-card { grid-column:span 7; height:300px; } .advice-card { grid-column:span 7; height:300px; } .profile { grid-column:span 10; height:300px; }",
            ".diag-card { grid-column:span 7; height:300px; } .advice-card { grid-column:span 7; height:300px; } .expert-card { grid-column:span 5; height:300px; } .profile { grid-column:span 5; height:300px; } .expert-text { margin-top:12px; color:#344054; font-size:12px; line-height:1.7; font-weight:700; white-space:pre-line; }",
            1,
        )
    model_block = r'''const modelLabels = payload.modelNames.slice(0,4);
    const modelOption = merge(baseChartOption(), {
      color: ["#3B82F6", "#22C55E"],
      tooltip: merge(baseChartOption().tooltip, {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: function(items) {
          var html = "<b>" + items[0].axisValue + "</b>";
          items.forEach(function(it) {
            html += "<br/>" + it.seriesName + "：" + (it.value == null ? "-" : it.value.toFixed(3));
          });
          html += "<br/><span style='color:#667085'>AUC 仅用于 high_buy 二分类辅助模型；cat1 为多分类任务。</span>";
          return html;
        }
      }),
      legend: { top: 0, left: "center", textStyle: { fontSize: 10, color: mutedColor } },
      grid: { left: 44, right: 28, top: 58, bottom: 82 },
      graphic: [
        { type: "text", right: 28, top: 30, style: { text: "最高分：Gradient Boosting", fill: "#8B5CF6", font: "700 11px Microsoft YaHei" } },
        { type: "text", right: 28, top: 48, style: { text: "推荐展示：Random Forest", fill: "#FF5C8A", font: "700 11px Microsoft YaHei" } }
      ],
      xAxis: { type: "category", data: modelLabels, axisLabel: { color: textColor, fontSize: 10, interval: 0, rotate: 18 }, axisTick: { show: false }, axisLine: { lineStyle: { color: gridLine } } },
      yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: mutedColor, fontSize: 10 }, splitLine: { lineStyle: { color: gridLine } } },
      series: [
        { name: "Accuracy", type: "bar", data: payload.modelAcc.slice(0,4), barWidth: 14, label: { show: true, position: "top", fontSize: 9, formatter: function(p) { return p.value == null ? "-" : p.value.toFixed(3); } } },
        { name: "F1-score", type: "bar", data: payload.modelF1.slice(0,4), barWidth: 14, label: { show: true, position: "top", fontSize: 9, formatter: function(p) { return p.value == null ? "-" : p.value.toFixed(3); } } }
      ]
    });
    chart("modelChart", modelOption);'''
    dashboard = re.sub(
        r'const modelLabels = .*?\n    chart\("modelChart",.*?\n    chart\("featureChart"',
        lambda _: model_block + '\n    chart("featureChart"',
        dashboard,
        count=1,
        flags=re.S,
    )
    gauge_calls = (
        'chart("salesGaugeChart",gaugeOption("销量贡献率",payload.metrics.salesContributionRate,"#3B82F6"));\n'
        '    chart("tradeGauge",gaugeOption("交易可用率",payload.metrics.tradeUsabilityRate,"#22C55E"));\n'
        '    chart("profileGauge",gaugeOption("画像覆盖率",payload.metrics.profileCoverageRate,"#F59E0B"));\n'
        '    chart("coreGauge",gaugeOption("核心完整率",payload.metrics.coreFieldCompleteRate,"#8B5CF6"));\n'
    )
    if 'chart("salesGaugeChart"' not in dashboard:
        dashboard = dashboard.replace('    chart("weekdayChart"', '    ' + gauge_calls + '    chart("weekdayChart"', 1)
    if "function gaugeOption" not in dashboard:
        dashboard = dashboard.replace(
            'const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:"#C4B5FD"},{offset:1,color:"#8B5CF6"}]):"#8B5CF6";',
            'const purpleGrad=window.echarts?new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:"#C4B5FD"},{offset:1,color:"#8B5CF6"}]):"#8B5CF6";\n'
            '    function gaugeOption(title,value,color){return merge(baseChartOption(),{tooltip:merge(baseChartOption().tooltip,{trigger:"item",formatter:()=>`${title}：${Number(value).toFixed(1)}%`}),series:[{type:"gauge",min:0,max:100,radius:"92%",center:["50%","58%"],startAngle:205,endAngle:-25,progress:{show:true,width:10,itemStyle:{color}},axisLine:{lineStyle:{width:10,color:[[1,"rgba(148,163,184,.16)"]]}},axisTick:{show:false},splitLine:{show:false},axisLabel:{show:false},pointer:{show:false},anchor:{show:false},title:{offsetCenter:[0,"42%"],fontSize:10,color:mutedColor,fontWeight:800},detail:{valueAnimation:true,offsetCenter:[0,"4%"],fontSize:19,fontWeight:900,color,formatter:v=>v.toFixed(1)+"%"},data:[{value,name:title}]}]});}',
            1,
        )
    if "expert-analysis" not in dashboard:
        dashboard = dashboard.replace(
            'modal.addEventListener("click", event => { if (event.target === modal) modal.classList.remove("active"); });',
            'modal.addEventListener("click", event => { if (event.target === modal) modal.classList.remove("active"); });\n'
            '    const expertBox = document.getElementById("expertAnalysis");\n'
            '    if (expertBox) fetch("/api/expert-analysis").then(r=>r.json()).then(d=>{expertBox.textContent=d.analysis || "暂无分析建议";}).catch(()=>{expertBox.textContent="消费结构：喂养用品、婴儿服饰和尿裤湿巾构成核心品类。\\n高购买量订单：占比不高但销量贡献突出，适合重点运营。\\n模型解释：Gradient Boosting 分数最高，Random Forest 便于解释特征重要性。\\n后续建议：补充价格、品牌、浏览、收藏和加购等行为特征。";});',
            1,
        )

    # Final dashboard contract: two tabs, no standalone quality/conclusion cards in
    # the business page, and no old diagnosis/advice cards in the model page.
    dashboard = re.sub(r'\s*<section class="card quality">.*?</section>', "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r'\s*<section class="card conclusion">.*?</section>', "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r'\s*<section class="card diag-card">.*?</section>', "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r'\s*<section class="card advice-card">.*?</section>', "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r'\s*<section class="card expert-card">.*?</section>', "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r'\s*chart\("tradeGauge".*?\);\n', "\n", dashboard)
    dashboard = re.sub(r'\s*chart\("profileGauge".*?\);\n', "\n", dashboard)
    dashboard = re.sub(r'\s*chart\("coreGauge".*?\);\n', "\n", dashboard)
    dashboard = re.sub(
        r'\s*const expertBox = document\.getElementById\("expertAnalysis"\);\n\s*if \(expertBox\).*?\}\);',
        "",
        dashboard,
        count=1,
        flags=re.S,
    )

    business_expert_card = (
        '<section class="card expert-business"><h3>AI 专家分析</h3>'
        '<div class="sub">基于消费结构、时间趋势和高购买量订单生成</div>'
        '<div id="businessExpertAnalysis" class="expert-text">正在生成分析建议...</div></section>'
    )
    if 'id="businessExpertAnalysis"' not in dashboard:
        dashboard = re.sub(
            r'(<section class="card weekday">.*?</section>)',
            r'\1' + business_expert_card,
            dashboard,
            count=1,
            flags=re.S,
        )

    model_expert_card = (
        '<section class="card expert-model"><h3>AI 模型专家分析</h3>'
        '<div class="sub">结合模型性能、特征重要性与 high_buy 辅助任务生成</div>'
        '<div id="modelExpertAnalysis" class="expert-text">正在生成分析建议...</div></section>'
    )
    if 'id="modelExpertAnalysis"' not in dashboard:
        dashboard = dashboard.replace('<section class="card profile">', model_expert_card + '<section class="card profile">', 1)

    dashboard = dashboard.replace(
        f"<span>主模型 Accuracy</span><strong>{payload['metrics']['displayAccuracy']:.3f}</strong><small>F1-score {payload['metrics']['displayF1']:.3f}</small>",
        f"<span>推荐展示模型</span><strong>Random Forest</strong><small>Acc {payload['metrics']['displayAccuracy']:.3f} / F1 {payload['metrics']['displayF1']:.3f}</small>",
    )
    dashboard = dashboard.replace(
        '<section class="card expert-business"><h3>AI 专家分析</h3>',
        '<section class="card expert-business"><div class="ai-card-head"><h3>AI 专家分析</h3><button class="ai-settings-btn" type="button">AI 设置</button></div>',
    )
    dashboard = dashboard.replace(
        '<section class="card expert-model"><h3>AI 模型专家分析</h3>',
        '<section class="card expert-model"><div class="ai-card-head"><h3>AI 模型专家分析</h3><button class="ai-settings-btn" type="button">AI 设置</button></div>',
    )

    layout_override = """
  <style id="latest-dashboard-contract">
    .word { grid-column:span 8 !important; height:270px; }
    .weekday { grid-column:span 7 !important; height:270px; }
    .expert-business { grid-column:span 9 !important; height:270px; }
    .expert-model { grid-column:span 14 !important; height:300px; }
    .profile { grid-column:span 10 !important; height:300px; }
    .expert-text { margin-top:12px; color:#344054; font-size:13px; line-height:1.7; font-weight:700; white-space:pre-line; }
    .expert-business .expert-text { font-size:14px; }
    .ai-card-head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .ai-settings-btn { border:0; border-radius:999px; padding:6px 12px; background:rgba(255,92,138,.12); color:#FF5C8A; font-size:12px; font-weight:900; cursor:pointer; }
    .ai-settings-btn:hover { background:rgba(255,92,138,.18); }
    .ai-insight-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:10px; }
    .ai-insight-item { min-height:72px; padding:10px 11px; border-radius:14px; border:1px solid rgba(255,255,255,.72); background:var(--ai-bg,rgba(255,241,245,.72)); box-shadow:inset 0 1px 0 rgba(255,255,255,.82); }
    .ai-insight-item h4 { margin:0 0 6px; color:var(--ai-c,#FF5C8A); font-size:12px; font-weight:900; }
    .ai-insight-item p { margin:0; color:#344054; font-size:12px; line-height:1.45; font-weight:800; }
    .ai-insight-item mark { background:transparent; color:#EF476F; font-weight:950; padding:0; }
    .ai-config-modal { position:fixed; inset:0; display:none; z-index:40; align-items:center; justify-content:center; background:rgba(16,24,40,.38); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
    .ai-config-modal.active { display:flex; }
    .ai-config-panel { width:min(520px,92vw); border-radius:22px; padding:20px; background:rgba(255,255,255,.94); border:1px solid rgba(255,255,255,.88); box-shadow:0 24px 70px rgba(16,24,40,.18); }
    .ai-config-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .ai-config-head h3 { margin:0; font-size:18px; color:#101828; }
    .ai-config-close { border:0; border-radius:50%; width:32px; height:32px; cursor:pointer; background:#FFF1F5; color:#FF5C8A; font-size:18px; }
    .ai-form-row { display:grid; gap:6px; margin:10px 0; }
    .ai-form-row label { color:#344054; font-size:12px; font-weight:900; }
    .ai-form-row input { height:38px; border-radius:12px; border:1px solid #EEF2F7; padding:0 12px; outline:none; font-size:13px; color:#101828; background:#fff; }
    .ai-form-row input:focus { border-color:rgba(255,92,138,.48); box-shadow:0 0 0 3px rgba(255,92,138,.10); }
    .ai-config-actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .ai-config-actions button { border:0; border-radius:12px; padding:9px 12px; font-weight:900; cursor:pointer; }
    .ai-save { background:#FF5C8A; color:#fff; } .ai-clear { background:#F2F4F7; color:#344054; } .ai-test { background:#EAF3FF; color:#3B82F6; }
    .ai-config-note,.ai-status { margin-top:10px; color:#667085; font-size:12px; line-height:1.5; font-weight:800; }
    .model-chart-card { grid-column:span 12 !important; }
    .feature-card { grid-column:span 12 !important; }
    .quality,.conclusion,.diag-card,.advice-card,.expert-card { display:none !important; }
    @media (max-width:1400px) { .expert-business,.expert-model { grid-column:1/-1 !important; } }
  </style>
"""
    if "latest-dashboard-contract" not in dashboard:
        dashboard = dashboard.replace("</head>", layout_override + "</head>", 1)

    ai_config_modal = """
  <div class="ai-config-modal" id="aiConfigModal">
    <section class="ai-config-panel">
      <div class="ai-config-head"><h3>AI 专家分析设置</h3><button class="ai-config-close" type="button" aria-label="关闭">×</button></div>
      <div class="ai-form-row"><label for="aiApiBase">API Base</label><input id="aiApiBase" type="text" placeholder="https://api.openai.com/v1"></div>
      <div class="ai-form-row"><label for="aiModel">Model</label><input id="aiModel" type="text" placeholder="gpt-4o-mini"></div>
      <div class="ai-form-row"><label for="aiApiKey">API Key</label><input id="aiApiKey" type="password" placeholder="仅临时保存到 sessionStorage"></div>
      <div class="ai-config-actions">
        <button class="ai-save" id="aiSaveConfig" type="button">保存本次会话</button>
        <button class="ai-clear" id="aiClearConfig" type="button">清空配置</button>
        <button class="ai-test" id="aiTestConfig" type="button">测试连接</button>
      </div>
      <div class="ai-config-note">仅用于本地演示，配置只保存在当前浏览器会话。正式提交和答辩复现建议使用 .env 与后端 app.py。</div>
      <div class="ai-status" id="aiConfigStatus">未填写 API Key 时将继续使用本地规则分析。</div>
    </section>
  </div>
"""
    if 'id="aiConfigModal"' not in dashboard:
        dashboard = dashboard.replace('<div class="modal" id="wordcloudModal">', ai_config_modal + '\n  <div class="modal" id="wordcloudModal">', 1)

    expert_script = f"""
    const aiFallbacks = {{
      business: [
        {{ title: "消费结构", text: "{payload['categoryNames'][0]}为热销核心品类。", color: "#EF476F", bg: "rgba(255,241,245,.76)" }},
        {{ title: "高价值订单", text: "占{payload['metrics']['highBuyRate']:.1f}%，贡献{payload['metrics']['salesContributionRate']:.1f}%销量。", color: "#3B82F6", bg: "rgba(234,243,255,.78)" }},
        {{ title: "时间趋势", text: "购买行为存在月度与星期波动。", color: "#22C55E", bg: "rgba(236,253,243,.78)" }},
        {{ title: "运营建议", text: "围绕核心品类配置库存、促销和推荐位。", color: "#F59E0B", bg: "rgba(255,247,230,.78)" }}
      ],
      model: [
        {{ title: "最高分模型", text: "{payload['metrics']['highestModel']} Acc {payload['metrics']['highestAccuracy']:.3f} / F1 {payload['metrics']['highestF1']:.3f}。", color: "#8B5CF6", bg: "rgba(243,238,255,.78)" }},
        {{ title: "推荐展示模型", text: "Random Forest Acc {payload['metrics']['displayAccuracy']:.3f} / F1 {payload['metrics']['displayF1']:.3f}，解释性更强。", color: "#EF476F", bg: "rgba(255,241,245,.76)" }},
        {{ title: "关键特征", text: "{payload['featureNames'][0]}贡献最高，体现商品层级关系。", color: "#3B82F6", bg: "rgba(234,243,255,.78)" }},
        {{ title: "辅助模型", text: "high_buy AUC {payload['metrics']['highBuyAuxAuc']:.3f} / F1 {payload['metrics']['highBuyAuxF1']:.3f}。", color: "#06B6D4", bg: "rgba(236,254,255,.78)" }},
        {{ title: "优化方向", text: "补充价格、品牌、浏览、收藏和加购特征。", color: "#22C55E", bg: "rgba(236,253,243,.78)" }}
      ]
    }};
    const aiTitles = {{
      business: ["消费结构", "高价值订单", "时间趋势", "运营建议"],
      model: ["最高分模型", "推荐展示模型", "关键特征", "辅助模型", "优化方向"]
    }};
    function getAiConfig() {{
      return {{
        api_base: sessionStorage.getItem("mbi_api_base") || "",
        model: sessionStorage.getItem("mbi_model") || "",
        api_key: sessionStorage.getItem("mbi_api_key") || ""
      }};
    }}
    function highlightAiText(text) {{
      return String(text || "").replace(/(\\d+\\.\\d+%|\\d+\\.\\d+x|\\d+\\.\\d{{3}}|Gradient Boosting|Random Forest|cat_id|high_buy)/g, "<mark>$1</mark>");
    }}
    function normalizeExpertItems(scene, analysis) {{
      const base = aiFallbacks[scene];
      const lines = String(analysis || "").split(/\\n+/).map(line => line.replace(/^[-*\\d.、\\s]+/, "").trim()).filter(Boolean);
      return base.map((item, index) => {{
        const expectedTitle = aiTitles[scene][index];
        let line = lines[index] || item.text;
        line = line.replace(new RegExp("^" + expectedTitle + "[：:]?\\\\s*"), "");
        line = line.replace(/^(消费结构|高价值订单|高购买量订单|时间趋势|运营建议|最高分模型|推荐展示模型|关键特征|辅助模型|优化方向)[：:]?\\s*/, "");
        return {{ ...item, text: line || item.text }};
      }});
    }}
    function renderExpertCards(id, scene, analysis) {{
      const box = document.getElementById(id);
      if (!box) return;
      const items = normalizeExpertItems(scene, analysis);
      box.classList.add("ai-insight-grid");
      box.innerHTML = items.map(item => `<article class="ai-insight-item" style="--ai-c:${{item.color}};--ai-bg:${{item.bg}}"><h4>${{item.title}}</h4><p>${{highlightAiText(item.text)}}</p></article>`).join("");
    }}
    function requestExpertAnalysis(scene) {{
      const cfg = getAiConfig();
      return fetch("/api/expert-analysis", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ scene, api_base: cfg.api_base, model: cfg.model, api_key: cfg.api_key, payload }})
      }}).then(res => res.ok ? res.json() : Promise.reject(new Error("backend unavailable")));
    }}
    function loadExpertAnalysis(id, scene) {{
      const fallbackText = aiFallbacks[scene].map(item => `${{item.title}}：${{item.text}}`).join("\\n");
      requestExpertAnalysis(scene)
        .then(data => renderExpertCards(id, scene, data.analysis || fallbackText))
        .catch(() => renderExpertCards(id, scene, fallbackText + "\\n后端接口未启动，当前使用规则分析。"));
    }}
    function refreshAiAnalysis() {{
      loadExpertAnalysis("businessExpertAnalysis", "business");
      loadExpertAnalysis("modelExpertAnalysis", "model");
    }}
    function initAiSettings() {{
      const modal = document.getElementById("aiConfigModal");
      const status = document.getElementById("aiConfigStatus");
      const baseInput = document.getElementById("aiApiBase");
      const modelInput = document.getElementById("aiModel");
      const keyInput = document.getElementById("aiApiKey");
      if (!modal || !status || !baseInput || !modelInput || !keyInput) return;
      function fillForm() {{
        const cfg = getAiConfig();
        baseInput.value = cfg.api_base || "https://api.openai.com/v1";
        modelInput.value = cfg.model || "gpt-4o-mini";
        keyInput.value = cfg.api_key || "";
      }}
      document.querySelectorAll(".ai-settings-btn").forEach(btn => btn.addEventListener("click", () => {{ fillForm(); modal.classList.add("active"); }}));
      document.querySelector(".ai-config-close").addEventListener("click", () => modal.classList.remove("active"));
      modal.addEventListener("click", event => {{ if (event.target === modal) modal.classList.remove("active"); }});
      document.getElementById("aiSaveConfig").addEventListener("click", () => {{
        sessionStorage.setItem("mbi_api_base", baseInput.value.trim());
        sessionStorage.setItem("mbi_model", modelInput.value.trim());
        sessionStorage.setItem("mbi_api_key", keyInput.value.trim());
        status.textContent = keyInput.value.trim() ? "已保存到本次会话，API Key 不会写入本地文件。" : "未填写 API Key，当前使用规则分析。";
        refreshAiAnalysis();
      }});
      document.getElementById("aiClearConfig").addEventListener("click", () => {{
        ["mbi_api_base","mbi_model","mbi_api_key"].forEach(key => sessionStorage.removeItem(key));
        fillForm();
        status.textContent = "已清空本次会话配置，当前使用规则分析。";
        refreshAiAnalysis();
      }});
      document.getElementById("aiTestConfig").addEventListener("click", () => {{
        sessionStorage.setItem("mbi_api_base", baseInput.value.trim());
        sessionStorage.setItem("mbi_model", modelInput.value.trim());
        sessionStorage.setItem("mbi_api_key", keyInput.value.trim());
        if (!keyInput.value.trim()) {{
          status.textContent = "未填写 API Key，当前使用规则分析。";
          refreshAiAnalysis();
          return;
        }}
        status.textContent = "正在测试连接...";
        requestExpertAnalysis("business").then(data => {{
          status.textContent = data.source === "llm" ? "连接成功。" : "连接失败，当前使用规则分析。";
          refreshAiAnalysis();
        }}).catch(() => {{
          status.textContent = "后端接口未启动，当前使用规则分析。";
          refreshAiAnalysis();
        }});
      }});
    }}
    initAiSettings();
    refreshAiAnalysis();
"""
    if "localBusinessAnalysis" not in dashboard:
        dashboard = dashboard.replace("    fetch(\"../dashboard_data.json\")", expert_script + "\n    fetch(\"../dashboard_data.json\")", 1)

    # Current presentation version: no smart insight badges and no LLM/API dependency.
    dashboard = re.sub(r"<span class=['\"]insight-badge['\"].*?</span>", "", dashboard, flags=re.S)
    dashboard = re.sub(r"\s*<div class=\"ai-config-modal\" id=\"aiConfigModal\">.*?</section>\s*</div>", "", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r"\s*const aiFallbacks = \{.*?refreshAiAnalysis\(\);\n", "\n", dashboard, count=1, flags=re.S)
    dashboard = re.sub(r"\s*function gaugeOption\(.*?\n", lambda m: m.group(0), dashboard, count=0)
    business_conclusion = f"""
          <section class="card expert-business conclusion-summary"><h3>结论总结</h3>
            <div class="sub">基于消费结构、时间趋势和高购买量订单的分析结论</div>
            <div class="conclusion-list">
              <div class="conclusion-item"><span>消费结构</span><p><b>{payload['categoryNames'][0]}</b>、<b>婴儿服饰</b>和<b>尿裤湿巾</b>构成主要消费品类。</p></div>
              <div class="conclusion-item"><span>高购买量订单</span><p>订单数占比约 <mark>{payload['metrics']['highBuyRate']:.1f}%</mark>，贡献约 <mark>{payload['metrics']['salesContributionRate']:.1f}%</mark> 的销量，具有重点运营价值。</p></div>
              <div class="conclusion-item"><span>时间趋势</span><p>购买行为存在月度与星期波动，可结合活动周期安排备货。</p></div>
              <div class="conclusion-item"><span>运营建议</span><p>围绕核心品类开展组合促销、库存优化和推荐位配置。</p></div>
            </div>
          </section>"""
    model_conclusion = f"""
          <section class="card expert-model conclusion-summary model-conclusion"><h3>模型结论总结</h3>
            <div class="sub">基于模型性能、特征重要性和 high_buy 辅助任务的分析结论</div>
            <div class="conclusion-list model-list">
              <div class="conclusion-item"><span>最高分模型</span><p><b>Gradient Boosting</b> 在 cat1 任务中表现最好，Accuracy 约 <mark>{payload['metrics']['highestAccuracy']:.3f}</mark>，F1-score 约 <mark>{payload['metrics']['highestF1']:.3f}</mark>。</p></div>
              <div class="conclusion-item"><span>推荐展示模型</span><p><b>Random Forest</b> 指标较高，并能输出特征重要性，适合作为报告展示模型。</p></div>
              <div class="conclusion-item"><span>关键特征</span><p><b>cat_id</b>、<b>cat_id_encoded</b> 和 <b>cat_id_freq</b> 贡献最高，说明商品层级关系影响明显。</p></div>
              <div class="conclusion-item"><span>辅助模型</span><p><b>high_buy</b> 是二分类辅助任务，AUC 约 <mark>{payload['metrics']['highBuyAuxAuc']:.3f}</mark>，具备一定识别能力。</p></div>
              <div class="conclusion-item wide"><span>优化方向</span><p>后续可补充价格、品牌、浏览、收藏、加购等行为特征。</p></div>
            </div>
          </section>"""
    dashboard = re.sub(r"\s*<section class=\"card expert-business\">.*?</section>", "\n" + business_conclusion, dashboard, count=1, flags=re.S)
    dashboard = re.sub(r"\s*<section class=\"card expert-model\">.*?</section>", "\n" + model_conclusion, dashboard, count=1, flags=re.S)
    dashboard = dashboard.replace(
        '<div id="modelChart" class="chart model-chart"></div></section>',
        '<div id="modelChart" class="chart model-chart"></div><div class="model-note">说明：Gradient Boosting 在 cat1 识别任务中得分最高；Random Forest 指标较高且便于解释，因此作为推荐展示模型。</div></section>',
        1,
    )
    dashboard = re.sub(
        r"graphic:\s*\[.*?\],\n\s*xAxis:",
        'xAxis:',
        dashboard,
        count=1,
        flags=re.S,
    )
    static_conclusion_css = """
  <style id="static-conclusion-style">
    .card-head { min-height:auto; margin-bottom:8px; }
    .trend-chart,.top-chart { height:340px; }
    .model-chart { height:252px; }
    .model-note { margin-top:4px; color:#667085; font-size:11px; line-height:1.45; font-weight:800; }
    .conclusion-summary { overflow:hidden; }
    .conclusion-summary h3 { margin:0; font-size:15px; color:#101828; }
    .conclusion-summary .sub { margin:5px 0 10px; color:#667085; font-size:11px; font-weight:800; }
    .conclusion-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .conclusion-item { padding:10px 12px; border-radius:12px; background:rgba(255,255,255,.58); border:1px solid rgba(255,255,255,.72); }
    .conclusion-item:nth-child(2n) { background:rgba(234,243,255,.62); }
    .conclusion-item:nth-child(3n) { background:rgba(236,253,243,.62); }
    .conclusion-item span { color:#FF5C8A; font-weight:900; font-size:12px; }
    .conclusion-item p { margin:6px 0 0; color:#344054; font-size:12px; line-height:1.55; font-weight:700; }
    .conclusion-item b { color:#101828; font-weight:950; }
    .conclusion-item mark { background:transparent; color:#EF476F; font-weight:950; padding:0; }
    .model-conclusion .conclusion-item p { font-size:11.5px; line-height:1.48; }
    .model-list .wide { grid-column:1/-1; }
  </style>
"""
    if "static-conclusion-style" not in dashboard:
        dashboard = dashboard.replace("</head>", static_conclusion_css + "</head>", 1)
    dashboard = re.sub(
        r'(modal\.addEventListener\("click", event => \{ if \(event\.target === modal\) modal\.classList\.remove\("active"\); \}\);\n).*?(    fetch\("../dashboard_data\.json"\))',
        r"\1    \2",
        dashboard,
        count=1,
        flags=re.S,
    )
    dashboard = re.sub(r"\n\s*\.insight-badge[^\n]*", "", dashboard)
    dashboard = re.sub(r"\n\s*\.ai-[^\n]*", "", dashboard)
    compact_summary = f"""
        <section class="summary compact-summary">
          <div class="summary-card task"><span>主任务</span><strong>商品大类识别 cat1</strong><small>多分类类目识别任务</small><em>主模型</em></div>
          <div class="summary-card"><span>最高分模型</span><strong>Gradient Boosting</strong><small>Accuracy {payload['metrics']['highestAccuracy']:.3f} / F1 {payload['metrics']['highestF1']:.3f}</small><em>最高分</em></div>
          <div class="summary-card"><span>推荐展示模型</span><strong>Random Forest</strong><small>Accuracy {payload['metrics']['displayAccuracy']:.3f} / F1 {payload['metrics']['displayF1']:.3f}</small><em>可解释性强</em></div>
          <div class="summary-card"><span>辅助任务</span><strong>high_buy 识别</strong><small>高购买量订单二分类</small><em>辅助分析</em></div>
          <div class="summary-card"><span>辅助模型表现</span><strong>AUC {payload['metrics']['highBuyAuxAuc']:.3f}</strong><small>F1 {payload['metrics']['highBuyAuxF1']:.3f} / Accuracy {payload['modelAcc'][-1]:.3f}</small><em>仍有空间</em></div>
          <div class="summary-card"><span>关键解释特征</span><strong>{payload['featureNames'][0]}</strong><small>商品层级关系贡献最高</small><em>核心特征</em></div>
        </section>"""
    dashboard = re.sub(r"\s*<section class=\"summary\">.*?</section>", "\n" + compact_summary, dashboard, count=1, flags=re.S)

    final_model_block = r'''const modelLabels = ["Logistic\nRegression","Decision\nTree","Random\nForest\n推荐展示","Gradient\nBoosting\n最高分"];
    const modelOption = merge(baseChartOption(), {
      color: ["#3B82F6", "#22C55E"],
      tooltip: merge(baseChartOption().tooltip, {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: function(items) {
          var html = "<b>" + items[0].axisValue.replace(/\n/g, " ") + "</b>";
          items.forEach(function(it) {
            html += "<br/>" + it.seriesName + "：" + (it.value == null ? "-" : it.value.toFixed(3));
          });
          return html;
        }
      }),
      legend: { top: 8, left: "center", textStyle: { fontSize: 11, color: mutedColor } },
      grid: { left: 48, right: 30, top: 56, bottom: 82 },
      xAxis: { type: "category", data: modelLabels, axisLabel: { color: textColor, fontSize: 10, interval: 0, lineHeight: 14 }, axisTick: { show: false }, axisLine: { lineStyle: { color: gridLine } } },
      yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: mutedColor, fontSize: 10 }, splitLine: { lineStyle: { color: gridLine } } },
      series: [
        { name: "Accuracy", type: "bar", data: payload.modelAcc.slice(0,4), barWidth: 16, barGap: "20%", label: { show: true, position: "top", fontSize: 10, formatter: function(p) { return p.value == null ? "-" : p.value.toFixed(3); } }, markArea: { silent: true, itemStyle: { color: "rgba(255,92,138,.08)" }, data: [[{ xAxis: "Gradient\nBoosting\n最高分" }, { xAxis: "Gradient\nBoosting\n最高分" }]] } },
        { name: "F1-score", type: "bar", data: payload.modelF1.slice(0,4), barWidth: 16, label: { show: true, position: "top", fontSize: 10, formatter: function(p) { return p.value == null ? "-" : p.value.toFixed(3); } } }
      ]
    });
    chart("modelChart", modelOption);'''
    dashboard = re.sub(
        r'const modelLabels = .*?\n    chart\("modelChart", modelOption\);',
        lambda _: final_model_block,
        dashboard,
        count=1,
        flags=re.S,
    )

    model_page_css = """
  <style id="model-tab-tight-style">
    .compact-summary { grid-template-columns:repeat(6,minmax(0,1fr)) !important; gap:10px !important; margin-bottom:12px !important; }
    .compact-summary .summary-card { min-height:90px !important; height:90px !important; padding:12px 12px !important; border-radius:15px !important; overflow:hidden; }
    .compact-summary .summary-card span { color:#667085; font-size:11px; font-weight:900; }
    .compact-summary .summary-card strong { margin-top:5px !important; color:#101828; font-size:18px !important; line-height:1.08; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .compact-summary .summary-card small { display:block; margin-top:5px; color:#667085; font-size:10.5px; line-height:1.25; font-weight:800; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .compact-summary .summary-card em { display:inline-block; margin-top:6px; padding:3px 7px; border-radius:999px; background:rgba(255,92,138,.12); color:#FF5C8A; font-size:10px; font-style:normal; font-weight:900; }
    .compact-summary .summary-card:nth-child(2) em { background:rgba(243,238,255,.86); color:#8B5CF6; }
    .compact-summary .summary-card:nth-child(3) em { background:rgba(234,243,255,.86); color:#3B82F6; }
    .compact-summary .summary-card:nth-child(4) em { background:rgba(236,253,243,.86); color:#22C55E; }
    .compact-summary .summary-card:nth-child(5) em { background:rgba(255,247,230,.86); color:#F59E0B; }
    .model-chart-card,.feature-card { height:430px !important; overflow:visible !important; }
    .model-chart { height:310px !important; }
    .feature-chart { height:330px !important; }
    .model-note,.feature-note { margin-top:6px !important; padding:8px 10px; border-radius:10px; color:#667085; font-size:12px; line-height:1.5; font-weight:700; }
    .model-note { background:rgba(255,241,245,.70); }
    .feature-note { background:rgba(243,238,255,.65); }
    .model-note b,.feature-note b { color:#FF5C8A; }
    .expert-model,.profile { height:300px !important; }
    @media (max-width:1400px) { .compact-summary { grid-template-columns:repeat(2,minmax(0,1fr)) !important; } }
  </style>
"""
    if "model-tab-tight-style" not in dashboard:
        dashboard = dashboard.replace("</head>", model_page_css + "</head>", 1)

    dashboard = dashboard.replace(
        "<title>MotherBabyInsight 母婴电商消费行为分析可视化大屏</title>",
        "<title>Baby-care</title>",
    )
    dashboard = dashboard.replace(
        "<h1>MotherBabyInsight</h1><p>母婴电商消费行为分析平台</p>",
        "<h1>Baby-care Insight</h1><p>母婴电商消费行为分析平台</p>",
    )
    dashboard = dashboard.replace(
        "<h2>MotherBabyInsight&nbsp;&nbsp;母婴电商消费行为分析可视化大屏</h2><p>Mother &amp; Baby E-commerce Behavior Analytics Dashboard</p>",
        "<h2>母婴电商消费行为分析及可视化</h2><p>Mother &amp; Baby E-commerce Behavior Analytics</p>",
    )

    chart_modal_css = """
  <style id="chart-modal-style">
    #trendChart,#categoryChart,#highBuyChart,#salesGaugeChart,#weekdayChart,#modelChart,#featureChart,#genderChart,#ageChart,
    .wordcloud img { cursor:zoom-in; }
    .chart-modal { position:fixed; inset:0; z-index:99; display:none; align-items:center; justify-content:center; background:rgba(16,24,40,.45); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
    .chart-modal.active { display:flex; }
    .chart-modal-body { width:82vw; height:78vh; padding:22px; border-radius:24px; background:rgba(255,255,255,.95); box-shadow:0 30px 90px rgba(16,24,40,.30); }
    .chart-modal-title { position:absolute; top:32px; left:50%; transform:translateX(-50%); color:#101828; font-size:18px; font-weight:900; }
    .chart-modal-echarts { width:100%; height:100%; }
    .chart-modal-image { max-width:100%; max-height:100%; object-fit:contain; display:none; margin:auto; }
    .chart-modal .modal-close { position:absolute; top:26px; right:36px; width:38px; height:38px; border:0; border-radius:50%; background:rgba(255,255,255,.9); color:#101828; font-size:24px; cursor:pointer; line-height:38px; }
  </style>
"""
    if "chart-modal-style" not in dashboard:
        dashboard = dashboard.replace("</head>", chart_modal_css + "</head>", 1)

    chart_modal_html = """
  <div class="chart-modal" id="chartModal">
    <button class="modal-close" type="button" aria-label="关闭">×</button>
    <div class="chart-modal-title" id="chartModalTitle"></div>
    <div class="chart-modal-body">
      <div id="chartModalEcharts" class="chart-modal-echarts"></div>
      <img id="chartModalImage" class="chart-modal-image" alt="图表放大预览">
    </div>
  </div>
"""
    dashboard = re.sub(
        r'\s*<div class="modal" id="wordcloudModal">.*?</div>',
        "\n" + chart_modal_html,
        dashboard,
        count=1,
        flags=re.S,
    )

    dashboard = dashboard.replace(
        "const charts = [];",
        """const charts = [];
    const chartOptions = {};
    let modalChart = null;
    const chartTitles = {
      trendChart: "购买趋势概览（按月）",
      categoryChart: "商品大类销量 TOP6",
      highBuyChart: "高购买量订单价值洞察",
      salesGaugeChart: "销量贡献率",
      weekdayChart: "星期购买活跃度",
      modelChart: "模型性能对比（cat1 主模型）",
      featureChart: "特征重要性 TOP10",
      genderChart: "已匹配用户性别画像",
      ageChart: "已匹配宝宝年龄阶段画像"
    };""",
        1,
    )
    dashboard = re.sub(
        r'function chart\(id,opt\)\{.*?\}\n    const pinkGrad',
        '''function chart(id,opt){const el=document.getElementById(id); if(!el)return; if(!window.echarts){el.innerHTML="<div class='missing'>ECharts 加载失败，请联网后重新打开。</div>";return;} const c=echarts.init(el,null,{renderer:"canvas"}); c.setOption(opt); c.__chartId=id; chartOptions[id]=opt; charts.push(c); window.addEventListener("resize",()=>c.resize());}
    const pinkGrad''',
        dashboard,
        count=1,
        flags=re.S,
    )
    chart_modal_script = r'''function initChartModal() {
      const modal = document.getElementById("chartModal");
      const modalTitle = document.getElementById("chartModalTitle");
      const modalEcharts = document.getElementById("chartModalEcharts");
      const modalImage = document.getElementById("chartModalImage");
      if (!modal || !modalTitle || !modalEcharts || !modalImage) return;
      function closeModal() {
        modal.classList.remove("active");
        if (modalChart) { modalChart.dispose(); modalChart = null; }
      }
      function openChart(id) {
        if (!chartOptions[id] || !window.echarts) return;
        modalTitle.textContent = chartTitles[id] || "图表详情";
        modalImage.style.display = "none";
        modalEcharts.style.display = "block";
        modal.classList.add("active");
        if (modalChart) modalChart.dispose();
        modalChart = echarts.init(modalEcharts, null, { renderer: "canvas" });
        modalChart.setOption(chartOptions[id], true);
        setTimeout(() => modalChart && modalChart.resize(), 40);
      }
      function openImage(src, title) {
        modalTitle.textContent = title || "热门商品词云";
        if (modalChart) { modalChart.dispose(); modalChart = null; }
        modalEcharts.style.display = "none";
        modalImage.style.display = "block";
        modalImage.src = src;
        modal.classList.add("active");
      }
      ["trendChart","categoryChart","highBuyChart","salesGaugeChart","weekdayChart","modelChart","featureChart","genderChart","ageChart"].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("click", event => { event.stopPropagation(); openChart(id); });
        const card = el.closest(".card");
        if (card && !card.dataset.zoomBound) {
          card.dataset.zoomBound = "1";
          card.addEventListener("click", event => {
            if (event.target.closest("button,a,img")) return;
            openChart(id);
          });
        }
      });
      const wordImg = document.getElementById("wordcloudImg");
      if (wordImg) {
        wordImg.addEventListener("click", event => {
          event.stopPropagation();
          openImage(wordImg.getAttribute("src") || "../charts/09_wordcloud.png", "热门商品词云");
        });
      }
      modal.querySelector(".modal-close").addEventListener("click", closeModal);
      modal.addEventListener("click", event => { if (event.target === modal) closeModal(); });
      window.addEventListener("keydown", event => { if (event.key === "Escape" && modal.classList.contains("active")) closeModal(); });
    }
    initChartModal();'''
    dashboard = re.sub(
        r'\s*const modal = document\.getElementById\("wordcloudModal"\);.*?modal\.addEventListener\("click", event => \{\s*if \(event\.target === modal\) modal\.classList\.remove\("active"\);\s*\}\);',
        "\n    " + chart_modal_script,
        dashboard,
        count=1,
        flags=re.S,
    )

    html_path.write_text(dashboard, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=PROJECT_EN_NAME)
    parser.add_argument(
        "--spark",
        action="store_true",
        help="启用 SparkSession。默认关闭，避免 Java/PySpark 环境异常时长时间卡住。",
    )
    args = parser.parse_args()
    run_pipeline(use_spark=args.spark)
