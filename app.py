from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HTML_DIR = OUTPUT_DIR / "html"
DASHBOARD_DATA = OUTPUT_DIR / "dashboard_data.json"

app = FastAPI(title="MotherBabyInsight Expert Analysis API")

if OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class ExpertRequest(BaseModel):
    scene: str = "business"
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    payload: dict[str, Any] | None = None


def load_dashboard_data(extra_payload: dict[str, Any] | None = None) -> dict:
    data: dict[str, Any] = {}
    if DASHBOARD_DATA.exists():
        data = json.loads(DASHBOARD_DATA.read_text(encoding="utf-8"))
    if extra_payload:
        data.update(extra_payload)
    return data


def local_business_analysis(data: dict) -> str:
    metrics = data.get("metrics", {})
    categories = data.get("categoryNames", [])
    months = data.get("monthLabels", [])
    month_buy = data.get("monthBuy", [])
    weekdays = data.get("weekdayLabels", [])
    weekday_buy = data.get("weekdayBuy", [])

    top_category = categories[0] if categories else "核心母婴品类"
    peak_month = months[month_buy.index(max(month_buy))] if months and month_buy else "峰值月份"
    peak_weekday = weekdays[weekday_buy.index(max(weekday_buy))] if weekdays and weekday_buy else "高活跃日"
    high_rate = float(metrics.get("highBuyRate", 0))
    sales_rate = float(metrics.get("salesContributionRate", 0))

    return (
        f"消费结构：{top_category}为热销核心品类。\n"
        f"高价值订单：占{high_rate:.1f}%，贡献{sales_rate:.1f}%销量。\n"
        f"时间趋势：{peak_month}与周{peak_weekday}购买更活跃。\n"
        "运营建议：围绕核心品类配置库存、促销和推荐位。"
    )


def local_model_analysis(data: dict) -> str:
    metrics = data.get("metrics", {})
    features = data.get("featureNames", [])
    top_feature = features[0] if features else "cat_id"
    highest = metrics.get("highestModel", "Gradient Boosting")
    highest_acc = float(metrics.get("highestAccuracy", 0))
    highest_f1 = float(metrics.get("highestF1", 0))
    rf_acc = float(metrics.get("displayAccuracy", 0))
    rf_f1 = float(metrics.get("displayF1", 0))
    high_auc = float(metrics.get("highBuyAuxAuc", 0))
    high_f1 = float(metrics.get("highBuyAuxF1", 0))

    return (
        f"最高分模型：{highest}，Acc {highest_acc:.3f} / F1 {highest_f1:.3f}。\n"
        f"推荐展示模型：Random Forest Acc {rf_acc:.3f} / F1 {rf_f1:.3f}。\n"
        f"关键特征：{top_feature}贡献最高，体现商品层级关系。\n"
        f"辅助模型：high_buy AUC {high_auc:.3f} / F1 {high_f1:.3f}。\n"
        "优化方向：补充价格、品牌、浏览、收藏和加购特征。"
    )


def local_expert_analysis(data: dict, scene: str) -> str:
    if scene == "model":
        return local_model_analysis(data)
    return local_business_analysis(data)


def build_prompt(scene: str, data: dict) -> str:
    metrics = data.get("metrics", {})
    summary = {
        "metrics": metrics,
        "top_categories": list(zip(data.get("categoryNames", [])[:6], data.get("categoryValues", [])[:6])),
        "feature_top": list(zip(data.get("featureNames", [])[:5], data.get("featureValues", [])[:5])),
        "month_labels": data.get("monthLabels", []),
        "month_buy": data.get("monthBuy", []),
        "weekday_labels": data.get("weekdayLabels", []),
        "weekday_buy": data.get("weekdayBuy", []),
    }
    if scene == "model":
        return (
            "你是一名机器学习模型分析专家。请根据当前模型指标和特征重要性，生成模型解释建议。\n"
            "分析重点：\n"
            "1. 说明 Gradient Boosting 是最高分模型；\n"
            "2. 说明 Random Forest 适合作为推荐展示模型的原因；\n"
            "3. 解释 cat_id 相关特征贡献最高的含义；\n"
            "4. 说明 high_buy 辅助模型的 AUC、F1 表现及局限；\n"
            "5. 给出后续特征补充建议。\n"
            "写作要求：\n"
            "1. 使用中文；\n"
            "2. 不超过 5 条；\n"
            "3. 每条不超过 45 字；\n"
            "4. 不要夸大模型能力；\n"
            "5. 明确区分 cat1 多分类任务和 high_buy 二分类任务。\n"
            f"当前真实数据：{json.dumps(summary, ensure_ascii=False)}"
        )
    return (
        "你是一名母婴电商数据分析专家。请根据当前项目数据，生成面向业务运营人员的简洁分析建议。\n"
        "分析重点：\n"
        "1. 消费结构：说明主要热销品类；\n"
        "2. 高购买量订单：说明订单占比、销量贡献和运营价值；\n"
        "3. 时间趋势：说明月度和星期购买波动；\n"
        "4. 运营建议：给出库存、促销、推荐位配置建议。\n"
        "写作要求：\n"
        "1. 使用中文；\n"
        "2. 不超过 4 条；\n"
        "3. 每条不超过 40 字；\n"
        "4. 不要编造没有提供的数据；\n"
        "5. 语气专业、简洁，适合放在可视化大屏中。\n"
        f"当前真实数据：{json.dumps(summary, ensure_ascii=False)}"
    )


def call_llm_if_configured(
    scene: str,
    data: dict,
    fallback: str,
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str]:
    resolved_key = api_key or os.getenv("LLM_API_KEY")
    resolved_base = (api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")).rstrip("/")
    resolved_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not resolved_key:
        return "local_rules", fallback + "\n当前使用规则分析。"

    body = json.dumps(
        {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": "你是电商数据分析实训答辩专家。"},
                {"role": "user", "content": build_prompt(scene, data)},
            ],
            "temperature": 0.2,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib_request.Request(
        f"{resolved_base}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {resolved_key}"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=18) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        return "llm", content or fallback
    except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
        return "local_rules_after_llm_failure", fallback + f"\n当前使用规则分析（连接失败：{exc.__class__.__name__}）。"


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(HTML_DIR / "dashboard.html")


@app.get("/api/expert-analysis")
def expert_analysis_get(
    scene: str = Query("business", pattern="^(business|model)$"),
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    data = load_dashboard_data()
    fallback = local_expert_analysis(data, scene)
    source, analysis = call_llm_if_configured(scene, data, fallback, api_base, model, api_key)
    return {"source": source, "scene": scene, "analysis": analysis}


@app.post("/api/expert-analysis")
def expert_analysis_post(req: ExpertRequest) -> dict:
    scene = req.scene if req.scene in {"business", "model"} else "business"
    data = load_dashboard_data(req.payload)
    fallback = local_expert_analysis(data, scene)
    source, analysis = call_llm_if_configured(scene, data, fallback, req.api_base, req.model, req.api_key)
    return {"source": source, "scene": scene, "analysis": analysis}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
