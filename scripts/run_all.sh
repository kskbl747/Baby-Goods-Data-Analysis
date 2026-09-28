#!/bin/bash
set -e

echo "开始运行 MotherBabyInsight 项目"
python src/mother_baby_insight.py
echo "项目运行完成"
echo "大屏位置：outputs/html/dashboard.html"
echo "报告图表位置：outputs/report_figures/"
echo "报告表格位置：outputs/report_tables/"
