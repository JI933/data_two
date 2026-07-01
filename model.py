import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.switch_backend('Agg')

# ===================== 1、读取建模数据集 =====================
df = pd.read_excel("建模标准化数据集.xlsx")
print(f"数据集加载成功，样本量：{len(df)}")

# ===================== 2、变量选择 =====================
y = df["ln票房"]

X_vars = [
    "猫眼评分", "评分人数", "平均票价", "场均人次",
    "上映年份", "是否国产", "导演影响力", "主演影响力",
    "类型_动作", "类型_剧情", "类型_喜剧", "类型_科幻",
    "类型_奇幻", "类型_爱情", "类型_动画", "类型_冒险"
]

X = df[X_vars]
X = sm.add_constant(X)


# ===================== 3、显著性判断函数（改成中文） =====================
def get_sig_text(p):
    if p < 0.01:
        return "高度显著（1%水平）"
    elif p < 0.05:
        return "显著（5%水平）"
    elif p < 0.1:
        return "边际显著（10%水平）"
    else:
        return "不显著"


# ===================== 4、基准回归 =====================
model = sm.OLS(y, X).fit()

base_result = pd.DataFrame({
    "变量名": model.params.index,
    "回归系数": model.params.values,
    "标准误": model.bse.values,
    "t值": model.tvalues.values,
    "P值": model.pvalues.values,
    "95%置信区间下限": model.conf_int()[0].values,
    "95%置信区间上限": model.conf_int()[1].values,
})

base_result["显著性"] = base_result["P值"].apply(get_sig_text)
base_result.loc[base_result["变量名"] == "const", "变量名"] = "常数项"

print("✓ 基准回归完成")

# ===================== 5、VIF多重共线性检验 =====================
vif_data = pd.DataFrame()
vif_data["变量名"] = X.columns
vif_data["VIF值"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data[vif_data["变量名"] != "const"]
vif_data = vif_data.sort_values("VIF值", ascending=False).reset_index(drop=True)


def vif_note(v):
    if v > 10:
        return "严重共线性（需处理）"
    elif v > 5:
        return "中度共线性（可接受）"
    else:
        return "正常（无共线性）"


vif_data["共线性判断"] = vif_data["VIF值"].apply(vif_note)

print("✓ VIF检验完成")

# ===================== 6、模型检验汇总 =====================
residuals = model.resid

jb_stat, jb_pvalue = stats.jarque_bera(residuals)
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)

test_summary = pd.DataFrame({
    "检验项目": [
        "样本量", "自变量个数", "R²", "调整R²",
        "F统计量", "F检验P值", "AIC", "BIC",
        "JB正态检验统计量", "JB检验P值", "正态性判断",
        "BP异方差检验统计量", "BP检验P值", "异方差判断"
    ],
    "检验结果": [
        len(df), len(X_vars),
        round(model.rsquared, 4), round(model.rsquared_adj, 4),
        round(model.fvalue, 2), f"{model.f_pvalue:.4e}",
        round(model.aic, 2), round(model.bic, 2),
        round(jb_stat, 4), round(jb_pvalue, 4),
        "满足正态性" if jb_pvalue > 0.05 else "不满足（大样本下仍可用）",
        round(bp_stat, 4), round(bp_pvalue, 4),
        "不存在异方差" if bp_pvalue > 0.05 else "存在异方差（可用稳健标准误修正）"
    ]
})

print("✓ 模型检验完成")


# ===================== 7、逐步回归 =====================
def backward_elimination(X, y, significance_level=0.05):
    vars_list = list(X.columns)
    step_log = []
    while len(vars_list) > 1:
        model_step = sm.OLS(y, X[vars_list]).fit()
        pvalues = model_step.pvalues.drop("const")
        max_pvalue = pvalues.max()
        if max_pvalue > significance_level:
            remove_var = pvalues.idxmax()
            step_log.append({"剔除变量": remove_var, "剔除时P值": round(max_pvalue, 4), "剔除原因": "不显著（P>0.05）"})
            vars_list.remove(remove_var)
        else:
            break
    return vars_list, model_step, step_log


selected_vars, model_step, step_log = backward_elimination(X, y, 0.05)

step_log_df = pd.DataFrame(step_log)
step_log_df.index = range(1, len(step_log_df) + 1)
step_log_df.index.name = "步骤"

step_result = pd.DataFrame({
    "变量名": model_step.params.index,
    "回归系数": model_step.params.values,
    "标准误": model_step.bse.values,
    "t值": model_step.tvalues.values,
    "P值": model_step.pvalues.values,
    "95%置信区间下限": model_step.conf_int()[0].values,
    "95%置信区间上限": model_step.conf_int()[1].values,
})
step_result["显著性"] = step_result["P值"].apply(get_sig_text)
step_result.loc[step_result["变量名"] == "const", "变量名"] = "常数项"

print("✓ 逐步回归完成")

# ===================== 8、模型对比表 =====================
compare_df = pd.DataFrame({
    "指标": ["R²", "调整R²", "F统计量", "F检验P值", "AIC", "BIC", "自变量个数", "样本量"],
    "全变量模型": [
        round(model.rsquared, 4), round(model.rsquared_adj, 4),
        round(model.fvalue, 2), f"{model.f_pvalue:.4e}",
        round(model.aic, 2), round(model.bic, 2),
        len(X_vars), len(df)
    ],
    "逐步回归模型": [
        round(model_step.rsquared, 4), round(model_step.rsquared_adj, 4),
        round(model_step.fvalue, 2), f"{model_step.f_pvalue:.4e}",
        round(model_step.aic, 2), round(model_step.bic, 2),
        len(selected_vars) - 1, len(df)
    ]
})

print("✓ 模型对比完成")

# ===================== 9、系数经济意义解释 =====================
eco_interpret = []
for var in selected_vars:
    if var == "const":
        continue
    coef = model_step.params[var]
    pval = model_step.pvalues[var]
    pct_change = (np.exp(coef) - 1) * 100

    if var == "评分人数":
        coef_10w = coef * 100000
        pct_10w = (np.exp(coef_10w) - 1) * 100
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 10),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"评分人数每增加10万人，票房约增加{pct_10w:.2f}%（话题热度效应）"
        })
    elif var.startswith("类型_"):
        type_name = var.replace("类型_", "")
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"相对于犯罪类型，{type_name}片的票房约{'低' if coef < 0 else '高'}{abs(pct_change):.2f}%"
        })
    elif var == "是否国产":
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"国产影片比进口影片票房约{'高' if coef > 0 else '低'}{abs(pct_change):.2f}%"
        })
    elif var == "上映年份":
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"每过一年，单部影片票房平均增长{pct_change:.2f}%（市场扩容效应）"
        })
    elif var == "平均票价":
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"平均票价每提高1元，票房约增加{pct_change:.2f}%"
        })
    elif var == "场均人次":
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"场均人次每增加1人，票房约增加{pct_change:.2f}%（放映效率效应）"
        })
    elif var == "猫眼评分":
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"猫眼评分每提高1分，票房约增加{pct_change:.2f}%（口碑质量效应）"
        })
    else:
        eco_interpret.append({
            "变量名": var,
            "回归系数": round(coef, 6),
            "P值": round(pval, 4),
            "显著性": get_sig_text(pval),
            "经济意义解释": f"该变量每增加1单位，票房约增加{pct_change:.2f}%"
        })

eco_df = pd.DataFrame(eco_interpret)

print("✓ 经济意义解释完成")

# ===================== 10、输出到Excel =====================
with pd.ExcelWriter("回归分析结果汇总.xlsx", engine="openpyxl") as writer:
    base_result.to_excel(writer, sheet_name="基准回归结果", index=False)
    step_result.to_excel(writer, sheet_name="逐步回归结果", index=False)
    step_log_df.to_excel(writer, sheet_name="逐步回归过程")
    compare_df.to_excel(writer, sheet_name="模型对比", index=False)
    vif_data.to_excel(writer, sheet_name="VIF共线性检验", index=False)
    test_summary.to_excel(writer, sheet_name="模型检验汇总", index=False)
    eco_df.to_excel(writer, sheet_name="系数经济意义解释", index=False)

# ===================== 11、残差分析图 =====================
residuals = model.resid
fitted = model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(fitted, residuals, alpha=0.6, color="#1f77b4")
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel("拟合值")
axes[0, 0].set_ylabel("残差")
axes[0, 0].set_title("(a) 残差 vs 拟合值")

sns.histplot(residuals, bins=25, kde=True, color="#2ca02c", ax=axes[0, 1])
axes[0, 1].set_xlabel("残差")
axes[0, 1].set_title("(b) 残差分布直方图")

sm.qqplot(residuals, line='45', fit=True, ax=axes[1, 0])
axes[1, 0].set_title("(c) 残差Q-Q图（正态性检验）")

standard_resid = residuals / np.std(residuals)
axes[1, 1].scatter(fitted, np.sqrt(np.abs(standard_resid)), alpha=0.6, color="#ff7f0e")
axes[1, 1].set_xlabel("拟合值")
axes[1, 1].set_ylabel("√|标准化残差|")
axes[1, 1].set_title("(d) 异方差检验（Scale-Location）")

plt.suptitle("图12 回归模型残差分析图", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("图12_残差分析图.png")
plt.close()

# ===================== 12、完成提示 =====================
print("\n" + "=" * 60)
print("✓ 全部完成！已生成文件：")
print("  - 回归分析结果汇总.xlsx（7个sheet，全中文）")
print("  - 图12_残差分析图.png")