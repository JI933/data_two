import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ===================== 全局绘图配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)
plt.switch_backend('Agg')

# ===================== 1、读取原始数据 =====================
df_raw = pd.read_excel("猫眼中国电影票房总榜.xlsx")
df = df_raw.copy()

# ===================== 2、数据预处理 =====================
# 2.1 删除无意义冗余字段
drop_cols = ["英文名", "电影ID"]
df = df.drop(columns=drop_cols)

# 2.2 根据电影名称去重
df = df.drop_duplicates(subset="电影名称", keep="first")

# 2.3 提取年份
df["上映日期"] = df["上映日期"].str.replace(" 上映", "", regex=False)
df["上映年份"] = pd.to_datetime(df["上映日期"]).dt.year

# 2.4 数值字段清洗：去除千分位逗号，转为浮点型
num_cols = ["票房(万元)", "猫眼评分", "时长(分钟)", "平均票价", "场均人次", "评分人数"]
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)

# 2.5 缺失值处理：核心数值缺失直接删除，文本缺失填充"其他"
df = df.dropna(subset=num_cols)
text_cols = ["电影类型", "导演", "主演", "上映地区", "语言", "电影简介"]
for col in text_cols:
    df[col] = df[col].fillna("其他")

# 2.6 自定义宽松规则异常值处理
# 票房字段：仅设下限（4倍IQR），不设上限，完整保留所有高票房头部影片
q1_box = df["票房(万元)"].quantile(0.25)
q3_box = df["票房(万元)"].quantile(0.75)
iqr_box = q3_box - q1_box
lower_box = q1_box - 4 * iqr_box
df = df[df["票房(万元)"] >= lower_box]

# 场均人次字段：4倍IQR宽松标准双向过滤，剔除极端异常值
q1_per = df["场均人次"].quantile(0.25)
q3_per = df["场均人次"].quantile(0.75)
iqr_per = q3_per - q1_per
lower_per = q1_per - 4 * iqr_per
upper_per = q3_per + 4 * iqr_per
df = df[(df["场均人次"] >= lower_per) & (df["场均人次"] <= upper_per)]

# 2.7 因变量构建：票房自然对数
df["ln票房"] = np.log(df["票房(万元)"])

# ===================== 3、特征工程 =====================
# 3.1 国产/进口二分类变量
def is_china(x):
    return 1 if "中国大陆" in str(x) else 0

df["是否国产"] = df["上映地区"].apply(is_china)

# 3.2 电影类型多标签独热编码
main_type = ["动作", "剧情", "喜剧", "科幻", "奇幻", "爱情", "犯罪", "动画", "冒险"]
for t in main_type:
    df[f"类型_{t}"] = df["电影类型"].apply(lambda x: 1 if t in x else 0)

# 3.3 导演影响力
dir_count = df["导演"].value_counts()
df["导演影响力"] = df["导演"].map(dir_count)

# 3.4 主演影响力
actor_dict = {}
for idx, actors in df["主演"].items():
    act_list = [a.strip() for a in actors.split(",")]
    for a in act_list:
        actor_dict[a] = actor_dict.get(a, 0) + 1

def get_top_actor_score(actor_str):
    acts = [a.strip() for a in actor_str.split(",")]
    return max([actor_dict.get(a, 1) for a in acts])

df["主演影响力"] = df["主演"].apply(get_top_actor_score)

# 3.5 口碑交互项
df["评分热度交互"] = df["猫眼评分"] * df["评分人数"]

# ===================== 4、探索性可视化全部图表 =====================
# ---------------------- 图2：票房分布特征图 ----------------------
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(data=df, x="票房(万元)", bins=30, color="#1f77b4", ax=axes[0, 0])
axes[0, 0].set_title("(a)原始票房分布（万元）", fontsize=12)
axes[0, 0].set_xlabel("总票房")

sns.histplot(data=df, x="ln票房", bins=30, color="#ff7f0e", ax=axes[0, 1])
axes[0, 1].set_title("(b)对数转换后票房分布", fontsize=12)
axes[0, 1].set_xlabel("ln(票房)")

sns.boxplot(data=df, y="票房(万元)", color="#2ca02c", ax=axes[1, 0])
axes[1, 0].set_title("(c)票房箱线图", fontsize=12)

top15 = df.sort_values("票房(万元)", ascending=False).head(15)
sns.barplot(data=top15, x="票房(万元)", y="电影名称", color="#d62728", ax=axes[1, 1])
axes[1, 1].set_title("(d)票房TOP15影片", fontsize=12)
plt.tight_layout()
plt.savefig("图2_票房分布特征.png")
plt.close()

# ---------------------- 图3：时间维度趋势图 ----------------------
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
year_cnt = df["上映年份"].value_counts().sort_index()
year_mean = df.groupby("上映年份")["票房(万元)"].mean()
year_sum = df.groupby("上映年份")["票房(万元)"].sum()

sns.barplot(x=year_cnt.index, y=year_cnt.values, ax=axes[0, 0])
axes[0, 0].set_title("(a)各年份上榜影片数量")
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].plot(year_mean.index, year_mean.values, marker="o", color="crimson")
axes[0, 1].set_title("(b)历年平均票房")

axes[1, 0].plot(year_sum.index, year_sum.values, marker="s", color="darkgreen")
axes[1, 0].set_title("(c)年度上榜影片总票房")

sns.scatterplot(data=df, x="上映年份", y="票房(万元)", alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title("(d)上映年份与票房散点")
plt.tight_layout()
plt.savefig("图3_时间维度趋势.png")
plt.close()

# ---------------------- 图4：电影类型分析图 ----------------------
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
type_stat = {}
for t in main_type:
    cnt = df[f"类型_{t}"].sum()
    mean_box = df[df[f"类型_{t}"] == 1]["票房(万元)"].mean()
    type_stat[t] = {"数量": cnt, "均值票房": mean_box}
type_df = pd.DataFrame(type_stat).T
type_df["类型"] = type_df.index

sns.barplot(data=type_df, x="类型", y="数量", ax=axes[0, 0])
axes[0, 0].set_title("(a)各类型影片数量")
axes[0, 0].tick_params(axis='x', rotation=30)

sns.barplot(data=type_df, x="类型", y="均值票房", color="orange", ax=axes[0, 1])
axes[0, 1].set_title("(b)各类型平均票房")
axes[0, 1].tick_params(axis='x', rotation=30)

axes[1, 0].pie(type_df["数量"], labels=type_df.index, autopct="%.1f%%")
axes[1, 0].set_title("(c)电影类型数量占比环形图")

box_data = []
box_label = []
for t in main_type:
    tmp = df[df[f"类型_{t}"] == 1]["票房(万元)"]
    if len(tmp) > 0:
        box_data.append(tmp)
        box_label.append(t)
axes[1, 1].boxplot(box_data, tick_labels=box_label)
axes[1, 1].set_title("(d)各类型票房分布箱线")
axes[1, 1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("图4_电影类型分析.png")
plt.close()

# ---------------------- 图5：口碑与热度分析图 ----------------------
fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.scatterplot(data=df, x="猫眼评分", y="票房(万元)", alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_title("(a)猫眼评分 vs 票房")

sns.scatterplot(data=df, x="评分人数", y="票房(万元)", alpha=0.5, color="green", ax=axes[0, 1])
axes[0, 1].set_title("(b)评分人数 vs 票房")

sns.scatterplot(data=df, x="评分热度交互", y="票房(万元)", alpha=0.5, color="red", ax=axes[1, 0])
axes[1, 0].set_title("(c)评分×评分人数交互项 vs 票房")

sns.histplot(data=df, x="猫眼评分", bins=20, color="purple", ax=axes[1, 1])
axes[1, 1].set_title("(d)猫眼评分分布")
plt.tight_layout()
plt.savefig("图5_口碑热度分析.png")
plt.close()

# ---------------------- 图6：放映指标分析图 ----------------------
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(data=df, x="平均票价", bins=25, ax=axes[0, 0])
axes[0, 0].set_title("(a)平均票价分布")

sns.histplot(data=df, x="场均人次", bins=25, color="orange", ax=axes[0, 1])
axes[0, 1].set_title("(b)场均人次分布")

sns.scatterplot(data=df, x="平均票价", y="票房(万元)", alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_title("(c)平均票价与票房关系")

sns.scatterplot(data=df, x="场均人次", y="票房(万元)", alpha=0.5, color="green", ax=axes[1, 1])
axes[1, 1].set_title("(d)场均人次与票房关系")
plt.tight_layout()
plt.savefig("图6_放映指标分析.png")
plt.close()

# ---------------------- 图7：主创影响力分析图 ----------------------
fig7, axes = plt.subplots(2, 2, figsize=(14, 10))

top_dir = dir_count.head(10)
sns.barplot(y=top_dir.index, x=top_dir.values, ax=axes[0, 0])
axes[0, 0].set_title("(a)上榜次数TOP10导演")

top_actor = pd.Series(actor_dict).sort_values(ascending=False).head(10)
sns.barplot(y=top_actor.index, x=top_actor.values, ax=axes[0, 1])
axes[0, 1].set_title("(b)上榜次数TOP10主演")

sns.boxplot(data=df, x="导演影响力", y="票房(万元)", ax=axes[1, 0])
axes[1, 0].set_title("(c)不同导演影响力票房分布")

sns.boxplot(data=df, x="主演影响力", y="票房(万元)", ax=axes[1, 1])
axes[1, 1].set_title("(d)不同主演影响力票房分布")
plt.tight_layout()
plt.savefig("图7_主创影响力分析.png")
plt.close()

# ---------------------- 图8：国产进口对比图 ----------------------
fig8, axes = plt.subplots(2, 2, figsize=(14, 10))
cn_cnt = df["是否国产"].value_counts()

axes[0, 0].pie(cn_cnt.values, labels=["进口", "国产"], autopct="%.1f%%")
axes[0, 0].set_title("(a)国产/进口影片数量占比")

sns.boxplot(data=df, x="是否国产", y="票房(万元)", ax=axes[0, 1])
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(["进口", "国产"])
axes[0, 1].set_title("(b)国产与进口票房对比")

year_cn = pd.crosstab(df["上映年份"], df["是否国产"])
year_cn.plot(kind="bar", stacked=True, ax=axes[1, 0])
axes[1, 0].set_title("(c)历年国产/进口影片数量")
axes[1, 0].legend(["进口", "国产"])
axes[1, 0].tick_params(axis='x', rotation=45)

year_mean_cn = df.groupby(["上映年份", "是否国产"])["票房(万元)"].mean().unstack()
year_mean_cn.plot(marker="o", ax=axes[1, 1])
axes[1, 1].set_title("(d)历年两类影片平均票房")
axes[1, 1].legend(["进口", "国产"])
plt.tight_layout()
plt.savefig("图8_国产进口对比.png")
plt.close()

# ---------------------- 图9：变量相关性热力图 ----------------------
fig9 = plt.figure(figsize=(12, 9))
corr_vars = ["ln票房", "猫眼评分", "评分人数", "评分热度交互", "平均票价", "场均人次", "上映年份", "是否国产",
             "导演影响力", "主演影响力"]
corr_mat = df[corr_vars].corr()
sns.heatmap(corr_mat, annot=True, cmap="RdBu_r", fmt=".3f", linewidths=0.5)
plt.title("多变量相关性热力图", fontsize=14)
plt.tight_layout()
plt.savefig("图9_变量相关性热力图.png")
plt.close()

# ---------------------- 图10：核心变量散点矩阵图 ----------------------
fig10 = plt.figure(figsize=(15, 12))
pair_vars = ["ln票房", "猫眼评分", "评分人数", "平均票价", "场均人次"]
pair_df = df[pair_vars].sample(n=200, random_state=42)
g = sns.PairGrid(pair_df, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.4, s=15)
g.map_lower(sns.regplot, scatter_kws={"alpha": 0.3, "s": 12}, line_kws={"color": "red"})
g.map_diag(sns.histplot, bins=20)
g.fig.suptitle("核心变量散点矩阵图", y=1.02, fontsize=15)
plt.tight_layout()
plt.savefig("图10_核心变量散点矩阵.png")
plt.close()

# ---------------------- 图11：票房TOP6雷达图（莫兰迪配色） ----------------------
radar_movies = df.nlargest(6, "票房(万元)")["电影名称"].tolist()

radar_features = ["猫眼评分", "评分人数", "平均票价", "场均人次", "时长(分钟)"]
feature_labels = ["猫眼评分", "话题热度\n(评分人数)", "平均票价", "场均人次", "影片时长"]

radar_df = df[df["电影名称"].isin(radar_movies)].copy()
for feat in radar_features:
    radar_df[feat + "_norm"] = (radar_df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())

morandi_colors = [
    '#FF6B35', '#00E676', '#D500F9', '#2979FF', '#FFEA00', '#FF4081'
]

N = len(radar_features)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig_radar, axes_radar = plt.subplots(2, 3, figsize=(18, 12),
                                     subplot_kw=dict(polar=True))
axes_radar = axes_radar.flatten()

for idx, movie in enumerate(radar_movies):
    ax = axes_radar[idx]
    movie_data = radar_df[radar_df["电影名称"] == movie]
    values = [movie_data[feat + "_norm"].values[0] for feat in radar_features]
    values += values[:1]

    color = morandi_colors[idx]

    ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color)
    ax.fill(angles, values, color=color, alpha=0.35)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=7)
    ax.set_ylim(0, 1.05)
    ax.grid(color='#dddddd', linestyle='--', linewidth=0.8)

    boxoffice = movie_data["票房(万元)"].values[0] / 10000
    ax.set_title(f"{movie}\n(票房 {boxoffice:.1f} 亿)", fontsize=11, pad=12, color=color)

plt.suptitle("图11 票房TOP6电影多维特征雷达图对比\n(指标已归一化至0-1区间，面积越大综合表现越强)",
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("图11_电影多维特征雷达图.png", bbox_inches='tight')
plt.close()

# ===================== 输出结果与导出数据集 =====================
df.to_excel("建模标准化数据集.xlsx", index=False)

print("===== 数据清洗完成，总样本量：", len(df))
print("\n===== 连续变量描述性统计 =====")
desc_cols = ["票房(万元)","ln票房","猫眼评分","时长(分钟)","平均票价","场均人次","评分人数","上映年份","导演影响力","主演影响力"]
print(df[desc_cols].describe())
print("\n===== 国产/进口影片数量统计 =====")
print(df["是否国产"].replace({0:"进口影片",1:"国产影片"}).value_counts())
print("\n===== 各题材影片数量 =====")
for t in main_type:
    print(f"{t}：{df[f'类型_{t}'].sum()} 部")

# ===================== 雷达图分析结论 =====================
print("\n" + "="*60)
print("【雷达图分析结论】")
print("="*60)
print("分析对象：票房TOP6影片")
print("选取指标：猫眼评分、评分人数(热度)、平均票价、场均人次、影片时长")
print("分析方法：Min-Max归一化后绘制单部电影雷达图")
print("\n各电影特征分析：")

for movie in radar_movies:
    movie_data = radar_df[radar_df["电影名称"] == movie]
    scores = [movie_data[feat + "_norm"].values[0] for feat in radar_features]
    avg_score = np.mean(scores)
    boxoffice = movie_data["票房(万元)"].values[0] / 10000

    max_idx = np.argmax(scores)
    min_idx = np.argmin(scores)

    print(f"\n▶ {movie}（票房{boxoffice:.1f}亿，综合得分{avg_score:.3f}）")
    print(f"  最强项：{feature_labels[max_idx].replace(chr(10), '')}（{scores[max_idx]:.2%}）")
    print(f"  最弱项：{feature_labels[min_idx].replace(chr(10), '')}（{scores[min_idx]:.2%}）")

    if scores[1] > 0.7 and scores[0] > 0.6:
        print("  特征判断：口碑热度双高型，属于爆款影片，宣发与质量兼备")
    elif scores[1] > 0.7:
        print("  特征判断：高热度话题型，宣发引流效果显著，观众关注度高")
    elif scores[0] > 0.7:
        print("  特征判断：高口碑品质型，靠内容质量取胜，长尾放映能力强")
    elif scores[3] > 0.7:
        print("  特征判断：高上座效率型，院线认可度高，单厅产出能力强")
    elif scores[2] > 0.7:
        print("  特征判断：高票价定位型，多为视效大片，IMAX/3D占比高")
    else:
        print("  特征判断：均衡发展型，各维度表现中规中矩，无明显短板")


print("="*60)
print("✓ 雷达图生成完毕：图11_电影多维特征雷达图.png")
print("="*60)