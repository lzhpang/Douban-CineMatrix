import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import logging
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fake_useragent import UserAgent
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================== 配置区 ==================
DOUBAN_COOKIE = '添加真正的cookie'   # ⚠️ 必须更新为有效 Cookie
API_KEY = "填入自己的apikey"
BASE_URL = "对应的模型地址"
MODEL_NAME = "模型名"

TARGET_MOVIES = ["肖申克的救赎", "盗梦空间", "泰坦尼克号", "阿甘正传", "千与千寻", "星际穿越"]   # 可增减，确保 CSV 中有对应行


MAX_REVIEWS = 250
BATCH_SIZE = 20
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================== 爬虫模块 ==================
class DoubanReviewScraper:
    def __init__(self, cookie):
        self.session = requests.Session()
        self.ua = UserAgent()
        self.cookie = cookie

    def scrape_reviews(self, movie_id, movie_name, max_reviews=250, force=False):
        """
        抓取电影短评。force=True 时会忽略缓存重新抓取。
        返回评论列表。
        """
        cache_file = os.path.join(CACHE_DIR, f"movie_{movie_id}_reviews.json")
        # 如果不强制，且缓存存在，直接读取
        if not force and os.path.exists(cache_file):
            logging.info(f"📦 缓存命中《{movie_name}》")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)[:max_reviews]

        logging.info(f"🌐 开始抓取《{movie_name}》(ID:{movie_id}) 的短评...")
        reviews = []
        headers = {
            "User-Agent": self.ua.random,
            "Cookie": self.cookie
        }

        for start in range(0, max_reviews, 20):
            url = f"https://movie.douban.com/subject/{movie_id}/comments?start={start}&limit=20&status=P&sort=new_score"
            try:
                res = self.session.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(res.text, 'html.parser')
                comment_nodes = soup.select('.short') or soup.select('.comment-content')
                # 第一页未提取到内容时保存调试页面
                if len(comment_nodes) == 0 and start == 0:
                    with open("debug_comments.html", "w", encoding="utf-8") as f:
                        f.write(res.text)
                    logging.warning("⚠️ 未找到评论节点，已保存 debug_comments.html，请检查 Cookie 或网络")
                for node in comment_nodes:
                    text = node.text.strip()
                    if len(text) > 5:
                        reviews.append(text)
                logging.info(f"  已抓取 {len(reviews)} 条")
                if len(comment_nodes) < 20:
                    break
                time.sleep(random.uniform(2, 4))
            except Exception as e:
                logging.error(f"抓取异常: {e}")
                break

        if reviews:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(reviews[:max_reviews], f, ensure_ascii=False)
            logging.info(f"💾 已缓存 {len(reviews[:max_reviews])} 条评论")
        return reviews[:max_reviews]

# ================== LLM 分析器 ==================
class LLMAnalyzer:
    def __init__(self):
        logging.info(f"🤖 已连接 MiMo: {MODEL_NAME}")

    def is_valid_cache(self, df):
        """检测 DataFrame 是否全是默认值 0.5 且无标签"""
        if df.empty:
            return False
        if df['positive_score'].nunique() == 1 and df['positive_score'].iloc[0] == 0.5:
            if all(len(t) == 0 for t in df['tags']):
                return False
        return True

    def batch_analyze(self, reviews, batch_size=20):
        all_results = []
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            prompt = self._build_prompt(batch)
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role":"system","content":"你是一位影评分析师，严格输出JSON。"},
                              {"role":"user","content":prompt}],
                    temperature=0.1,
                    response_format={"type":"json_object"}
                )
                data = json.loads(resp.choices[0].message.content)
                all_results.extend(data.get("results", []))
            except Exception as e:
                logging.error(f"LLM调用失败: {e}")
                for j in range(len(batch)):
                    all_results.append({"index":i+j, "positive_score":0.5, "tags":[], "is_controversial":False})
            time.sleep(1)
        df = pd.DataFrame(all_results)
        if 'index' in df.columns:
            df = df.sort_values('index').reset_index(drop=True)
            df['text'] = [reviews[int(idx)] for idx in df['index']]
        return df

    def _build_prompt(self, batch):
        texts = "\n".join([f"{i}. {text}" for i,text in enumerate(batch)])
        return f"""分析每条短评，返回JSON：
- positive_score: 0-1，1为很正面
- tags: 不超过3个关键词
- is_controversial: 是否有对立/反讽/强情绪

格式：{{"results":[{{"index":0,"positive_score":0.8,"tags":["特效"],"is_controversial":false}}]}}

评论：
{texts}"""

    def negative_summary(self, texts):
        if not texts:
            return "无显著差评"
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":f"用一句话总结差评核心不满：\n" + "\n".join(texts[:50])}],
                temperature=0.3)
            return resp.choices[0].message.content.strip()
        except:
            return "总结失败"

    def cloud_words(self, texts):
        if not texts:
            return []
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":f"提取差评中最具代表性的名词/形容词（逗号分隔，最多20个）：\n" + "\n".join(texts[:80])}],
                temperature=0.2)
            return [w.strip() for w in resp.choices[0].message.content.split(',')]
        except:
            return []

# ================== 仪表盘可视化 ==================
def build_dashboard(df_summary, all_movie_dfs):
    if df_summary.empty:
        logging.error("无数据，不生成图表")
        return

    n = len(df_summary)
    movies = df_summary['电影名称'].tolist()
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#C084FC'][:n]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "histogram"}, {"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "table"}, {"type": "xy"}]],
        subplot_titles=("情感得分分布", "口碑指标对比", "争议评论占比",
                        "评分vs情感定位", "高频关键词", "词云（另存图片）"),
        horizontal_spacing=0.1, vertical_spacing=0.15
    )

    # 1. 情感直方图
    for i, (name, mdf) in enumerate(all_movie_dfs.items()):
        fig.add_trace(go.Histogram(x=mdf['positive_score'], name=name,
                                   marker_color=colors[i % len(colors)],
                                   opacity=0.6, nbinsx=20), row=1, col=1)

    # 2. 口碑指标柱状图
    for j, met in enumerate(['平均正面得分', '争议度(标准差)', '豆瓣评分']):
        fig.add_trace(go.Bar(x=movies, y=df_summary[met],
                             name=met, marker_color=colors[j] if j<len(colors) else colors[0],
                             showlegend=False), row=1, col=2)

    # 3. 争议评论环形图
    for i, row in df_summary.iterrows():
        fig.add_trace(go.Pie(labels=['争议', '非争议'],
                             values=[row['争议评论比例'], 1-row['争议评论比例']],
                             hole=0.6, marker_colors=['coral','lightgray'],
                             name=row['电影名称'], showlegend=False), row=1, col=3)

    # 4. 评分 vs 情感散点
    fig.add_trace(go.Scatter(x=df_summary['豆瓣评分'], y=df_summary['平均正面得分'],
                             mode='markers+text', text=movies, textposition='top center',
                             marker=dict(size=df_summary['评论数']/5, color='#FFD93D'),
                             name='口碑定位', showlegend=False), row=2, col=1)

    # 5. 高频关键词表
    all_tags = Counter()
    for mdf in all_movie_dfs.values():
        for taglist in mdf['tags']:
            if isinstance(taglist, list):
                all_tags.update(taglist)
    top_tags = all_tags.most_common(10)
    fig.add_trace(go.Table(
    header=dict(
        values=['<b>关键词</b>', '<b>频次</b>'],
        fill_color='#1f2c3d',        # 深色表头背景
        font=dict(color='white', size=14),
        align='center'
    ),
    cells=dict(
        values=[[t[0] for t in top_tags], [t[1] for t in top_tags]],
        fill_color=[['#2a3f54', '#3a4f64'] * 5],  # 交替行底色
        font=dict(color='white', size=13),
        align='center',
        height=30
    )), row=2, col=2)

    # 6. 词云占位
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='text',
                             text=["📄 词云已单独保存为\nnegative_wordcloud.png"],
                             textfont=dict(size=20, color='gray'),
                             showlegend=False), row=2, col=3)

    fig.update_layout(template='plotly_dark', title_text=f"🎬 豆瓣口碑深度对比仪表盘（{n}部电影）", height=900)
    fig.write_html("dashboard.html", include_plotlyjs='cdn')
    logging.info("📊 仪表盘已保存: dashboard.html")

    # 差评词云
    all_neg = []
    for mdf in all_movie_dfs.values():
        all_neg.extend(mdf[mdf['positive_score'] < 0.4]['text'].tolist())
    if all_neg:
        analyzer = LLMAnalyzer()
        words = analyzer.cloud_words(all_neg)
        if words:
            freqs = Counter(words)
            try:
                wc = WordCloud(font_path='simhei.ttf', width=800, height=400,
                               background_color='black', colormap='Reds').generate_from_frequencies(freqs)
            except:
                wc = WordCloud(width=800, height=400, background_color='black',
                               colormap='Reds').generate_from_frequencies(freqs)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title("差评核心概念 (LLM提取)", fontsize=18, color='white')
            plt.savefig("negative_wordcloud.png", dpi=150, bbox_inches='tight')
            plt.close()
            logging.info("☁️ 词云已保存")
        else:
            logging.warning("未提取到有效词云词汇")

# ================== 主流程 ==================
def process_movies(movies_to_process, scraper, analyzer):
    """给定电影列表，执行评论获取和LLM分析，返回汇总DataFrame和明细dict"""
    all_summaries = []
    all_dfs = {}
    for movie in movies_to_process:
        mid = movie['movie_id']
        name = movie['name']
        score = movie['douban_score']

        # 获取评论（force参数由外部决定，scraper内部分发）
        reviews = scraper.scrape_reviews(mid, name, MAX_REVIEWS)
        if len(reviews) < 20:
            logging.warning(f"⛔ 《{name}》评论不足（{len(reviews)}条），跳过")
            continue

        # LLM 分析缓存
        llm_cache = os.path.join(CACHE_DIR, f"movie_{mid}_llm.csv")
        if os.path.exists(llm_cache):
            df_rev = pd.read_csv(llm_cache)
            df_rev['tags'] = df_rev['tags'].apply(lambda x: eval(x) if isinstance(x,str) and x.startswith('[') else x)
            if not analyzer.is_valid_cache(df_rev):
                logging.warning(f"🗑️ 《{name}》LLM缓存无效，重新分析")
                os.remove(llm_cache)
                df_rev = analyzer.batch_analyze(reviews, BATCH_SIZE)
                if not df_rev.empty:
                    df_rev.to_csv(llm_cache, index=False, encoding='utf-8-sig')
        else:
            df_rev = analyzer.batch_analyze(reviews, BATCH_SIZE)
            if not df_rev.empty:
                df_rev.to_csv(llm_cache, index=False, encoding='utf-8-sig')

        if df_rev.empty:
            continue

        avg_pos = df_rev['positive_score'].mean()
        std_pos = df_rev['positive_score'].std()
        contr_ratio = df_rev['is_controversial'].mean()
        low_texts = df_rev[df_rev['positive_score'] < 0.4]['text'].tolist()
        neg_sum = analyzer.negative_summary(low_texts)

        all_summaries.append({
            "电影名称": name,
            "豆瓣评分": score,
            "评论数": len(reviews),
            "平均正面得分": round(avg_pos, 3),
            "争议度(标准差)": round(std_pos, 3),
            "争议评论比例": round(contr_ratio, 3),
            "差评总结": neg_sum
        })
        all_dfs[name] = df_rev
        logging.info(f"✅ 《{name}》: 正面{avg_pos:.2f} 争议{std_pos:.2f}")

    return pd.DataFrame(all_summaries), all_dfs

def get_movie_info_from_csv(csv_path, movie_names):
    """从CSV中读取指定电影的基本信息，返回列表[dict]"""
    if not os.path.exists(csv_path):
        logging.error(f"找不到 {csv_path}，请先运行 main.py 抓取 Top250")
        return []
    df = pd.read_csv(csv_path)
    # 安全转换电影ID
    df['电影ID'] = df['电影ID'].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")
    movies_info = []
    for name in movie_names:
        match = df[df['电影名称'] == name]
        if not match.empty:
            row = match.iloc[0]
            movies_info.append({
                "movie_id": row['电影ID'],
                "name": row['电影名称'],
                "douban_score": row['评分']
            })
        else:
            logging.warning(f"CSV中没有找到《{name}》，跳过")
    return movies_info

def discover_cached_movies(cache_dir, csv_path):
    """扫描缓存文件夹，返回有评论缓存的电影信息列表"""
    if not os.path.exists(cache_dir):
        return []
    movies = []
    for fname in os.listdir(cache_dir):
        if fname.startswith("movie_") and fname.endswith("_reviews.json"):
            movie_id = fname.replace("movie_", "").replace("_reviews.json", "")
            # 从 CSV 中获取电影名称和评分
            df = pd.read_csv(csv_path)
            df['电影ID'] = df['电影ID'].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")
            match = df[df['电影ID'] == movie_id]
            if not match.empty:
                row = match.iloc[0]
                movies.append({
                    "movie_id": movie_id,
                    "name": row['电影名称'],
                    "douban_score": row['评分']
                })
            else:
                movies.append({
                    "movie_id": movie_id,
                    "name": f"电影 {movie_id}",
                    "douban_score": 0.0
                })
    return movies

if __name__ == "__main__":
    csv_path = "douban_top250_deep_mining.csv"
    if not os.path.exists(csv_path):
        logging.error("请先运行 main.py 生成 douban_top250_deep_mining.csv")
        exit()

    print("\n======================================")
    print("   豆瓣短评情感分析流水线")
    print("======================================")
    print("请选择数据获取方式：")
    print("  1 - 仅使用缓存（不爬取，直接分析已有评论）")
    print("  2 - 重新爬取所有目标电影（忽略缓存）")
    print("  3 - 缓存优先，缺失时爬取（推荐）")
    choice = input("请输入数字 (1/2/3): ").strip()

    if choice == '1':
        # 仅缓存模式：自动发现缓存中的电影
        logging.info("选择了『仅使用缓存』模式")
        movies_to_process = discover_cached_movies(CACHE_DIR, csv_path)
        if not movies_to_process:
            logging.error("缓存目录中没有任何评论文件，退出。")
            exit()
        # 爬虫实例（不需要有效Cookie，因为不会爬取）
        scraper = DoubanReviewScraper(DOUBAN_COOKIE)
        # 修改scraper的scrape_reviews方法，强制只用缓存：设置force=False且不抓取（缓存不存在就返回空）
        # 可以通过猴子补丁或临时修改实例方法，这里简单：我们调用时传入force=False，内部已处理。
        # 但内部即使缓存不存在也会尝试抓取，为了防止抓取，可以临时将cookie置空或重写方法。
        # 简单方案：如果缓存不存在，直接返回空列表，不尝试抓取。
        original_scrape = scraper.scrape_reviews
        def cache_only_scrape(mid, name, max_reviews=250, force=False):
            cache_file = os.path.join(CACHE_DIR, f"movie_{mid}_reviews.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)[:max_reviews]
            else:
                logging.warning(f"缓存不存在《{name}》，跳过")
                return []
        scraper.scrape_reviews = cache_only_scrape

    elif choice == '2':
        # 强制重新爬取：忽略缓存，爬取TARGET_MOVIES
        logging.info("选择了『重新爬取所有目标电影』模式")
        if not DOUBAN_COOKIE or DOUBAN_COOKIE == '你的豆瓣Cookie':
            logging.error("请先填入有效的豆瓣Cookie！")
            exit()
        movies_to_process = get_movie_info_from_csv(csv_path, TARGET_MOVIES)
        if not movies_to_process:
            logging.error("目标电影列表为空，退出。")
            exit()
        scraper = DoubanReviewScraper(DOUBAN_COOKIE)
        # 设置为强制抓取模式：删除对应缓存文件，然后调用scrape时传入force=True
        for movie in movies_to_process:
            cache_file = os.path.join(CACHE_DIR, f"movie_{movie['movie_id']}_reviews.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logging.info(f"已删除缓存 {cache_file}")
        # 修改scrape_reviews使force=True时忽略缓存（内部已经支持，但force参数默认False，需要传递）
        # 我们在调用scraper.scrape_reviews时传入force=True，所以需要修改调用处。
        # 为方便，重新绑定一个爬取方法
        original_scrape = scraper.scrape_reviews
        def force_scrape(mid, name, max_reviews=250):
            return original_scrape(mid, name, max_reviews, force=True)
        scraper.scrape_reviews = force_scrape

    elif choice == '3':
        # 缓存优先，缺失时抓取（默认方式）
        logging.info("选择了『缓存优先，缺失时爬取』模式")
        if not DOUBAN_COOKIE or DOUBAN_COOKIE == '你的豆瓣Cookie':
            logging.error("请先填入有效的豆瓣Cookie！")
            exit()
        movies_to_process = get_movie_info_from_csv(csv_path, TARGET_MOVIES)
        if not movies_to_process:
            logging.error("目标电影列表为空，退出。")
            exit()
        scraper = DoubanReviewScraper(DOUBAN_COOKIE)
        # 不需要特殊处理，scrape_reviews默认force=False，缓存存在即用，不存在则抓取
    else:
        logging.error("无效选择，退出。")
        exit()

    # 公共部分：使用上面定义的movies_to_process和scraper
    analyzer = LLMAnalyzer()
    df_summary, all_dfs = process_movies(movies_to_process, scraper, analyzer)

    if df_summary.empty:
        logging.error("未能分析任何电影，程序结束。")
        exit()

    df_summary.to_csv("movies_final_summary.csv", index=False, encoding='utf-8-sig')
    build_dashboard(df_summary, all_dfs)
    logging.info("🎉 全部分析完成！请查看 dashboard.html 和词云图片。")