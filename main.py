import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
import ast
import plotly.express as px
import logging
from fake_useragent import UserAgent  # 如果没有安装先 pip install fake-useragent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ⚠️ 把你的 Cookie 粘贴到下面
DOUBAN_COOKIE = 'll="118124"; bid=o-InxSVcKZk; dbcl2="294675203:O2/IVssV1dM"; push_noty_num=0; push_doumail_num=0; ck=_rDx; _pk_id.100001.4cf6=2ec6e7f651ab8065.1777604202.; _pk_ses.100001.4cf6=1; ap_v=0,6.0'

class DoubanHardcoreMiner:
    def __init__(self):
        self.base_url = "https://movie.douban.com/top250?start={}"
        self.session = requests.Session()
        self.ua = UserAgent()
        self.session.headers.update({
            "Cookie": DOUBAN_COOKIE,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://movie.douban.com/",
            "Connection": "keep-alive",
        })

    def _get_headers(self):
        """每次请求随机 User‑Agent"""
        return {"User-Agent": self.ua.random}

    def _parse_movie_item(self, item):
        """防御性提取，已修复所有已知 bug"""
        try:
            title_node = item.select_one('.title')
            if not title_node: return None
            title = title_node.text.strip()

            rating_node = item.select_one('.rating_num')
            rating = float(rating_node.text) if rating_node and rating_node.text else 0.0

            # 修复评价人数提取
            star_block = item.select_one('.star')
            review_count = 0
            if star_block:
                # 评价人数通常在第四个 span 中，文本格式为 “123456人评价”
                eval_node = star_block.select_one('span:nth-child(4)')
                if eval_node:
                    match = re.search(r'\d+', eval_node.text)
                    if match:
                        review_count = int(match.group())
                else:
                    # 降级：直接用最后一个 span
                    spans = star_block.find_all('span')
                    if spans:
                        match = re.search(r'\d+', spans[-1].text)
                        if match: review_count = int(match.group())

            bd = item.select_one('.bd')
            director, year, genres = "未知", None, []
            if bd:
                p_node = bd.find('p')
                if p_node:
                    p_text = p_node.text.strip().split('\n')
                    # 导演提取（直接取第一个斜杠前的部分，不再用正则删英文）
                    if len(p_text) > 0:
                        dir_match = re.search(r'导演:\s*(.*?)(?:\s+主演:|$)', p_text[0].strip())
                        if dir_match:
                            director_raw = dir_match.group(1).split('/')[0].strip()
                            # 不再正则抹掉英文，保留原名
                            director = director_raw

                    # 年份和类型提取
                    if len(p_text) > 1:
                        meta_info = p_text[1].strip().split('/')
                        if len(meta_info) >= 3:
                            # 年份
                            year_match = re.search(r'\d{4}', meta_info[0])
                            year = int(year_match.group()) if year_match else None
                            # 核心修复：类型用 '/' 分割
                            genres = [g.strip() for g in meta_info[2].split('/') if g.strip()]

            # 电影 ID 提取（方便后续短评抓取）
            movie_id = None
            link_node = item.select_one('a')
            if link_node:
                href = link_node.get('href', '')
                match = re.search(r'subject/(\d+)/', href)
                movie_id = match.group(1) if match else None

            return {
                "电影名称": title,
                "导演": director,
                "上映年份": year,
                "评分": rating,
                "评价人数": review_count,
                "类型": genres,
                "电影ID": movie_id
            }
        except Exception:
            return None

    def execute_mining(self, pages=10):
        dataset = []
        last_page_saved = 0
        # 断点续抓：如果已有部分数据，跳过已爬页
        if os.path.exists("douban_top250_deep_mining.csv"):
            try:
                existing = pd.read_csv("douban_top250_deep_mining.csv")
                if not existing.empty:
                    last_page_saved = len(existing) // 25  # 每页25条
                    logging.info(f"发现已有数据 {len(existing)} 条，将从第 {last_page_saved+1} 页继续")
                    dataset = existing.to_dict('records')
            except:
                pass

        logging.info("🚀 启动硬核数据挖掘流水线...")
        for i in range(last_page_saved, pages):
            url = self.base_url.format(i * 25)
            logging.info(f"正在抓取第 {i+1}/{pages} 页...")
            try:
                res = self.session.get(url, headers=self._get_headers(), timeout=10)
                res.encoding = 'utf-8'
                soup = BeautifulSoup(res.text, 'html.parser')
                items = soup.find_all('div', class_='item')

                if len(items) == 0 and i == 0:
                    logging.error("❌ 未匹配到任何电影数据！正在保存调试页面...")
                    with open("debug_error_page.html", "w", encoding="utf-8") as f:
                        f.write(res.text)
                    return pd.DataFrame()

                for item in items:
                    data = self._parse_movie_item(item)
                    if data:
                        dataset.append(data)

                # 每爬完一页就存一次，防止中断丢失
                pd.DataFrame(dataset).to_csv("douban_top250_deep_mining.csv", index=False, encoding='utf-8-sig')
            except Exception as e:
                logging.error(f"第 {i+1} 页请求异常: {e}")
            time.sleep(random.uniform(2.0, 4.0))

        return pd.DataFrame(dataset)

class AdvancedVisualizer:
    @staticmethod
    def plot_director_matrix(df):
        logging.info("正在渲染 [导演权力矩阵]...")
        df_clean = df.dropna(subset=['导演', '评分', '电影名称']).copy()
        dir_stats = df_clean.groupby('导演').agg(
            电影数量=('电影名称', 'count'),
            平均评分=('评分', 'mean'),
            代表作=('电影名称', lambda x: "<br>".join(list(x)[:3]))
        ).reset_index()
        elite_dirs = dir_stats[dir_stats['电影数量'] >= 2]

        fig = px.scatter(
            elite_dirs,
            x="电影数量", y="平均评分", size="电影数量",
            color="平均评分",
            hover_name="导演",
            hover_data={"代表作": True, "电影数量": True, "平均评分": ':.2f'},
            size_max=40,
            template="plotly_dark",
            title="<b>豆瓣神级导演权力矩阵</b><br><sup>气泡大小代表入榜作品数</sup>",
            color_continuous_scale=px.colors.sequential.Sunsetdark
        )
        fig.update_layout(xaxis_title="入榜电影数量", yaxis_title="平均豆瓣评分")
        fig.write_html("director_matrix.html", include_plotlyjs='cdn')

    @staticmethod
    def plot_era_sunburst(df):
        logging.info("正在渲染 [年代旭日图]...")
        df_sun = df.copy()
        # 修复：类型字段可能是列表/字符串，统一处理
        df_sun['类型'] = df_sun['类型'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_sun['年代'] = (df_sun['上映年份'] // 10 * 10).fillna(0).astype(int).astype(str) + "s"
        df_sun = df_sun.explode('类型').dropna(subset=['年代', '类型', '电影名称', '评分'])

        # 取数量前10的类型
        top_genres = df_sun['类型'].value_counts().head(10).index
        df_sun = df_sun[df_sun['类型'].isin(top_genres)]

        # 气泡大小用评价人数，但若为0则设为1避免消失
        df_sun['显示人数'] = df_sun['评价人数'].apply(lambda x: x if x > 0 else 1)

        fig = px.sunburst(
            df_sun,
            path=['年代', '类型', '电影名称'],
            values='显示人数',
            color='评分',
            color_continuous_scale='RdBu_r',
            template="plotly_dark",
            title="<b>时空透视：电影的黄金时代与题材变迁</b><br><sup>点击扇区可动态下钻放大</sup>"
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>评分: %{color:.2f}<br>评价人数: %{value}")
        fig.write_html("era_sunburst.html", include_plotlyjs='cdn')


if __name__ == "__main__":
    miner = DoubanHardcoreMiner()
    df = miner.execute_mining(pages=10)

    if not df.empty:
        logging.info(f"✅ 爬取成功！共 {len(df)} 条数据，保存至 douban_top250_deep_mining.csv")
        viz = AdvancedVisualizer()
        viz.plot_director_matrix(df)
        viz.plot_era_sunburst(df)
        logging.info("🎉 可视化完成，请打开 HTML 文件查看。")
    else:
        logging.error("❌ 未抓取到任何数据。")