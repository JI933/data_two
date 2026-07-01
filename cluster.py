import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re


class MaoyanBoxOfficeSpider:
    """猫眼票房排行榜爬虫类"""

    def __init__(self, year=None, fetch_detail=True):
        """
        初始化爬虫
        :param year: 年份，如2026；为None时爬取历史总票房榜
        :param fetch_detail: 是否爬取电影详情信息（类型、时长、评分、导演等）
        """
        self.year = year
        self.fetch_detail = fetch_detail

        if year:
            self.base_url = f"https://piaofang.maoyan.com/rankings/year?year={year}"
            self.title = f"{year}年票房排行榜"
        else:
            self.base_url = "https://piaofang.maoyan.com/rankings/year"
            self.title = "中国电影票房总榜"

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://piaofang.maoyan.com/'
        }

        # API接口的headers
        self.api_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.movies_data = []

    def get_page(self, url):
        """获取页面内容"""
        try:
            time.sleep(random.uniform(0.5, 2))
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.RequestException as e:
            print(f"请求页面失败: {e}")
            return None

    def extract_movie_id(self, row):
        """从行元素中提取电影ID"""
        data_com = row.get('data-com', '')
        match = re.search(r"/movie/(\d+)", data_com)
        if match:
            return match.group(1)
        return None

    def get_movie_detail(self, movie_id):
        """通过API获取电影详细信息"""
        if not movie_id:
            return {}
        try:
            time.sleep(random.uniform(0.3, 1))
            url = f"https://m.maoyan.com/ajax/detailmovie?movieId={movie_id}"
            response = requests.get(url, headers=self.api_headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            movie = data.get('detailMovie', {})

            detail = {
                '电影类型': movie.get('cat', ''),
                '导演': movie.get('dir', ''),
                '主演': movie.get('star', ''),
                '时长(分钟)': movie.get('dur', ''),
                '猫眼评分': movie.get('sc', ''),
                '评分人数': movie.get('snum', ''),
                '上映地区': movie.get('src', ''),
                '语言': movie.get('oriLang', ''),
                '电影简介': movie.get('dra', ''),
                '英文名': movie.get('enm', ''),
            }
            return detail
        except Exception as e:
            print(f"  获取电影详情失败 (ID: {movie_id}): {e}")
            return {}

    def parse_movie_list(self, soup):
        """解析电影列表数据"""
        movies = []
        rows = soup.find_all('ul', class_='row')

        for idx, row in enumerate(rows):
            try:
                rank_elem = row.find('li', class_='col0')
                rank = rank_elem.get_text(strip=True) if rank_elem else ''
                if not rank.isdigit():
                    continue

                movie_id = self.extract_movie_id(row)

                col1_elem = row.find('li', class_='col1')
                movie_name = ''
                release_date = ''
                if col1_elem:
                    name_elem = col1_elem.find('p', class_='first-line')
                    if name_elem:
                        movie_name = name_elem.get_text(strip=True)
                    date_elem = col1_elem.find('p', class_='second-line')
                    if date_elem:
                        release_date = date_elem.get_text(strip=True)

                box_office_elem = row.find('li', class_='col2')
                box_office = box_office_elem.get_text(strip=True) if box_office_elem else ''

                avg_price_elem = row.find('li', class_='col3')
                avg_price = avg_price_elem.get_text(strip=True) if avg_price_elem else ''

                avg_people_elem = row.find('li', class_='col4')
                avg_people = avg_people_elem.get_text(strip=True) if avg_people_elem else ''

                if rank.isdigit() and movie_name:
                    movie_data = {
                        '排名': int(rank),
                        '电影名称': movie_name,
                        '上映日期': release_date,
                        '票房(万元)': box_office,
                        '平均票价': avg_price,
                        '场均人次': avg_people,
                        '电影ID': movie_id,
                    }
                    movies.append(movie_data)
            except Exception as e:
                print(f"解析电影数据失败: {e}")
                continue
        return movies

    def crawl(self):
        """执行爬取"""
        print(f"开始爬取 {self.title}...")
        print(f"目标URL: {self.base_url}")

        soup = self.get_page(self.base_url)
        if not soup:
            print("获取页面失败")
            return []

        movies = self.parse_movie_list(soup)
        print(f"成功获取 {len(movies)} 部电影的基本信息")

        if self.fetch_detail and movies:
            print(f"\n开始爬取电影详情信息（共 {len(movies)} 部）...")
            for i, movie in enumerate(movies):
                movie_id = movie.get('电影ID')
                movie_name = movie.get('电影名称', '未知')
                print(f"  [{i + 1}/{len(movies)}] 正在获取: {movie_name} (ID: {movie_id})")
                detail = self.get_movie_detail(movie_id)
                movie.update(detail)
                if (i + 1) % 10 == 0:
                    print(f"  已完成 {i + 1}/{len(movies)} 部电影详情爬取")

        self.movies_data = movies
        print(f"\n爬取完成！共获取 {len(movies)} 部电影数据")
        return movies

    def save_to_excel(self, filename=None):
        """保存数据到Excel"""
        if not self.movies_data:
            print("没有数据可以保存")
            return None

        if not filename:
            if self.year:
                filename = f"猫眼{self.year}年票房排行榜.xlsx"
            else:
                filename = "猫眼中国电影票房总榜.xlsx"

        df = pd.DataFrame(self.movies_data)
        if '排名' in df.columns:
            df = df.sort_values('排名')

        preferred_columns = [
            '排名', '电影名称', '票房(万元)', '猫眼评分', '电影类型',
            '导演', '主演', '时长(分钟)', '上映日期', '平均票价',
            '场均人次', '上映地区', '语言', '英文名', '评分人数',
            '电影简介', '电影ID'
        ]
        existing_columns = [col for col in preferred_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in existing_columns]
        df = df[existing_columns + other_columns]

        sheet_name = self.title[:30]
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            column_widths = {
                '排名': 8, '电影名称': 25, '票房(万元)': 12, '猫眼评分': 10,
                '电影类型': 20, '导演': 15, '主演': 30, '时长(分钟)': 12,
                '上映日期': 18, '平均票价': 10, '场均人次': 10, '上映地区': 15,
                '语言': 10, '英文名': 20, '评分人数': 12, '电影简介': 50, '电影ID': 12,
            }
            for col_name, width in column_widths.items():
                if col_name in df.columns:
                    col_idx = df.columns.get_loc(col_name)
                    col_letter = chr(65 + col_idx)
                    worksheet.column_dimensions[col_letter].width = width

        print(f"数据已保存到: {filename}")
        return filename

    def print_summary(self):
        """打印数据摘要"""
        if not self.movies_data:
            print("没有数据")
            return

        print("\n" + "=" * 70)
        print(f"{self.title}数据摘要")
        print("=" * 70)
        print(f"总计电影数量: {len(self.movies_data)} 部")

        if self.movies_data:
            total_box_office = 0
            for movie in self.movies_data:
                try:
                    box_office = float(movie.get('票房(万元)', '0').replace(',', ''))
                    total_box_office += box_office
                except:
                    pass
            print(f"总票房: {total_box_office:,.0f} 万元 ({total_box_office / 10000:.2f} 亿元)")

            print("\n票房TOP 10:")
            print("-" * 70)
            print(f"{'排名':>4} {'电影名称':<20} {'票房(万)':>10} {'评分':>6} {'类型':<15} {'导演':<10}")
            print("-" * 70)
            for i, movie in enumerate(self.movies_data[:10], 1):
                print(f"{i:>4} {movie.get('电影名称', '未知'):<20} "
                      f"{movie.get('票房(万元)', '未知'):>10} "
                      f"{str(movie.get('猫眼评分', '-')):>6} "
                      f"{movie.get('电影类型', '-'):<15} "
                      f"{movie.get('导演', '-'):<10}")
        print("=" * 70 + "\n")


def main():
    """主函数"""
    print("猫眼票房排行榜爬虫（增强版）")
    print("=" * 60)
    print("功能：爬取票房排行榜 + 电影详情信息（类型、时长、评分、导演等）")
    print("=" * 60)

    # 爬取历史总票房榜（含详情）
    spider = MaoyanBoxOfficeSpider(year=None, fetch_detail=True)

    # 如果要爬取年度榜单，使用下面这行：
    # spider = MaoyanBoxOfficeSpider(year=2026, fetch_detail=True)

    # 如果只需要基本信息，不需要详情，使用：
    # spider = MaoyanBoxOfficeSpider(year=None, fetch_detail=False)

    movies = spider.crawl()
    if movies:
        spider.print_summary()
        filename = spider.save_to_excel()
        if filename:
            print(f"\n爬取完成！数据已保存到 {filename}")
    else:
        print("\n爬取失败，未获取到数据")


if __name__ == '__main__':
    main()