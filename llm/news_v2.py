import json
import csv
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from client import LocalLLMClient, create_client  # 假设LLM客户端类已存在

# 预设细分行业列表（保持不变）
INDUSTRY_LIST = [
    "园林工程", "冰洗", "粮油加工", "显示器件", "商业地产", "其他专业工程", "IT服务", "固废治理", "光伏发电", "其他建材",
    "炼油化工", "国际工程承包", "综合环境治理", "化学制剂", "电视广播", "空调", "电网自动化", "家电零部件", "燃气", "机床工具",
    "多业态零售", "粘胶", "自然景点", "其他生物制品", "旅游综合", "输变电设备", "铁路运输", "房产租赁经纪", "航空装备", "信托",
    "综合乘用车", "机器人", "铜", "图片媒体", "汽车经销商", "板材", "大众出版", "线缆部件及其他", "餐饮", "化学原料药",
    "磁性材料", "海洋捕捞", "电能综合服务", "纸包装", "专业连锁", "冶金矿采化工设备", "其它专用机械", "其他汽车零部件", "摩托车", "电机",
    "激光设备", "预加工食品", "工程机械器件", "运动服装", "工控设备", "印刷", "成品家居", "光学元件", "疫苗", "公路货运",
    "通信终端及配件", "民爆用品", "装修装饰", "光伏电池组件", "集成电路封测", "学历教育", "其他塑料制品", "硅料硅片", "化学工程", "有机硅",
    "安防设备", "火电设备", "公交", "楼宇设备", "医疗耗材", "仓储物流", "卫浴电器", "油气及炼化工程", "电子化学品", "电动乘用车",
    "其他家电", "线下药店", "院线", "医疗研发外包", "航天装备", "个护小家电", "大气治理", "视频媒体", "集成电路制造", "人力资源服务",
    "品牌化妆品", "会展服务", "产业地产", "旅游零售", "文字媒体", "燃料电池", "商业物业经营", "铁路设备", "钟表珠宝", "锂电池",
    "铅锌", "通信网络设备及器件", "油品石化贸易", "氮肥", "高速公路", "其他医疗服务", "资产管理", "水务及水治理", "环保设备", "轮胎轮毂",
    "林业", "广告媒体", "医美服务", "冶钢辅料", "金属新材料", "纯碱", "其他化学原料", "热电", "车身附件及饰件", "特钢",
    "啤酒", "生猪养殖", "汽车综合服务", "其它视听器材", "光伏加工设备", "稀土", "软饮料", "其他酒类", "涤纶", "其他纺织",
    "油气开采", "炭黑", "其他自动化设备", "期货", "厨房小家电", "数字芯片设计", "洗护用品", "纺织服装设备", "体外诊断", "综合电商",
    "磨具磨料", "纺织化学用品", "水产养殖", "化妆品制造及其他", "门户网站", "计量仪表", "电商服务", "游戏", "医疗设备", "肉鸡养殖",
    "其他纤维", "磷肥", "金融信息服务", "管材", "金属包装", "彩电", "调味发酵品", "蓄电池及其他电池", "胶黏剂及胶带", "熟食",
    "白银", "中间产品及消费品供应链服务", "农商行", "综合包装", "无机盐", "核力发电", "教育运营及其他", "锂电专用设备", "其他通信设备", "清洁小家电",
    "逆变器", "鞋帽", "其他种植业", "其他饰品", "房地产综合服务", "国有大型银行", "涂料", "镍", "股份制银行", "房地产开发",
    "横向通用软件", "综合", "火电", "医药流通", "底盘与发动机系统", "通信线缆及配套", "港口", "机场", "贸易", "风力发电",
    "工程机械整机", "中药", "酒店", "氯碱", "医院", "农药", "照明设备", "商用载货车", "动力煤", "金融控股",
    "原材料供应链服务", "保险", "其他石化", "钨", "塑料包装", "种子生产", "长材", "棉纺", "超市", "钢铁管材",
    "工程咨询服务", "钾肥", "影视动漫制作", "其他化学制品", "通信工程及服务", "果蔬加工", "跨境物流", "非金属新材料", "水产饲料", "乳品",
    "其它通用机械", "食品及饲料添加剂", "辅料", "品牌消费电子", "氨纶", "半导体材料", "快递", "LED", "动物保健", "聚氨酯",
    "水泥制品", "油田服务", "家纺", "其他养殖", "氟化工及制冷剂", "半导体设备", "其他能源发电", "食用菌", "农用机械", "橡胶助剂",
    "钴", "模拟芯片设计", "医美耗材", "其他专业服务", "电池化学品", "玻璃制造", "消费电子零部件及组装", "金属制品", "物业管理", "其他电子",
    "其他计算机设备", "航空运输", "光伏辅材", "证券", "水泥制造", "血液制品", "锂", "租赁", "大宗用纸", "基建市政工程",
    "百货", "垂直应用软件", "黄金", "地面兵装", "航运", "培训教育", "肉制品", "制冷空调设备", "钛白粉", "军工电子",
    "涂料油漆油墨制造", "白酒", "水电", "铝", "房屋建设", "被动元件", "保健品", "铁矿石", "营销代理", "畜禽饲料",
    "诊断服务", "零食", "焦炭", "锦纶", "印制电路板", "煤化工", "通信应用增值服务", "膜材料", "商用载客车", "汽车电子电气系统",
    "复合肥", "瓷砖地板", "其他农产品加工", "焦煤", "其他多元金融", "其他稀有小金属", "生活用纸", "非运动服装", "其他家居用品", "烘焙食品",
    "娱乐用品", "城商行", "合成树脂", "印刷包装机械", "卫浴制品", "特种纸", "厨房电器", "检测服务", "仪器仪表", "人工景点",
    "耐火材料", "玻纤制造", "文化用品", "其他交运设备", "钢结构", "配电设备", "风电整机", "其他橡胶制品", "其它电源设备", "防水材料",
    "跨境电商", "改性塑料", "风电零部件", "农业综合", "定制家居", "体育", "宠物食品", "船舶制造", "分立器件", "教育出版",
    "纺织鞋类制造", "其他数字媒体", "电信运营商", "印染", "粮食种植", "综合电力设备商", "钼"
]
INDUSTRY_STR = ", ".join(INDUSTRY_LIST)


class NewsDebateAnalyzer:
    """新闻行业影响辩论式分析器，裁决方同时参考新闻原文和辩论内容"""
    
    def __init__(self, llm_client: Optional[LocalLLMClient] = None, **llm_kwargs):
        self.llm_client = llm_client or create_client(** llm_kwargs)
        self.prompts = self._build_role_prompts()

    def _build_role_prompts(self) -> Dict[str, str]:
        """构建角色提示词，特别强化裁决方对新闻原文的依赖"""
        return {
            "pro": f"""你是专业财经分析师（正方），负责从新闻中挖掘利好行业/个股。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利好的行业；
2. 每个行业说明理由（≤40字），推荐1-2只A股个股（含代码）及理由（≤25字）；
3. 只输出利好，无利好返回空列表。
严格按JSON格式返回：
{{"positive": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}]}}""",

            "anti": f"""你是专业财经分析师（反方），负责挖掘利空并驳斥正方。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利空的行业；
2. 每个行业说明理由（≤40字），指出1-2只风险个股（含代码）及理由（≤25字）；
3. 针对性驳斥正方观点（≤50字），无驳斥则填空字符串；
4. 只输出利空，无利空返回空列表。
严格按JSON格式返回：
{{"negative": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"refute":"驳斥内容"}}""",

            "judge": f"""你是资深财经裁决专家，需同时结合新闻原文和正反方观点进行裁决。
任务要求：
1. 仔细阅读新闻原文，分析其核心信息和潜在影响；
2. 参考所有正方的利好分析和反方的利空分析及驳斥观点；
3. 筛选出真实利好的行业（排除反方合理驳斥的虚假利好）；
4. 每个行业需说明综合理由（结合新闻原文和辩论，≤80字）；
5. 为每个行业保留1-2只最具代表性的A股个股（含代码）；
6. 只输出最终确认的利好，无利好返回空列表。
严格按JSON格式返回：
{{"final_positive": [{{"industry":"行业名","comprehensive_reason":"综合理由","stocks":[{{"name":"股名","code":"代码","reason":"推荐理由"}}]}}]}}"""
        }

    def _call_llm(self, role: str, content: str, extra_context: str = "") -> Dict:
        """通用LLM调用函数，确保裁决方同时获取新闻原文和辩论上下文"""
        # 关键修改：裁决方输入同时包含新闻原文和辩论汇总
        if role == "judge" and extra_context:
            user_input = f"新闻原文：{content}\n\n辩论汇总：{extra_context}\n请基于新闻原文和上述辩论给出最终裁决"
        elif role == "anti" and extra_context:
            user_input = f"正方观点：{extra_context}\n新闻内容：{content}"
        else:
            user_input = content

        try:
            temp = 0.3 if role != "judge" else 0.2
            result = self.llm_client.single_chat(
                user_message=user_input,
                system_prompt=self.prompts[role],
                temperature=temp,
                max_tokens=10000
            )

            if result["status"] != "success" or not result["response"]:
                return {"error": f"LLM调用失败：{result.get('error', '未知错误')}"}

            clean_response = result["response"].strip().strip("`").strip("json").strip()
            return json.loads(clean_response)

        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败：{str(e)}，原始输出：{result['response'][:200]}..."}
        except Exception as e:
            return {"error": f"处理失败：{str(e)}"}

    def analyze_single_news(self, news_content: str, pro_count: int = 2, anti_count: int = 3) -> Dict:
        """单条新闻分析流程，保持角色数量参数化"""
        # 1. 多正方分析
        pro_opinions = []
        for i in range(pro_count):
            pro_res = self._call_llm(role="pro", content=news_content)
            pro_opinions.append({
                "pro_id": f"pro_{i+1}",
                "opinion": pro_res.get("positive", []),
                "error": pro_res.get("error", "")
            })

        # 2. 多反方分析
        pro_summary = json.dumps(pro_opinions, ensure_ascii=False)
        anti_opinions = []
        for i in range(anti_count):
            anti_res = self._call_llm(role="anti", content=news_content, extra_context=pro_summary)
            anti_opinions.append({
                "anti_id": f"anti_{i+1}",
                "opinion": anti_res.get("negative", []),
                "refute": anti_res.get("refute", ""),
                "error": anti_res.get("error", "")
            })

        # 3. 裁决方分析（关键修改：同时传入新闻原文和辩论汇总）
        debate_summary = json.dumps({
            "pro_opinions": pro_opinions,
            "anti_opinions": anti_opinions
        }, ensure_ascii=False)
        # 裁决方同时接收新闻原文（content）和辩论汇总（debate_summary）
        judge_res = self._call_llm(role="judge", content=news_content, extra_context=debate_summary)

        return {
            "news_content": news_content,
            "debate_process": {
                "pro_count": pro_count,
                "anti_count": anti_count,
                "pro_opinions": pro_opinions,
                "anti_opinions": anti_opinions
            },
            "final_verdict": judge_res.get("final_positive", []),
            "error": judge_res.get("error", "")
        }

    def batch_analyze_news(self, news_df: pd.DataFrame, pro_count: int = 2, anti_count: int = 3) -> List[Dict]:
        """批量分析新闻，保持原有逻辑"""
        batch_result = []
        total_news = len(news_df)
        for idx, row in news_df.iterrows():
            news = f"标题：{row['title']}\n内容：{row['content']}".replace("\n\n", "\n").strip()
            print(f"正在分析第 {idx+1}/{total_news} 条新闻...")

            single_result = self.analyze_single_news(
                news_content=news,
                pro_count=pro_count,
                anti_count=anti_count
            )
            single_result["news_date"] = row["date"]
            batch_result.append(single_result)

        print(f"批量分析完成！共处理 {total_news} 条新闻")
        return batch_result

    def export_verdict_to_csv(self, batch_result: List[Dict], output_path: str) -> bool:
        """导出裁决结果到CSV，格式保持不变"""
        csv_fields = [
            "news_date", "industry", "comprehensive_reason",
            "stock_name", "stock_code", "stock_reason"
        ]

        try:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()

                for news_res in batch_result:
                    news_date = news_res.get("news_date", "")
                    final_industries = news_res.get("final_verdict", [])

                    if not final_industries:
                        continue

                    for industry_info in final_industries:
                        industry = industry_info.get("industry", "")
                        comp_reason = industry_info.get("comprehensive_reason", "")
                        stocks = industry_info.get("stocks", [])

                        if not stocks:
                            writer.writerow({
                                "news_date": news_date,
                                "industry": industry,
                                "comprehensive_reason": comp_reason,
                                "stock_name": "",
                                "stock_code": "",
                                "stock_reason": ""
                            })
                        else:
                            for stock in stocks:
                                writer.writerow({
                                    "news_date": news_date,
                                    "industry": industry,
                                    "comprehensive_reason": comp_reason,
                                    "stock_name": stock.get("name", ""),
                                    "stock_code": stock.get("code", ""),
                                    "stock_reason": stock.get("reason", "")
                                })

            print(f"裁决结果已成功导出到：{output_path}")
            return True

        except Exception as e:
            print(f"CSV导出失败：{str(e)}")
            return False


# 使用示例
if __name__ == "__main__":
    PRO_COUNT = 2
    ANTI_COUNT = 3
    NEWS_START_DAYS = 5
    CSV_OUTPUT_PATH = f"./news_final_verdict_{datetime.today().strftime('%Y%m%d')}.csv"

    analyzer = NewsDebateAnalyzer()

    try:
        news_df = pd.read_parquet('../data/stock_daily_cctvnews.parquet')
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce").dt.strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")
        start_date = (datetime.today() - timedelta(days=NEWS_START_DAYS)).strftime("%Y%m%d")
        filtered_df = news_df[(news_df["date"] >= start_date) & (news_df["date"] <= end_date)].reset_index(drop=True)
        print(f"筛选出 {len(filtered_df)} 条新闻（{start_date} ~ {end_date}）")
    except Exception as e:
        print(f"新闻数据读取失败：{str(e)}")
        exit(1)

    if len(filtered_df) > 0:
        batch_result = analyzer.batch_analyze_news(
            news_df=filtered_df,
            pro_count=PRO_COUNT,
            anti_count=ANTI_COUNT
        )
        analyzer.export_verdict_to_csv(batch_result, CSV_OUTPUT_PATH)
    else:
        print("无符合条件的新闻，无需分析")
