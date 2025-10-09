import json
from typing import Dict, List, Optional
from client import LocalLLMClient, create_client  # 假设LLM客户端类在LocalLLMClient模块中

# 定义需要分析的行业列表
# INDUSTRY_LIST = [
#     "电气设备I", "电子I", "公用事业I", "轻工制造I", "国防军工I", 
#     "食品饮料I", "银行I", "计算机I", "建筑材料I", "纺织服装I", 
#     "化工I", "休闲服务I", "煤炭I", "商业贸易I", "机械设备I", 
#     "建筑装饰I", "医药生物I", "石油石化I", "交通运输I", "传媒I", 
#     "美容护理I", "钢铁I", "房地产I", "家用电器I", "农林牧渔I", 
#     "综合I", "汽车I", "环保I", "有色金属I", "通信I", "非银金融I"
# ]

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

class NewsIndustryAnalyzer:
    """新闻行业影响分析器，用于分析新闻对行业和个股的利好影响"""
    
    def __init__(self, llm_client: Optional[LocalLLMClient] = None, **llm_kwargs):
        """
        初始化分析器
        
        Args:
            llm_client: 已实例化的LLM客户端，若为None则自动创建
            llm_kwargs: 创建LLM客户端的参数
        """
        self.llm_client = llm_client or create_client(** llm_kwargs)
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词，指导LLM进行分析"""
        industries_str = ", ".join(INDUSTRY_LIST)
        
        return f"""你是一位专业的财经分析师，擅长分析新闻对股市行业板块和个股的影响。

请根据提供的新闻内容，完成以下任务：
1. 从列表中[{industries_str}]识别出受到重大利好影响的行业板块，(必须是非常重大的利好! 受到机构和广大散户认同的，宁缺毋滥)，最多选择5个最相关的行业
2. 对每个利好行业，简要说明理由（不超过50字）
3. 为每个利好行业推荐1-3只相关的代表性个股（需是A股上市公司，注明股票代码）
4. 对每个推荐的个股，简要说明推荐理由（不超过30字）

分析结果请严格按照以下JSON格式返回，不要添加任何额外内容：
{{
    "positive_industries": [
        {{
            "industry": "行业名称（必须从提供的列表中选择）",
            "reason": "利好理由",
            "related_stocks": [
                {{
                    "stock_name": "股票名称",
                    "stock_code": "股票代码（如600000.XSHG）",
                    "reason": "推荐理由"
                }}
            ]
        }}
    ]
}}

注意：
- 只返回利好行业，不提及利空行业
- 确保行业名称与提供的列表完全一致
- 股票代码格式必须正确，确保是A股上市公司
- 如果没有明显利好的行业，返回空列表
"""
    
    def analyze_news(self, news_content: str) -> Dict:
        """
        分析新闻内容，识别利好行业和相关个股
        
        Args:
            news_content: 新闻内容文本
            
        Returns:
            包含利好行业和个股的分析结果字典
        """
        try:
            # 调用LLM进行分析
            result = self.llm_client.single_chat(
                user_message=news_content,
                system_prompt=self.system_prompt,
                temperature=0.3,  # 降低随机性，提高结果稳定性
                max_tokens=100000
            )
            
            if result["status"] != "success" or not result["response"]:
                return {"error": f"LLM调用失败: {result.get('error', '未知错误')}", "positive_industries": []}
            
            # 解析LLM返回的JSON结果
            analysis_result = json.loads(result["response"])
            
            # 验证结果格式
            if "positive_industries" not in analysis_result:
                return {"error": "LLM返回格式错误，缺少positive_industries字段", "positive_industries": []}
                
            return analysis_result
            
        except json.JSONDecodeError as e:
            return {"error": f"解析LLM结果失败: {str(e)}", "positive_industries": []}
        except Exception as e:
            return {"error": f"分析过程出错: {str(e)}", "positive_industries": []}
    
    def format_analysis_result(self, analysis_result: Dict) -> str:
        """
        将分析结果格式化为易读的字符串
        
        Args:
            analysis_result: 分析结果字典
            
        Returns:
            格式化后的字符串
        """
        if "error" in analysis_result and analysis_result["error"]:
            return f"分析出错: {analysis_result['error']}"
        
        industries = analysis_result.get("positive_industries", [])
        
        if not industries:
            return "未发现对任何行业板块有明显利好影响。"
        
        output = []
        output.append("新闻利好行业及个股分析：")
        output.append("=" * 50)
        
        for i, industry_info in enumerate(industries, 1):
            output.append(f"\n{i}. 利好行业：{industry_info['industry']}")
            output.append(f"   利好理由：{industry_info['reason']}")
            output.append("   相关个股：")
            
            for stock in industry_info["related_stocks"]:
                output.append(f"   - {stock['stock_name']}（{stock['stock_code']}）：{stock['reason']}")
        
        output.append("\n" + "=" * 50)
        output.append("分析结束")
        
        return "\n".join(output)


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = NewsIndustryAnalyzer()
    import pandas as pd
    from datetime import datetime
    
    df = pd.read_parquet('../data/stock_daily_cctvnews.parquet')
    
    # 获取date 等于今日的新闻
    today = datetime.today().strftime('%Y%m%d')
    df = df[df['date'] == today]
    
    # 循环遍历df，将title、content组合为news，传入大模型进行分析
    # 将news 和 result 保存在列表中，然后将列表保存为md文件
    analysis_list = []
    
    for index, row in df.iterrows():
        news = f"{row['title']} {row['content']}"
        result = analyzer.analyze_news(news)
        analysis_list.append({"news": news, "result": result})
    
    with open(f"./news_analysis_{today}.md", "w", encoding="utf-8") as f:
        for item in analysis_list:
            if(item["result"]=='未发现对任何行业板块有明显利好影响。'):
                continue
            f.write(item["news"]+"\n\n" )
            f.write(analyzer.format_analysis_result(item["result"]))
            f.write("\n\n" + "=" * 50 + "\n\n")
    
    with open(f"./news_analysis_result_{today}.md", "w", encoding="utf-8") as f:
        for item in analysis_list:
            if(item["result"]=='未发现对任何行业板块有明显利好影响。'):
                continue
            f.write(analyzer.format_analysis_result(item["result"]))
            f.write("\n\n" + "=" * 50 + "\n\n")
    