# 封装为函数

import json
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from client import LocalLLMClient, create_client  # 假设LLM客户端类已存在

# --------------------------
# 1. 基础配置（固定参数）
# --------------------------
# 日志配置（仅初始化一次）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 预设细分行业列表
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

# 创建日志目录
Path("debate_logs").mkdir(exist_ok=True)


# --------------------------
# 2. 核心封装函数：输入标题+内容，输出裁决结果
# --------------------------
def analyze_news_single(
    news_title: str, 
    news_content: str, 
    debate_rounds: int = 2,
    llm_client: Optional[LocalLLMClient] = None,
    **llm_kwargs
) -> Dict[str, Any]:
    """
    单条新闻分析函数：输入新闻标题和内容，返回大模型裁决的行业列表（含信念值、个股推荐）
    
    参数：
    - news_title: 新闻标题（str）
    - news_content: 新闻正文（str）
    - debate_rounds: 辩论轮次（默认2轮，每轮含正方回应+反方驳斥）
    - llm_client: 已实例化的LLM客户端（可选，无则自动创建）
    - **llm_kwargs: 创建LLM客户端的参数（如api_key、model等）
    
    返回：
    - Dict: 包含以下key：
      - "status": 分析状态（"success"/"failed"）
      - "error_msg": 错误信息（status为failed时非空）
      - "log_path": 辩论日志文件路径（便于追溯）
      - "ruled_industries": 裁决行业列表（status为success时非空，每个元素含：
        - industry: 行业名称
        - impact: 影响方向（"利好"/"利空"）
        - confidence: 信念强度（1-10分）
        - comprehensive_reason: 综合理由
        - stocks: 个股推荐列表（含name/code/reason）
      ）
    """
    # 初始化LLM客户端
    if not llm_client:
        try:
            llm_client = create_client(**llm_kwargs)
        except Exception as e:
            return {
                "status": "failed",
                "error_msg": f"LLM客户端初始化失败：{str(e)}",
                "log_path": "",
                "ruled_industries": []
            }

    # --------------------------
    # 内部辅助方法（复用原代码核心逻辑）
    # --------------------------
    def _build_prompts() -> Dict[str, str]:
        """构建角色提示词"""
        return {
            "pro_initial": f"""你是专业游资，以读取新闻预判消息炒股为生（正方），你需要从机构能否认同的角度，以宁缺毋滥的态度，负责从新闻中挖掘利好行业/个股。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利好的行业；
2. 每个行业说明理由（≤100字），推荐1-2只A股个股（含代码）及理由（≤100字）；
3. 只输出利好，无利好返回空列表。
严格按JSON格式返回，不添加任何额外内容：
{{"positive": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}]}}""",

            "anti_initial": f"""你是专业游资，以读取新闻预判消息炒股为生（反方），你需要从机构能否认同的角度，以宁缺毋滥的态度，负责挖掘利空并驳斥正方。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利空的行业；
2. 每个行业说明理由（≤100字），指出1-2只风险个股（含代码）及理由（≤100字）；
3. 针对性驳斥正方首次提出的利好观点（≤80字）；
4. 只输出利空，无利空返回空列表。
严格按JSON格式返回，不添加任何额外内容：
{{"negative": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"refute":"驳斥内容"}}""",

            "pro_rebuttal": f"""你是专业游资，以读取新闻预判消息炒股为生（正方），你需要从游资、散户能否认同的角度，以宁缺毋滥的态度，负责回应反方驳斥并强化观点。
任务要求：
1. 针对反方的驳斥内容进行有理有据的回应（≤100字）；
2. 进一步强化你的利好观点，可补充新的理由或个股；
3. 格式与首次观点一致，只输出利好相关内容。
严格按JSON格式返回，不添加任何额外内容：
{{"positive": [{{"industry":"行业名","reason":"理由（含回应）","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"response":"对反方的回应"}}""",

            "anti_rebuttal": f"""你是专业游资，以读取新闻预判消息炒股为生（反方），你需要从游资、散户能否认同的角度，以宁缺毋滥的态度，负责继续驳斥正方观点。
任务要求：
1. 针对正方的回应内容进行进一步驳斥（≤100字）；
2. 可补充新的利空理由或强化原有观点；
3. 格式与首次观点一致。
严格按JSON格式返回，不添加任何额外内容：
{{"negative": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"refute":"进一步驳斥内容"}}""",

            "judge": f"""你是资深财经裁决专家，需同时结合新闻原文和完整辩论过程进行裁决。
任务要求：
1. 仔细阅读新闻原文，分析其核心信息和潜在影响；
2. 完整回顾正反方多轮辩论的全部观点和驳斥内容；
3. 对每个行业明确判断是利好还是利空；
4. 对判断结果给出信念强度（1-10分，10分为最确定）；
5. 每个行业需说明综合理由（结合新闻原文和全部辩论，≤150字）；
6. 为每个行业保留1-2只最具代表性的A股个股（含代码）；
7. 只输出最终确认的判断，无明确判断返回空列表。
严格按JSON格式返回，不添加任何额外内容，确保JSON格式正确：
{{"final_judgment": [{{
    "industry":"行业名",
    "impact":"利好或利空",
    "confidence":1-10的数字,
    "comprehensive_reason":"综合理由",
    "stocks":[{{"name":"股名","code":"代码","reason":"推荐理由"}}]
}}]}}"""
        }

    def _extract_json(response: str) -> Optional[str]:
        """提取并修复JSON"""
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
        json_str = response[start_idx:end_idx+1]
        json_str = json_str.replace("'", '"').replace('\n', ' ').replace(', ]', ' ]').replace(', }', ' }')
        return json_str

    def _call_llm(role: str, content: str, extra_context: str = "") -> Dict:
        """LLM调用（带重试）"""
        prompts = _build_prompts()
        if role not in prompts:
            return {"error": f"无效角色：{role}"}

        # 构建输入
        if role == "judge" and extra_context:
            user_input = f"新闻原文：{content}\n\n完整辩论过程：{extra_context}\n请基于新闻原文和上述完整辩论给出最终裁决"
        elif "rebuttal" in role and extra_context:
            user_input = f"之前的辩论内容：{extra_context}\n新闻内容：{content}\n请根据上述信息进行回应"
        elif role == "anti_initial" and extra_context:
            user_input = f"正方首次观点：{extra_context}\n新闻内容：{content}\n请针对正方观点进行驳斥并提出利空"
        else:
            user_input = content

        # 温度配置
        temp_map = {"pro_initial": 0.4, "anti_initial": 0.4, "pro_rebuttal": 0.5, "anti_rebuttal": 0.5, "judge": 0.2}
        temperature = temp_map.get(role, 0.3)

        # 3次重试
        for attempt in range(3):
            try:
                result = llm_client.single_chat(
                    user_message=user_input,
                    system_prompt=prompts[role],
                    temperature=temperature,
                    max_tokens=20000
                )
                if result["status"] != "success" or not result["response"]:
                    raise Exception(f"LLM调用失败：{result.get('error', '未知错误')}")

                # 解析JSON
                clean_json = _extract_json(result["response"]) or result["response"].strip().strip("`").strip("json").strip()
                return json.loads(clean_json)
            except Exception as e:
                if attempt == 2:
                    return {"error": f"LLM调用失败（{attempt+1}次重试）：{str(e)}"}
                logger.warning(f"LLM调用尝试 {attempt+1} 失败，重试中...")

    # --------------------------
    # 核心分析流程
    # --------------------------
    # 1. 生成唯一新闻ID
    news_id = f"news_single_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(news_title[:20]) % 1000:03d}"
    full_news = f"标题：{news_title}\n内容：{news_content}".replace("\n\n", "\n").strip()
    logger.info(f"开始分析单条新闻（ID：{news_id}，标题：{news_title[:30]}...）")

    # 2. 存储辩论历史
    debate_history = {
        "news_id": news_id,
        "news_title": news_title,
        "news_content": news_content,
        "rounds": debate_rounds,
        "debate_process": [],
        "final_verdict": [],
        "errors": []
    }

    try:
        # 3. 第一轮：正方初始观点 → 反方初始驳斥
        # 正方初始
        pro_initial = _call_llm(role="pro_initial", content=full_news)
        debate_history["debate_process"].append({"round": 1, "role": "pro_initial", "content": pro_initial})
        if "error" in pro_initial:
            raise Exception(f"正方初始观点失败：{pro_initial['error']}")

        # 反方初始驳斥
        anti_initial = _call_llm(
            role="anti_initial",
            content=full_news,
            extra_context=json.dumps(pro_initial, ensure_ascii=False)
        )
        debate_history["debate_process"].append({"round": 1, "role": "anti_initial", "content": anti_initial})
        if "error" in anti_initial:
            raise Exception(f"反方初始驳斥失败：{anti_initial['error']}")

        # 4. 多轮辩论：正方回应 → 反方再驳斥
        current_context = json.dumps({"pro_initial": pro_initial, "anti_initial": anti_initial}, ensure_ascii=False)
        for round_num in range(2, debate_rounds + 1):
            # 正方回应
            pro_resp = _call_llm(role="pro_rebuttal", content=full_news, extra_context=current_context)
            debate_history["debate_process"].append({"round": round_num, "role": "pro_rebuttal", "content": pro_resp})
            if "error" in pro_resp:
                raise Exception(f"第 {round_num} 轮正方回应失败：{pro_resp['error']}")

            # 反方再驳斥
            current_context = json.dumps({"previous": current_context, "pro_resp": pro_resp}, ensure_ascii=False)
            anti_resp = _call_llm(role="anti_rebuttal", content=full_news, extra_context=current_context)
            debate_history["debate_process"].append({"round": round_num, "role": "anti_rebuttal", "content": anti_resp})
            if "error" in anti_resp:
                raise Exception(f"第 {round_num} 轮反方驳斥失败：{anti_resp['error']}")
            current_context = json.dumps({"previous": current_context, "anti_resp": anti_resp}, ensure_ascii=False)

        # 5. 裁决方最终裁决
        judge_result = _call_llm(
            role="judge",
            content=full_news,
            extra_context=json.dumps(debate_history["debate_process"], ensure_ascii=False)
        )
        if "error" in judge_result:
            raise Exception(f"裁决失败：{judge_result['error']}")
        debate_history["final_verdict"] = judge_result.get("final_judgment", [])

        # 6. 保存辩论日志
        log_path = Path(f"debate_logs/news_{news_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(debate_history, f, ensure_ascii=False, indent=2)
        logger.info(f"单条新闻分析完成，日志保存至：{log_path}")

        # 7. 整理返回结果
        return {
            "status": "success",
            "error_msg": "",
            "log_path": str(log_path),
            "ruled_industries": debate_history["final_verdict"]  # 含industry/impact/confidence/stocks
        }

    except Exception as e:
        # 分析失败处理
        error_msg = str(e)
        logger.error(f"单条新闻分析失败：{error_msg}")
        # 保存错误日志
        log_path = Path(f"debate_logs/news_{news_id}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        debate_history["errors"].append(error_msg)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(debate_history, f, ensure_ascii=False, indent=2)
        return {
            "status": "failed",
            "error_msg": error_msg,
            "log_path": str(log_path),
            "ruled_industries": []
        }


# --------------------------
# 3. 使用示例
# --------------------------
if __name__ == "__main__":
    # 示例1：直接调用函数（输入新闻标题+内容）
    test_news_title = "国家加大半导体国产替代支持力度，2024年专项补贴提升30%"
    test_news_content = "近日，工信部发布《半导体产业发展规划（2024-2026）》，明确加大国产替代支持力度，2024年专项补贴资金较去年提升30%，重点支持半导体设备、材料等关键环节企业，鼓励企业突破14nm以下先进制程技术。同时，将建立半导体产业链协同平台，促进上下游企业合作。"

    # 调用封装函数
    result = analyze_news_single(
        news_title=test_news_title,
        news_content=test_news_content,
        debate_rounds=2  # 辩论2轮
        # 若需自定义LLM客户端，可添加参数：llm_client=your_client 或 model="gpt-4", api_key="xxx"
    )

    # 打印结果
    print("="*50)
    print(f"分析状态：{result['status']}")
    if result["status"] == "success":
        print(f"裁决行业数量：{len(result['ruled_industries'])}")
        for idx, industry in enumerate(result["ruled_industries"], 1):
            print(f"\n{idx}. 行业：{industry['industry']}")
            print(f"   影响方向：{industry['impact']}")
            print(f"   信念强度：{industry['confidence']}分")
            print(f"   综合理由：{industry['comprehensive_reason']}")
            print(f"   推荐个股：")
            for stock in industry['stocks']:
                print(f"     - {stock['name']}（{stock['code']}）：{stock['reason']}")
    else:
        print(f"错误信息：{result['error_msg']}")
    print(f"日志路径：{result['log_path']}")
    print("="*50)