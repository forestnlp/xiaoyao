# 增加了分值。

import json
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from client import LocalLLMClient, create_client  # 假设LLM客户端类已存在

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    """新闻行业影响辩论式分析器，支持多轮辩论和详细日志记录"""
    
    def __init__(self, llm_client: Optional[LocalLLMClient] = None, **llm_kwargs):
        self.llm_client = llm_client or create_client(** llm_kwargs)
        self.prompts = self._build_role_prompts()
        # 创建日志存储目录
        Path("debate_logs").mkdir(exist_ok=True)

    def _build_role_prompts(self) -> Dict[str, str]:
        """构建角色提示词，支持多轮辩论和确定性输出"""
        return {
            "pro_initial": f"""你是专业财经分析师（正方），负责从新闻中挖掘利好行业/个股。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利好的行业；
2. 每个行业说明理由（≤40字），推荐1-2只A股个股（含代码）及理由（≤25字）；
3. 只输出利好，无利好返回空列表。
严格按JSON格式返回，不添加任何额外内容：
{{"positive": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}]}}""",

            "anti_initial": f"""你是专业财经分析师（反方），负责挖掘利空并驳斥正方。
任务要求：
1. 从行业列表[{INDUSTRY_STR}]中，识别最多3个受新闻重大利空的行业；
2. 每个行业说明理由（≤40字），指出1-2只风险个股（含代码）及理由（≤25字）；
3. 针对性驳斥正方首次提出的利好观点（≤80字）；
4. 只输出利空，无利空返回空列表。
严格按JSON格式返回，不添加任何额外内容：
{{"negative": [{{"industry":"行业名","reason":"理由","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"refute":"驳斥内容"}}""",

            "pro_rebuttal": f"""你是专业财经分析师（正方），负责回应反方驳斥并强化观点。
任务要求：
1. 针对反方的驳斥内容进行有理有据的回应（≤100字）；
2. 进一步强化你的利好观点，可补充新的理由或个股；
3. 格式与首次观点一致，只输出利好相关内容。
严格按JSON格式返回，不添加任何额外内容：
{{"positive": [{{"industry":"行业名","reason":"理由（含回应）","stocks":[{{"name":"股名","code":"代码","reason":"理由"}}]}}],"response":"对反方的回应"}}""",

            "anti_rebuttal": f"""你是专业财经分析师（反方），负责继续驳斥正方观点。
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

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """从LLM响应中提取JSON部分，解决格式错误问题"""
        # 尝试找到JSON开始和结束的位置
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
            
        # 提取可能的JSON部分
        json_str = response[start_idx:end_idx+1]
        
        # 尝试修复常见的JSON错误
        json_str = json_str.replace("'", '"')  # 替换单引号为双引号
        json_str = json_str.replace('\n', ' ')  # 移除换行符
        
        # 检查是否有多余的逗号
        json_str = json_str.replace(', ]', ' ]').replace(', }', ' }')
        
        return json_str

    def _call_llm_with_retry(self, role: str, content: str, extra_context: str = "", max_retries: int = 3) -> Dict:
        """带重试机制的LLM调用函数，解决JSON解析错误"""
        for attempt in range(max_retries):
            try:
                # 根据角色构建不同的用户输入
                if role == "judge" and extra_context:
                    user_input = f"新闻原文：{content}\n\n完整辩论过程：{extra_context}\n请基于新闻原文和上述完整辩论给出最终裁决"
                elif "rebuttal" in role and extra_context:
                    user_input = f"之前的辩论内容：{extra_context}\n新闻内容：{content}\n请根据上述信息进行回应"
                elif role == "anti_initial" and extra_context:
                    user_input = f"正方首次观点：{extra_context}\n新闻内容：{content}\n请针对正方观点进行驳斥并提出利空"
                else:
                    user_input = content

                # 根据角色设置不同温度
                temp_map = {
                    "pro_initial": 0.4,
                    "anti_initial": 0.4,
                    "pro_rebuttal": 0.5,
                    "anti_rebuttal": 0.5,
                    "judge": 0.2
                }
                temperature = temp_map.get(role, 0.3)
                
                # 增大token限制以处理长文本
                result = self.llm_client.single_chat(
                    user_message=user_input,
                    system_prompt=self.prompts[role],
                    temperature=temperature,
                    max_tokens=20000  # 大幅增加token容量
                )

                if result["status"] != "success" or not result["response"]:
                    error_msg = f"LLM调用失败：{result.get('error', '未知错误')}"
                    logger.error(f"尝试 {attempt+1}/{max_retries} {error_msg}")
                    if attempt == max_retries - 1:
                        return {"error": error_msg}
                    continue

                # 提取并清理响应中的JSON部分
                response_text = result["response"].strip()
                logger.debug(f"LLM {role} 原始响应: {response_text[:500]}...")
                
                # 尝试从响应中提取有效的JSON
                json_str = self._extract_json_from_response(response_text)
                if not json_str:
                    json_str = response_text.strip().strip("`").strip("json").strip()
                
                # 尝试解析JSON
                return json.loads(json_str)

            except json.JSONDecodeError as e:
                error_msg = f"JSON解析失败：{str(e)}，原始输出：{response_text[:500]}..."
                logger.error(f"尝试 {attempt+1}/{max_retries} {error_msg}")
                if attempt == max_retries - 1:
                    return {"error": error_msg}
            except Exception as e:
                error_msg = f"处理失败：{str(e)}"
                logger.error(f"尝试 {attempt+1}/{max_retries} {error_msg}")
                if attempt == max_retries - 1:
                    return {"error": error_msg}
        
        return {"error": "达到最大重试次数，调用失败"}

    def analyze_single_news(self, news_content: str, news_id: str, debate_rounds: int = 2) -> Dict:
        """单条新闻分析流程，支持多轮辩论"""
        logger.info(f"开始分析新闻 {news_id}，辩论轮次: {debate_rounds}")
        
        # 存储完整辩论过程和日志
        debate_history = {
            "news_id": news_id,
            "news_content": news_content,
            "rounds": debate_rounds,
            "debate_process": [],
            "final_verdict": None,
            "errors": []
        }

        # 1. 第一轮：正方初始观点
        logger.info(f"新闻 {news_id} - 第一轮：正方初始观点")
        pro_initial = self._call_llm_with_retry(role="pro_initial", content=news_content)
        debate_step = {
            "round": 1,
            "role": "pro_initial",
            "content": pro_initial
        }
        debate_history["debate_process"].append(debate_step)
        
        if "error" in pro_initial:
            error_msg = f"正方初始观点生成失败: {pro_initial['error']}"
            debate_history["errors"].append(error_msg)
            logger.error(error_msg)

        # 2. 第一轮：反方初始驳斥
        logger.info(f"新闻 {news_id} - 第一轮：反方初始驳斥")
        pro_initial_summary = json.dumps(pro_initial, ensure_ascii=False)
        anti_initial = self._call_llm_with_retry(
            role="anti_initial", 
            content=news_content, 
            extra_context=pro_initial_summary
        )
        debate_step = {
            "round": 1,
            "role": "anti_initial",
            "content": anti_initial
        }
        debate_history["debate_process"].append(debate_step)
        
        if "error" in anti_initial:
            error_msg = f"反方初始驳斥生成失败: {anti_initial['error']}"
            debate_history["errors"].append(error_msg)
            logger.error(error_msg)

        # 3. 多轮辩论：正方回应与反方再驳斥
        current_context = json.dumps({
            "pro_initial": pro_initial,
            "anti_initial": anti_initial
        }, ensure_ascii=False)
        
        for round_num in range(2, debate_rounds + 1):
            # 正方回应
            logger.info(f"新闻 {news_id} - 第 {round_num} 轮：正方回应")
            pro_response = self._call_llm_with_retry(
                role="pro_rebuttal",
                content=news_content,
                extra_context=current_context
            )
            debate_step = {
                "round": round_num,
                "role": "pro_rebuttal",
                "content": pro_response
            }
            debate_history["debate_process"].append(debate_step)
            
            if "error" in pro_response:
                error_msg = f"第 {round_num} 轮正方回应失败: {pro_response['error']}"
                debate_history["errors"].append(error_msg)
                logger.error(error_msg)
                # 如果正方回应失败，本轮反方驳斥也跳过
                continue

            # 反方再驳斥
            logger.info(f"新闻 {news_id} - 第 {round_num} 轮：反方再驳斥")
            current_context = json.dumps({
                "previous_context": current_context,
                "pro_response": pro_response
            }, ensure_ascii=False)
            
            anti_response = self._call_llm_with_retry(
                role="anti_rebuttal",
                content=news_content,
                extra_context=current_context
            )
            debate_step = {
                "round": round_num,
                "role": "anti_rebuttal",
                "content": anti_response
            }
            debate_history["debate_process"].append(debate_step)
            
            if "error" in anti_response:
                error_msg = f"第 {round_num} 轮反方驳斥失败: {anti_response['error']}"
                debate_history["errors"].append(error_msg)
                logger.error(error_msg)
                
            current_context = json.dumps({
                "previous_context": current_context,
                "anti_response": anti_response
            }, ensure_ascii=False)

        # 4. 裁决方最终裁决
        logger.info(f"新闻 {news_id} - 裁决方最终裁决")
        debate_summary = json.dumps(debate_history["debate_process"], ensure_ascii=False)
        judge_result = self._call_llm_with_retry(
            role="judge",
            content=news_content,
            extra_context=debate_summary
        )
        debate_history["final_verdict"] = judge_result.get("final_judgment", [])
        debate_history["error"] = judge_result.get("error", "")
        
        if debate_history["error"]:
            logger.error(f"新闻 {news_id} 裁决失败: {debate_history['error']}")
        else:
            logger.info(f"新闻 {news_id} 分析完成，识别到 {len(debate_history['final_verdict'])} 个行业判断")

        # 保存完整辩论日志
        log_path = Path(f"debate_logs/news_{news_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(debate_history, f, ensure_ascii=False, indent=2)
        logger.info(f"新闻 {news_id} 辩论日志已保存至 {log_path}")

        return {
            "news_id": news_id,
            "news_content": news_content,
            "debate_rounds": debate_rounds,
            "final_verdict": debate_history["final_verdict"],
            "error": debate_history["error"],
            "log_path": str(log_path)
        }

    def batch_analyze_news(self, news_df: pd.DataFrame, debate_rounds: int = 2) -> List[Dict]:
        """批量分析新闻"""
        batch_result = []
        total_news = len(news_df)
        required_columns = ['title', 'content', 'date']
        missing = [c for c in required_columns if c not in news_df.columns]
        
        if missing:
            error_msg = f"数据缺少必要列：{', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        for idx, row in news_df.iterrows():
            try:
                news_id = f"news_{idx+1}_{row['date']}_{hash(row['title'][:20]) % 1000:03d}"
                news_content = f"标题：{row['title']}\n内容：{row['content']}".replace("\n\n", "\n").strip()
                logger.info(f"开始分析第 {idx+1}/{total_news} 条新闻 (ID: {news_id})")

                # 分析单条新闻
                single_result = self.analyze_single_news(
                    news_content=news_content,
                    news_id=news_id,
                    debate_rounds=debate_rounds
                )
                
                # 补充新闻元数据
                single_result["news_date"] = row["date"]
                single_result["news_title"] = row["title"]
                single_result["status"] = "completed" if not single_result["error"] else "failed"
                
                batch_result.append(single_result)
                logger.info(f"第 {idx+1}/{total_news} 条新闻分析完成")
                
            except Exception as e:
                error_msg = f"第 {idx+1}/{total_news} 条新闻分析过程出错: {str(e)}"
                logger.error(error_msg)
                batch_result.append({
                    "news_id": f"error_{idx+1}",
                    "news_title": row.get("title", ""),
                    "news_date": row.get("date", ""),
                    "status": "failed",
                    "error": error_msg,
                    "final_verdict": []
                })

        logger.info(f"批量分析完成！共处理 {total_news} 条新闻")
        return batch_result

    def export_verdict_to_csv(self, batch_result: List[Dict], output_path: str) -> bool:
        """导出裁决结果到CSV，包含利好/利空标识和信念强度"""
        csv_fields = [
            "news_id", "news_date", "news_title", "industry", "impact", "confidence",
            "comprehensive_reason", "stock_name", "stock_code", "stock_reason", "error"
        ]

        try:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()

                for news_res in batch_result:
                    news_id = news_res.get("news_id", "")
                    news_date = news_res.get("news_date", "")
                    news_title = news_res.get("news_title", "")
                    final_judgments = news_res.get("final_verdict", [])
                    error = news_res.get("error", "")

                    # 处理分析失败的新闻
                    if news_res.get("status") == "failed":
                        writer.writerow({
                            "news_id": news_id,
                            "news_date": news_date,
                            "news_title": news_title,
                            "industry": "",
                            "impact": "",
                            "confidence": "",
                            "comprehensive_reason": "",
                            "stock_name": "",
                            "stock_code": "",
                            "stock_reason": "",
                            "error": error
                        })
                        continue

                    # 处理没有明确判断的新闻
                    if not final_judgments:
                        writer.writerow({
                            "news_id": news_id,
                            "news_date": news_date,
                            "news_title": news_title,
                            "industry": "无明确判断",
                            "impact": "",
                            "confidence": "",
                            "comprehensive_reason": "",
                            "stock_name": "",
                            "stock_code": "",
                            "stock_reason": "",
                            "error": ""
                        })
                        continue

                    # 处理有明确判断的行业
                    for judgment in final_judgments:
                        industry = judgment.get("industry", "")
                        impact = judgment.get("impact", "")  # 利好或利空
                        confidence = judgment.get("confidence", "")  # 信念强度1-10
                        comp_reason = judgment.get("comprehensive_reason", "")
                        stocks = judgment.get("stocks", [])

                        if not stocks:
                            writer.writerow({
                                "news_id": news_id,
                                "news_date": news_date,
                                "news_title": news_title,
                                "industry": industry,
                                "impact": impact,
                                "confidence": confidence,
                                "comprehensive_reason": comp_reason,
                                "stock_name": "",
                                "stock_code": "",
                                "stock_reason": "",
                                "error": ""
                            })
                        else:
                            for stock in stocks:
                                writer.writerow({
                                    "news_id": news_id,
                                    "news_date": news_date,
                                    "news_title": news_title,
                                    "industry": industry,
                                    "impact": impact,
                                    "confidence": confidence,
                                    "comprehensive_reason": comp_reason,
                                    "stock_name": stock.get("name", ""),
                                    "stock_code": stock.get("code", ""),
                                    "stock_reason": stock.get("reason", ""),
                                    "error": ""
                                })

            logger.info(f"裁决结果已成功导出到：{output_path}")
            print(f"裁决结果已成功导出到：{output_path}")
            return True

        except Exception as e:
            error_msg = f"CSV导出失败：{str(e)}"
            logger.error(error_msg)
            print(error_msg)
            return False


# 使用示例
if __name__ == "__main__":
    DEBATE_ROUNDS = 2  # 辩论轮次，每轮包含正方回应和反方再驳斥
    NEWS_START_DAYS = 2
    CSV_OUTPUT_PATH = f"./news_final_verdict_{datetime.today().strftime('%Y%m%d')}.csv"

    try:
        # 初始化分析器
        analyzer = NewsDebateAnalyzer()
        logger.info("分析器初始化成功")
    except Exception as e:
        logger.error(f"分析器初始化失败：{str(e)}")
        exit(1)

    try:
        # 读取并筛选新闻数据
        news_df = pd.read_parquet('../data/stock_daily_cctvnews.parquet')
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce").dt.strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")
        start_date = (datetime.today() - timedelta(days=NEWS_START_DAYS)).strftime("%Y%m%d")
        filtered_df = news_df[(news_df["date"] >= start_date) & (news_df["date"] <= end_date)].reset_index(drop=True)
        # 测试时可以限制只分析少量新闻
        # filtered_df = filtered_df.head(3)
        logger.info(f"筛选出 {len(filtered_df)} 条新闻（{start_date} ~ {end_date}）")
        print(f"筛选出 {len(filtered_df)} 条新闻（{start_date} ~ {end_date}）")
    except Exception as e:
        logger.error(f"新闻数据读取失败：{str(e)}")
        print(f"新闻数据读取失败：{str(e)}")
        exit(1)

    if len(filtered_df) > 0:
        try:
            # 批量分析新闻
            batch_result = analyzer.batch_analyze_news(
                news_df=filtered_df,
                debate_rounds=DEBATE_ROUNDS
            )
            # 导出结果
            analyzer.export_verdict_to_csv(batch_result, CSV_OUTPUT_PATH)
        except Exception as e:
            logger.error(f"批量分析失败：{str(e)}")
            print(f"批量分析失败：{str(e)}")
            exit(1)
    else:
        logger.info("无符合条件的新闻，无需分析")
        print("无符合条件的新闻，无需分析")
    