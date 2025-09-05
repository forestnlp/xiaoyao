# factor_analysis_demo.py
# 演示如何使用Qlib计算因子的IC和IR值

import qlib
from qlib.data import D


def run_factor_analysis():
    """执行因子分析流程"""
    try:
        # 1. 初始化Qlib环境
        # provider_uri 指向我们刚刚准备好的数据
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
        print("Qlib 初始化成功。")

        # 2. 定义因子表达式列表
        factor_expressions = [
            'Mean($close, 5)',      # 因子1: 5日收盘价均线
            '($close / Ref($close, 5)) - 1' # 因子2: 5日动量/收益率
        ]
        factor_names = ['MA5', 'Momentum5D']

        # 3. 定义标签（预测目标）
        # 我们用今天的因子，预测未来1天的收益率
        label_formula = 'Ref($close, -2) / Ref($close, -1) - 1'
        label_name = 'NextDayReturn'

        # 4. 获取并配置因子分析器
        from qlib.contrib.analyzer import FactorAnalyzer

        # 配置分析器
        # 我们在沪深300成分股上，对2021年全年的因子表现进行分析
        analyzer_config = {
            "market": "csi300",
            "start_time": "2021-01-01",
            "end_time": "2021-12-31",
        }

        print(f"\n准备在 {analyzer_config['market']}市场上分析因子...")
        analyzer = FactorAnalyzer(
            factor_list=factor_expressions,
            factor_names=factor_names,
            label=[(label_formula, label_name)],
            **analyzer_config
        )

        # 5. 运行分析
        print("正在运行因子分析，这可能需要一点时间...")
        analyzer.run()

        # 6. 打印分析报告
        print("\n--- IC (信息系数) 报告 ---")
        print(analyzer.get_ic_report())

        print("\n--- IR (信息比率) 报告 ---")
        print(analyzer.get_ir_report())

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请确保Qlib已正确安装，并且数据已通过上一步成功初始化。")


if __name__ == "__main__":
    run_factor_analysis()
