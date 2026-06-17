import pandas as pd
import numpy as np
from binning_service import BinningService


def example_1_basic_score_binning():
    print("=" * 70)
    print("示例 1: 成绩分箱（默认区间标签）")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    scores = [0, 45, 59, 60, 60.1, 75, 79.9, 80, 95, 100]

    binner = BinningService(bins=bins)
    result = binner.transform(scores)

    print(f"分箱边界: {bins}")
    print(f"区间类型: 右闭区间 (左开右闭, include_lowest=True)")
    print(f"原始数据: {scores}")
    print()
    print("分箱结果:")
    for score, label in zip(scores, result):
        print(f"  {score:>6}  ->  {label}")
    print()


def example_2_custom_labels():
    print("=" * 70)
    print("示例 2: 成绩分箱（自定义标签 + 边界归属明确）")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]
    scores = [0, 45, 59, 60, 75, 80, 95, 100]

    binner = BinningService(bins=bins, labels=labels)

    print(f"分箱边界: {bins}")
    print(f"自定义标签: {labels}")
    print(f"区间类型: 右闭区间 (左开右闭, include_lowest=True)")
    print()
    print("查看分箱规则:")
    print(binner.get_bin_info().to_string(index=False))
    print()

    result = binner.transform(scores)
    print("分箱结果:")
    for score, label in zip(scores, result):
        print(f"  {score:>3} 分  ->  {label}")
    print()


def example_3_check_value_belong():
    print("=" * 70)
    print("示例 3: 边界值归属查询 (check_value_belong)")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]
    test_values = [-1, 0, 59, 60, 60.0001, 79.9999, 80, 100, 101]

    binner = BinningService(bins=bins, labels=labels)

    print(f"分箱边界: {bins}")
    print(f"测试边界值: {test_values}")
    print()
    print("边界归属详情:")
    print(binner.check_value_belong(test_values).to_string(index=False))
    print()


def example_4_right_vs_left_closed():
    print("=" * 70)
    print("示例 4: 右闭区间 vs 左闭区间 对比")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    test_values = [59, 60, 79, 80]

    print(f"分箱边界: {bins}")
    print(f"测试数据: {test_values}")
    print()

    print("【模式 A】右闭区间 (right=True, 默认) - 左开右闭")
    print("  区间规则: [0, 60], (60, 80], (80, 100]")
    binner_right = BinningService(bins=bins, right=True, include_lowest=True)
    print(binner_right.get_bin_info()[["interval", "description"]].to_string(index=False))
    result_right = binner_right.transform(test_values)
    print("  分箱结果:")
    for val, label in zip(test_values, result_right):
        print(f"    {val} -> {label}")
    print()

    print("【模式 B】左闭区间 (right=False) - 左闭右开")
    print("  区间规则: [0, 60), [60, 80), [80, 100)")
    binner_left = BinningService(bins=bins, right=False, include_lowest=True)
    print(binner_left.get_bin_info()[["interval", "description"]].to_string(index=False))
    result_left = binner_left.transform(test_values)
    print("  分箱结果:")
    for val, label in zip(test_values, result_left):
        print(f"    {val} -> {label}")
    print()

    print("关键差异点 (60 和 80 这两个边界值):")
    print(f"  60  在右闭模式 -> {result_right.iloc[1]}  (归属第一个区间)")
    print(f"  60  在左闭模式 -> {result_left.iloc[1]}  (归属第二个区间)")
    print(f"  80  在右闭模式 -> {result_right.iloc[3]}  (归属第二个区间)")
    print(f"  80  在左闭模式 -> {result_left.iloc[3]}  (归属第三个区间)")
    print()


def example_5_include_lowest_effect():
    print("=" * 70)
    print("示例 5: include_lowest 参数的影响")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    test_values = [0, 0.0001, 60]

    print(f"分箱边界: {bins}")
    print(f"测试数据: {test_values}")
    print()

    print("include_lowest=True (默认):")
    binner_with = BinningService(bins=bins, right=True, include_lowest=True)
    print(binner_with.get_bin_info()[["interval", "description"]].to_string(index=False))
    result_with = binner_with.transform(test_values)
    for val, label in zip(test_values, result_with):
        print(f"  {val:<8} -> {label}")
    print()

    print("include_lowest=False:")
    binner_without = BinningService(bins=bins, right=True, include_lowest=False)
    print(binner_without.get_bin_info()[["interval", "description"]].to_string(index=False))
    result_without = binner_without.transform(test_values)
    for val, label in zip(test_values, result_without):
        nan_note = " (NaN! 超出范围)" if pd.isna(label) else ""
        print(f"  {val:<8} -> {label}{nan_note}")
    print()


def example_6_dataframe_with_bounds():
    print("=" * 70)
    print("示例 6: DataFrame 输入 + include_bounds 双输出")
    print("=" * 70)

    data = pd.DataFrame({
        "student_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"],
        "score": [0, 59, 60, 75, 79, 80, 100, 45],
    })

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]

    binner = BinningService(bins=bins, labels=labels, right=True)
    result = binner.transform(data, column="score", include_bounds=True)

    print("输入 DataFrame + 分箱结果:")
    print(result.to_string(index=False))
    print()


def example_7_age_binning_left_closed():
    print("=" * 70)
    print("示例 7: 年龄分箱 (左闭右开模式)")
    print("=" * 70)

    bins = [0, 18, 30, 45, 60, 100]
    labels = ["儿童", "青年", "中年", "中老年", "老年"]
    ages = [0, 17, 18, 29, 30, 44, 45, 59, 60, 80, 99]

    binner = BinningService(bins=bins, labels=labels, right=False)

    print(f"分箱边界: {bins}")
    print(f"标签: {labels}")
    print(f"区间类型: 左闭右开 (right=False)")
    print()
    print("分箱规则:")
    print(binner.get_bin_info()[["interval", "description", "custom_label"]].to_string(index=False))
    print()

    result = binner.transform(ages)
    print("分箱结果:")
    for age, label in zip(ages, result):
        print(f"  {age:>3} 岁 -> {label}")
    print()


def example_8_edge_case_test():
    print("=" * 70)
    print("示例 8: 边界极端值测试")
    print("=" * 70)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]

    test_cases = [
        ("刚好在左边界", 0),
        ("刚好在右边界", 100),
        ("左边界 + 极小值", 0.000001),
        ("右边界 - 极小值", 99.999999),
        ("第一区间临界值", 59.999999),
        ("第一第二区间边界", 60),
        ("第二区间临界值", 79.999999),
        ("第二第三区间边界", 80),
        ("超出左边界", -0.000001),
        ("超出右边界", 100.000001),
    ]

    print("【右闭区间模式】")
    binner = BinningService(bins=bins, labels=labels, right=True, include_lowest=True)
    values = [v for _, v in test_cases]
    details = binner.check_value_belong(values)
    for i, (name, val) in enumerate(test_cases):
        row = details.iloc[i]
        status = "✓" if row["bin_index"] != -1 else "✗ 超出范围"
        print(f"  {name:<18} ({val:>12}) -> {str(row['bin_label']):<6} {str(row['bin_interval']):<12} {status}")
    print()

    print("【左闭区间模式】")
    binner2 = BinningService(bins=bins, labels=labels, right=False, include_lowest=True)
    details2 = binner2.check_value_belong(values)
    for i, (name, val) in enumerate(test_cases):
        row = details2.iloc[i]
        status = "✓" if row["bin_index"] != -1 else "✗ 超出范围"
        print(f"  {name:<18} ({val:>12}) -> {str(row['bin_label']):<6} {str(row['bin_interval']):<12} {status}")
    print()


def example_9_auto_bins_quantile():
    print("=" * 70)
    print("示例 9: 自动分箱 - 等频分箱（基于百分位数）")
    print("=" * 70)

    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(50, 15, 500),
        np.random.normal(75, 10, 300),
        np.random.normal(90, 5, 200),
    ])
    data = np.clip(data, 0, 100)
    scores = pd.Series(data, name="score")

    print(f"生成模拟数据: {len(scores)} 个成绩样本")
    print(f"数据分布: min={scores.min():.2f}, max={scores.max():.2f}, "
          f"mean={scores.mean():.2f}, median={scores.median():.2f}")
    print()

    print("【方法 A】自动推荐 5 个等频分箱:")
    suggestion = BinningService.suggest_bins_quantile(scores, n_bins=5)
    print(f"  方法描述: {suggestion['description']}")
    print(f"  推荐边界: {suggestion['bins']}")
    print(f"  使用百分位数: {[round(q, 2) for q in suggestion['quantiles_used']]}")
    print()
    print("  各箱分布统计:")
    print(suggestion['bin_stats'].to_string(index=False))
    print()

    print("【方法 B】使用自定义百分位数 [0, 0.25, 0.5, 0.75, 1.0] (四分位数):")
    suggestion_q4 = BinningService.suggest_bins_quantile(
        scores,
        quantiles=[0.25, 0.5, 0.75],
    )
    print(f"  推荐边界: {suggestion_q4['bins']}")
    print("  各箱分布统计:")
    print(suggestion_q4['bin_stats'].to_string(index=False))
    print()


def example_10_auto_bins_equal_width():
    print("=" * 70)
    print("示例 10: 自动分箱 - 等宽分箱")
    print("=" * 70)

    np.random.seed(123)
    incomes = np.random.exponential(scale=30000, size=1000) + 10000
    incomes = pd.Series(incomes, name="income")

    print(f"生成模拟数据: {len(incomes)} 个收入样本 (指数分布)")
    print(f"数据分布: min={incomes.min():.0f}, max={incomes.max():.0f}, "
          f"mean={incomes.mean():.0f}, median={incomes.median():.0f}")
    print()

    print("【方法 A】指定 n_bins=5 自动等宽分箱:")
    suggestion = BinningService.suggest_bins_equal_width(incomes, n_bins=5)
    print(f"  方法描述: {suggestion['description']}")
    print(f"  推荐边界: {suggestion['bins']}")
    print(f"  实际箱宽: {suggestion['actual_bin_width']:.0f}")
    print("  各箱分布统计:")
    print(suggestion['bin_stats'].to_string(index=False))
    print()

    print("【方法 B】指定 bin_width=20000 自动计算箱数:")
    suggestion_w = BinningService.suggest_bins_equal_width(incomes, bin_width=20000)
    print(f"  推荐边界: {suggestion_w['bins']}")
    print(f"  实际分箱数: {suggestion_w['n_bins']}")
    print("  各箱分布统计:")
    print(suggestion_w['bin_stats'].to_string(index=False))
    print()


def example_11_from_auto_bins_one_shot():
    print("=" * 70)
    print("示例 11: 一键自动分箱 (from_auto_bins)")
    print("=" * 70)

    np.random.seed(456)
    data = np.random.lognormal(mean=3, sigma=0.8, size=800)
    amounts = pd.Series(data, name="amount")

    print(f"生成模拟数据: {len(amounts)} 个消费金额 (对数正态)")
    print(f"数据范围: [{amounts.min():.1f}, {amounts.max():.1f}], "
          f"均值={amounts.mean():.1f}, 中位数={amounts.median():.1f}")
    print()

    print("【场景 1】等频分箱 + 自动生成标签:")
    binner_q, info_q = BinningService.from_auto_bins(
        data=amounts,
        n_bins=5,
        method="quantile",
        auto_generate_labels=True,
        label_prefix="档位",
    )
    print(f"  推荐分箱边界: {info_q['bins']}")
    print(f"  自动标签: {binner_q.labels}")
    print("  分箱规则:")
    print(binner_q.get_bin_info()[["interval", "custom_label", "description"]].to_string(index=False))
    print()

    print("【场景 2】等宽分箱 + 自定义标签:")
    custom_labels = ["极低消费", "低消费", "中消费", "高消费", "极高消费"]
    binner_w, info_w = BinningService.from_auto_bins(
        data=amounts,
        n_bins=5,
        method="equal_width",
        labels=custom_labels,
    )
    print(f"  推荐分箱边界: {info_w['bins']}")
    print("  分箱结果抽样 (前 10 个样本):")
    sample_result = binner_w.transform(amounts.head(10).tolist(), include_bounds=True)
    labels_s, bounds_s = sample_result
    for i, (amt, lab, bnd) in enumerate(zip(amounts.head(10), labels_s, bounds_s)):
        print(f"    金额 {amt:>8.1f}  -> {lab:<8} ({bnd})")
    print()


def example_12_compare_binning_methods():
    print("=" * 70)
    print("示例 12: 三种自动分箱方法对比")
    print("=" * 70)

    np.random.seed(789)
    data = np.concatenate([
        np.random.normal(20, 5, 300),
        np.random.normal(50, 8, 400),
        np.random.normal(85, 5, 300),
    ])
    data = np.clip(data, 0, 100)
    scores = pd.Series(data, name="score")

    print(f"测试数据: {len(scores)} 个样本 (三峰分布)")
    print(f"  min={scores.min():.1f}, max={scores.max():.1f}, "
          f"mean={scores.mean():.1f}, median={scores.median():.1f}")
    print()

    methods = ["quantile", "equal_width", "kmeans"]
    method_names = {
        "quantile": "等频分箱 (Quantile)",
        "equal_width": "等宽分箱 (Equal Width)",
        "kmeans": "K-Means 聚类分箱",
    }

    for method in methods:
        print(f"--- {method_names[method]} ---")
        try:
            suggestion = BinningService.suggest_bins(
                scores, n_bins=5, method=method
            )
            print(f"  推荐边界: {suggestion['bins']}")
            print("  各箱样本分布:")
            for _, row in suggestion['bin_stats'].iterrows():
                bar = "█" * int(row['ratio'] * 50)
                print(f"    [{row['lower_bound']:>6.1f}, {row['upper_bound']:>6.1f}]  "
                      f"count={row['count']:>4} ({row['ratio']*100:>5.1f}%)  {bar}")
            print()
        except ImportError as e:
            print(f"  跳过: {e}")
            print()


def example_13_evaluate_woe_iv():
    print("=" * 70)
    print("示例 13: 分箱质量评估 (WOE/IV 指标)")
    print("=" * 70)

    np.random.seed(999)
    n = 1000
    ages = np.random.randint(18, 70, n)
    default_prob = 1 / (1 + np.exp(0.08 * ages - 4))
    defaults = (np.random.random(n) < default_prob).astype(int)

    print(f"模拟信贷数据: {n} 个样本")
    print(f"  年龄范围: {ages.min()} - {ages.max()} 岁")
    print(f"  违约样本数: {defaults.sum()} ({defaults.mean()*100:.1f}%)")
    print()

    print("【步骤 1】自动推荐等频分箱边界:")
    suggestion = BinningService.suggest_bins_quantile(pd.Series(ages), n_bins=6)
    print(f"  推荐边界: {suggestion['bins']}")
    print()

    print("【步骤 2】创建分箱服务:")
    labels = ["青年组", "青年中", "中青年", "中年组", "中老年", "老年组"]
    binner = BinningService(bins=suggestion["bins"], labels=labels)
    print(f"  分箱服务: {binner}")
    print()

    print("【步骤 3】评估分箱质量 (含 WOE/IV):")
    eval_df = binner.evaluate(pd.Series(ages), target=pd.Series(defaults))
    print(eval_df.to_string(index=False))
    print()

    print("【IV 值参考标准】:")
    print("  IV < 0.02: 无预测能力")
    print("  0.02 ≤ IV < 0.1: 预测能力较弱")
    print("  0.1 ≤ IV < 0.3: 预测能力中等")
    print("  IV ≥ 0.3: 预测能力较强")
    print()


def example_14_suggest_bins_fallback():
    print("=" * 70)
    print("示例 14: 使用通用 suggest_bins 接口 + 异常处理")
    print("=" * 70)

    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print("使用 suggest_bins() 统一接口，可切换 method 参数:")
    print()

    print("1. method='quantile' (默认，等频):")
    s1 = BinningService.suggest_bins(data, n_bins=4, method="quantile")
    print(f"   边界: {s1['bins']}")
    print()

    print("2. method='equal_width' (等宽):")
    s2 = BinningService.suggest_bins(data, n_bins=4, method="equal_width")
    print(f"   边界: {s2['bins']}")
    print()

    print("3. method='q' (简写形式):")
    s3 = BinningService.suggest_bins(data, n_bins=3, method="q")
    print(f"   边界: {s3['bins']}")
    print()

    print("4. 错误方法名的异常提示:")
    try:
        BinningService.suggest_bins(data, n_bins=3, method="invalid_method")
    except ValueError as e:
        print(f"   ValueError: {e}")
    print()


if __name__ == "__main__":
    example_1_basic_score_binning()
    example_2_custom_labels()
    example_3_check_value_belong()
    example_4_right_vs_left_closed()
    example_5_include_lowest_effect()
    example_6_dataframe_with_bounds()
    example_7_age_binning_left_closed()
    example_8_edge_case_test()
    example_9_auto_bins_quantile()
    example_10_auto_bins_equal_width()
    example_11_from_auto_bins_one_shot()
    example_12_compare_binning_methods()
    example_13_evaluate_woe_iv()
    example_14_suggest_bins_fallback()

    print("=" * 70)
    print("所有示例运行完成！自动分箱推荐功能验证通过 ✓")
    print("=" * 70)
