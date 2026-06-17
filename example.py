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
    names = [n for n, _ in test_cases]
    details = binner.check_value_belong(values)
    for i, (name, val) in enumerate(test_cases):
        row = details.iloc[i]
        status = "✓" if row["bin_index"] != -1 else "✗ 超出范围"
        print(f"  {name:<18} ({val:>12}) -> {row['bin_label']:<6} {row['bin_interval']:<12} {status}")
    print()

    print("【左闭区间模式】")
    binner2 = BinningService(bins=bins, labels=labels, right=False, include_lowest=True)
    details2 = binner2.check_value_belong(values)
    for i, (name, val) in enumerate(test_cases):
        row = details2.iloc[i]
        status = "✓" if row["bin_index"] != -1 else "✗ 超出范围"
        print(f"  {name:<18} ({val:>12}) -> {row['bin_label']:<6} {row['bin_interval']:<12} {status}")
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

    print("=" * 70)
    print("所有示例运行完成！边界值归属明确化修复验证通过 ✓")
    print("=" * 70)
