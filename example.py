import pandas as pd
import numpy as np
from binning_service import BinningService


def example_1_basic_score_binning():
    print("=" * 60)
    print("示例 1: 成绩分箱（默认标签）")
    print("=" * 60)

    bins = [0, 60, 80, 100]
    scores = [45, 59, 60, 75, 80, 95, 100]

    binner = BinningService(bins=bins)
    result = binner.transform(scores)

    print(f"分箱边界: {bins}")
    print(f"原始数据: {scores}")
    print(f"分箱结果:")
    print(result)
    print()


def example_2_custom_labels():
    print("=" * 60)
    print("示例 2: 成绩分箱（自定义标签）")
    print("=" * 60)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]
    scores = [45, 59, 60, 75, 80, 95, 100]

    binner = BinningService(bins=bins, labels=labels)
    result = binner.transform(scores)

    print(f"分箱边界: {bins}")
    print(f"自定义标签: {labels}")
    print(f"原始数据: {scores}")
    print(f"分箱结果:")
    for score, label in zip(scores, result):
        print(f"  {score} 分 -> {label}")
    print()


def example_3_dataframe_input():
    print("=" * 60)
    print("示例 3: DataFrame 输入")
    print("=" * 60)

    data = pd.DataFrame({
        "student_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十", "郑十一", "王十二"],
        "score": [55, 62, 78, 85, 92, 59, 60, 80, 100, 45],
    })

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]

    binner = BinningService(bins=bins, labels=labels)
    result = binner.transform(data, column="score")

    print("原始 DataFrame:")
    print(result)
    print()
    print("按等级分组统计:")
    print(result.groupby("score_binned", observed=True)["score"].agg(["count", "mean", "min", "max"]))
    print()


def example_4_bin_info():
    print("=" * 60)
    print("示例 4: 查看分箱信息")
    print("=" * 60)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]

    binner = BinningService(bins=bins, labels=labels)

    print("分箱信息:")
    print(binner.get_bin_info())
    print()


def example_5_different_bounds():
    print("=" * 60)
    print("示例 5: 不同的边界开闭设置")
    print("=" * 60)

    bins = [0, 60, 80, 100]
    scores = [59, 60, 79, 80]

    print(f"测试数据: {scores}")
    print()

    print("右闭区间 (right=True, 默认):")
    binner_right = BinningService(bins=bins, right=True, include_lowest=True)
    result_right = binner_right.transform(scores)
    for score, label in zip(scores, result_right):
        print(f"  {score} -> {label}")
    print()

    print("左闭区间 (right=False):")
    binner_left = BinningService(bins=bins, right=False, include_lowest=True)
    result_left = binner_left.transform(scores)
    for score, label in zip(scores, result_left):
        print(f"  {score} -> {label}")
    print()


def example_6_include_bounds():
    print("=" * 60)
    print("示例 6: 同时返回标签和区间边界")
    print("=" * 60)

    bins = [0, 60, 80, 100]
    labels = ["不及格", "良好", "优秀"]
    scores = [45, 60, 75, 90]

    binner = BinningService(bins=bins, labels=labels)
    result_labels, result_bounds = binner.transform(scores, include_bounds=True)

    print(f"原始数据: {scores}")
    print()
    print("分箱结果对比:")
    for score, label, bound in zip(scores, result_labels, result_bounds):
        print(f"  {score} 分 -> {label} ({bound})")
    print()


def example_7_age_binning():
    print("=" * 60)
    print("示例 7: 年龄分箱案例")
    print("=" * 60)

    bins = [0, 18, 30, 45, 60, 100]
    labels = ["儿童", "青年", "中年", "中老年", "老年"]
    ages = [5, 18, 25, 35, 50, 65, 80]

    binner = BinningService(bins=bins, labels=labels)
    result = binner.transform(ages)

    print(f"分箱边界: {bins}")
    print(f"标签: {labels}")
    print()
    for age, label in zip(ages, result):
        print(f"  {age} 岁 -> {label}")
    print()


if __name__ == "__main__":
    example_1_basic_score_binning()
    example_2_custom_labels()
    example_3_dataframe_input()
    example_4_bin_info()
    example_5_different_bounds()
    example_6_include_bounds()
    example_7_age_binning()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
