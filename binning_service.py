import pandas as pd
from typing import List, Union, Optional, Tuple


class BinningService:
    def __init__(
        self,
        bins: List[Union[int, float]],
        labels: Optional[List[str]] = None,
        include_lowest: bool = True,
        right: bool = True,
        ordered: bool = True,
    ):
        self.bins = self._validate_bins(bins)
        self.labels = self._validate_labels(labels)
        self.include_lowest = include_lowest
        self.right = right
        self.ordered = ordered

    def _validate_bins(self, bins: List[Union[int, float]]) -> List[Union[int, float]]:
        if len(bins) < 2:
            raise ValueError("分箱边界至少需要2个值")
        if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
            raise ValueError("分箱边界必须严格递增")
        return bins

    def _validate_labels(self, labels: Optional[List[str]]) -> Optional[List[str]]:
        if labels is None:
            return None
        expected_len = len(self.bins) - 1
        if len(labels) != expected_len:
            raise ValueError(f"标签数量 ({len(labels)}) 必须等于分箱数量 ({expected_len})")
        return labels

    def _generate_default_labels(self) -> List[str]:
        labels = []
        for i in range(len(self.bins) - 1):
            left = self.bins[i]
            right = self.bins[i + 1]
            if self.right:
                if self.include_lowest and i == 0:
                    left_bracket = "["
                else:
                    left_bracket = "("
                right_bracket = "]"
            else:
                left_bracket = "["
                right_bracket = ")"
            labels.append(f"{left_bracket}{left}, {right}{right_bracket}")
        return labels

    def _get_effective_labels(self, return_labels: bool) -> Optional[List[str]]:
        if return_labels:
            return self.labels if self.labels is not None else self._generate_default_labels()
        return self._generate_default_labels()

    def transform(
        self,
        data: Union[List[Union[int, float]], pd.Series, pd.DataFrame],
        column: Optional[str] = None,
        return_labels: bool = True,
        include_bounds: bool = False,
    ) -> Union[pd.Series, pd.DataFrame, Tuple[pd.Series, pd.Series]]:
        if isinstance(data, list):
            data = pd.Series(data)
            data_series = data
        elif isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("当输入为 DataFrame 时，必须指定 column 参数")
            data_series = data[column]
        else:
            data_series = data

        effective_labels = self._get_effective_labels(return_labels)

        binned = pd.cut(
            data_series,
            bins=self.bins,
            labels=effective_labels,
            include_lowest=self.include_lowest,
            right=self.right,
            ordered=self.ordered,
        )

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result[f"{column}_binned"] = binned
            if include_bounds:
                bounds_series = pd.cut(
                    data_series,
                    bins=self.bins,
                    labels=self._generate_default_labels(),
                    include_lowest=self.include_lowest,
                    right=self.right,
                    ordered=self.ordered,
                )
                result[f"{column}_bounds"] = bounds_series
            return result

        if include_bounds:
            bounds = pd.cut(
                data_series,
                bins=self.bins,
                labels=self._generate_default_labels(),
                include_lowest=self.include_lowest,
                right=self.right,
                ordered=self.ordered,
            )
            return binned, bounds

        return binned

    def get_bin_info(self) -> pd.DataFrame:
        info = []
        default_labels = self._generate_default_labels()
        for i in range(len(self.bins) - 1):
            left = self.bins[i]
            right_val = self.bins[i + 1]
            if self.right:
                left_inclusive = self.include_lowest if i == 0 else False
                right_inclusive = True
            else:
                left_inclusive = True
                right_inclusive = False
            info.append({
                "bin_index": i,
                "lower_bound": left,
                "upper_bound": right_val,
                "left_inclusive": left_inclusive,
                "right_inclusive": right_inclusive,
                "interval": default_labels[i],
                "custom_label": self.labels[i] if self.labels else None,
                "description": self._describe_bin(i),
            })
        return pd.DataFrame(info)

    def _describe_bin(self, index: int) -> str:
        left = self.bins[index]
        right = self.bins[index + 1]
        if self.right:
            if self.include_lowest and index == 0:
                return f"{left} ≤ x ≤ {right}"
            else:
                return f"{left} < x ≤ {right}"
        else:
            return f"{left} ≤ x < {right}"

    def check_value_belong(
        self,
        values: Union[int, float, List[Union[int, float]]]
    ) -> pd.DataFrame:
        if isinstance(values, (int, float)):
            values = [values]
        series = pd.Series(values)
        bin_info = self.get_bin_info()
        binned = self.transform(series, return_labels=True, include_bounds=True)
        labels_series, bounds_series = binned
        results = []
        for idx, val in enumerate(values):
            label = labels_series.iloc[idx]
            bound = bounds_series.iloc[idx]
            if pd.isna(label):
                bin_idx = -1
                desc = "超出分箱范围"
            else:
                bin_rows = bin_info[bin_info["interval"] == bound]
                if len(bin_rows) > 0:
                    bin_idx = bin_rows.iloc[0]["bin_index"]
                    desc = bin_rows.iloc[0]["description"]
                else:
                    bin_idx = -1
                    desc = "未知"
            results.append({
                "value": val,
                "bin_index": bin_idx,
                "bin_label": None if pd.isna(label) else label,
                "bin_interval": None if pd.isna(bound) else bound,
                "belong_rule": desc,
            })
        return pd.DataFrame(results)

    def __repr__(self) -> str:
        interval_type = "右闭(左开右闭)" if self.right else "左闭(左闭右开)"
        return (
            f"BinningService(bins={self.bins}, labels={self.labels}, "
            f"区间类型={interval_type}, include_lowest={self.include_lowest})"
        )
