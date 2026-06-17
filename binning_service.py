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
            if self.include_lowest and i == 0:
                left_bracket = "["
            else:
                left_bracket = "(" if self.right else "["
            right_bracket = "]" if self.right else ")"
            labels.append(f"{left_bracket}{left}, {right}{right_bracket}")
        return labels

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

        labels = self.labels if return_labels else None

        binned = pd.cut(
            data_series,
            bins=self.bins,
            labels=labels,
            include_lowest=self.include_lowest,
            right=self.right,
            ordered=self.ordered,
        )

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result[f"{column}_binned"] = binned
            if include_bounds:
                result[f"{column}_bounds"] = self._generate_default_labels()
                result[f"{column}_bounds"] = result[f"{column}_bounds"].astype("category")
                result[f"{column}_bounds"] = result[f"{column}_bounds"].cat.set_categories(
                    self._generate_default_labels(), ordered=True
                )
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
            info.append({
                "bin_index": i,
                "lower_bound": self.bins[i],
                "upper_bound": self.bins[i + 1],
                "interval": default_labels[i],
                "custom_label": self.labels[i] if self.labels else None,
            })
        return pd.DataFrame(info)

    def __repr__(self) -> str:
        return (
            f"BinningService(bins={self.bins}, labels={self.labels}, "
            f"include_lowest={self.include_lowest}, right={self.right})"
        )
