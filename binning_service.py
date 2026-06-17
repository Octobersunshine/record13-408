import pandas as pd
import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any


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

    def evaluate(
        self,
        data: Union[List[Union[int, float]], pd.Series],
        target: Optional[Union[List[int], pd.Series]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = pd.Series(data)
        data_clean = data.dropna()

        binned = self.transform(data_clean, return_labels=False, include_bounds=True)
        labels_series, bounds_series = binned

        bin_info = self.get_bin_info()
        intervals = self._generate_default_labels()
        total_count = len(data_clean)

        eval_rows = []
        for interval in intervals:
            mask = (bounds_series == interval)
            bin_count = mask.sum()
            bin_data = data_clean[mask]
            bin_ratio = bin_count / total_count if total_count > 0 else 0

            row = {
                "bin_index": bin_info[bin_info["interval"] == interval].iloc[0]["bin_index"],
                "interval": interval,
                "custom_label": bin_info[bin_info["interval"] == interval].iloc[0]["custom_label"],
                "count": int(bin_count),
                "ratio": round(bin_ratio, 4),
                "min": round(bin_data.min(), 4) if bin_count > 0 else None,
                "max": round(bin_data.max(), 4) if bin_count > 0 else None,
                "mean": round(bin_data.mean(), 4) if bin_count > 0 else None,
                "median": round(bin_data.median(), 4) if bin_count > 0 else None,
                "std": round(bin_data.std(), 4) if bin_count > 1 else None,
            }

            if target is not None:
                if isinstance(target, list):
                    target = pd.Series(target)
                target_clean = target.reset_index(drop=True)
                bin_target = target_clean[mask.reset_index(drop=True)]
                event = int(bin_target.sum()) if bin_count > 0 else 0
                non_event = int(bin_count - event)
                total_event = int(target_clean.sum())
                total_non_event = int(len(target_clean) - total_event)

                event_rate = event / bin_count if bin_count > 0 else 0

                if total_event > 0 and total_non_event > 0:
                    dist_event = event / total_event if total_event > 0 else 1e-10
                    dist_non_event = non_event / total_non_event if total_non_event > 0 else 1e-10
                    dist_event = max(dist_event, 1e-10)
                    dist_non_event = max(dist_non_event, 1e-10)
                    woe = np.log(dist_non_event / dist_event)
                    iv = (dist_non_event - dist_event) * woe
                else:
                    woe = 0
                    iv = 0

                row["event_count"] = event
                row["non_event_count"] = non_event
                row["event_rate"] = round(event_rate, 4)
                row["woe"] = round(woe, 4)
                row["iv_contribution"] = round(iv, 4)

            eval_rows.append(row)

        result_df = pd.DataFrame(eval_rows).sort_values("bin_index").reset_index(drop=True)

        if target is not None:
            total_iv = result_df["iv_contribution"].sum()
            print(f"分箱 IV 值 (Information Value): {round(total_iv, 4)}")

        return result_df

    def __repr__(self) -> str:
        interval_type = "右闭(左开右闭)" if self.right else "左闭(左闭右开)"
        return (
            f"BinningService(bins={self.bins}, labels={self.labels}, "
            f"区间类型={interval_type}, include_lowest={self.include_lowest})"
        )

    # ==================== 自动分箱推荐方法 ====================

    @staticmethod
    def suggest_bins_quantile(
        data: Union[List[Union[int, float]], pd.Series],
        n_bins: int = 5,
        quantiles: Optional[List[float]] = None,
        include_bounds: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(data, list):
            data = pd.Series(data)
        data_clean = data.dropna()

        if quantiles is not None:
            if not all(0 <= q <= 1 for q in quantiles):
                raise ValueError("quantiles 必须在 [0, 1] 范围内")
            if quantiles[0] != 0:
                quantiles = [0.0] + list(quantiles)
            if quantiles[-1] != 1:
                quantiles = list(quantiles) + [1.0]
            q_values = sorted(list(set(quantiles)))
        else:
            q_values = np.linspace(0, 1, n_bins + 1).tolist()

        bin_edges = data_clean.quantile(q_values).tolist()
        bin_edges = sorted(list(set(bin_edges)))

        if include_bounds:
            if bin_edges[0] > data_clean.min():
                bin_edges[0] = float(data_clean.min())
            if bin_edges[-1] < data_clean.max():
                bin_edges[-1] = float(data_clean.max())

        if len(bin_edges) < 2:
            bin_edges = [float(data_clean.min()), float(data_clean.max())]

        bin_edges = [round(x, 6) for x in bin_edges]

        quantile_info = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            count = ((data_clean >= left) & (data_clean <= right)).sum() if i == 0 else \
                    ((data_clean > left) & (data_clean <= right)).sum()
            ratio = count / len(data_clean) if len(data_clean) > 0 else 0
            quantile_info.append({
                "bin_index": i,
                "lower_bound": left,
                "upper_bound": right,
                "count": int(count),
                "ratio": round(ratio, 4),
            })

        return {
            "method": "quantile",
            "description": "等频分箱（基于百分位数，各箱样本数尽量相等）",
            "n_bins": len(bin_edges) - 1,
            "bins": bin_edges,
            "quantiles_used": q_values,
            "bin_stats": pd.DataFrame(quantile_info),
            "data_summary": {
                "total_samples": int(len(data_clean)),
                "min": float(data_clean.min()),
                "max": float(data_clean.max()),
                "mean": round(float(data_clean.mean()), 4),
                "median": round(float(data_clean.median()), 4),
                "std": round(float(data_clean.std()), 4),
            },
        }

    @staticmethod
    def suggest_bins_equal_width(
        data: Union[List[Union[int, float]], pd.Series],
        n_bins: int = 5,
        bin_width: Optional[Union[int, float]] = None,
        round_digits: int = 2,
    ) -> Dict[str, Any]:
        if isinstance(data, list):
            data = pd.Series(data)
        data_clean = data.dropna()

        data_min = float(data_clean.min())
        data_max = float(data_clean.max())
        data_range = data_max - data_min

        if data_range <= 0:
            raise ValueError("数据范围为 0，无法进行等宽分箱")

        if bin_width is not None:
            n_bins_calc = int(np.ceil(data_range / bin_width))
            n_bins = max(n_bins_calc, 1)
            bin_edges = np.arange(data_min, data_min + bin_width * (n_bins + 1), bin_width).tolist()
            bin_edges[-1] = max(bin_edges[-1], data_max)
        else:
            step = data_range / n_bins
            bin_edges = [data_min + i * step for i in range(n_bins + 1)]
            bin_edges[0] = data_min
            bin_edges[-1] = data_max

        bin_edges = [round(x, round_digits) for x in bin_edges]
        bin_edges = sorted(list(set(bin_edges)))

        if len(bin_edges) < 2:
            bin_edges = [round(data_min, round_digits), round(data_max, round_digits)]

        actual_width = round(bin_edges[1] - bin_edges[0], round_digits) if len(bin_edges) > 1 else 0

        bin_info = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            count = ((data_clean >= left) & (data_clean <= right)).sum() if i == 0 else \
                    ((data_clean > left) & (data_clean <= right)).sum()
            ratio = count / len(data_clean) if len(data_clean) > 0 else 0
            bin_info.append({
                "bin_index": i,
                "lower_bound": left,
                "upper_bound": right,
                "count": int(count),
                "ratio": round(ratio, 4),
            })

        return {
            "method": "equal_width",
            "description": "等宽分箱（各区间宽度相等）",
            "n_bins": len(bin_edges) - 1,
            "bins": bin_edges,
            "actual_bin_width": actual_width,
            "bin_stats": pd.DataFrame(bin_info),
            "data_summary": {
                "total_samples": int(len(data_clean)),
                "min": data_min,
                "max": data_max,
                "range": round(data_range, round_digits),
                "mean": round(float(data_clean.mean()), 4),
                "std": round(float(data_clean.std()), 4),
            },
        }

    @staticmethod
    def suggest_bins_kmeans(
        data: Union[List[Union[int, float]], pd.Series],
        n_bins: int = 5,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ) -> Dict[str, Any]:
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("使用 K-Means 分箱需要安装 scikit-learn: pip install scikit-learn")

        if isinstance(data, list):
            data = pd.Series(data)
        data_clean = data.dropna()
        data_values = data_clean.values.reshape(-1, 1)

        actual_n_bins = min(n_bins, len(data_clean.unique()))
        if actual_n_bins < 2:
            actual_n_bins = 2

        kmeans = KMeans(
            n_clusters=actual_n_bins,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        kmeans.fit(data_values)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        cluster_info = []
        for cluster_id in range(actual_n_bins):
            mask = (labels == cluster_id)
            cluster_data = data_clean[mask]
            cluster_info.append({
                "cluster_id": cluster_id,
                "centroid": float(centroids[cluster_id]),
                "min": float(cluster_data.min()),
                "max": float(cluster_data.max()),
                "count": int(mask.sum()),
            })

        cluster_info_sorted = sorted(cluster_info, key=lambda x: x["centroid"])

        bin_edges = [float(cluster_info_sorted[0]["min"])]
        for i in range(len(cluster_info_sorted) - 1):
            mid_point = (cluster_info_sorted[i]["max"] + cluster_info_sorted[i + 1]["min"]) / 2
            bin_edges.append(round(float(mid_point), 6))
        bin_edges.append(float(cluster_info_sorted[-1]["max"]))

        bin_edges = sorted(list(set(bin_edges)))
        if len(bin_edges) < 2:
            bin_edges = [float(data_clean.min()), float(data_clean.max())]

        bin_stats = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            count = ((data_clean >= left) & (data_clean <= right)).sum() if i == 0 else \
                    ((data_clean > left) & (data_clean <= right)).sum()
            ratio = count / len(data_clean) if len(data_clean) > 0 else 0
            bin_stats.append({
                "bin_index": i,
                "lower_bound": left,
                "upper_bound": right,
                "count": int(count),
                "ratio": round(ratio, 4),
            })

        return {
            "method": "kmeans",
            "description": "K-Means 聚类分箱（基于数据分布的自然聚类）",
            "n_bins": len(bin_edges) - 1,
            "bins": bin_edges,
            "cluster_centroids": [round(c, 4) for c in centroids.tolist()],
            "inertia": round(float(kmeans.inertia_), 4),
            "cluster_details": pd.DataFrame(cluster_info_sorted),
            "bin_stats": pd.DataFrame(bin_stats),
            "data_summary": {
                "total_samples": int(len(data_clean)),
                "min": float(data_clean.min()),
                "max": float(data_clean.max()),
                "unique_values": int(data_clean.nunique()),
            },
        }

    @staticmethod
    def suggest_bins(
        data: Union[List[Union[int, float]], pd.Series],
        n_bins: int = 5,
        method: str = "quantile",
        **kwargs,
    ) -> Dict[str, Any]:
        method = method.lower()
        valid_methods = ["quantile", "equal_width", "equal", "width", "kmeans"]

        if method in ["quantile", "q", "percentile"]:
            return BinningService.suggest_bins_quantile(data, n_bins=n_bins, **kwargs)
        elif method in ["equal_width", "equal", "width", "w"]:
            return BinningService.suggest_bins_equal_width(data, n_bins=n_bins, **kwargs)
        elif method in ["kmeans", "k-means", "cluster"]:
            return BinningService.suggest_bins_kmeans(data, n_bins=n_bins, **kwargs)
        else:
            raise ValueError(f"不支持的分箱方法: {method}。支持的方法: {valid_methods}")

    @classmethod
    def from_auto_bins(
        cls,
        data: Union[List[Union[int, float]], pd.Series],
        n_bins: int = 5,
        method: str = "quantile",
        labels: Optional[List[str]] = None,
        auto_generate_labels: bool = False,
        label_prefix: str = "区间",
        include_lowest: bool = True,
        right: bool = True,
        ordered: bool = True,
        **kwargs,
    ) -> Tuple["BinningService", Dict[str, Any]]:
        suggestion = BinningService.suggest_bins(
            data=data,
            n_bins=n_bins,
            method=method,
            **kwargs,
        )
        bins = suggestion["bins"]

        if auto_generate_labels and labels is None:
            n_intervals = len(bins) - 1
            labels = [f"{label_prefix}{i + 1}" for i in range(n_intervals)]

        binner = cls(
            bins=bins,
            labels=labels,
            include_lowest=include_lowest,
            right=right,
            ordered=ordered,
        )

        return binner, suggestion
