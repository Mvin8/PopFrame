import geopandas as gpd
import re
from typing import Optional, Tuple, Dict
from popframe.method.base_method import BaseMethod  # замените на ваш реальный импорт

class SpatialInequalityCalculator(BaseMethod):
    """
    Калькулятор пространственных показателей неравенства.
    Содержит методы для переноса показателей неравенства из точек в полигоны,
    поиска оптимальной территории и определения «лучшей» группы.
    """

    def transfer_inequality_metrics_to_polygons(
        self,
        gdf_cities: gpd.GeoDataFrame,
        gdf_polygons: gpd.GeoDataFrame,
        inequality_keyword: str = "Неравенство"
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Dict[str, float]]]:
        """
        Переносит все колонки, содержащие показатели неравенства
        из точек (городов) на полигоны (агломераций) усреднением по территории.

        Возвращает:
            gdf_polygons_with_metrics: GeoDataFrame полигонами с новыми колонками метрик
            stats: словарь со средними значениями внутри и вне полигонов
        """
        # 1) Колонки с нужным ключевым словом
        pattern = re.compile(rf".*\b{re.escape(inequality_keyword)}\b.*", re.IGNORECASE)
        metric_cols = [c for c in gdf_cities.columns if isinstance(c, str) and pattern.match(c)]
        if not metric_cols:
            raise KeyError(f"Не найдено колонок с '{inequality_keyword}'")

        # 2) Гео‑объединение точек и полигонов
        cities_with_idx = gpd.sjoin(
            gdf_cities[metric_cols + ['geometry']],
            gdf_polygons[['geometry']],
            how='left',
            predicate='within'
        )

        # 3) Усреднение по индексам полигонов
        grouped = (
            cities_with_idx
            .groupby('index_right')[metric_cols]
            .mean()
            .rename_axis('poly_index')
        )

        # 4) Присоединяем к полигонам
        gdf_polygons_with_metrics = (
            gdf_polygons
            .reset_index()
            .rename(columns={'index': 'poly_index'})
            .merge(grouped.reset_index(), on='poly_index', how='left')
            .set_index('poly_index')
        )

        # 5) Статистика внутри/вне
        inside  = cities_with_idx.dropna(subset=['index_right'])
        outside = cities_with_idx[cities_with_idx['index_right'].isna()]

        mean_within = inside[metric_cols].mean().to_dict()
        mean_outside = outside[metric_cols].mean().to_dict()
        stats = {
            'mean_within': mean_within,
            'mean_outside': mean_outside
        }

        return gdf_polygons_with_metrics, stats


    def get_best_territory(
        self,
        gdf: gpd.GeoDataFrame,
        group_name: Optional[str] = None,
        spatial_suffix: str = " - Неравенство",
        default_col: str = "Пространственное неравенство",
        top_n: int = 5
    ) -> gpd.GeoDataFrame:
        """
        Возвращает до `top_n` территорий (строк) с минимальным значением пространственного неравенства.

        Если задана группа, отбирает только:
        • все колонки с «Неравенство» для этой группы
        • все прочие колонки, НЕ содержащие «Неравенство»
        Если группа не указана — возвращаются первые `top_n` по общему показателю.

        Параметры:
            gdf (GeoDataFrame): должен содержать geometry и
                колонки с суффиксом spatial_suffix.
            group_name (str|None): название соц‑группы без суффикса.
            spatial_suffix (str): суффикс для поиска основной метрики.
            default_col (str): имя столбца общего неравенства.
            top_n (int): сколько лучших территорий вернуть (по возрастанию показателя).

        Возвращает:
            GeoDataFrame: до `top_n` строк с минимальными значениями.
        """
        if top_n < 1:
            raise ValueError("Параметр top_n должен быть >= 1")

        if group_name:
            primary_col = f"{group_name.strip()}{spatial_suffix}"
            if primary_col not in gdf.columns:
                raise KeyError(f"В GeoDataFrame нет колонки «{primary_col}»")

            # сортируем по нужной метрике и берём до top_n
            top_df = gdf.sort_values(primary_col, ascending=True).head(top_n).copy()

            # колонки-метрики для этой группы
            pattern = re.compile(
                rf"^{re.escape(group_name.strip())}.*\bНеравенство\b", re.IGNORECASE
            )
            group_metrics = [c for c in top_df.columns if pattern.match(c)]
            non_metrics = [c for c in top_df.columns if "Неравенство" not in c]

            keep_cols = non_metrics + group_metrics
            return top_df[keep_cols]

        else:
            if default_col not in gdf.columns:
                raise KeyError(f"В GeoDataFrame нет колонки «{default_col}»")

            # сортируем по default_col и берём до top_n
            return gdf.sort_values(default_col, ascending=True).head(top_n).copy()




    def get_best_group_for_territory(
        self,
        gdf: gpd.GeoDataFrame,
        suffix: str = " - Неравенство",
        new_col: str = "Наименьшее неравенство для соц‑группы"
    ) -> gpd.GeoDataFrame:
        """
        Добавляет столбец с названием соц‑группы,
        у которой минимальный показатель неравенства.
        """
        all_cols = [
            c for c in gdf.columns
            if isinstance(c, str)
               and c.endswith(suffix)
               and c[: -len(suffix)].strip().lower() != "итоговое"
        ]
        if not all_cols:
            raise KeyError(f"Колонки с суффиксом '{suffix}' не найдены")

        df_vals = gdf[all_cols]
        best_col = df_vals.idxmin(axis=1)
        pattern = re.escape(suffix) + r"$"
        best_group = best_col.str.replace(pattern, "", regex=True).str.strip()

        result = gdf.copy()
        result[new_col] = best_group
        return result
