import geopandas as gpd
import pandas as pd
import geopandas as gpd
import numpy as np
import pandas as pd
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import branca.colormap as cm
import json
from .base_method import BaseMethod


# Определение списка столбцов с оценками
SCORE_COLUMNS = [
    'provision', 'basic', 'additional', 'comfort',
    'soc_workers_dev', 'soc_workers_soc', 'soc_workers_bas',
    'soc_old_dev', 'soc_old_soc', 'soc_old_bas',
    'soc_parents_dev', 'soc_parents_soc', 'soc_parents_bas',
]


class SpatialInequalityCalculator(BaseMethod):
    def calculate_spatial_inequality(self, tows_info_gdf):
        
        spatial_inequality_gdf = tows_info_gdf.copy()

        for col in SCORE_COLUMNS:
            spatial_inequality_gdf[col] = 1 - spatial_inequality_gdf[col]

        # Шаг 1: Распределение оценок на города
        spatial_inequality_gdf = self.distribute_scores_by_population(spatial_inequality_gdf)

        spatial_inequality_gdf['spatial_inequality'] = spatial_inequality_gdf[SCORE_COLUMNS].mean(axis=1)

        return spatial_inequality_gdf

    

    def distribute_scores_by_population(self, towns_gdf):
        """
        Распределяет оценки из полигонов неравенства на города с учетом их населения.
        
        Параметры:
        towns_gdf (GeoDataFrame): GeoDataFrame с городами.

        Возвращает:
        GeoDataFrame: Копия исходного GeoDataFrame с городами и распределенными оценками.
        """
        # Копия GeoDataFrame с городами для записи распределённых оценок
        towns_result = towns_gdf.copy()

        soc_scores = [
            'soc_workers_dev', 'soc_workers_soc', 'soc_workers_bas',
            'soc_old_dev', 'soc_old_soc', 'soc_old_bas',
            'soc_parents_dev', 'soc_parents_soc', 'soc_parents_bas',
        ]
        
        # Инициализируем новые столбцы для распределённых оценок
        for col in soc_scores:
            towns_result[col] = np.nan

        # Задаём запас, чтобы оценки не получались ровно 0 или 1
        eps = 1e-6

        # Определяем агрегирующую функцию
        agg_dict = {col: 'first' for col in soc_scores}

        # Группировка по parent_id и агрегация
        gdf_aggregated = towns_gdf.groupby('parent_id', as_index=False).agg(agg_dict)

        # Проходим по каждому полигону из inequality_gdf
        for poly_idx, poly in gdf_aggregated.iterrows():

            # Выбираем города, которые полностью лежат внутри полигона
            towns_in_poly = towns_gdf[towns_gdf['parent_id'] == poly['parent_id']]

            # Если городов в полигоне нет или суммарное население равно 0 — пропускаем
            if towns_in_poly.empty:
                continue

            if (towns_in_poly['population'] <= 0).all():
                continue

            # Для каждого оценочного столбца рассчитываем распределение
            for col in soc_scores:
                polygon_score = poly[col]
                # Если значение оценок полигона не лежит в (eps, 1-eps), оставляем его без изменений
                if not (eps < polygon_score < 1 - eps):
                    towns_result.loc[towns_in_poly.index, col] = polygon_score
                    continue

                # Нормализуем население внутри полигона: r_i от 0 до 1
                pop = towns_in_poly['population']
                pop_min = pop.min()
                pop_max = pop.max()

                # Если все города имеют одинаковое население, просто присваиваем polygon_score
                if pop_max == pop_min:
                    towns_result.loc[towns_in_poly.index, col] = polygon_score
                    continue

                r = (pop - pop_min) / (pop_max - pop_min)
                mean_r = r.mean()

                # Вычисляем максимально допустимый delta, чтобы итоговые оценки не вышли за (eps, 1-eps)
                # Для города с максимальным r_i (r_i = 1): s_max = polygon_score + delta*(1 - mean_r) <= 1 - eps
                delta_max_pos = (1 - polygon_score - eps) / (1 - mean_r) if (1 - mean_r) > 0 else np.inf
                # Для города с минимальным r_i (r_i = 0): s_min = polygon_score - delta*(mean_r) >= eps
                delta_max_neg = (polygon_score - eps) / (mean_r) if mean_r > 0 else np.inf

                delta = min(delta_max_pos, delta_max_neg)

                # Вычисляем итоговую оценку для каждого города
                computed_scores = polygon_score + delta * (r - mean_r)
                # Обеспечиваем, чтобы значения точно были в диапазоне (eps, 1-eps)
                computed_scores = computed_scores.clip(lower=eps, upper=1-eps)

                # Записываем вычисленные оценки в towns_result
                towns_result.loc[towns_in_poly.index, col] = computed_scores

        return towns_result


    def calculate_polygon_spatial_inequality(self, spatial_inequality_gdf, settlement_boundaries):
        """
        Вычисляет средние оценки для полигонов на основе оценок городов внутри них.
        
        Параметры:
        spatial_inequality_gdf (GeoDataFrame): GeoDataFrame с городами и их оценками.
        settlement_boundaries (GeoDataFrame): GeoDataFrame с полигонами для записи средних оценок.

        Возвращает:
        GeoDataFrame: Копия исходного GeoDataFrame с полигонами и средними оценками.
        """
        # Создаём копию result_gdf для записи новых оценок
        polygon_spatial_inequality = settlement_boundaries.copy()

        # Добавляем в polygon_spatial_inequality столбцы для оценок (инициализируем NaN)
        for col in SCORE_COLUMNS:
            polygon_spatial_inequality[col] = np.nan

        local_crs = self.region.region.crs
        spatial_inequality_gdf = spatial_inequality_gdf.to_crs(local_crs)
        settlement_boundaries = settlement_boundaries.to_crs(local_crs)
        
        # Проходим по каждому полигону в polygon_spatial_inequality
        for idx, poly in polygon_spatial_inequality.iterrows():
            poly_geom = poly.geometry
            # Отбираем города, которые полностью находятся внутри полигона
            towns_in_poly = spatial_inequality_gdf[spatial_inequality_gdf.within(poly_geom)]
            
            # Если городов внутри полигона нет, переходим к следующему
            if towns_in_poly.empty:
                continue
            
            # Для каждого оценочного столбца вычисляем среднее значение по городам внутри полигона
            for col in SCORE_COLUMNS:
                mean_score = towns_in_poly[col].mean()
                polygon_spatial_inequality.loc[idx, col] = mean_score

        
        # Шаг 3: Анализ распределения городов внутри и вне полигонов
        stats_json = self.analyze_town_distribution(spatial_inequality_gdf, polygon_spatial_inequality)

        polygon_spatial_inequality['spatial_inequality'] = polygon_spatial_inequality[SCORE_COLUMNS].mean(axis=1)

        return polygon_spatial_inequality, stats_json



    def analyze_town_distribution(self, towns_result, polygon_spatial_inequality):
        """
        Анализирует распределение городов внутри и вне полигонов и вычисляет статистику по оценкам.
        
        Параметры:
        towns_result (GeoDataFrame): GeoDataFrame с городами и их оценками.
        polygon_spatial_inequality (GeoDataFrame): GeoDataFrame с полигонами и их оценками.

        Возвращает:
        str: JSON-строка с двумя словарями статистики (inside_stats и outside_stats).
        """
        # Создаём булеву маску: для каждого города проверяем, пересекается ли его геометрия хотя бы с одним полигоном
        inside_mask = towns_result.geometry.apply(lambda pt: polygon_spatial_inequality.intersects(pt).any())

        # Разбиваем на две группы
        inside_towns = towns_result[inside_mask]
        outside_towns = towns_result[~inside_mask]

        # Собираем статистику по каждому атрибуту
        inside_stats = {}
        outside_stats = {}
        
        for col in SCORE_COLUMNS+['spatial_inequality']:
            inside_stats[col] = inside_towns[col].mean()
            outside_stats[col] = outside_towns[col].mean()
        
        # Создаем общий словарь и преобразуем его в JSON
        result = {
            "inside_stats": inside_stats,
            "outside_stats": outside_stats
        }
        
        return result

    def create_interactive_map(self, spatial_inequality_gdf, polygon_spatial_inequality, selected_attribute=None, color_by_average=True, vmin=None, vmax=None):
        """
        Создает интерактивную карту с отображением полигонов и точек с возможностью выбора атрибута для отображения
        или раскраски по колонке spatial_inequality.
        
        Параметры:
        spatial_inequality_gdf (GeoDataFrame): GeoDataFrame с данными о неравенстве (города).
        polygon_spatial_inequality (GeoDataFrame): GeoDataFrame с полигонами регионов.
        selected_attribute (str, optional): Атрибут для раскраски объектов. Если None и color_by_average=True, используется 'spatial_inequality'.
        color_by_average (bool): Если True, раскрашивает объекты по колонке 'spatial_inequality', иначе - по selected_attribute.
        vmin (float, optional): Минимальное значение для цветовой шкалы. Если None, вычисляется автоматически.
        vmax (float, optional): Максимальное значение для цветовой шкалы. Если None, вычисляется автоматически.
        
        Возвращает:
        folium.Map: Интерактивная карта с визуализацией данных.
        """
        # Убеждаемся, что все GeoDataFrame имеют систему координат EPSG:4326
        if spatial_inequality_gdf.crs != "EPSG:4326":
            spatial_inequality_gdf = spatial_inequality_gdf.to_crs(epsg=4326)
        if polygon_spatial_inequality.crs != "EPSG:4326":
            polygon_spatial_inequality = polygon_spatial_inequality.to_crs(epsg=4326)

        # Создаем карту с центром на средних координатах
        m = folium.Map(location=[spatial_inequality_gdf.geometry.y.mean(), 
                                spatial_inequality_gdf.geometry.x.mean()], 
                        zoom_start=10)
        
        # Список переменных
        columns = ["soc_workers_dev", "soc_workers_soc", "soc_workers_bas", 
                "soc_old_dev", "soc_old_soc", "soc_old_bas", 
                "soc_parents_dev", "soc_parents_soc", "soc_parents_bas",
                'provision', 'basic', 'additional', 'comfort', 'spatial_inequality']
        
        # Группировка колонок с изменением отображаемого имени для 'provision'
        groups = {
            'soc_workers': ['soc_workers_dev', 'soc_workers_soc', 'soc_workers_bas'],
            'soc_old': ['soc_old_dev', 'soc_old_soc', 'soc_old_bas'],
            'soc_parents': ['soc_parents_dev', 'soc_parents_soc', 'soc_parents_bas'],
            'provision': ['provision', 'basic', 'additional', 'comfort']
        }
        
        # Создаем группу для точек
        points_layer = folium.FeatureGroup(name="Cities")
        
        # Находим минимальные и максимальные значения, если vmin и vmax не заданы
        if vmin is None or vmax is None:
            if color_by_average:
                # Используем значения из колонки 'spatial_inequality'
                all_values = []
                all_values.extend(spatial_inequality_gdf['spatial_inequality'].dropna().tolist())
                all_values.extend(polygon_spatial_inequality['spatial_inequality'].dropna().tolist())
            else:
                # Используем все значения выбранного атрибута или всех атрибутов, если selected_attribute не указан
                all_values = []
                target_columns = [selected_attribute] if selected_attribute else columns
                for col in target_columns:
                    if col in polygon_spatial_inequality.columns and col in spatial_inequality_gdf.columns:
                        all_values.extend(polygon_spatial_inequality[col].dropna().tolist())
                        all_values.extend(spatial_inequality_gdf[col].dropna().tolist())
            
            calc_vmin = min(all_values) if all_values else 0
            calc_vmax = max(all_values) if all_values else 1
            
            # Сужаем диапазон для большей выразительности только если color_by_average=True
            if color_by_average:
                calc_vmin = calc_vmin + (calc_vmax - calc_vmin) * 0.25  # Сдвигаем нижнюю границу на 25% вверх
                calc_vmax = calc_vmax - (calc_vmax - calc_vmin) * 0.25  # Сдвигаем верхнюю границу на 25% вниз
            
            # Устанавливаем vmin и vmax, если они не заданы
            vmin = calc_vmin if vmin is None else vmin
            vmax = calc_vmax if vmax is None else vmax
        
        # Проверяем, что vmin меньше vmax
        if vmin >= vmax:
            raise ValueError("vmin должен быть меньше vmax")
        
        # Цветовая шкала для карты (зеленый -> оранжевый -> красный)
        colormap = cm.LinearColormap(['green', 'orange', 'red'], vmin=vmin, vmax=vmax)
        
        # Цветовая карта для гистограмм (зеленый -> красный для нормализованных значений)
        bar_colormap = cm.LinearColormap(['green', 'orange', 'red'], vmin=0, vmax=1)
        
        # Общая функция для создания HTML гистограммы
        def create_histogram_html(row, groups, include_group_means=False):
            html = """
            <style>
                .popup-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    max-width: 600px;
                }
                .group-container {
                    width: 48%;
                    margin-bottom: 10px;
                    font-family: Arial, sans-serif;
                    color: #333;
                    background-color: #f9f9f9;
                    padding: 5px;
                    border-radius: 3px;
                }
                .group-title {
                    font-weight: bold;
                    color: #222;
                    margin-bottom: 5px;
                    font-size: 14px;
                }
                .bar-container {
                    margin: 2px 0;
                    font-size: 11px;
                }
                .bar {
                    height: 12px;
                    border-radius: 3px;
                    display: inline-block;
                    vertical-align: middle;
                    color: white;
                    text-align: right;
                    padding-right: 5px;
                    box-sizing: border-box;
                    min-width: 30px;
                }
                .label {
                    display: inline-block;
                    width: 120px;
                    margin-right: 5px;
                    color: #333;
                }
                .average-bar {
                    font-weight: bold;
                }
            </style>
            <div class="popup-container">
            """
            
            # Используем предвычисленное значение spatial_inequality
            overall_avg = row.get('spatial_inequality', np.nan)
            
            # Определяем максимальное значение для нормализации гистограммы
            all_values = [row[col] for col in columns if col in row.index and pd.notnull(row[col])]
            max_val = max(all_values, default=1) if all_values else 1
            
            # Добавляем общее среднее значение в начале
            if pd.notnull(overall_avg):
                avg_width = (overall_avg / max_val) * 150
                normalized_avg = overall_avg / max_val if max_val > 0 else 0
                avg_color = bar_colormap(normalized_avg)  # Используем bar_colormap для цвета
                html += f"""
                <div class="group-container" style="width: 100%;">
                    <div class="group-title">Пространственное неравенство</div>
                    <div class="bar-container">
                        <span class="label">Значение неравенства</span>
                        <span class="bar average-bar" style="width: {avg_width}px; background: {avg_color};">{round(overall_avg, 3)}</span>
                    </div>
                </div>
                """
            
            # Добавляем остальные группы
            for group_name, group_cols in groups.items():
                display_name = group_name + ' inequality'
                html += f'<div class="group-container"><div class="group-title">{display_name}</div>'
                
                # Добавляем среднее по группе, если указано (для полигонов, кроме provision)
                if include_group_means and group_name != 'provision' and f'{group_name}_mean' in row.index:
                    mean_value = row[f'{group_name}_mean']
                    if pd.notnull(mean_value):
                        normalized_mean = mean_value / max_val if max_val > 0 else 0
                        mean_color = bar_colormap(normalized_mean)
                        mean_width = (mean_value / max_val) * 150
                        html += f"""
                        <div class="bar-container">
                            <span class="label">{group_name}_mean</span>
                            <span class="bar" style="width: {mean_width}px; background: {mean_color};">{round(mean_value, 3)}</span>
                        </div>
                        """
                
                # Добавляем отдельные атрибуты
                for col in group_cols:
                    if col in row.index:
                        value = row[col]
                        if pd.notnull(value):
                            normalized_value = value / max_val if max_val > 0 else 0
                            bar_color = bar_colormap(normalized_value)
                            width = (value / max_val) * 150
                            html += f"""
                            <div class="bar-container">
                                <span class="label">{col}</span>
                                <span class="bar" style="width: {width}px; background: {bar_color};">{round(value, 3)}</span>
                            </div>
                            """
                html += '</div>'
            
            html += '</div>'
            return html
        
        # Добавляем точки (города)
        for _, row in spatial_inequality_gdf.iterrows():
            # Определяем цвет точки
            if color_by_average or selected_attribute is None:
                if 'spatial_inequality' in row.index and pd.notnull(row['spatial_inequality']):
                    color = colormap(row['spatial_inequality'])
                else:
                    color = '#888888'  # Серый цвет для точек без данных
            else:
                if selected_attribute in row.index and pd.notnull(row[selected_attribute]):
                    color = colormap(row[selected_attribute])
                else:
                    color = '#888888'
            
            # Название города (для всплывающей подсказки)
            city_name = row.get('name', f"Город {_}")
            
            # Создаем всплывающую подсказку для точек (без среднего по группам)
            popup_html = f"""
            <b>Город: {city_name}</b><br>
            Население: {row.get('population', 'Н/Д')}<br>
            {create_histogram_html(row, groups, include_group_means=False)}
            """
            
            # Добавляем точку на карту
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                stroke=False,
                color='black',
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=600)
            ).add_to(points_layer)
        
        # Добавляем полигоны
        territories_layer = folium.FeatureGroup(name="Territories")
        
        for _, row in polygon_spatial_inequality.iterrows():
            # Определяем цвет полигона
            if color_by_average or selected_attribute is None:
                if 'spatial_inequality' in row.index and pd.notnull(row['spatial_inequality']):
                    fill_color = colormap(row['spatial_inequality'])
                else:
                    fill_color = '#888888'  # Серый цвет для полигонов без данных
            else:
                if selected_attribute in row.index and pd.notnull(row[selected_attribute]):
                    fill_color = colormap(row[selected_attribute])
                else:
                    fill_color = '#888888'
            
            # Создаем GeoJSON объект с полигоном
            geo_json = folium.GeoJson(
                data=row.geometry.__geo_interface__,
                style_function=lambda x, fill_color=fill_color: {
                    'fillColor': fill_color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.4
                }
            )
            
            # Название территории (для всплывающей подсказки)
            territory_name = row.get('name', f"Территория {row['anchor_name']}")
            
            # Создаем всплывающую подсказку для полигонов (с средним по группам)
            popup_html = f"""
            <b>Название: {territory_name}</b><br>
            {create_histogram_html(row, groups, include_group_means=True)}
            """
            
            folium.Popup(popup_html, max_width=600).add_to(geo_json)
            geo_json.add_to(territories_layer)
        
        territories_layer.add_to(m)
        points_layer.add_to(m)
        
        # Добавляем переключатель слоев и легенду
        folium.LayerControl().add_to(m)
        
        # Добавляем цветовую шкалу с соответствующей подписью
        if color_by_average:
            colormap.caption = "Пространственное неравенство"
        elif selected_attribute:
            colormap.caption = f"Значение атрибута: {selected_attribute}"
        else:
            colormap.caption = "Значения атрибутов"
        
        m.add_child(colormap)
        
        return m