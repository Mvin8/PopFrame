from typing import List
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely.ops import nearest_points
from itertools import combinations
import numpy as np

import json
from io import StringIO


from ..models.region_old import Town
from .base_method import BaseMethod


class TerritoryEvaluation(BaseMethod):

    @classmethod
    def _is_criterion_satisfied(cls, profile_value, criterion_value):
        if isinstance(profile_value, tuple):
            return profile_value[0] <= criterion_value <= profile_value[1]
        return criterion_value >= profile_value

    @classmethod
    def _calculate_exceedance(cls, profile_value, criterion_value):
        if isinstance(profile_value, tuple):
            if profile_value[0] <= criterion_value <= profile_value[1]:
                return criterion_value - profile_value[0]
            return 0
        return max(0, criterion_value - profile_value)

    def calculate_potential(self, criteria_values):
        profiles = {
            "Жилая застройка - ИЖС": {
                "criteria": {"Население": 1, "Транспорт": 2, "Экология": 4, "Соц-об": 4, "Инж инф": 3},
                "weights": {"Население": 0, "Транспорт": 0, "Экология": 1, "Соц-об": 1, "Инж инф": 0}
            },
            "Жилая застройка - Малоэтажная": {
                "criteria": {"Население": 3, "Транспорт": 3, "Экология": 4, "Соц-об": 3, "Инж инф": 4},
                "weights": {"Население": 0, "Транспорт": 0, "Экология": 1, "Соц-об": 1, "Инж инф": 1}
            },
            "Жилая застройка - Среднеэтажная": {
                "criteria": {"Население": 4, "Транспорт": 4, "Экология": 4, "Соц-об": 3, "Инж инф": 5},
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 1, "Инж инф": 1}
            },
            "Жилая застройка - Многоэтажная": {
                "criteria": {"Население": 5, "Транспорт": 5, "Экология": 4, "Соц-об": 3, "Инж инф": 5},
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 1, "Инж инф": 1}
            },
            "Общественно-деловая": {
                "criteria": {"Население": 4, "Транспорт": 5, "Экология": 4, "Соц-об": 2, "Инж инф": 4},
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 0, "Инж инф": 1}
            },
            "Рекреационная": {
                "criteria": {"Население": 0, "Транспорт": 0, "Экология": 4, "Соц-об": 0, "Инж инф": 0},
                "weights": {"Население": 0, "Транспорт": 0, "Экология": 0, "Соц-об": 0, "Инж инф": 0}
            },
            "Специального назначения": {
                "criteria": {"Население": 0, "Транспорт": 3, "Экология": 1, "Соц-об": 0, "Инж инф": 2},
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 0, "Инж инф": 1}
            },
            "Промышленная": {
                "criteria": {"Население": 3, "Транспорт": 4, "Экология": 0, "Соц-об": 2, "Инж инф": 4},
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 0, "Соц-об": 0, "Инж инф": 1}
            },
            "Сельско-хозяйственная": {
                "criteria": {"Население": 3, "Транспорт": 4, "Экология": 4, "Соц-об": 2, "Инж инф": 3},
                "weights": {"Население": 0, "Транспорт": 0, "Экология": 1, "Соц-об": 0, "Инж инф": 0, "Кул-дос-сп": 0}
            },
            "Транспортная инженерная": {
                "criteria": {"Население": 2, "Транспорт": 2, "Экология": 0, "Соц-об": 1, "Инж инф": 2},
                "weights": {"Население": 0, "Транспорт": 0, "Экология": 0, "Соц-об": 0, "Инж инф": 0}
            }
        }

        potential_scores = {}
        for profile, data in profiles.items():
            criteria = data["criteria"]
            weights = data["weights"]
            potential = sum(
                self._is_criterion_satisfied(criteria.get(criterion, -1), value)
                for criterion, value in criteria_values.items()
            )
            weighted_score = sum(
                self._calculate_exceedance(criteria.get(criterion, -1), value) * weights.get(criterion, 1)
                for criterion, value in criteria_values.items()
            )
            potential_scores[profile] = (potential, weighted_score)

        ranked_profiles = sorted(potential_scores.items(), key=lambda x: (x[0] != "Рекреационная", x[1][0], x[1][1]), reverse=True)
        ranked_profiles = [item for item in ranked_profiles if item[1][0] > 0] + [item for item in ranked_profiles if item[1][0] == 0]

        return ranked_profiles


    def evaluate_territory_location(self, territories):
        settlements_gdf = self.region.get_towns_gdf()
        if territories is None:
            territories_gdf = self.region.get_territories_gdf()
        else:
            territories = json.dumps(territories)
            territories_gdf = gpd.read_file(StringIO(territories))

        # Преобразование системы координат
        settlements_gdf = settlements_gdf.to_crs(epsg=3857)
        territories_gdf = territories_gdf.to_crs(epsg=3857)

        results = []

        # Определение буферов для каждого уровня
        buffer_distances = {
            "Сверхкрупный город": 10000,
            "Крупнейший город": 9000,
            "Крупный город": 8000,
            "Большой город": 7000,
            "Средний город": 6000,
            "Малый город": 5000,
            "Крупное сельское поселение": 4000,
            "Большое сельское поселение": 3000,
            "Среднее сельское поселение": 2000,
            "Малое сельское поселение": 1000,
        }

        for territory in territories_gdf.itertuples():
            # Получение геометрии территории
            territory_geom = territory.geometry

            # Инициализация переменных для результатов
            highest_score = 0
            min_distance = float('inf')
            closest_settlement1 = None
            closest_settlement2 = None
            interpretation = ""
            closest_settlement_name = ""

            # Поиск населенных пунктов внутри буферов
            for idx, row in settlements_gdf.iterrows():
                level = row['level']
                buffer_distance = buffer_distances.get(level, 5000)
                buffer = territory_geom.buffer(buffer_distance)
                point_within_buffer = row['geometry'].intersects(buffer)

                if point_within_buffer:
                    level_scores = {
                        "Сверхкрупный город": 10,
                        "Крупнейший город": 9,
                        "Крупный город": 8,
                        "Большой город": 7,
                        "Средний город": 6,
                        "Малый город": 5,
                        "Крупное сельское поселение": 4,
                        "Большое сельское поселение": 3,
                        "Среднее сельское поселение": 2,
                        "Малое сельское поселение": 1,
                    }
                    score = level_scores.get(level, 0)
                    if score > highest_score:
                        highest_score = score
                        interpretation = f"территория находится внутри или непосредственной близости населенного пункта уровня {level}"
                        closest_settlement_name = row['name']

            if highest_score > 0:
                results.append({
                    "territory": territory.name,
                    "score": highest_score,
                    "interpretation": interpretation,
                    "closest_settlement": closest_settlement_name,
                    "closest_settlement1": None,
                    "closest_settlement2": None
                })
                continue

            # Проверка пар населенных пунктов
            for settlement1, settlement2 in combinations(settlements_gdf.itertuples(), 2):
                distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)
                distance_to_settlement1 = territory_geom.distance(settlement1.geometry)
                distance_to_settlement2 = territory_geom.distance(settlement2.geometry)
                total_distance = distance_to_settlement1 + distance_to_settlement2

                if (distance_to_settlement1 > 5000 and distance_to_settlement2 > 5000 and
                        total_distance <= 1.2 * distance_between_settlements and
                        total_distance < min_distance):
                    min_distance = total_distance
                    closest_settlement1 = settlement1
                    closest_settlement2 = settlement2
                    highest_score = 1
                    interpretation = "территория находится между основными ядрами системы расселения"

            # Возвращение результатов, если найдена подходящая пара
            if closest_settlement1 and closest_settlement2:
                results.append({
                    "territory": territory.name,
                    "score": highest_score,
                    "interpretation": interpretation,
                    "closest_settlement": None,
                    "closest_settlement1": closest_settlement1.name,
                    "closest_settlement2": closest_settlement2.name
                })
            else:
                results.append({
                    "territory": territory.name,
                    "score": 0,
                    "interpretation": "Территория находится за границей агломерации",
                    "closest_settlement": None,
                    "closest_settlement1": None,
                    "closest_settlement2": None
                })

        return results

    # def evaluate_territory_location(self, territories):
    #     # Mapping new levels to the original three levels
    #     level_mapping = {
    #         "Сверхкрупный город": "Первый уровень",
    #         "Крупнейший город": "Первый уровень",
    #         "Крупный город": "Первый уровень",
    #         "Большой город": "Второй уровень",
    #         "Средний город": "Второй уровень",
    #         "Малый город": "Второй уровень",
    #         "Крупное сельское поселение": "Третий уровень",
    #         "Большое сельское поселение": "Третий уровень",
    #         "Среднее сельское поселение": "Третий уровень",
    #         "Малое сельское поселение": "Третий уровень"
    #     }

    #     settlements_gdf = self.region.get_towns_gdf()
    #     settlements_gdf['level'] = settlements_gdf['level'].map(level_mapping)

    #     if territories is None:
    #         territories_gdf = self.region.get_territories_gdf()
    #     else:
    #         territories = json.dumps(territories)
    #         territories_gdf = gpd.read_file(StringIO(territories))

    #     # Преобразование системы координат
    #     settlements_gdf = settlements_gdf.to_crs(epsg=3857)
    #     territories_gdf = territories_gdf.to_crs(epsg=3857)

    #     results = []

    #     for territory in territories_gdf.itertuples():
    #         # Получение геометрии территории
    #         territory_geom = territory.geometry

    #         # Фильтрация населенных пунктов второго и третьего уровней
    #         settlements_filtered = settlements_gdf[
    #             settlements_gdf['level'].isin(['Второй уровень', 'Третий уровень'])
    #         ]

    #         # Создание буфера вокруг геометрии территории для разных уровней населенных пунктов
    #         buffer_7km = territory_geom.buffer(7000)  # для 1 уровня
    #         buffer_5km = territory_geom.buffer(5000)  # для 2 и 3 уровней

    #         # Поиск населенных пунктов внутри буферов
    #         highest_score = 0
    #         min_distance = float('inf')
    #         closest_settlement1 = None
    #         closest_settlement2 = None
    #         interpretation = ""
    #         closest_settlement_name = ""

    #         for idx, row in settlements_gdf.iterrows():
    #             point_within_7km = row['geometry'].intersects(buffer_7km)
    #             point_within_5km = row['geometry'].intersects(buffer_5km)

    #             if point_within_7km and row['level'] == "Первый уровень":
    #                 if highest_score < 5:
    #                     highest_score = 5
    #                     interpretation = "территория находится внутри или непосредственной близости наиболее крупного населенного пункта"
    #                     closest_settlement_name = row['name']
    #             if point_within_5km and row['level'] == "Второй уровень":
    #                 if highest_score < 4:
    #                     highest_score = 4
    #                     interpretation = "территория находится внутри или непосредственной близости крупного населенного пункта"
    #                     closest_settlement_name = row['name']
    #             if point_within_5km and row['level'] == "Третий уровень":
    #                 if highest_score < 3:
    #                     highest_score = 3
    #                     interpretation = "территория находится внутри или непосредственной близости небольшого населенного пункта"
    #                     closest_settlement_name = row['name']

    #         if highest_score > 0:
    #             results.append({
    #                 "territory": territory.name,
    #                 "score": highest_score,
    #                 "interpretation": interpretation,
    #                 "closest_settlement": closest_settlement_name,
    #                 "closest_settlement1": None,
    #                 "closest_settlement2": None
    #             })
    #             continue

    #         # Приоритет отдаётся населенным пунктам второго уровня с расстоянием между ними не более 50,000 метров
    #         for settlement1, settlement2 in combinations(settlements_filtered.itertuples(), 2):
    #             distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)

    #             # Рассчитывать только для второго уровня, если расстояние между пунктами <= 50,000 метров
    #             if settlement1.level == 'Второй уровень' and settlement2.level == 'Второй уровень' and distance_between_settlements <= 50000:
    #                 distance_to_settlement1 = territory_geom.distance(settlement1.geometry)
    #                 distance_to_settlement2 = territory_geom.distance(settlement2.geometry)
    #                 total_distance = distance_to_settlement1 + distance_to_settlement2

    #                 # Проверка условий и обновление минимального расстояния
    #                 if (distance_to_settlement1 > 5000 and distance_to_settlement2 > 5000 and
    #                         total_distance <= 1.2 * distance_between_settlements and
    #                         total_distance < min_distance):
    #                     min_distance = total_distance
    #                     closest_settlement1 = settlement1
    #                     closest_settlement2 = settlement2
    #                     highest_score = 2
    #                     interpretation = "территория находится между основными ядрами системы расселения"

    #         # Проверка населенных пунктов третьего уровня, если не найдено подходящих пар второго уровня
    #         if not closest_settlement1:
    #             for settlement1, settlement2 in combinations(settlements_filtered.itertuples(), 2):
    #                 distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)
    #                 if settlement1.level == 'Третий уровень' and settlement2.level == 'Третий уровень' and distance_between_settlements <= 30000:
    #                     distance_to_settlement1 = territory_geom.distance(settlement1.geometry)
    #                     distance_to_settlement2 = territory_geom.distance(settlement2.geometry)
    #                     total_distance = distance_to_settlement1 + distance_to_settlement2

    #                     if (distance_to_settlement1 > 5000 and distance_to_settlement2 > 5000 and
    #                             total_distance <= 1.2 * distance_between_settlements and
    #                             total_distance < min_distance):
    #                         min_distance = total_distance
    #                         closest_settlement1 = settlement1
    #                         closest_settlement2 = settlement2
    #                         highest_score = 1
    #                         interpretation = "территория находится между незначительными ядрами системы расселения"

    #         # Возвращение результатов, если найдена подходящая пара
    #         if closest_settlement1 and closest_settlement2:
    #             results.append({
    #                 "territory": territory.name,
    #                 "score": highest_score,
    #                 "interpretation": interpretation,
    #                 "closest_settlement": None,
    #                 "closest_settlement1": closest_settlement1.name,
    #                 "closest_settlement2": closest_settlement2.name
    #             })
    #         else:
    #             results.append({
    #                 "territory": territory.name,
    #                 "score": 0,
    #                 "interpretation": "Территория находится за границей агломерации",
    #                 "closest_settlement": None,
    #                 "closest_settlement1": None,
    #                 "closest_settlement2": None
    #             })

    #     return results
    
    def population_criterion(self, merged_gdf, territories):
        # Преобразование данных в систему координат 3857
        assessment_score = 3
        if territories is None:
            gdf_territory = self.region.get_territories_gdf().to_crs(epsg=3857)
        else:
            territories = json.dumps(territories)
            gdf_territory = gpd.read_file(StringIO(territories)).to_crs(epsg=3857)
        gdf_municipalities = merged_gdf.to_crs(epsg=3857)

        # Функция для вычисления средней плотности населения и среднего прироста населения в радиусе 60 км
        def _calculate_density_growth(gdf_territory, gdf_municipalities, radius_m=60000):
            results = []
            
            for _, territory in gdf_territory.iterrows():
                buffer = territory.geometry.buffer(radius_m)
                municipalities_in_buffer = gdf_municipalities[gdf_municipalities.intersects(buffer)]

                municipalities_filtered = [m for _, m in municipalities_in_buffer.iterrows()
                                        if m.geometry.intersection(buffer).area / m.geometry.area >= 0.15]

                if municipalities_filtered:
                    densities = [m['density_2022'] for m in municipalities_filtered]
                    growths = [m['growth_2022'] for m in municipalities_filtered]
                    avg_density = np.mean(densities)
                    avg_growth = np.mean(growths)
                    results.append({
                        'project': territory['name'],
                        'average_population_density': round(avg_density, 1),
                        'average_population_growth': round(avg_growth, 1)
                    })
            return results

        results = _calculate_density_growth(gdf_territory, gdf_municipalities)

        score_table = {
            (5, 'Меньше 10 чел на кв. км.', 'Убыль'): 2,
            (5, 'Меньше 10 чел на кв. км.', 'Рост'): 4,
            (5, '10 – 50 чел на кв. км.', 'Убыль'): 3,
            (5, '10 – 50 чел на кв. км.', 'Рост'): 5,
            (5, 'Более 50 чел. на кв. км.', 'Убыль'): 4,
            (5, 'Более 50 чел. на кв. км.', 'Рост'): 5,
            (4, 'Меньше 10 чел на кв. км.', 'Убыль'): 2,
            (4, 'Меньше 10 чел на кв. км.', 'Рост'): 4,
            (4, '10 – 50 чел на кв. км.', 'Убыль'): 3,
            (4, '10 – 50 чел на кв. км.', 'Рост'): 4,
            (4, 'Более 50 чел. на кв. км.', 'Убыль'): 4,
            (4, 'Более 50 чел. на кв. км.', 'Рост'): 5,
            (3, 'Меньше 10 чел на кв. км.', 'Убыль'): 2,
            (3, 'Меньше 10 чел на кв. км.', 'Рост'): 3,
            (3, '10 – 50 чел на кв. км.', 'Убыль'): 2,
            (3, '10 – 50 чел на кв. км.', 'Рост'): 3,
            (3, 'Более 50 чел. на кв. км.', 'Убыль'): 3,
            (3, 'Более 50 чел. на кв. км.', 'Рост'): 4,
            (2, 'Меньше 10 чел на кв. км.', 'Убыль'): 1,
            (2, 'Меньше 10 чел на кв. км.', 'Рост'): 2,
            (2, '10 – 50 чел на кв. км.', 'Убыль'): 1,
            (2, '10 – 50 чел на кв. км.', 'Рост'): 2,
            (2, 'Более 50 чел. на кв. км.', 'Убыль'): 2,
            (2, 'Более 50 чел. на кв. км.', 'Рост'): 3,
        }

        def _assess_territory(density, growth, assessment_score):
            growth_status = 'Убыль' if growth < 0 else 'Рост'
            if density < 10:
                density_status = 'Меньше 10 чел на кв. км.'
            elif 10 <= density <= 50:
                density_status = '10 – 50 чел на кв. км.'
            else:
                density_status = 'Более 50 чел. на кв. км.'

            return score_table.get((assessment_score, density_status, growth_status), 0)
        
        for result in results:
            result['score'] = _assess_territory(result['average_population_density'], result['average_population_growth'], assessment_score)

        return results

        

