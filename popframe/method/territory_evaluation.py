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
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 1, "Инж инф": 1}
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
                "weights": {"Население": 1, "Транспорт": 1, "Экология": 1, "Соц-об": 0, "Инж инф": 1}
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
        territories_gdf = territories
        if territories is None:
            territories_gdf = self.region.get_territories_gdf()

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
            # Получение имени территории или "nan", если имя отсутствует
            territory_name = getattr(territory, "name", "None")
            
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
                    "territory": territory_name,
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
                    "territory": territory_name,
                    "score": highest_score,
                    "interpretation": interpretation,
                    "closest_settlement": None,
                    "closest_settlement1": closest_settlement1.name,
                    "closest_settlement2": closest_settlement2.name
                })
            else:
                results.append({
                    "territory": territory_name,
                    "score": 0,
                    "interpretation": "Территория находится за границей агломерации",
                    "closest_settlement": None,
                    "closest_settlement1": None,
                    "closest_settlement2": None
                })

        return results
       
    def population_criterion(self, territories):
        if territories is None:
            gdf_territory = self.region.get_territories_gdf().to_crs(epsg=3857)
        else: 
            gdf_territory = territories.to_crs(epsg=3857)

        towns_gdf = self.region.get_towns_gdf().to_crs(epsg=3857)

        def _calculate_density_population(gdf_territory, towns_gdf, radius_m=20000):
            results = []
            for _, territory in gdf_territory.iterrows():
                buffer = territory.geometry.buffer(radius_m)
                towns_in_buffer = towns_gdf[towns_gdf.intersects(buffer)]

                if not towns_in_buffer.empty:
                    total_population = towns_in_buffer['population'].sum()
                    buffer_area = buffer.area / 1e6  # в квадратных километрах
                    population_density = total_population / buffer_area if buffer_area > 0 else 0

                    results.append({
                        'project': territory.get('name'),
                        'average_population_density': round(population_density, 1),
                        'total_population': total_population
                    })
            return results

        results = _calculate_density_population(gdf_territory, towns_gdf)

        # Оценочная таблица только по плотности и численности населения
        score_table = {
            (0, 10, 0, 1000): 1, (0, 10, 1000, 5000): 2, (0, 10, 5000, float('inf')): 3,
            (10, 50, 0, 1000): 2, (10, 50, 1000, 5000): 3, (10, 50, 5000, float('inf')): 4,
            (50, float('inf'), 0, 1000): 3, (50, float('inf'), 1000, 5000): 4, (50, float('inf'), 5000, float('inf')): 5
        }



        def _assess_territory(density, population):
            for (min_dens, max_dens, min_pop, max_pop), score in score_table.items():
                if min_dens <= density < max_dens and min_pop <= population < max_pop:
                    return score
            return 0  # если не попадает в диапазон

        def _interpret_score(score):
            interpretations = {
                0: "Территория имеет очень низкую численность и плотность населения, что может усложнить ее развитие.",
                1: "Территория имеет малую численность и плотность населения для активного развития.",
                2: "Территория имеет умеренную численность и плотность населения, что указывает на потенциал для развития.",
                3: "Территория имеет показатели выше среднего, что указывает на возможность роста и развития территории.",
                4: "Территория имеет хорошие демографические характеристики, что способствует ее активному развитию.",
                5: "Территория с высокой плотностью и численностью населения, что показывает высокий потенциал экономического роста и развития."
            }
            return interpretations.get(score, "Неизвестный показатель.")

        for result in results:
            score = _assess_territory(result['average_population_density'], result['total_population'])
            result['score'] = score
            result['interpretation'] = _interpret_score(score)

        return results

