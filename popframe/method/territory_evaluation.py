from typing import List
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely.ops import nearest_points
from itertools import combinations


from ..models.region import Town
from .base_method import BaseMethod


class TerritoryEvaluation(BaseMethod):

    @classmethod
    def _is_criterion_satisfied(self, profile_value, criterion_value):
        if isinstance(profile_value, tuple):
            return profile_value[0] <= criterion_value <= profile_value[1]
        return criterion_value >= profile_value
        
    @classmethod
    def _calculate_exceedance(self, profile_value, criterion_value):
        if isinstance(profile_value, tuple):
            if profile_value[0] <= criterion_value <= profile_value[1]:
                return criterion_value - profile_value[0]
            return 0
        return max(0, criterion_value - profile_value)

    def calculate_potential(self, profiles, criteria_values):
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


    def evaluate_territory_location(self):
        settlements_gdf = self.region.get_towns_gdf()
        territory_gdf = self.region.territory
        # Преобразование системы координат
        settlements_gdf = settlements_gdf.to_crs(epsg=3857)
        territory_gdf = territory_gdf.to_crs(epsg=3857)

        # Получение геометрии территории
        territory_geom = territory_gdf.geometry.unary_union

        # Фильтрация населенных пунктов второго и третьего уровней
        settlements_filtered = settlements_gdf[
            settlements_gdf['level'].isin(['Второй уровень', 'Третий уровень'])
        ]

        # Создание буфера вокруг геометрии территории для разных уровней населенных пунктов
        buffer_7km = territory_geom.buffer(7000)  # для 1 уровня
        buffer_5km = territory_geom.buffer(5000)  # для 2 и 3 уровней

        # Поиск населенных пунктов внутри буферов
        highest_score = 0
        min_distance = float('inf')
        closest_settlement1 = None
        closest_settlement2 = None
        interpretation = ""
        closest_settlement_name = ""

        for idx, row in settlements_gdf.iterrows():
            point_within_7km = row['geometry'].intersects(buffer_7km)
            point_within_5km = row['geometry'].intersects(buffer_5km)

            if point_within_7km and row['level'] == "Первый уровень":
                if highest_score < 5:
                    highest_score = 5
                    interpretation = "территория находится внутри или непосредственной близости наиболее крупного населенного пункта"
                    closest_settlement_name = row['name']
            if point_within_5km and row['level'] == "Второй уровень":
                if highest_score < 4:
                    highest_score = 4
                    interpretation = "территория находится внутри или непосредственной близости крупного населенного пункта"
                    closest_settlement_name = row['name']
            if point_within_5km and row['level'] == "Третий уровень":
                if highest_score < 3:
                    highest_score = 3
                    interpretation = "территория находится внутри или непосредственной близости небольшого населенного пункта"
                    closest_settlement_name = row['name']

        if highest_score > 0:
            return highest_score, interpretation, closest_settlement_name, None, None
        else:
            # Приоритет отдаётся населенным пунктам второго уровня с расстоянием между ними не более 50,000 метров
            for settlement1, settlement2 in combinations(settlements_filtered.itertuples(), 2):
                distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)

                # Рассчитывать только для второго уровня, если расстояние между пунктами <= 50,000 метров
                if settlement1.level == 'Второй уровень' and settlement2.level == 'Второй уровень' and distance_between_settlements <= 50000:
                    distance_to_settlement1 = territory_geom.distance(settlement1.geometry)
                    distance_to_settlement2 = territory_geom.distance(settlement2.geometry)
                    total_distance = distance_to_settlement1 + distance_to_settlement2

                    # Проверка условий и обновление минимального расстояния
                    if (distance_to_settlement1 > 5000 and distance_to_settlement2 > 5000 and
                            total_distance <= 1.2 * distance_between_settlements and
                            total_distance < min_distance):
                        min_distance = total_distance
                        closest_settlement1 = settlement1
                        closest_settlement2 = settlement2
                        highest_score = 2
                        interpretation = "территория находится между основными ядрами системы расселения"

            # Проверка населенных пунктов третьего уровня, если не найдено подходящих пар второго уровня
            if not closest_settlement1:
                for settlement1, settlement2 in combinations(settlements_filtered.itertuples(), 2):
                    distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)
                    if settlement1.level == 'Третий уровень' and settlement2.level == 'Третий уровень' and distance_between_settlements <= 30000:
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
                            interpretation = "территория находится между незначительными ядрами системы расселения"

            # Возвращение результатов, если найдена подходящая пара
            if closest_settlement1 and closest_settlement2:
                return highest_score, interpretation, None, closest_settlement1.name,  closest_settlement2.name,
            
            # Если территория не соответствует условиям, возвращаем только False
            return 0, "Территория находится за границей агломерации", None, None, None


    # def population_criterion(self):

