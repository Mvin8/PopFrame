from typing import List
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely.ops import nearest_points
from itertools import combinations
import numpy as np
import pandas as pd


from ..models.region import Town
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

    def evaluate_territory_location(self, territories_gdf):
        """
        Main function to evaluate the location of a territory relative to settlements.

        Parameters:
            self: The class object containing methods to retrieve settlements and territories.
            territories (GeoDataFrame): Geospatial data with territories. If None, territories from self.region will be used.

        Returns:
            list: A list of evaluation results for each territory.
        """
        settlements_gdf = self.region.get_towns_gdf()

        # Coordinate system transformation
        settlements_gdf = settlements_gdf.to_crs(epsg=3857)
        territories_gdf = territories_gdf.to_crs(epsg=3857)

        # Dictionary with settlement level scores
        level_scores = self._get_level_scores()

        results = []

        # Process each territory
        for territory in territories_gdf.itertuples():
            territory_name = getattr(territory, "name", "None")
            territory_geom = territory.geometry

            # Territory evaluation
            result = self._evaluate_single_territory(territory_geom, territory_name, settlements_gdf, level_scores)
            results.append(result)

        return results


    def _get_level_scores(self):
        """
        Returns a dictionary with scores for different levels of settlements.

        Returns:
            dict: A dictionary where the key is the settlement level and the value is the score.
        """
        return {
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


    def _evaluate_single_territory(self, territory_geom, territory_name, settlements_gdf, level_scores):
        """
        Evaluates a single territory, determining its nearest settlements and interpretation.

        Parameters:
            territory_geom (geometry): The geometry of the territory.
            territory_name (str): The name of the territory.
            settlements_gdf (GeoDataFrame): Geospatial data with settlements.
            level_scores (dict): A dictionary with settlement level scores.

        Returns:
            dict: The results of the territory evaluation.
        """
        buffer = territory_geom.buffer(10000)
        settlements_in_buffer = settlements_gdf[settlements_gdf.geometry.intersects(buffer)]

        if not settlements_in_buffer.empty:
            return self._evaluate_nearby_settlement(territory_name, settlements_in_buffer, level_scores)
        else:
            return self._evaluate_between_settlements(territory_geom, territory_name, settlements_gdf)


    def _evaluate_nearby_settlement(self, territory_name, settlements_in_buffer, level_scores):
        """
        Evaluates the territory if it is near settlements.

        Parameters:
            territory_name (str): The name of the territory.
            settlements_in_buffer (GeoDataFrame): The settlements within the buffer.
            level_scores (dict): A dictionary with settlement level scores.

        Returns:
            dict: The results of the territory evaluation.
        """
        settlements_in_buffer['score'] = settlements_in_buffer['level'].map(level_scores)
        max_settlement = settlements_in_buffer.loc[settlements_in_buffer['score'].idxmax()]
        max_level = max_settlement['level']
        max_score = max_settlement['score']
        closest_settlement_name = max_settlement['name']

        interpretation = f"Территория находится внутри или непосредственной близости населенного пункта уровня {max_level}"
        return {
            "territory": territory_name,
            "score": max_score,
            "interpretation": interpretation,
            "closest_settlement": closest_settlement_name,
            "closest_settlement1": None,
            "closest_settlement2": None
        }


    def _evaluate_between_settlements(self, territory_geom, territory_name, settlements_gdf):
        """
        Evaluates the territory if it is located between settlements.

        Parameters:
            territory_geom (geometry): The geometry of the territory.
            territory_name (str): The name of the territory.
            settlements_gdf (GeoDataFrame): Geospatial data with settlements.

        Returns:
            dict: The results of the territory evaluation.
        """
        buffer_20km = territory_geom.buffer(30000)
        nearby_settlements = settlements_gdf[settlements_gdf.geometry.intersects(buffer_20km)]

        if len(nearby_settlements) > 1:
            closest_settlement1, closest_settlement2 = self._find_closest_settlement_pair(territory_geom, nearby_settlements)

            if closest_settlement1 and closest_settlement2:
                interpretation = "Территория находится между основными ядрами системы расселения"
                return {
                    "territory": territory_name,
                    "score": 1,
                    "interpretation": interpretation,
                    "closest_settlement": None,
                    "closest_settlement1": closest_settlement1.name,
                    "closest_settlement2": closest_settlement2.name
                }
        
        return {
            "territory": territory_name,
            "score": 0,
            "interpretation": "Территория находится за границей агломерации",
            "closest_settlement": None,
            "closest_settlement1": None,
            "closest_settlement2": None
        }


    def _find_closest_settlement_pair(self, territory_geom, nearby_settlements):
        """
        Finds the pair of nearest settlements around the territory.

        Parameters:
            territory_geom (geometry): The geometry of the territory.
            nearby_settlements (GeoDataFrame): The nearby settlements.

        Returns:
            tuple: The pair of nearest settlements and the minimum distance.
        """

        min_distance = float('inf')
        closest_settlement1 = None
        closest_settlement2 = None

        for settlement1, settlement2 in combinations(nearby_settlements.itertuples(), 2):
            distance_between_settlements = settlement1.geometry.distance(settlement2.geometry)
            distance_to_settlement1 = territory_geom.distance(settlement1.geometry)
            distance_to_settlement2 = territory_geom.distance(settlement2.geometry)
            total_distance = distance_to_settlement1 + distance_to_settlement2

            if (distance_to_settlement1 > 10000 and distance_to_settlement2 > 10000 and
                    total_distance <= 1.2 * distance_between_settlements and
                    total_distance < min_distance):
                min_distance = total_distance
                closest_settlement1 = settlement1
                closest_settlement2 = settlement2

        return closest_settlement1, closest_settlement2
       
    def population_criterion(self, territories_gdf):
        """
        Calculates population density and assesses territories based on demographic characteristics.

        Parameters:
        territories (GeoDataFrame or None): GeoDataFrame of territories or None to use default region territories.

        Returns:
        list: A list of dictionaries containing the assessment results for each territory.
        """
        gdf_territory = territories_gdf.to_crs(epsg=3857)
        towns_gdf = self.region.get_towns_gdf().to_crs(epsg=3857)
        results = self._calculate_density_population(gdf_territory, towns_gdf)
        
        for result in results:
            score = self._assess_territory(result['average_population_density'], result['total_population'])
            result['score'] = score
            result['interpretation'] = self._interpret_score(score)

        return results

    def _calculate_density_population(self, gdf_territory, towns_gdf, radius_m=20000):
        """
        Calculates population density within a specified buffer radius for each territory.

        Parameters:
        gdf_territory (GeoDataFrame): GeoDataFrame of territories.
        towns_gdf (GeoDataFrame): GeoDataFrame of towns with population data.
        radius_m (int, optional): Radius in meters for the buffer around each territory. Default is 20000.

        Returns:
        list: A list of dictionaries containing population density and total population for each territory.
        """
        results = []
        for _, territory in gdf_territory.iterrows():
            buffer = territory.geometry.buffer(radius_m)
            towns_in_buffer = gpd.sjoin(towns_gdf, gpd.GeoDataFrame(geometry=[buffer], crs=towns_gdf.crs), predicate='intersects')

            if not towns_in_buffer.empty:
                total_population = towns_in_buffer['population'].sum()
                buffer_area = buffer.area / 1e6  # in square kilometers
                population_density = total_population / buffer_area if buffer_area > 0 else 0
            else:
                total_population = 0
                population_density = 0

            results.append({
                'project': territory.get('name'),
                'average_population_density': round(population_density, 1),
                'total_population': total_population
            })

        return results

    def _assess_territory(self, density, population):
        """
        Assesses the territory based on population density and total population.

        Parameters:
        density (float): The average population density of the territory.
        population (int): The total population of the territory.

        Returns:
        int: The score representing the assessment of the territory.
        """
        if density == 0 and population == 0:
            return 0
        score_df = pd.DataFrame([
            {'min_dens': 0, 'max_dens': 10, 'min_pop': 0, 'max_pop': 1000, 'score': 1},
            {'min_dens': 0, 'max_dens': 10, 'min_pop': 1000, 'max_pop': 5000, 'score': 2},
            {'min_dens': 0, 'max_dens': 10, 'min_pop': 5000, 'max_pop': float('inf'), 'score': 3},
            {'min_dens': 10, 'max_dens': 50, 'min_pop': 0, 'max_pop': 1000, 'score': 2},
            {'min_dens': 10, 'max_dens': 50, 'min_pop': 1000, 'max_pop': 5000, 'score': 3},
            {'min_dens': 10, 'max_dens': 50, 'min_pop': 5000, 'max_pop': float('inf'), 'score': 4},
            {'min_dens': 50, 'max_dens': float('inf'), 'min_pop': 0, 'max_pop': 1000, 'score': 3},
            {'min_dens': 50, 'max_dens': float('inf'), 'min_pop': 1000, 'max_pop': 5000, 'score': 4},
            {'min_dens': 50, 'max_dens': float('inf'), 'min_pop': 5000, 'max_pop': float('inf'), 'score': 5}
        ])

        result = score_df[
            (score_df['min_dens'] <= density) & (density < score_df['max_dens']) &
            (score_df['min_pop'] <= population) & (population < score_df['max_pop'])
        ]
        if not result.empty:
            return result.iloc[0]['score']
        return 0

    def _interpret_score(self, score):
        """
        Interprets the score assigned to a territory.

        Parameters:
        score (int): The score representing the assessment of the territory.

        Returns:
        str: The interpretation of the score.
        """
        interpretations = {
            0: "Территория имеет нулевые показатели численности и плотности населения, что может усложнить ее развитие.",
            1: "Территория имеет малые показатели численности и плотности населения для активного развития.",
            2: "Территория имеет умеренные показатели численности и плотности населения, что указывает на потенциал для развития.",
            3: "Территория имеет показатели численности и плотности населения выше среднего, что указывает на возможность развития территории.",
            4: "Территория имеет хорошие показатели численности и плотности населения, что способствует ее активному развитию.",
            5: "Территория с высокими показателями численности и плотности населения, что указывает высокий потенциал развития."
        }
        return interpretations.get(score, "Неизвестный показатель.")

