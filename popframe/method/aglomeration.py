from .base_method import BaseMethod
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from popframe.preprocessing.level_filler import LevelFiller
import pandas as pd
from shapely.ops import unary_union

RADIUS = 400
MIN_POPULATION = 15000
IN_AGGLOMERATION = {}

CITY_LEVELS = [
            "Малый город",
            "Средний город",
            "Большой город",
            "Крупный город",
            "Крупнейший город",
            "Сверхкрупный город"
        ]


class AgglomerationBuilder(BaseMethod):

    def _build_agglomeration(self, towns, time):
        """
        Builds agglomerations for cities based on the accessibility matrix and travel time.
        
        Parameters:
        - time_threshold: Maximum travel time to consider cities in the same agglomeration (default is 80 minutes).
        
        Returns:
        - A GeoDataFrame of the agglomerations.
        """
        node_population = towns.set_index('id')['population']
        node_names = towns.set_index('id')['name']
        agglomerations = []

        for level_index, level in enumerate(reversed(CITY_LEVELS)):
            max_time = time - 10 * level_index
            level_nodes = towns[towns['level'] == level].sort_values(by='population', ascending=False)

            for node, population in level_nodes[['id', 'population']].itertuples(index=False):
                if node in IN_AGGLOMERATION or population < MIN_POPULATION:
                    continue

                agglomeration = self._get_agglomeration_around_node(node, max_time, towns)

                if agglomeration:
                    agglomeration["name"] = node_names.get(node)
                    agglomerations.append(agglomeration)

                    for member_node in agglomeration["nodes_in_agglomeration"]:
                        if node_population.get(member_node) < MIN_POPULATION:
                            IN_AGGLOMERATION[member_node] = True

        if agglomerations:
            agglomeration_gdf = gpd.GeoDataFrame(
                agglomerations, columns=["name",  "geometry"]
            ).set_geometry('geometry')
            agglomeration_gdf.set_crs(self.region.crs, inplace=True)
        else:
            agglomeration_gdf = gpd.GeoDataFrame(columns=["name", "geometry"])

        return agglomeration_gdf

    def _get_agglomeration_around_node(self, start_node, max_time, towns):
        """
        Finds the agglomeration around a given city node within the specified time limit using the accessibility matrix.
        
        Parameters:
        - start_node: The node to start agglomeration search from.
        - max_time: The maximum time to travel from the start node.
        
        Returns:
        - A dictionary containing the geometry of the agglomeration and the nodes within it.
        """
        accessibility_matrix = self.region.accessibility_matrix

        # Get the distances from the start_node to all others from the matrix
        distances_from_start = accessibility_matrix.loc[start_node]
        within_time_nodes = distances_from_start[distances_from_start <= max_time].index

        if within_time_nodes.empty:
            return None

        nodes_data = towns.set_index('id').loc[within_time_nodes]
        nodes_data['geometry'] = nodes_data.apply(lambda row: Point(row['geometry'].x, row['geometry'].y), axis=1)
        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs=self.region.crs)

        # Calculate the remaining distance buffer
        distance = {node: (max_time - distances_from_start[node]) * RADIUS for node in within_time_nodes}
        nodes_gdf["left_distance"] = nodes_gdf.index.map(distance)
        agglomeration_geom = nodes_gdf.buffer(nodes_gdf["left_distance"]).unary_union

        return {
            "geometry": agglomeration_geom,
            "nodes_in_agglomeration": list(within_time_nodes)
        }

    def _merge_intersecting_agglomerations(self, gdf, towns):
        """
        Merges intersecting agglomerations into larger polygons and sums up the population.
        
        Parameters:
        - gdf: GeoDataFrame of the agglomerations.
        - towns: GeoDataFrame of the towns.
        
        Returns:
        - A GeoDataFrame of the merged agglomerations with updated population data and agglomeration level.
        """
        merged_geometries = []
        processed_indices = set()

        # Основной цикл по агломерациям
        for i, row_i in gdf.iterrows():
            if i in processed_indices:
                continue

            overlapping_agglomerations = [row_i]
            geometry = row_i['geometry']
            merged_names = {row_i['name']}

            # Первый цикл: проверка пересечений и объединение агломераций
            for j, row_j in gdf.iterrows():
                if i != j and j not in processed_indices:
                    if geometry.intersects(row_j['geometry']):
                        overlapping_agglomerations.append(row_j)
                        geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                        merged_names.add(row_j['name'])
                        processed_indices.add(j)

            # Второй цикл: дополнительная проверка для объединения с новыми полигонами
            still_intersecting = True
            while still_intersecting:
                still_intersecting = False
                for j, row_j in gdf.iterrows():
                    if j not in processed_indices:
                        if geometry.intersects(row_j['geometry']):
                            overlapping_agglomerations.append(row_j)
                            geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                            merged_names.add(row_j['name'])
                            processed_indices.add(j)
                            still_intersecting = True

            # Проверяем валидность геометрии и исправляем ее, если необходимо
            if not geometry.is_valid:
                geometry = geometry.buffer(0)

            towns_in_agglomeration = towns[towns.intersects(geometry)]
            population_from_towns = towns_in_agglomeration['population'].sum()

            # Определение уровня агломерации на основе населения
            if population_from_towns <= 250000:
                agglomeration_level = 1
            elif population_from_towns <= 500000:
                agglomeration_level = 2
            elif population_from_towns <= 1000000:
                agglomeration_level = 3
            elif population_from_towns <= 5000000:
                agglomeration_level = 4
            else:
                agglomeration_level = 5

            merged_agglomeration = {
                'geometry': geometry,
                'type': 'Polycentric' if len(merged_names) > 1 else 'Monocentric',
                'core_cities': ', '.join(merged_names),
                'population': population_from_towns,
                'agglomeration_level': agglomeration_level
            }
            merged_geometries.append(merged_agglomeration)

            processed_indices.add(i)

        return gpd.GeoDataFrame(merged_geometries, crs=gdf.crs)


    def _simplify_multipolygons(self, gdf):
        """
        Simplifies multipolygons to keep only the largest polygon for each agglomeration.
        
        Parameters:
        - gdf: GeoDataFrame containing the agglomeration geometries.
        
        Returns:
        - A GeoDataFrame with simplified geometries.
        """
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom
        )
        return gdf


    def evaluate_city_agglomeration_status(self, towns, agglomeration_gdf):
        """
        Evaluates cities according to their position in the agglomeration.
        
        Adds the 'agglomeration_status' and 'agglomeration_level' attributes to towns:
        - 'agglomeration_status': 'Outside the agglomeration', 'Agglomeration Center', 'In the agglomeration'.
        - 'agglomeration_level': 0 for cities outside agglomerations, and agglomeration level (1 to 5) for cities within agglomerations.
        
        Parameters:
        - towns: GeoDataFrame with cities.
        - agglomeration_gdf: GeoDataFrame with agglomerations.
        
        Returns:
        - Updated GeoDataFrame with added 'agglomeration_status' (str) and 'agglomeration_level' (int) attributes.
        """
        agglomeration_status = []
        agglomeration_level = []

        for idx, town in towns.iterrows():
            town_point = town['geometry']
            town_name = town['name']
            
            in_agglomeration = False
            is_core_city = False
            current_agglomeration_level = 0  # Default level for towns outside agglomerations

            for agg_idx, agg in agglomeration_gdf.iterrows():
                if town_point.intersects(agg['geometry']):
                    in_agglomeration = True
                    current_agglomeration_level = agg['agglomeration_level']

                    # Проверяем, является ли город основным в агломерации
                    core_cities = agg['core_cities'].split(', ')
                    if town_name in core_cities:
                        is_core_city = True
                        current_status = 'Центр агломерации'
                        break

            if is_core_city:
                agglomeration_status.append('Центр агломерации')
                agglomeration_level.append(current_agglomeration_level)
            elif not in_agglomeration:
                agglomeration_status.append('Вне агломерации')
                agglomeration_level.append(0)  # 0 for cities outside agglomerations
            else:
                agglomeration_status.append('В агломерации')
                agglomeration_level.append(current_agglomeration_level)

        towns['agglomeration_status'] = agglomeration_status
        towns['agglomeration_level'] = agglomeration_level
        return towns


    def get_agglomerations(self, update_df: pd.DataFrame | None = None, time: int = 80):
        """
        The main function that orchestrates the creation, merging, and finalization of agglomerations.
        
        Returns:
        - A GeoDataFrame with the finalized agglomerations, merged, simplified, and overlaid on region boundaries.
        """
        # towns = self._get_towns_gdf(update_df)
        towns = self.region.get_update_towns_gdf(update_df)

        region_boundary = self.region.region

        # Step 1: Build agglomerations
        agglomeration_gdf = self._build_agglomeration(towns, time)
                # Step 4: Simplify multipolygons
        agglomeration_gdf = self._simplify_multipolygons(agglomeration_gdf)
        # Step 2: Merge intersecting agglomerations and update population data
        agglomeration_gdf = self._merge_intersecting_agglomerations(agglomeration_gdf, towns)

        # Step 3: Overlay agglomerations on region boundaries
        agglomeration_gdf = gpd.overlay(agglomeration_gdf, region_boundary, how='intersection')

        # Step 4: Simplify multipolygons
        agglomeration_gdf = self._simplify_multipolygons(agglomeration_gdf)

        # Step 5: Final geometry corrections
        agglomeration_gdf['geometry'] = agglomeration_gdf['geometry'].apply(
            lambda geom: Polygon(geom.exterior) if geom.is_valid else geom
        )

        return agglomeration_gdf
