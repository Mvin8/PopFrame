from typing import List
import numpy as np
import networkx as nx
import geopandas as gpd
import folium
import json
from ..models.region import Town
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon
import pandas as pd

from .base_method import BaseMethod

class PopulationFrame(BaseMethod):

    def _create_circle(self, center, size):
        return center.buffer(size)

    def _size_from_population(self, population, level):
        if level in ["Малое сельское поселение", "Среднее сельское поселение", "Большое сельское поселение"]:
            return 0.0001 * (population ** 0.5)   # Логарифмическая шкала для малых населенных пунктов
        elif level == "Сверхкрупный город":
            return 0.00006 * (population ** 0.5)  # Уменьшенная линейная шкала для сверхкрупных городов
        return 0.0001 * (population ** 0.5)  # Линейная шкала для крупных населенных пунктов

    def _convert_points_to_circles(self, gdf):
        gdf['size'] = gdf.apply(lambda row: self._size_from_population(row['population'], row['level']), axis=1)
        gdf['size_in_meters'] = gdf['size'] * 111320
        gdf['geometry'] = gdf.apply(lambda row: self._create_circle(row['geometry'], row['size_in_meters']) if isinstance(row['geometry'], Point) else row['geometry'], axis=1)
        gdf = gdf.drop(columns=['size', 'size_in_meters'])
        return gdf

    def build_circle_frame(self, update_df: pd.DataFrame | None = None):
        towns = self.region.get_update_towns_gdf(update_df)
        gdf = self._convert_points_to_circles(towns)
        return gdf
