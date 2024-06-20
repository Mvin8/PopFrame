from typing import List
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrying import retry
import pandas as pd
import json
from io import StringIO
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
from ..models.region_old import Town
from .base_method import BaseMethod


class UrbanisationLevel(BaseMethod):

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def fetch_osm_data(self, polygon, tags):
        return ox.features_from_polygon(polygon, tags=tags)

    def get_landuse_data(self, territories):
        if isinstance(territories, gpd.GeoDataFrame):
            territories_gdf = territories.to_crs(4326)
        elif isinstance(territories, (str, dict)):
            if isinstance(territories, dict):
                territories = json.dumps(territories)
            territories_gdf = gpd.read_file(StringIO(territories))
            territories_gdf.set_crs(epsg=4326, inplace=True)
        elif territories is None:
            territories_gdf = self.region.get_territories_gdf()
        else:
            raise ValueError("Invalid type for territories. Expected GeoDataFrame, JSON string, dictionary, or None.")

        landuse_tags = {
            '1.3.1 Процент застройки жилищным строительством': ['residential', 'apartments', 'detached', 'construction'],
            '1.3.2 Процент земель сельскохозяйственного назначения': ['farmland', 'farmyard', 'orchard', 'vineyard', 'greenhouse_horticulture', 'meadow', 'plant_nursery', 'aquaculture', 'animal_keeping', 'breeding'],
            '1.3.3 Процент земель промышленного назначения': ['industrial', 'quarry', 'landfill'],
            '1.3.4 Процент земель, занимаемыми лесными массивами': ['forest', 'wood'],
            '1.3.5 Процент земель специального назначения': ['military', 'railway', 'cemetery', 'landfill', 'brownfield'],
            '1.3.6 Процент земель населенных пунктов': ['place_city', 'place_town'],
            '1.3.7 Процент земель, занимаемых особо охраняемыми природными территориями': ['national_park', 'protected_area', 'nature_reserve', 'conservation'],
            '1.3.8 Процент земель, занимаемых водным фондом': ['basin', 'reservoir', 'water', 'salt_pond']
        }

        unique_tags = set(tag for tags in landuse_tags.values() for tag in tags)
        tag_filters = [{'landuse': tag} for tag in unique_tags if not tag.startswith('place_')] + \
                      [{'place': 'city'}, {'place': 'town'}] + \
                      [{'natural': 'water'}, {'natural': 'wood'}]

        def process_polygon(polygon):
            unique_gdfs = {}
            with ThreadPoolExecutor() as executor:
                future_to_tag_filter = {executor.submit(self.fetch_osm_data, polygon, tag_filter): tag_filter for tag_filter in tag_filters}
                for future in as_completed(future_to_tag_filter):
                    tag_filter = future_to_tag_filter[future]
                    try:
                        gdf = future.result()
                        if not gdf.empty:
                            gdf = gdf.set_geometry('geometry')
                            if gdf.crs is None:
                                gdf.set_crs(epsg=4326, inplace=True)
                            unique_gdfs[frozenset(tag_filter.items())] = gdf
                    except Exception as e:
                        print(f"Error fetching data for tag filter {tag_filter}: {e}")

            combined_geometries = []
            for category, tags in landuse_tags.items():
                category_geoms = []
                for tag in tags:
                    if tag.startswith('place_'):
                        tag_filter = {'place': 'city' if tag == 'place_city' else 'town'}
                    else:
                        tag_filter = {'landuse': tag} if tag not in ['wood', 'water'] else {'natural': tag}

                    gdf = unique_gdfs.get(frozenset(tag_filter.items()))
                    if gdf is not None:
                        category_geoms.extend(gdf.geometry)

                if category_geoms:
                    valid_geoms = [geom for geom in category_geoms if geom.is_valid]
                    valid_geoms = [geom.buffer(0) for geom in valid_geoms]
                    category_union = unary_union(valid_geoms)
                    category_intersect = category_union.intersection(polygon)
                    if isinstance(category_intersect, (Polygon, MultiPolygon)):
                        combined_geometries.append({'indicator': category, 'geometry': category_intersect})
                    elif isinstance(category_intersect, GeometryCollection):
                        for geom in category_intersect.geoms:
                            if isinstance(geom, (Polygon, MultiPolygon)):
                                combined_geometries.append({'indicator': category, 'geometry': geom})

            return combined_geometries

        all_combined_geometries = []
        with ThreadPoolExecutor() as executor:
            future_to_polygon = {executor.submit(process_polygon, polygon): polygon for polygon in territories_gdf.geometry}
            for future in as_completed(future_to_polygon):
                all_combined_geometries.extend(future.result())

        landuse_gdf = gpd.GeoDataFrame(all_combined_geometries, columns=['indicator', 'geometry'], crs='EPSG:4326')

        # Calculate the area percentage
        region_area_km2 = territories_gdf.to_crs(territories_gdf.estimate_utm_crs()).geometry.area.sum() / 1e6
        landuse_gdf['urbanization'] = (landuse_gdf.to_crs(territories_gdf.estimate_utm_crs()).geometry.area / 1e6 / region_area_km2 * 100).round().astype(int)

        # Reorder columns
        landuse_gdf = landuse_gdf[['indicator', 'urbanization', 'geometry']]

        return landuse_gdf

    def plot_landuse(self, region_gdf, landuse_gdf):
        if region_gdf.crs is None:
            region_gdf.set_crs(epsg=4326, inplace=True)
        crs = region_gdf.estimate_utm_crs()
        region_gdf_utm = region_gdf.to_crs(crs)  # Преобразование региона в UTM

        fig, ax = plt.subplots(figsize=(10, 10))  # Увеличим размер фигуры
        ax.set_axis_off()  # Отключение осей
        
        region_gdf_utm.boundary.plot(ax=ax, linewidth=1, color='black', label='Граница региона')

        colors = {
            'Застройка жилищным строительством': 'blue',
            'Сельскохозяйственные земли': 'yellow',
            'Промышленные земли': 'gray',
            'Лесные массивы': 'green',
            'Земли специального назначения': 'brown',
            'Земли населенных пунктов': 'orange',
            'Особо охраняемые природные территории': 'purple',
            'Водный фонд': 'cyan'
        }
        
        landuse_mapping = {
            '1.3.1 Процент застройки жилищным строительством': 'Застройка жилищным строительством',
            '1.3.2 Процент земель сельскохозяйственного назначения': 'Сельскохозяйственные земли',
            '1.3.3 Процент земель промышленного назначения': 'Промышленные земли',
            '1.3.4 Процент земель, занимаемыми лесными массивами': 'Лесные массивы',
            '1.3.5 Процент земель специального назначения': 'Земли специального назначения',
            '1.3.6 Процент земель населенных пунктов': 'Земли населенных пунктов',
            '1.3.7 Процент земель, занимаемых особо охраняемыми природными территориями': 'Особо охраняемые природные территории',
            '1.3.8 Процент земель, занимаемых водным фондом': 'Водный фонд'
        }
        
        for key, label in landuse_mapping.items():
            gdf = landuse_gdf[landuse_gdf['indicator'] == key]
            if not gdf.empty:
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                gdf_utm = gdf.to_crs(crs)  # Преобразование каждого типа земельного использования в UTM
                gdf_utm.plot(ax=ax, color=colors.get(label, 'gray'), alpha=0.5, label=label)
        
        # Добавление подложки карты
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron)

        # Создание пользовательских меток для легенды
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) 
                           for label, color in colors.items()]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))  # Перемещение легенды в верхнюю часть справа
        plt.show()