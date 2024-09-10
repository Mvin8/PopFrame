import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from shapely.ops import unary_union
import numpy as np
from .base_method import BaseMethod

class AgglomerationAnalysis(BaseMethod):
    
    def detect_agglomerations(self, G):
        """
        Применение алгоритма Лувена для обнаружения сообществ.
        
        Параметры:
        G (networkx.Graph): Входной граф с узлами и рёбрами.
        
        Возвращает:
        dict: Словарь, где ключи — индексы узлов, а значения — id агломераций.
        """
        partition = community_louvain.best_partition(G)
        return partition

    def get_largest_city_name(self, towns_gdf, nodes):
        """
        Нахождение названия крупнейшего города среди заданных узлов.
        
        Параметры:
        towns_gdf (GeoDataFrame): GeoDataFrame с информацией о городах.
        nodes (list): Список узлов для анализа.
        
        Возвращает:
        str: Название крупнейшего города.
        """
        largest_city = towns_gdf.loc[nodes].sort_values(by='population', ascending=False).iloc[0]
        return largest_city['name']

    def create_voronoi_polygons(self, towns_gdf, boundary_gdf):
        """
        Создает полигоны Вороного и обрезает их по границе региона.
        
        Параметры:
        towns_gdf (GeoDataFrame): GeoDataFrame с информацией о городах.
        boundary_gdf (GeoDataFrame): GeoDataFrame с границами региона.
        
        Возвращает:
        GeoDataFrame: GeoDataFrame с обрезанными полигонами Вороного и индексами узлов.
        """
        points = np.array([point.coords[0] for point in towns_gdf.geometry])
        vor = Voronoi(points)
        polygons = []
        node_indices = []
        
        for point_index, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not -1 in region and len(region) > 0:
                polygon = Polygon([vor.vertices[i] for i in region])
                clipped_polygon = polygon.intersection(boundary_gdf.geometry.unary_union)
                if not clipped_polygon.is_empty:
                    polygons.append(clipped_polygon)
                    node_indices.append(point_index)
        
        return gpd.GeoDataFrame({'geometry': polygons, 'node_index': node_indices}, crs=towns_gdf.crs)

    def build_agglomeration_voronoi(self, G):
        """
        Строит полигоны Вороного для агломераций и возвращает GeoDataFrame с крупнейшими городами.
        
        Параметры:
        G (networkx.Graph): Входной граф с узлами и рёбрами.
        towns_gdf (GeoDataFrame): GeoDataFrame с информацией о городах.
        boundary_gdf (GeoDataFrame): GeoDataFrame с границами региона.
        
        Возвращает:
        GeoDataFrame: GeoDataFrame с полигонами агломераций и именами крупнейших городов.
        """
        boundary_gdf = self.region.region
        towns_gdf = self.region.get_towns_gdf()

        partition = self.detect_agglomerations(G)
        voronoi_gdf = self.create_voronoi_polygons(towns_gdf, boundary_gdf)
        agglomerations = {}
        
        for node, agglomeration_id in partition.items():
            if agglomeration_id not in agglomerations:
                agglomerations[agglomeration_id] = []
            agglomerations[agglomeration_id].append(node)
        
        polygons = []
        names = []
        
        for agglomeration_id, nodes in agglomerations.items():
            agglomeration_polygons = voronoi_gdf[voronoi_gdf['node_index'].isin(nodes)].geometry
            if len(agglomeration_polygons) > 0:
                union_polygon = unary_union(agglomeration_polygons)
                name = self.get_largest_city_name(towns_gdf, nodes)
                polygons.append(union_polygon)
                names.append(name)
        
        agglomeration_gdf = gpd.GeoDataFrame({'name': names, 'geometry': polygons}, crs=towns_gdf.crs)
        return agglomeration_gdf

    def visualize_agglomerations(self, G):
        """
        Визуализирует агломерации на графе.
        
        Параметры:
        G (networkx.Graph): Входной граф с узлами и рёбрами.
        """
        partition = self.detect_agglomerations(G)
        pos = nx.get_node_attributes(G, 'pos')
        cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
        node_colors = [cmap(partition[node]) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, cmap=cmap)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        plt.axis('off')
        plt.show()
