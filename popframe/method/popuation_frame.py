from typing import List
import numpy as np
import networkx as nx
import geopandas as gpd
import folium
import json
from ..models.region import Town
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon

from .base_method import BaseMethod

class PopulationFrame(BaseMethod):
    """Class provides methods for network analysis on geographic data"""

    def _initialize_graph(self, towns: gpd.GeoDataFrame):
        """Initializes the graph with nodes"""
        G = nx.Graph()
        for idx, row in towns.iterrows():
            G.add_node(
                idx, 
                pos=(row.geometry.x, row.geometry.y), 
                level=row['level'], 
                name=row['name'],
                population=row['population']
            )
        return G

    def _connect_levels(self, towns: gpd.GeoDataFrame, G: nx.Graph, from_level: str, to_levels: List[str]):
        """Connects nodes across specified levels using time from accessibility_matrix"""
        from_towns = towns[towns['level'] == from_level].index
        to_towns = towns[towns['level'].isin(to_levels)].index

        if to_towns.empty:
            return

        for idx in from_towns:
            times = self.region.accessibility_matrix.loc[idx, to_towns]  # Используем время из матрицы
            if times.empty:
                continue
            nearest_idx = times.idxmin()  # Находим ближайший по времени город
            min_time = times.min()  # Находим минимальное время

            if nearest_idx and not G.has_edge(idx, nearest_idx):
                # Добавляем ребро с атрибутом времени
                G.add_edge(idx, nearest_idx, level=from_level, time=min_time)

    def _connect_same_level(self, towns: gpd.GeoDataFrame, G: nx.Graph, level: str):
        """Connects nodes within the same level using Kruskal's algorithm with time from accessibility_matrix"""
        level_idxs = towns[towns['level'] == level].index

        if len(level_idxs) > 1:
            H = nx.Graph()

            for idx in level_idxs:
                H.add_node(idx)

            for idx1 in level_idxs:
                for idx2 in level_idxs:
                    if idx1 != idx2:
                        time = self.region.accessibility_matrix.at[idx1, idx2]  # Используем время из матрицы
                        H.add_edge(idx1, idx2, weight=time)

            mst = nx.minimum_spanning_tree(H)
            for edge in mst.edges(data=True):
                # Добавляем ребро с атрибутом времени
                G.add_edge(edge[0], edge[1], level=level, time=edge[2]['weight'])

    def build_network_frame(self) -> nx.Graph:
        towns = self.region.get_towns_gdf().to_crs(4326)
        G = self._initialize_graph(towns)

        levels = [
            "Малое сельское поселение",
            "Среднее сельское поселение",
            "Большое сельское поселение",
            "Крупное сельское поселение",
            "Малый город",
            "Средний город",
            "Большой город",
            "Крупный город",
            "Крупнейший город",
            "Сверхкрупный город"
        ]

        for i, level in enumerate(levels[:-1]):
            from_level = levels[i]
            to_levels = levels[i + 1:]
            self._connect_levels(towns, G, from_level, to_levels)

        highest_level = max(towns['level'], key=lambda level: levels.index(level))
        self._connect_same_level(towns, G, highest_level)

        return G

    def get_color_map(self, levels):
        # Define the base colors for the heatmap starting from yellow to red
        base_colors = ['#FFC100', '#FF6500', '#C40C0C', '#6C0345']
        
        # Create a colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("heatmap", base_colors, N=len(levels))
        
        # Normalize the color map
        norm = mcolors.Normalize(vmin=0, vmax=len(levels) - 1)
        
        # Generate the colors for each level
        colors = [mcolors.to_hex(cmap(norm(i))) for i in range(len(levels))]
        
        return colors

    def get_graph_html(self, G: nx.Graph, filepath: str):
        """Saves the graph as an HTML file with a map visualization"""
        towns = self.region.get_towns_gdf()
        
        # Define levels
        levels = [
            "Малое сельское поселение",
            "Среднее сельское поселение",
            "Большое сельское поселение",
            "Крупное сельское поселение",
            "Малый город",
            "Средний город",
            "Большой город",
            "Крупный город",
            "Крупнейший город",
            "Сверхкрупный город"
        ]
        
        # Get the colors for each level
        level_colors = dict(zip(levels, self.get_color_map(levels)))

        m = folium.Map(zoom_start=3, tiles='CartoDB Positron')

        # Add edges to the map
        for edge in G.edges(data=True):
            start_pos = G.nodes[edge[0]]['pos']
            end_pos = G.nodes[edge[1]]['pos']
            start_level = G.nodes[edge[0]]['level']
            end_level = G.nodes[edge[1]]['level']
            
            lower_level = min(start_level, end_level, key=lambda x: list(level_colors.keys()).index(x))
            edge_color = level_colors[lower_level]

            folium.PolyLine(
                locations=[[start_pos[1], start_pos[0]], [end_pos[1], end_pos[0]]],
                color=edge_color,
                weight=2
            ).add_to(m)

        # Add nodes to the map
        for node, data in G.nodes(data=True):
            pos = data['pos']
            name = towns.loc[node, 'name']
            level = data['level']
            population = towns.loc[node, 'population']
            popup_text = f"{name}<br>Размер: {level}<br>Население: {population}"

            color = level_colors[level]
            radius = 7 if level == 'Сверхкрупный город' else 5  # Bigger radius for the largest city

            folium.CircleMarker(
                location=[pos[1], pos[0]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                popup=popup_text
            ).add_to(m)

        # Create legend
        legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 350px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;"><b>Легенда</b><br>'
        for level, color in level_colors.items():
            legend_html += f'&nbsp; {level} &nbsp; <i class="fa fa-circle" style="font-size:20px;color:{color}"></i><br>'
        legend_html += '</div>'

        # Add legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save map to HTML file
        m.save(filepath)

    def save_graph_to_geojson(self, G: nx.Graph, filepath: str = None):
        """Saves the graph as a GeoJSON file or returns it as a GeoDataFrame"""
        towns = self.region.get_towns_gdf()
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for node, attrs in G.nodes(data=True):
            name = towns.loc[node, 'name']
            population = int(towns.loc[node, 'population']) 
            
            point_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [attrs['pos'][0], attrs['pos'][1]]
                },
                "properties": {
                    "name": name,
                    "level": attrs['level'], 
                    "population": population
                }
            }
            geojson["features"].append(point_feature)

        for edge in G.edges(data=True):
            source_pos = G.nodes[edge[0]]['pos']
            target_pos = G.nodes[edge[1]]['pos']
            line_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[source_pos[0], source_pos[1]], [target_pos[0], target_pos[1]]]
                },
                "properties": {k: (int(v) if isinstance(v, np.int64) else v) for k, v in edge[2].items()}  # Преобразуем все int64 в int
            }
            geojson["features"].append(line_feature)

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(geojson, f)
            print(f"Graph successfully saved to {filepath} with all node attributes.")
        else:
            # Convert geojson to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson)
            return gdf
        

    def create_circle(self, center, size):
        return center.buffer(size)

    def size_from_population(self, population, level):
        if level in ["Малое сельское поселение", "Среднее сельское поселение", "Большое сельское поселение"]:
            return 0.0001 * (population ** 0.5)   # Логарифмическая шкала для малых населенных пунктов
        elif level == "Сверхкрупный город":
            return 0.00006 * (population ** 0.5)  # Уменьшенная линейная шкала для сверхкрупных городов
        return 0.0001 * (population ** 0.5)  # Линейная шкала для крупных населенных пунктов

    def convert_points_to_circles(self, gdf):
        new_geometries = []
        for idx, row in gdf.iterrows():
            if isinstance(row['geometry'], Point):
                size = self.size_from_population(row['population'], row['level'])
                size_in_meters = size * 111320  # Преобразование градусов в метры (примерный коэффициент)
                circle = self.create_circle(row['geometry'], size_in_meters)
                new_geometries.append(circle)
            else:
                new_geometries.append(row['geometry'])
        gdf['geometry'] = new_geometries
        return gdf

    def _get_color_map(self, levels):
        base_colors = ['#FFC100', '#FF6500', '#C40C0C', '#6C0345']
        cmap = mcolors.LinearSegmentedColormap.from_list("heatmap", base_colors, N=len(levels))
        norm = mcolors.Normalize(vmin=0, vmax=len(levels) - 1)
        colors = [mcolors.to_hex(cmap(norm(i))) for i in range(len(levels))]
        return colors

    def generate_map(self, towns):
        gdf = self.convert_points_to_circles(towns)
        gdf = gdf.to_crs(4326)
        levels = list(gdf['level'].unique())
        levels.sort(key=lambda x: ["Малое сельское поселение", "Среднее сельское поселение", "Большое сельское поселение", "Крупное сельское поселение", "Малый город", "Средний город", "Большой город", "Крупный город", "Крупнейший город", "Сверхкрупный город"].index(x))
        level_colors = dict(zip(levels, self._get_color_map(levels)))

        m = folium.Map(zoom_start=10, tiles='CartoDB Positron')

        for _, row in gdf.iterrows():
            color = level_colors[row['level']]
            popup_text = f"{row['name']}<br>Размер: {row['level']}<br>Население: {row['population']}"

            folium.GeoJson(
                row['geometry'],
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                tooltip=popup_text
            ).add_to(m)

        legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 350px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;"><b>Легенда</b><br>'
        for level, color in level_colors.items():
            legend_html += f'&nbsp; {level} &nbsp; <i class="fa fa-circle" style="font-size:20px;color:{color}"></i><br>'
        legend_html += '</div>'

        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def build_circle_frame(self, output_type='html'):
        towns = self.region.get_towns_gdf()
        if output_type == 'html':
            m = self.generate_map(towns)
            m.save('final_circle.html')
            return 'HTML map saved as final_circle.html'
        elif output_type == 'gdf':
            gdf = self.convert_points_to_circles(towns)
            return gdf
        else:
            raise ValueError("Unsupported output type. Choose either 'html' or 'gdf'.")