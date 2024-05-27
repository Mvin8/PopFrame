from typing import List
import networkx as nx
import geopandas as gpd
import folium
import json
from ..models.region import Town

from .base_method import BaseMethod

class PopFrame(BaseMethod):
    """Class provides methods for network analysis on geographic data"""

    def _initialize_graph(self, towns: gpd.GeoDataFrame):

        """Initializes the graph with nodes"""
        G = nx.Graph()
        for idx, row in towns.iterrows():
            G.add_node(idx, pos=(row.geometry.x, row.geometry.y), level=row['level'])
        return G

    def _connect_levels(self, towns: gpd.GeoDataFrame, G: nx.Graph,  from_level: str, to_levels: List[str]):
        """Connects nodes across specified levels"""
        from_towns = towns[towns['level'] == from_level].index
        to_towns = towns[towns['level'].isin(to_levels)].index

        for idx in from_towns:
            distances = self.region.accessibility_matrix.loc[idx, to_towns]
            nearest_idx = distances.idxmin()

            if nearest_idx and not G.has_edge(idx, nearest_idx):
                G.add_edge(idx, nearest_idx, level=from_level)

    def _connect_same_level(self, towns: gpd.GeoDataFrame, G: nx.Graph , level: str):
        """Connects nodes within the same level using Kruskal's algorithm"""
        level_idxs = towns[towns['level'] == level].index
        H = nx.Graph()

        for idx in level_idxs:
            H.add_node(idx)

        for idx1 in level_idxs:
            for idx2 in level_idxs:
                if idx1 != idx2:
                    distance = self.region.accessibility_matrix.at[idx1, idx2]
                    H.add_edge(idx1, idx2, weight=distance)

        mst = nx.minimum_spanning_tree(H)
        for edge in mst.edges(data=True):
            G.add_edge(edge[0], edge[1], level=level)

    def build_network(self) -> nx.Graph:
        towns = self.region.get_towns_gdf().to_crs(4326)
        G = self._initialize_graph(towns)

        self._connect_levels( towns, G , 'Пятый уровень', ['Второй уровень', 'Третий уровень', 'Четвертый уровень'])
        self._connect_levels(towns, G ,'Четвертый уровень', ['Третий уровень', 'Второй уровень'])
        self._connect_levels(towns, G ,'Третий уровень', ['Второй уровень'])
        self._connect_same_level(towns, G ,'Второй уровень')

        return  G

    def get_graph_html(self, G: nx.Graph, filepath: str):
        """Saves the graph as an HTML file with a map visualization"""
        towns = self.region.get_towns_gdf()

        m = folium.Map(zoom_start=3, tiles='CartoDB Positron')

        level_colors = {
            'Пятый уровень': '#6996B3',  # Синий для 1 уровня
            'Четвертый уровень': '#89AC76',  # Зеленый для 2 уровня
            'Третий уровень': '#FFC107',  # Красный для 3 уровня
            'Второй уровень': '#B71C1C'  # Желтый для 4 уровня
        }


        # Добавление рёбер графа на карту с логикой цвета из изначального кода
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

        # Добавление узлов графа на карту
        for node, data in G.nodes(data=True):
            pos = data['pos']
            name = towns.loc[node, 'name']
            # types = towns.loc[node, 'rural settlement']
            level = data['level']
            population = towns.loc[node, 'population']
            popup_text = f"{name}<br>Размер: {level}<br>Население: {population}"

            color = level_colors[level]
            radius = 7 if level == 'Второй уровень' else 5  # Больший радиус для уровня 2

            folium.CircleMarker(
                location=[pos[1], pos[0]],
                radius=radius,  # Стандартный радиус для всех узлов, можно настроить если нужно
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                popup=popup_text  # Упрощенный popup, можно дополнить информацией
            ).add_to(m)

        # Создание легенды
        legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: 130px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white;
                        ">
            &nbsp; <b>Легенда</b> <br>
            &nbsp; Пятый уровень &nbsp; <i class="fa fa-circle" style="font-size:20px;color:#6996B3"></i><br>
            &nbsp; Четвертый уровень &nbsp; <i class="fa fa-circle" style="font-size:20px;color:#89AC76"></i><br>
            &nbsp; Третий уровень &nbsp; <i class="fa fa-circle" style="font-size:20px;color:#FFC107"></i><br>
            &nbsp; Второй уровень &nbsp; <i class="fa fa-circle" style="font-size:20px;color:#B71C1C"></i><br>
            </div>
            """

        # Добавление легенды на карту
        m.get_root().html.add_child(folium.Element(legend_html))

        # Сохранение карты в HTML-файл
        m.save(filepath)

    def save_graph_to_geojson(self, G: nx.Graph, filepath: str = None):
        """Saves the graph as a GeoJSON file or returns it as a GeoDataFrame"""
        towns = self.region.get_towns_gdf()
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        # Добавление узлов графа как точек с сохранением всех атрибутов
        for node, attrs in G.nodes(data=True):
            name = towns.loc[node, 'name']
            population = towns.loc[node, 'population']
            
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

        # Добавление рёбер графа как линий
        for edge in G.edges(data=True):
            source_pos = G.nodes[edge[0]]['pos']
            target_pos = G.nodes[edge[1]]['pos']
            line_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[source_pos[0], source_pos[1]], [target_pos[0], target_pos[1]]]
                },
                "properties": edge[2]
            }
            geojson["features"].append(line_feature)

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(geojson, f)
            print(f"Graph successfully saved to {filepath} with all node attributes.")
        else:
            # Convert geojson to GeoDataFrame
            gdf= gpd.GeoDataFrame.from_features(geojson)
            return gdf