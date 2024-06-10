from typing import List
import networkx as nx
import geopandas as gpd
import folium
import json
from ..models.region import Town
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .base_method import BaseMethod

class PopFrame(BaseMethod):
    """Class provides methods for network analysis on geographic data"""

    def _initialize_graph(self, towns: gpd.GeoDataFrame):
        """Initializes the graph with nodes"""
        G = nx.Graph()
        for idx, row in towns.iterrows():
            G.add_node(idx, pos=(row.geometry.x, row.geometry.y), level=row['level'])
        return G

    def _connect_levels(self, towns: gpd.GeoDataFrame, G: nx.Graph, from_level: str, to_levels: List[str]):
        """Connects nodes across specified levels"""
        from_towns = towns[towns['level'] == from_level].index
        to_towns = towns[towns['level'].isin(to_levels)].index

        if to_towns.empty:
            return

        unconnected_from_towns = set(from_towns)

        for idx in from_towns:
            distances = self.region.accessibility_matrix.loc[idx, to_towns]
            if distances.empty:
                continue
            nearest_idx = distances.idxmin()

            if nearest_idx and not G.has_edge(idx, nearest_idx):
                G.add_edge(idx, nearest_idx, level=from_level)
                if idx in unconnected_from_towns:
                    unconnected_from_towns.remove(idx)

        # Connect remaining unconnected nodes using Euclidean distance
        if unconnected_from_towns:
            to_coords = towns.loc[to_towns].geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
            for idx in unconnected_from_towns:
                from_coord = (towns.loc[idx].geometry.x, towns.loc[idx].geometry.y)
                nearest_idx = min(to_towns, key=lambda to_idx: self._euclidean_distance(from_coord, (towns.loc[to_idx].geometry.x, towns.loc[to_idx].geometry.y)))
                G.add_edge(idx, nearest_idx, level=from_level)

    def _euclidean_distance(self, coord1, coord2):
        """Calculates Euclidean distance between two coordinates"""
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    def _connect_same_level(self, towns: gpd.GeoDataFrame, G: nx.Graph, level: str):
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

        # for level in levels:
        #     self._connect_same_level(towns, G, level)

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