from .base_method import BaseMethod
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import pandas as pd
from shapely.ops import unary_union

RADIUS = 260
MAX_ANCHOR_TIME = 50

class AnchorSettlementBuilder(BaseMethod):
    
    def _build_anchor_settlement_boundaries(self, towns):
        """
        Builds boundaries for anchor settlements iteratively, starting from the closest town based on travel time.
        
        Parameters:
        - towns: GeoDataFrame of towns.
        
        Returns:
        - A GeoDataFrame of the anchor settlement boundaries.
        """
        anchor_towns = towns[towns['is_anchor_settlement'] == True]
        boundaries = []
        
        accessibility_matrix = self.region.accessibility_matrix
        
        for node in anchor_towns['id']:
            boundary = self._get_boundary_around_node(node, MAX_ANCHOR_TIME, towns, accessibility_matrix)
            
            if boundary:
                boundary["name"] = towns.loc[towns['id'] == node, 'name'].values[0]
                boundaries.append(boundary)
        
        if boundaries:
            boundary_gdf = gpd.GeoDataFrame(boundaries, columns=["name", "geometry"]).set_geometry('geometry')
            boundary_gdf.set_crs(towns.crs, inplace=True)
        else:
            boundary_gdf = gpd.GeoDataFrame(columns=["name", "geometry"])
        
        return boundary_gdf

    def _get_boundary_around_node(self, start_node, max_time, towns, accessibility_matrix):
        distances_from_start = accessibility_matrix.loc[start_node]
        within_time_nodes = distances_from_start[distances_from_start <= max_time].index
        
        if within_time_nodes.empty:
            return None
        
        nodes_gdf = towns.set_index('id').loc[within_time_nodes]
        
        distance = {node: (max_time - distances_from_start[node]) * RADIUS for node in within_time_nodes}
        # distance = {node: (max_time) * RADIUS for node in within_time_nodes}
        nodes_gdf["left_distance"] = nodes_gdf.index.map(distance)
        boundary_geom = nodes_gdf.buffer(nodes_gdf["left_distance"]).unary_union
        
        return {
            "geometry": boundary_geom,
            "nodes_in_boundary": list(within_time_nodes)
        }
    
    def _simplify_multipolygons(self, gdf, towns):
        anchor_towns = towns[towns['is_anchor_settlement'] == True]
        
        def process_geometry(geom):
            if isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
                
                # Фильтруем только те полигоны, которые пересекаются с anchor_towns
                filtered_polygons = [p for p in polygons if any(p.intersects(at) for at in anchor_towns.geometry)]
                
                if filtered_polygons:
                    return MultiPolygon(filtered_polygons) if len(filtered_polygons) > 1 else filtered_polygons[0]
                else:
                    return None  # Удаляем геометрию, если ни один полигон не пересекается
            
            return geom if any(geom.intersects(at) for at in anchor_towns.geometry) else None
        
        gdf['geometry'] = gdf['geometry'].apply(process_geometry)
        gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)  # Удаляем строки без геометрии
        return gdf
                
    def _merge_intersecting_boundaries(self, gdf):
        merged_geometries = []
        processed_indices = set()

        for i, row_i in gdf.iterrows():
            if i in processed_indices:
                continue

            overlapping_boundaries = [row_i]
            geometry = row_i['geometry']
            merged_names = {row_i['name']}

            for j, row_j in gdf.iterrows():
                if i != j and j not in processed_indices:
                    if geometry.intersects(row_j['geometry']):
                        overlapping_boundaries.append(row_j)
                        geometry = unary_union([geometry, row_j['geometry']]).buffer(0)
                        merged_names.add(row_j['name'])
                        processed_indices.add(j)
            
            still_merging = True
            while still_merging:
                still_merging = False
                for j, row_j in gdf.iterrows():
                    if j not in processed_indices and geometry.intersects(row_j['geometry']):
                        overlapping_boundaries.append(row_j)
                        geometry = unary_union([geometry, row_j['geometry']]).buffer(0)
                        merged_names.add(row_j['name'])
                        processed_indices.add(j)
                        still_merging = True
            
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            
            merged_boundary = {
                'geometry': geometry,
                'type': 'Merged' if len(merged_names) > 1 else 'Single',
                'anchor_settlements': ', '.join(merged_names)
            }
            merged_geometries.append(merged_boundary)
            
            processed_indices.add(i)

        return gpd.GeoDataFrame(merged_geometries, crs=gdf.crs)
    
    def get_anchor_settlement_boundaries(self, towns, update_df: pd.DataFrame | None = None):
        """
        The main function that orchestrates the creation, merging, and finalization of anchor settlement boundaries.
        
        Parameters:
        - towns: GeoDataFrame of towns.
        - update_df: Optional DataFrame for updates.
        
        Returns:
        - A GeoDataFrame with finalized anchor settlement boundaries.
        """
        if update_df is not None:
            towns = self.region.get_update_towns_gdf(update_df)
        
        region_boundary = self.region.region
        
        boundary_gdf = self._build_anchor_settlement_boundaries(towns)
  
        boundary_gdf = self._simplify_multipolygons(boundary_gdf, towns)
        
        boundary_gdf = self._merge_intersecting_boundaries(boundary_gdf)

        boundary_gdf = boundary_gdf.explode(index_parts=False)

        # Сбрасываем индекс (если необходимо)
        boundary_gdf = boundary_gdf.reset_index(drop=True)
        
        boundary_gdf = gpd.overlay(boundary_gdf, region_boundary, how='intersection')
        
        boundary_gdf = self._simplify_multipolygons(boundary_gdf, towns)
        
        boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(
            lambda geom: Polygon(geom.exterior) if geom.is_valid else geom
        )
        
        return boundary_gdf

