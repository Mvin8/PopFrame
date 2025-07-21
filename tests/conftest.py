"""
Common test fixtures for PopFrame
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import MagicMock

from popframe.config.constants import AgglomerationConfig, PopulationThresholds, InfrastructureConfig


@pytest.fixture
def sample_polygon():
    """Create a sample polygon"""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def sample_points():
    """Create sample points"""
    return [Point(1, 1), Point(5, 5), Point(8, 8)]


@pytest.fixture
def sample_region_gdf(sample_polygon):
    """Create a sample region GeoDataFrame"""
    return gpd.GeoDataFrame([{'geometry': sample_polygon}], crs='EPSG:4326')


@pytest.fixture
def sample_towns_gdf(sample_points):
    """Create a sample towns GeoDataFrame"""
    data = {
        'name': ['Town A', 'Town B', 'Town C'],
        'population': [100000, 50000, 25000],
        'level': ['Большой город', 'Средний город', 'Малый город'],
        'geometry': sample_points
    }
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    gdf.index = [1, 2, 3]
    return gdf


@pytest.fixture
def sample_accessibility_matrix():
    """Create a sample accessibility matrix"""
    return pd.DataFrame([
        [0.0, 30.0, 60.0],
        [30.0, 0.0, 45.0],
        [60.0, 45.0, 0.0]
    ], index=[1, 2, 3], columns=[1, 2, 3])


@pytest.fixture
def mock_region(sample_region_gdf, sample_towns_gdf, sample_accessibility_matrix):
    """Create a mock Region object"""
    region = MagicMock()
    region.region = sample_region_gdf
    region.get_update_towns_gdf.return_value = sample_towns_gdf
    region.accessibility_matrix = sample_accessibility_matrix
    region.crs = 'EPSG:4326'
    return region


@pytest.fixture
def agglomeration_config():
    """Create AgglomerationConfig for testing"""
    return AgglomerationConfig()


@pytest.fixture
def population_thresholds():
    """Create PopulationThresholds for testing"""
    return PopulationThresholds()


@pytest.fixture
def infrastructure_config():
    """Create InfrastructureConfig for testing"""
    return InfrastructureConfig()


@pytest.fixture
def sample_infrastructure_gdf():
    """Create sample infrastructure GeoDataFrame"""
    data = {
        'type': ['hospital', 'school', 'power_plant'],
        'physical_object_type': [
            {'name': 'Больница'},
            {'name': 'Школа'},
            {'name': 'Атомная электростанция'}
        ],
        'geometry': [Point(2, 2), Point(4, 4), Point(6, 6)]
    }
    return gpd.GeoDataFrame(data, crs='EPSG:4326')


@pytest.fixture
def sample_assessment_areas_gdf():
    """Create sample assessment areas GeoDataFrame"""
    polygons = [
        Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        Polygon([(3, 3), (6, 3), (6, 6), (3, 6)]),
        Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])
    ]
    data = {
        'area_id': [1, 2, 3],
        'geometry': polygons
    }
    return gpd.GeoDataFrame(data, crs='EPSG:4326')