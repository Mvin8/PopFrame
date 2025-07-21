"""
Tests for Region model
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import MagicMock

from popframe.models.region import Region
from popframe.models.town import Town


class TestRegion:
    """Test cases for Region class"""
    
    @pytest.fixture
    def sample_region_gdf(self):
        """Create sample region GeoDataFrame"""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        return gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
    
    @pytest.fixture
    def sample_towns_gdf(self):
        """Create sample towns GeoDataFrame"""
        data = {
            'name': ['Town A', 'Town B'],
            'population': [50000, 30000],
            'level': ['Средний город', 'Малый город'],
            'geometry': [Point(1, 1), Point(2, 2)]
        }
        gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
        gdf.index = [1, 2]
        return gdf
    
    @pytest.fixture
    def sample_accessibility_matrix(self):
        """Create sample accessibility matrix"""
        return pd.DataFrame(
            [[0.0, 45.0], [45.0, 0.0]],
            index=[1, 2],
            columns=[1, 2]
        )
    
    def test_region_initialization(self, sample_region_gdf, sample_towns_gdf, sample_accessibility_matrix):
        """Test Region initialization"""
        region = Region(
            region=sample_region_gdf,
            towns=sample_towns_gdf,
            accessibility_matrix=sample_accessibility_matrix
        )
        
        assert region.crs == sample_towns_gdf.crs
        assert len(region.towns) == 2
        assert isinstance(region.accessibility_matrix, pd.DataFrame)
    
    def test_region_validation_errors(self, sample_region_gdf, sample_towns_gdf):
        """Test validation errors"""
        # Invalid accessibility matrix
        wrong_matrix = pd.DataFrame([[0.0]], index=[1], columns=[1])
        
        with pytest.raises(AssertionError):
            Region(
                region=sample_region_gdf,
                towns=sample_towns_gdf,
                accessibility_matrix=wrong_matrix
            )
    
    def test_town_access(self, sample_region_gdf, sample_towns_gdf, sample_accessibility_matrix):
        """Test town access by ID"""
        region = Region(
            region=sample_region_gdf,
            towns=sample_towns_gdf,
            accessibility_matrix=sample_accessibility_matrix
        )
        
        town = region[1]
        assert isinstance(town, Town)
        assert town.name == 'Town A'
        
        # Test accessibility access
        time = region[1, 2]
        assert time == 45.0
    
    def test_get_towns_gdf(self, sample_region_gdf, sample_towns_gdf, sample_accessibility_matrix):
        """Test getting towns as GeoDataFrame"""
        region = Region(
            region=sample_region_gdf,
            towns=sample_towns_gdf,
            accessibility_matrix=sample_accessibility_matrix
        )
        
        towns_gdf = region.get_towns_gdf()
        assert isinstance(towns_gdf, gpd.GeoDataFrame)
        assert len(towns_gdf) == 2
        assert 'id' in towns_gdf.columns