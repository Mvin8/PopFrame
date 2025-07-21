"""
Validation utilities for PopFrame
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Union, List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon


class RegionValidator:
    """Валидатор для данных регионов"""
    
    @staticmethod
    def validate_accessibility_matrix(
        matrix: pd.DataFrame, 
        towns: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Расширенная валидация матрицы доступности
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Матрица доступности
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
            
        Returns
        -------
        pd.DataFrame
            Валидированная матрица
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if matrix.empty:
            raise ValueError("Матрица доступности не может быть пустой")
        
        if not matrix.index.equals(matrix.columns):
            raise ValueError("Индексы и колонки матрицы должны совпадать")
            
        if not matrix.index.equals(towns.index):
            raise ValueError("Индексы матрицы должны соответствовать индексам городов")
            
        if (matrix < 0).any().any():
            raise ValueError("Матрица доступности не может содержать отрицательные значения")
            
        if not pd.api.types.is_numeric_dtype(matrix.values):
            raise ValueError("Матрица доступности должна содержать числовые значения")
            
        return matrix
    
    @staticmethod 
    def validate_towns_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Валидация GeoDataFrame с городами
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame с городами
            
        Returns
        -------
        gpd.GeoDataFrame
            Валидированный GeoDataFrame
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError('Towns должен быть экземпляром gpd.GeoDataFrame')
            
        required_columns = ['geometry']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
            
        if not all(isinstance(geom, Point) for geom in gdf.geometry):
            raise ValueError("Все геометрии городов должны быть Point")
            
        if 'population' in gdf.columns:
            if (gdf['population'] < 0).any():
                raise ValueError("Население не может быть отрицательным")
                
        return gdf
    
    @staticmethod
    def validate_region_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Валидация GeoDataFrame региона
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame региона
            
        Returns
        -------
        gpd.GeoDataFrame
            Валидированный GeoDataFrame
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError('Region должен быть экземпляром gpd.GeoDataFrame')
            
        valid_geom_types = ['Polygon', 'MultiPolygon']
        if not gdf.geom_type.isin(valid_geom_types).all():
            raise ValueError(f'Геометрия региона должна быть {valid_geom_types}')
            
        return gdf[['geometry']]


class DataValidator:
    """Общие валидаторы данных"""
    
    @staticmethod
    def validate_population_data(data: Union[pd.Series, np.ndarray, List]) -> bool:
        """
        Валидация данных о населении
        
        Parameters
        ----------
        data : Union[pd.Series, np.ndarray, List]
            Данные о населении
            
        Returns
        -------
        bool
            True если данные валидны
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values
            
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Данные о населении должны быть числовыми")
            
        if (data < 0).any():
            raise ValueError("Население не может быть отрицательным")
            
        return True
    
    @staticmethod
    def validate_time_parameter(time: Union[int, float]) -> int:
        """
        Валидация параметра времени
        
        Parameters
        ----------
        time : Union[int, float]
            Время в минутах
            
        Returns
        -------
        int
            Валидированное время
            
        Raises
        ------
        ValueError
            При некорректном времени
        """
        if not isinstance(time, (int, float)):
            raise ValueError("Время должно быть числом")
            
        if time <= 0:
            raise ValueError("Время должно быть положительным")
            
        if time > 300:  # 5 часов
            raise ValueError("Время не может превышать 5 часов (300 минут)")
            
        return int(time)
    
    @staticmethod
    def validate_crs_compatibility(*gdfs: gpd.GeoDataFrame) -> bool:
        """
        Проверка совместимости систем координат
        
        Parameters
        ----------
        *gdfs : gpd.GeoDataFrame
            GeoDataFrames для проверки
            
        Returns
        -------
        bool
            True если CRS совместимы
            
        Raises
        ------
        ValueError
            При несовместимых CRS
        """
        if len(gdfs) < 2:
            return True
            
        reference_crs = gdfs[0].crs
        
        for i, gdf in enumerate(gdfs[1:], 1):
            if gdf.crs != reference_crs:
                raise ValueError(f"CRS GeoDataFrame {i} не соответствует базовому CRS")
                
        return True