"""
Infrastructure analysis methods for PopFrame
"""

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from typing import Dict, Any, Set, List, Optional

from .base_method import BaseAnalyzer
from popframe.config.constants import InfrastructureConfig


class InfrastructureAnalyzer(BaseAnalyzer):
    """
    Анализатор инфраструктуры для оценки территорий
    
    Attributes
    ----------
    infrastructure_gdf : gpd.GeoDataFrame
        GeoDataFrame с объектами инфраструктуры
    assessment_areas_gdf : gpd.GeoDataFrame
        GeoDataFrame с территориями для оценки
    config : InfrastructureConfig
        Конфигурация анализатора
    """

    def __init__(
        self, 
        infrastructure_gdf: gpd.GeoDataFrame, 
        assessment_areas_gdf: gpd.GeoDataFrame,
        config: Optional[InfrastructureConfig] = None
    ) -> None:
        """
        Инициализация анализатора инфраструктуры

        Parameters
        ----------
        infrastructure_gdf : geopandas.GeoDataFrame
            GeoDataFrame с объектами инфраструктуры
        assessment_areas_gdf : geopandas.GeoDataFrame
            GeoDataFrame с территориями для оценки
        config : InfrastructureConfig, optional
            Конфигурация анализатора
        """
        super().__init__()
        self.config = config or InfrastructureConfig()
        
        # Валидация и подготовка данных
        self.infrastructure_gdf = self._prepare_infrastructure_data(infrastructure_gdf)
        self.assessment_areas_gdf = self._prepare_assessment_data(assessment_areas_gdf)

    def _prepare_infrastructure_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Подготовка данных об инфраструктуре
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Исходные данные об инфраструктуре
            
        Returns
        -------
        gpd.GeoDataFrame
            Подготовленные данные
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("infrastructure_gdf должен быть GeoDataFrame")
            
        if gdf.empty:
            raise ValueError("infrastructure_gdf не может быть пустым")
            
        # Приведение к метрической системе координат для расчетов
        return gdf.to_crs(epsg=3857)

    def _prepare_assessment_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Подготовка данных для оценки
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Исходные данные территорий
            
        Returns
        -------
        gpd.GeoDataFrame
            Подготовленные данные
            
        Raises
        ------
        ValueError
            При некорректных данных
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("assessment_areas_gdf должен быть GeoDataFrame")
            
        if gdf.empty:
            raise ValueError("assessment_areas_gdf не может быть пустым")
            
        # Приведение к метрической системе координат
        gdf = gdf.to_crs(epsg=3857)
        
        # Инициализация колонок для результатов
        gdf = gdf.copy()
        gdf['score'] = 0
        gdf['types_in_radius'] = None
        
        return gdf

    def validate_input(self, **kwargs) -> bool:
        """
        Валидация входных параметров
        
        Parameters
        ----------
        **kwargs
            Параметры для валидации
            
        Returns
        -------
        bool
            True если валидация прошла успешно
        """
        # Проверка наличия необходимых колонок
        required_infra_cols = ['physical_object_type', 'type']
        missing_infra_cols = [col for col in required_infra_cols 
                             if col not in self.infrastructure_gdf.columns]
        if missing_infra_cols:
            raise ValueError(f"Отсутствуют колонки в infrastructure_gdf: {missing_infra_cols}")
            
        return True

    def analyze(self, **kwargs) -> gpd.GeoDataFrame:
        """
        Выполнение анализа инфраструктуры
        
        Parameters
        ----------
        **kwargs
            Дополнительные параметры анализа
            
        Returns
        -------
        gpd.GeoDataFrame
            Результаты анализа в CRS 4326
        """
        self.validate_input(**kwargs)
        
        # Выполнение анализа
        self._analyze_infrastructure()
        
        # Возврат результатов
        return self.get_results()

    def get_radius(self, physical_object_type: Dict[str, Any]) -> float:
        """
        Определение радиуса влияния на основе типа объекта

        Parameters
        ----------
        physical_object_type : dict
            Словарь с информацией о типе физического объекта

        Returns
        -------
        float
            Радиус в метрах для данного типа объекта
        """
        name = physical_object_type.get('name', '')
        
        # Проверка на ядерные объекты
        for keyword in self.config.NUCLEAR_KEYWORDS:
            if keyword in name:
                return self.config.NUCLEAR_PLANT_RADIUS
        
        # Проверка на гидроэлектростанции
        for keyword in self.config.HYDRO_KEYWORDS:
            if keyword in name:
                return self.config.HYDRO_PLANT_RADIUS
        
        # Значение по умолчанию
        return self.config.DEFAULT_RADIUS

    def _analyze_infrastructure(self) -> None:
        """
        Анализ инфраструктуры для каждой территории
        """
        # Для каждой территории оценки проверяем наличие уникальных типов объектов в радиусе
        for index, area in self.assessment_areas_gdf.iterrows():
            unique_types_in_radius: Set[str] = set()

            # Проверяем каждый объект инфраструктуры на попадание в буфер
            for _, obj in self.infrastructure_gdf.iterrows():
                # Извлекаем информацию о типе объекта для определения радиуса
                physical_object_info: Dict[str, Any] = obj['physical_object_type']
                buffer_distance: float = self.get_radius(physical_object_info)
                
                # Создаем буфер для текущей территории оценки
                area_buffer: BaseGeometry = area.geometry.buffer(buffer_distance)
                
                # Если объект попадает в буфер, добавляем его тип в множество
                if obj.geometry.within(area_buffer) or obj.geometry.intersects(area_buffer):
                    unique_types_in_radius.add(obj['type'])

            # Подсчитываем уникальные типы и записываем результаты
            self.assessment_areas_gdf.at[index, 'score'] = len(unique_types_in_radius)
            self.assessment_areas_gdf.at[index, 'types_in_radius'] = list(unique_types_in_radius)

    def get_results(self) -> gpd.GeoDataFrame:
        """
        Возвращение результатов анализа

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame с результатами в CRS 4326
        """
        # Приведение обратно к CRS 4326 перед возвратом
        result_columns = ['score', 'types_in_radius', 'geometry']
        return self.assessment_areas_gdf[result_columns].to_crs(epsg=4326)

    def get_detailed_results(self) -> gpd.GeoDataFrame:
        """
        Получение детальных результатов с дополнительной статистикой
        
        Returns
        -------
        gpd.GeoDataFrame
            Детальные результаты анализа
        """
        results = self.get_results()
        
        # Добавляем дополнительную статистику
        results['infrastructure_density'] = results['score'] / results.geometry.area * 1000000  # на кв.км
        results['has_nuclear'] = results['types_in_radius'].apply(
            lambda types: any(keyword in str(types) for keyword in self.config.NUCLEAR_KEYWORDS)
        )
        results['has_hydro'] = results['types_in_radius'].apply(
            lambda types: any(keyword in str(types) for keyword in self.config.HYDRO_KEYWORDS)
        )
        
        return results
