"""
Agglomeration building methods for PopFrame
"""

from typing import Optional, Set
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from .base_method import BaseMethod
from popframe.preprocessing.level_filler import LevelFiller
from popframe.config.constants import AgglomerationConfig
from popframe.utils.validators import DataValidator


class AgglomerationBuilder(BaseMethod):
    """
    Класс для построения агломераций на основе временной доступности
    
    Attributes
    ----------
    config : AgglomerationConfig
        Конфигурация параметров агломерации
    """

    def __init__(self, region, config: Optional[AgglomerationConfig] = None):
        """
        Инициализация построителя агломераций
        
        Parameters
        ----------
        region : Region
            Регион для анализа
        config : AgglomerationConfig, optional
            Конфигурация параметров
        """
        super().__init__(region=region)
        self.config = config or AgglomerationConfig()
        self._agglomeration_tracker: Set[int] = set()

    def validate_input(self, **kwargs) -> bool:
        """
        Валидация входных параметров для построения агломераций
        
        Parameters
        ----------
        **kwargs
            Параметры включают time, update_df
            
        Returns
        -------
        bool
            True если валидация прошла успешно
            
        Raises
        ------
        ValueError
            При некорректных входных данных
        """
        time = kwargs.get('time', 80)
        update_df = kwargs.get('update_df')
        
        # Валидация времени
        DataValidator.validate_time_parameter(time)
        
        if time < self.config.MIN_TIME_THRESHOLD:
            raise ValueError(f"Минимально допустимое значение времени: {self.config.MIN_TIME_THRESHOLD} минут")
            
        if time > self.config.MAX_TIME_THRESHOLD:
            raise ValueError(f"Максимально допустимое значение времени: {self.config.MAX_TIME_THRESHOLD} минут")
        
        # Валидация update_df если присутствует
        if update_df is not None:
            if not isinstance(update_df, pd.DataFrame):
                raise ValueError("update_df должен быть DataFrame")
            if 'population' not in update_df.columns:
                raise ValueError("update_df должен содержать колонку 'population'")
                
        return True

    def run(self, time: int = 80, update_df: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """
        Основной метод построения агломераций
        
        Parameters
        ----------
        time : int, default 80
            Максимальное время доступности в минутах
        update_df : pd.DataFrame, optional
            DataFrame с обновлениями данных о населении
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame с агломерациями
        """
        # Валидация входных данных
        self.validate_input(time=time, update_df=update_df)
        
        # Подготовка данных
        towns = self._prepare_towns_data(update_df)
        
        # Построение агломераций
        agglomerations = self._build_agglomerations(towns, time)
        
        # Финализация результатов
        return self._finalize_agglomerations(agglomerations, towns)

    def _prepare_towns_data(self, update_df: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """
        Подготовка данных о городах
        
        Parameters
        ----------
        update_df : pd.DataFrame, optional
            DataFrame с обновлениями
            
        Returns
        -------
        gpd.GeoDataFrame
            Подготовленные данные о городах
            
        Raises
        ------
        ValueError
            При недостаточном количестве данных
        """
        towns = self.region.get_update_towns_gdf(update_df)
        
        if towns is None or len(towns) < 2:
            raise ValueError("Для построения агломерации требуется минимум два города.")
        
        # Проверяем, что хотя бы у двух городов есть ненулевое население
        valid_pop = towns['population'].notnull() & (towns['population'] > 0)
        if valid_pop.sum() < 2:
            raise ValueError("Требуются данные о населении минимум у двух разных городов.")
            
        return towns

    def _build_agglomerations(self, towns: gpd.GeoDataFrame, time: int) -> list:
        """
        Построение агломераций для городов
        
        Parameters
        ----------
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
        time : int
            Максимальное время доступности
            
        Returns
        -------
        list
            Список словарей с данными агломераций
        """
        self._agglomeration_tracker.clear()
        node_population = towns.set_index('id')['population']
        node_names = towns.set_index('id')['name']
        agglomerations = []

        # Обрабатываем города по уровням (от крупных к мелким)
        for level_index, level in enumerate(reversed(self.config.CITY_LEVELS)):
            max_time = time - self.config.LEVEL_TIME_REDUCTION * level_index
            level_nodes = towns[towns['level'] == level].sort_values(by='population', ascending=False)

            for node, population in level_nodes[['id', 'population']].itertuples(index=False):
                if node in self._agglomeration_tracker or population < self.config.MIN_POPULATION:
                    continue

                agglomeration = self._get_agglomeration_around_node(node, max_time, towns)

                if agglomeration:
                    agglomeration["name"] = node_names.get(node)
                    agglomerations.append(agglomeration)

        return agglomerations

    def _get_agglomeration_around_node(
        self, 
        central_node: int, 
        max_time: int, 
        towns: gpd.GeoDataFrame
    ) -> Optional[dict]:
        """
        Построение агломерации вокруг центрального узла
        
        Parameters
        ----------
        central_node : int
            ID центрального города
        max_time : int
            Максимальное время доступности
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
            
        Returns
        -------
        dict or None
            Данные агломерации или None если не удалось построить
        """
        # Получаем города в пределах временной доступности
        accessible_nodes = self._get_accessible_nodes(central_node, max_time)
        
        if len(accessible_nodes) < 2:  # Нужен хотя бы центральный город + 1
            return None
            
        # Создаем геометрию агломерации
        agglomeration_geometry = self._create_agglomeration_geometry(accessible_nodes, towns)
        
        if agglomeration_geometry is None:
            return None
            
        # Отмечаем города как включенные в агломерацию
        self._agglomeration_tracker.update(accessible_nodes)
        
        # Вычисляем характеристики агломерации
        return self._calculate_agglomeration_metrics(accessible_nodes, towns, agglomeration_geometry)

    def _get_accessible_nodes(self, central_node: int, max_time: int) -> list:
        """
        Получение списка доступных узлов в пределах времени
        
        Parameters
        ----------
        central_node : int
            ID центрального узла
        max_time : int
            Максимальное время доступности
            
        Returns
        -------
        list
            Список ID доступных узлов
        """
        if central_node not in self.region.accessibility_matrix.index:
            return [central_node]
            
        accessibility_row = self.region.accessibility_matrix.loc[central_node]
        accessible_mask = accessibility_row <= max_time
        
        return accessibility_row[accessible_mask].index.tolist()

    def _create_agglomeration_geometry(
        self, 
        node_ids: list, 
        towns: gpd.GeoDataFrame
    ) -> Optional[Polygon]:
        """
        Создание геометрии агломерации из точек городов
        
        Parameters
        ----------
        node_ids : list
            Список ID городов в агломерации
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
            
        Returns
        -------
        Polygon or None
            Геометрия агломерации
        """
        try:
            # Получаем точки городов
            town_points = towns[towns['id'].isin(node_ids)]['geometry'].tolist()
            
            if len(town_points) < 2:
                return None
                
            # Создаем буферы вокруг точек
            buffered_points = [point.buffer(self.config.DEFAULT_RADIUS) for point in town_points]
            
            # Объединяем буферы
            agglomeration_geom = unary_union(buffered_points)
            
            if isinstance(agglomeration_geom, MultiPolygon):
                # Берем самый большой полигон
                agglomeration_geom = max(agglomeration_geom.geoms, key=lambda x: x.area)
                
            return agglomeration_geom
            
        except Exception:
            return None

    def _calculate_agglomeration_metrics(
        self, 
        node_ids: list, 
        towns: gpd.GeoDataFrame, 
        geometry: Polygon
    ) -> dict:
        """
        Расчет метрик агломерации
        
        Parameters
        ----------
        node_ids : list
            Список ID городов
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
        geometry : Polygon
            Геометрия агломерации
            
        Returns
        -------
        dict
            Словарь с метриками агломерации
        """
        agglomeration_towns = towns[towns['id'].isin(node_ids)]
        
        total_population = agglomeration_towns['population'].sum()
        core_cities = agglomeration_towns.nlargest(3, 'population')['name'].tolist()
        
        # Определение уровня агломерации по населению
        agglomeration_level = self._determine_agglomeration_level(total_population)
        
        return {
            'geometry': geometry,
            'population': total_population,
            'core_cities': ', '.join(core_cities),
            'cities_count': len(node_ids),
            'agglomeration_level': agglomeration_level,
            'city_ids': node_ids
        }

    def _determine_agglomeration_level(self, population: int) -> int:
        """
        Определение уровня агломерации по населению
        
        Parameters
        ----------
        population : int
            Общая численность населения
            
        Returns
        -------
        int
            Уровень агломерации (1-5)
        """
        if population >= 3000000:
            return 5  # Мегаполис
        elif population >= 1000000:
            return 4  # Крупнейшая агломерация
        elif population >= 500000:
            return 3  # Крупная агломерация
        elif population >= 250000:
            return 2  # Средняя агломерация
        else:
            return 1  # Малая агломерация

    def _finalize_agglomerations(
        self, 
        agglomerations: list, 
        towns: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Финализация и постобработка агломераций
        
        Parameters
        ----------
        agglomerations : list
            Список данных агломераций
        towns : gpd.GeoDataFrame
            GeoDataFrame с городами
            
        Returns
        -------
        gpd.GeoDataFrame
            Финальный GeoDataFrame с агломерациями
        """
        if not agglomerations:
            # Возвращаем пустой GeoDataFrame с нужными колонками
            return gpd.GeoDataFrame(
                columns=['geometry', 'name', 'population', 'core_cities', 
                        'cities_count', 'agglomeration_level'],
                crs=self.region.crs
            )
            
        # Создаем GeoDataFrame
        agglomeration_gdf = gpd.GeoDataFrame(agglomerations, crs=self.region.crs)
        
        # Обрезаем по границам региона
        agglomeration_gdf = gpd.overlay(
            agglomeration_gdf, 
            self.region.region, 
            how='intersection'
        )
        
        # Упрощаем геометрии
        agglomeration_gdf = self._simplify_geometries(agglomeration_gdf)
        
        return agglomeration_gdf

    def _simplify_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Упрощение и очистка геометрий
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame с агломерациями
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame с упрощенными геометриями
        """
        # Убираем слишком маленькие геометрии
        gdf = gdf[gdf.geometry.area > 1000]  # 1000 кв. метров
        
        # Упрощаем сложные геометрии
        gdf['geometry'] = gdf.geometry.simplify(100)  # 100 метров tolerance
        
        # Удаляем пустые геометрии
        gdf = gdf[~gdf.geometry.is_empty]
        
        return gdf.reset_index(drop=True)

    def get_agglomerations(
        self, 
        update_df: Optional[pd.DataFrame] = None, 
        time: int = 80
    ) -> gpd.GeoDataFrame:
        """
        Главный метод получения агломераций (для обратной совместимости)
        
        Parameters
        ----------
        update_df : pd.DataFrame, optional
            DataFrame с обновлениями населения
        time : int, default 80
            Максимальное время доступности
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame с агломерациями
        """
        return self.run(time=time, update_df=update_df)