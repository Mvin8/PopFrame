PopFrame
========

.. logo-start

.. figure:: https://i.ibb.co/8bD3qr0/DALL-E-2024-05-22-16-49.png
   :alt: PopFrame logo

.. logo-end

|PythonVersion| |Black|

.. description-start

**EN:**
PopFrame is an open source library for modeling regional settlement systems and territory assessment. It provides tools for building a universal information model of a region based on localities, as well as for scenario modeling and analysis.

**RU:**
PopFrame — это open source библиотека для моделирования каркаса расселения и оценки территорий. Она предоставляет инструменты для построения универсальной информационной модели региона на основе населённых пунктов, а также для сценарного моделирования и анализа.

.. description-end

What's New in v0.1.0
--------------------

🔧 **Complete Architecture Refactoring**

- **Unified base classes**: All analysis methods now inherit from `BaseMethod` or `BaseAnalyzer`
- **Configuration system**: Centralized configuration with `AgglomerationConfig`, `PopulationThresholds`, etc.
- **Enhanced validation**: Comprehensive input validation with detailed error messages
- **Improved API**: Direct imports from main package - `from popframe import AgglomerationBuilder`
- **Flexible dependencies**: Version ranges instead of pinned versions
- **Better error handling**: More informative error messages and proper exception handling

Table of Contents / Содержание
-----------------------------

- `Core features <#core-features>`_
- `Installation <#installation>`_
- `Quick Start <#quick-start>`_
- `Examples <#examples>`_
- `Configuration <#configuration>`_
- `Project Structure <#project-structure>`_
- `Documentation <#documentation>`_
- `Developing <#developing>`_
- `License <#license>`_
- `Acknowledgments <#acknowledgments>`_
- `Contacts <#contacts>`_

Core features / Основные возможности
-----------------------------------

.. features-start

**EN:**
- Calculate indicators (population, birth rate, mortality) by municipal districts and municipalities.
- Evaluate territories in relation to the settlement system framework.
- Calculate the level of urbanization of the territory.
- Build agglomerations based on the settlement system framework.
- Analyze infrastructure coverage and accessibility.
- Assess spatial inequality and territorial development.

**RU:**
- Расчёт показателей (население, рождаемость, смертность) по муниципальным районам и муниципалитетам.
- Оценка территорий относительно каркаса расселения.
- Расчёт уровня урбанизации территории.
- Построение агломераций на основе каркаса расселения.
- Анализ покрытия и доступности инфраструктуры.
- Оценка пространственного неравенства и территориального развития.

.. features-end

Installation / Установка
-----------------------

.. installation-start

**EN:**
PopFrame can be installed with ``pip``:

::
   pip install popframe

For development with additional dependencies:

::
   pip install popframe[dev]

For documentation building:

::
   pip install popframe[docs]

**RU:**
Установить PopFrame можно через ``pip``:

::
   pip install popframe

Для разработки с дополнительными зависимостями:

::
   pip install popframe[dev]

Для сборки документации:

::
   pip install popframe[docs]

.. installation-end

Quick Start / Быстрый старт
---------------------------

**EN:**
Here's a quick example of building agglomerations:

**RU:**
Вот быстрый пример построения агломераций:

```python
import popframe as pf

# Load your region data
region = pf.Region.from_pickle('data/region.pickle')

# Create agglomeration builder with custom configuration
config = pf.AgglomerationConfig(
    MIN_POPULATION=20000,
    DEFAULT_RADIUS=500
)
builder = pf.AgglomerationBuilder(region, config=config)

# Build agglomerations
agglomerations = builder.run(time=90)

# Visualize results
agglomerations.plot()
```

Examples / Примеры
------------------

**EN:**
Below are examples for the main modules of PopFrame with the new architecture.

**RU:**
Ниже приведены примеры для основных модулей PopFrame с новой архитектурой.

**1. Region Model with Enhanced Validation:**
```python
import popframe as pf

# Load region with automatic validation
region = pf.Region.from_pickle('data/region.pickle')

# Access towns and accessibility
print(f"Total towns: {len(region.towns)}")
travel_time = region[town_id_1, town_id_2]
```

**2. Agglomeration Building with Configuration:**
```python
# Custom configuration
config = pf.AgglomerationConfig(
    MIN_POPULATION=15000,
    DEFAULT_RADIUS=400,
    MIN_TIME_THRESHOLD=60,
    MAX_TIME_THRESHOLD=120
)

builder = pf.AgglomerationBuilder(region=region, config=config)

# Run analysis with validation
try:
    agglos = builder.run(time=80)
    print(f"Built {len(agglos)} agglomerations")
except ValueError as e:
    print(f"Validation error: {e}")
```

**3. Infrastructure Analysis with New Architecture:**
```python
# Enhanced infrastructure analyzer
config = pf.InfrastructureConfig(
    DEFAULT_RADIUS=1000.0,
    NUCLEAR_PLANT_RADIUS=50000.0
)

analyzer = pf.InfrastructureAnalyzer(
    infrastructure_gdf=infrastructure_data,
    assessment_areas_gdf=assessment_areas,
    config=config
)

# Run analysis
results = analyzer.analyze()
detailed_results = analyzer.get_detailed_results()
```

**4. Population Level Classification:**
```python
# Custom population thresholds
thresholds = pf.PopulationThresholds()
filler = pf.LevelFiller(towns=gdf_towns, population_thresholds=thresholds)
leveled_towns = filler.fill_levels()
```

**5. Data Validation:**
```python
# Use built-in validators
try:
    pf.RegionValidator.validate_accessibility_matrix(matrix, towns)
    pf.DataValidator.validate_population_data(population_series)
    pf.DataValidator.validate_time_parameter(time_minutes)
except ValueError as e:
    print(f"Validation failed: {e}")
```

Configuration / Конфигурация
----------------------------

**EN:**
PopFrame now supports comprehensive configuration for all analysis methods:

**RU:**
PopFrame теперь поддерживает комплексную конфигурацию для всех методов анализа:

```python
import popframe as pf

# Agglomeration configuration
agglomeration_config = pf.AgglomerationConfig(
    DEFAULT_RADIUS=400,
    MIN_POPULATION=15000,
    LEVEL_TIME_REDUCTION=10,
    MIN_TIME_THRESHOLD=50,
    MAX_TIME_THRESHOLD=120,
    CITY_LEVELS=[
        "Малый город",
        "Средний город", 
        "Большой город",
        "Крупный город",
        "Крупнейший город",
        "Сверхкрупный город"
    ]
)

# Population thresholds
population_config = pf.PopulationThresholds()
print(population_config.THRESHOLDS)

# Infrastructure analysis configuration  
infra_config = pf.InfrastructureConfig(
    DEFAULT_RADIUS=1000.0,
    NUCLEAR_PLANT_RADIUS=100000.0,
    HYDRO_PLANT_RADIUS=10000.0
)
```

Project Structure / Структура проекта
------------------------------------

**EN:**
- ``popframe`` — library code:
  - ``config`` — configuration classes and constants
  - ``preprocessing`` — data preprocessing
  - ``models`` — core data models
  - ``method`` — analytical methods
  - ``utils`` — utility functions and validators
- ``examples`` — usage examples
- ``docs`` — documentation sources
- ``tests`` — comprehensive test suite

**RU:**
- ``popframe`` — код библиотеки:
  - ``config`` — классы конфигурации и константы
  - ``preprocessing`` — предобработка данных
  - ``models`` — основные модели данных
  - ``method`` — аналитические методы
  - ``utils`` — утилиты и валидаторы
- ``examples`` — примеры использования
- ``docs`` — исходники документации
- ``tests`` — комплексный набор тестов

API Changes / Изменения в API
-----------------------------

**EN:**
Major improvements in v0.1.0:

**RU:**
Основные улучшения в v0.1.0:

- **Simplified imports**: `from popframe import AgglomerationBuilder` instead of `from popframe.method.agglomeration import AgglomerationBuilder`
- **Configuration objects**: Replace magic numbers with structured configuration
- **Enhanced validation**: Better error messages and input validation
- **Consistent method interface**: All analysis methods follow the same pattern with `run()` method
- **Backward compatibility**: Old methods still work but show deprecation warnings

Migration Guide / Руководство по миграции
-----------------------------------------

**EN:**
To migrate from older versions:

**RU:**
Для миграции со старых версий:

```python
# Old way (still works with warnings)
from popframe.method.agglomeration import AgglomerationBuilder
builder = AgglomerationBuilder(region)
result = builder.get_agglomerations(time=80)

# New way (recommended)
import popframe as pf
config = pf.AgglomerationConfig()
builder = pf.AgglomerationBuilder(region, config=config)
result = builder.run(time=80)
```

Documentation / Документация
----------------------------

**EN:**
Full documentation is available at: https://mvin8.github.io/PopFrame/

**RU:**
Полная документация: https://mvin8.github.io/PopFrame/

Developing / Разработка
----------------------

.. developing-start

**EN:**
To start developing PopFrame:
1. Clone the repository:
   ::
       $ git clone https://github.com/Mvin8/PopFrame
2. (Optional) Create a virtual environment:
   ::
       $ make venv
       $ source .venv/bin/activate
3. Install the library in editable mode with development dependencies:
   ::
       $ make install-dev
4. Install pre-commit hooks:
   ::
       $ pre-commit install
5. Create a new branch:
   ::
       $ git checkout -b develop <new_branch_name>
6. Make changes, update tests and documentation, and run:
   ::
       $ make test
7. Commit and push your changes, then open a Pull Request.

**RU:**
Для начала разработки PopFrame:
1. Клонируйте репозиторий:
   ::
       $ git clone https://github.com/Mvin8/PopFrame
2. (Опционально) создайте виртуальное окружение:
   ::
       $ make venv
       $ source .venv/bin/activate
3. Установите библиотеку в editable-режиме с dev-зависимостями:
   ::
       $ make install-dev
4. Установите pre-commit хуки:
   ::
       $ pre-commit install
5. Создайте новую ветку:
   ::
       $ git checkout -b develop <new_branch_name>
6. Вносите изменения, обновляйте тесты и документацию, запускайте:
   ::
       $ make test
7. Зафиксируйте и отправьте изменения, откройте Pull Request.

.. developing-end

Testing / Тестирование
---------------------

**EN:**
Run tests with pytest:

**RU:**
Запуск тестов через pytest:

::
    # Run all tests
    pytest
    
    # Run with coverage
    pytest --cov=popframe
    
    # Run specific test file
    pytest tests/test_methods/test_agglomeration.py

License / Лицензия
------------------

BSD-3-Clause license. See LICENSE file.

Acknowledgments / Благодарности
-------------------------------

.. acknowledgments-start

**EN:**
PopFrame was developed as part of the ITMO University project by the Institute of Design and Urban Studies (IDU).

**RU:**
PopFrame разработан в рамках проекта Университета ИТМО Институтом дизайна и урбанистики (IDU).

.. acknowledgments-end

Contacts / Контакты
-------------------

.. contacts-start

- `NCCR <https://actcognitive.org/o-tsentre/kontakty>`__ — National Center for Cognitive Research
- `IDU <https://idu.itmo.ru/en/contacts/contacts.htm>`__ — Institute of Design and Urban Studies
- `Maksim Natykin <https://t.me/Mvin98>`__ — lead software engineer

.. contacts-end





