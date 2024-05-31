from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region
import geopandas as gpd
from typing import Any, Dict, List, Optional, Tuple

app = FastAPI()

class GeoJSONRequest(BaseModel):
    type: str
    name: str
    crs: dict
    features: list

class CriteriaRequest(BaseModel):
    population: int
    transport: int
    ecology: int
    social_objects: int
    engineering_infrastructure: int
    culture_leisure_sport: int

class EvaluateTerritoryLocationResult(BaseModel):
    territory: str
    score: int
    interpretation: str
    closest_settlement: str
    closest_settlement1: Optional[str]
    closest_settlement2: Optional[str]

class EvaluateTerritoryLocationResponse(BaseModel):
    result: List[EvaluateTerritoryLocationResult]

class PopulationCriterionResult(BaseModel):
    project: str
    average_population_density: float
    average_population_growth: float
    score: int

class PopulationCriterionResponse(BaseModel):
    result: List[PopulationCriterionResult]

class CalculatePotentialResult(BaseModel):
    category: str
    scores: List[int]

class CalculatePotentialResponse(BaseModel):
    result: List[CalculatePotentialResult]

region_model = Region.from_pickle('/Users/mvin/Code/PopFrame/examples/data/model_data/region.pickle')
gdf = gpd.read_parquet('/Users/mvin/Code/PopFrame/examples/data/model_data/mo_desteny_growth.parquet')

if not isinstance(region_model, Region):
    raise Exception("Invalid region model")

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "Welcome to PopFrame Service"}

@app.post("/evaluate_territory_location", response_model=EvaluateTerritoryLocationResponse)
async def evaluate_territory_location_endpoint(request: GeoJSONRequest):
    try:
        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.evaluate_territory_location(territories=request.dict())
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/population_criterion", response_model=PopulationCriterionResponse)
async def population_criterion_endpoint(request: GeoJSONRequest):
    try:
        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.population_criterion(gdf, territories=request.dict())
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/calculate_potential", response_model=CalculatePotentialResponse)
async def calculate_potential_endpoint(request: CriteriaRequest):
    try:
        territories_criteria = {
            "Население": request.population,
            "Транспорт": request.transport,
            "Экология": request.ecology,
            "Соц-об": request.social_objects,
            "Инж инф": request.engineering_infrastructure,
            "Кул-дос-сп": request.culture_leisure_sport
        }

        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.calculate_potential(territories_criteria)
        
        formatted_result = [
            {"category": category, "scores": scores}
            for category, scores in result
        ]
        
        return {"result": formatted_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)