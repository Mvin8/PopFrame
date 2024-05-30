from fastapi import FastAPI, HTTPException, File, UploadFile
from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region
from pydantic import BaseModel
import pickle

app = FastAPI()

class EvaluationRequest(BaseModel):
    region: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to PopFrame Service"}

@app.post("/evaluate_territory_location")
async def evaluate_territory_location_endpoint(file: UploadFile = File(...)):
    try:
        # Загружаем файл и десериализуем его
        contents = await file.read()
        region_model = pickle.loads(contents)
        
        # Проверка типа данных
        if not isinstance(region_model, Region):
            raise HTTPException(status_code=400, detail="Invalid region file")

        # Создаем экземпляр TerritoryEvaluation и оцениваем территорию
        evaluation = TerritoryEvaluation(region=region_model)
        result = evaluation.evaluate_territory_location()
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
