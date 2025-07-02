from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import plotly.express as px

app = FastAPI()

templates = Jinja2Templates(directory="templates")

try:
    data = pd.read_csv("dataset.csv", encoding="windows-1252")
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    data = pd.DataFrame()


def generate_map():
    if data.empty:
        return "<p>Нет данных для отображения</p>"

    fig = px.scatter_map(data_frame=data, lat="Latitude", lon="Longitude", color="Collision Type", center={"lat": 39, "lon":-86.5}, hover_data=["Injury Type"], height=700, width=1200)
    fig.update_layout(mapbox_style="open-street-map")
    return fig.to_html(full_html=False)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    map_html = generate_map()
    return templates.TemplateResponse("index.html", {"request": request, "map": map_html})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)