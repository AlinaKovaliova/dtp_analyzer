from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import List

app = FastAPI()

templates = Jinja2Templates(directory="templates")

try:
    data = pd.read_csv("results.csv", encoding="windows-1252")
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    data = pd.DataFrame()


def generate_map():
    if data.empty:
        return "<p>Нет данных для отображения</p>"

    fig = px.scatter_map(data_frame=data, lat="Latitude", lon="Longitude", color="Collision Type", center={"lat": 39, "lon":-86.5}, hover_data=["Injury Type"], height=700, width=1200)
    fig.update_layout(mapbox_style="open-street-map")
    return fig.to_html(full_html=False)

def generate_time_plots():
    datetime_list = ['Year', 'Month', 'Day', 'Hour']
    plots = {}

    for time_period in datetime_list:
        if data.empty:
            plots[time_period] = "<p>Нет данных для отображения</p>"
            continue

        # Группировка данных
        grouped = data[time_period].value_counts().reset_index()
        grouped.columns = [time_period, 'count']

        # Определение цветов
        max_count = grouped['count'].max()
        grouped['color'] = grouped['count'].apply(lambda x: '#FFA500' if x == max_count else '#008000')

        # Создание графика
        fig = px.bar(
            grouped,
            x=time_period,
            y='count',
            color='color',
            color_discrete_map={c: c for c in grouped['color'].unique()},
            labels={time_period: ""},
            title=f"Number of accidents by {time_period}s"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=60,
            height=400,
            width=700,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        plots[time_period] = fig.to_html(full_html=False)

    return plots


def generate_primary_factors():
    if data.empty or 'Primary Factor' not in data.columns:
        return "<p>Нет данных для отображения</p>"

    top_factors = data['Primary Factor'].value_counts().head(10).reset_index()
    top_factors.columns = ['Primary Factor', 'count']

    fig = px.bar(
        top_factors,
        y='Primary Factor',
        x='count',
        orientation='h',
        title='Топ-10 основных причин ДТП',
        labels={'Primary Factor': 'Основная причина', 'count': 'Количество'}
    )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)

def generate_weekend_pie():
    if data.empty or 'Weekend?' not in data.columns:
        return "<p>Нет данных для отображения</p>"

    weekend_counts = data['Weekend?'].value_counts().reset_index()
    weekend_counts.columns = ['Weekend', 'count']

    fig = px.pie(
        weekend_counts,
        values='count',
        names='Weekend',
        title='Распределение ДТП по дням недели',
        hole=0.3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)

def generate_injury_pie():
    if data.empty or 'Injury Type' not in data.columns:
        return "<p>Нет данных для отображения</p>"

    injury_counts = data['Injury Type'].value_counts().reset_index()
    injury_counts.columns = ['Injury Type', 'count']

    fig = px.pie(
        injury_counts,
        values='count',
        names='Injury Type',
        title='Типы травм в ДТП',
        hole=0.3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1)),
        showlegend=True
    )

    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.85),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)


def kmeans_cluster_map():
    if data.empty:
        return "<p>Нет данных для кластеризации</p>"

    try:
        cluster_data = data.copy()

        fig = px.scatter_map(
            data_frame=cluster_data,
            lat="Latitude",
            lon="Longitude",
            color="KMeans_clusters",
            center={"lat": 39, "lon": -86.5},
            height=700,
            width=1200,
            color_continuous_scale=px.colors.qualitative.Prism
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Кластер"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        return f"<p>Ошибка кластеризации: {str(e)}</p>"


def kmeans_cluster_distribution():
    if 'KMeans_clusters' not in data.columns:
        return "<p>Кластеры не были созданы</p>"

    cluster_counts = data['KMeans_clusters'].value_counts().reset_index()
    cluster_counts.columns = ['KMeans_clusters', 'count']

    fig = px.pie(
        cluster_counts,
        values='count',
        names='KMeans_clusters',
        title='Распределение ДТП по кластерам',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)

def agglo_cluster_map():
    if data.empty:
        return "<p>Нет данных для кластеризации</p>"

    try:
        cluster_data = data.copy()

        fig = px.scatter_map(
            data_frame=cluster_data,
            lat="Latitude",
            lon="Longitude",
            color="Agglo_clusters",
            center={"lat": 39, "lon": -86.5},
            height=700,
            width=1200,
            color_continuous_scale=px.colors.qualitative.Prism
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Кластер"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        return f"<p>Ошибка кластеризации: {str(e)}</p>"


def agglo_cluster_distribution():
    if 'Agglo_clusters' not in data.columns:
        return "<p>Кластеры не были созданы</p>"

    cluster_counts = data['Agglo_clusters'].value_counts().reset_index()
    cluster_counts.columns = ['Agglo_clusters', 'count']

    fig = px.pie(
        cluster_counts,
        values='count',
        names='Agglo_clusters',
        title='Распределение ДТП по кластерам',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)


def birch_cluster_map():
    if data.empty:
        return "<p>Нет данных для кластеризации</p>"

    try:
        cluster_data = data.copy()

        fig = px.scatter_map(
            data_frame=cluster_data,
            lat="Latitude",
            lon="Longitude",
            color="Birch_clusters",
            center={"lat": 39, "lon": -86.5},
            height=700,
            width=1200,
            color_continuous_scale=px.colors.qualitative.Prism
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Кластер"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        return f"<p>Ошибка кластеризации: {str(e)}</p>"


def birch_cluster_distribution():
    if 'Birch_clusters' not in data.columns:
        return "<p>Кластеры не были созданы</p>"

    cluster_counts = data['Birch_clusters'].value_counts().reset_index()
    cluster_counts.columns = ['Birch_clusters', 'count']

    fig = px.pie(
        cluster_counts,
        values='count',
        names='Birch_clusters',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)


def gauss_cluster_map():
    if data.empty:
        return "<p>Нет данных для кластеризации</p>"

    try:
        cluster_data = data.copy()

        fig = px.scatter_map(
            data_frame=cluster_data,
            lat="Latitude",
            lon="Longitude",
            color="Gauss_clusters",
            center={"lat": 39, "lon": -86.5},
            height=700,
            width=1200,
            color_continuous_scale=px.colors.qualitative.Prism
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Кластер"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        return f"<p>Ошибка кластеризации: {str(e)}</p>"


def gauss_cluster_distribution():
    if 'Gauss_clusters' not in data.columns:
        return "<p>Кластеры не были созданы</p>"

    cluster_counts = data['Gauss_clusters'].value_counts().reset_index()
    cluster_counts.columns = ['Gauss_clusters', 'count']

    fig = px.pie(
        cluster_counts,
        values='count',
        names='Gauss_clusters',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)



def cluster_factor_analysis(cluster_type: str, cluster_num: int):
    if cluster_type not in data.columns:
        return "<p>Данные кластеризации не найдены</p>"
    
    try:
        factors = data[data[cluster_type] == cluster_num]['Primary Factor'].value_counts().reset_index(name='count').head(20)
        
        fig = px.bar(
            factors,
            x='count',
            y='Primary Factor',
            orientation='h',
            title=f'Топ факторов ДТП для кластера {cluster_num}',
            color='count',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=100, r=20, t=40, b=20)
        )
        
        return fig.to_html(full_html=False)
    
    except Exception as e:
        return f"<p>Ошибка анализа факторов: {str(e)}</p>"


def cluster_injury_analysis(cluster_type: str, cluster_num: int):
    if cluster_type not in data.columns:
        return "<p>Данные кластеризации не найдены</p>"
    
    try:
        injuries = data[data[cluster_type] == cluster_num]['Injury Type'].value_counts().reset_index(name='count').head(10)
        
        fig = px.pie(
            injuries,
            values='count',
            names='Injury Type',
            title=f'Типы травм для кластера {cluster_num}',
            hole=0.3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#000000', width=1))
        )
        
        return fig.to_html(full_html=False)
    
    except Exception as e:
        return f"<p>Ошибка анализа травм: {str(e)}</p>"


def cluster_month_analysis(cluster_type: str, cluster_num: int, year: int):
    if cluster_type not in data.columns:
        return "<p>Данные кластеризации не найдены</p>"
    
    try:
        months_data = data[(data[cluster_type] == cluster_num) & (data['Year'] == year)]
        if months_data.empty:
            return "<p>Нет данных для выбранного года</p>"
            
        months = months_data['Month'].value_counts().reset_index(name='count')
        
        fig = px.bar(
            months,
            x='Month',
            y='count',
            title=f'ДТП по месяцам для кластера {cluster_num} в {year} году',
            color='count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        return fig.to_html(full_html=False)
    
    except Exception as e:
        return f"<p>Ошибка анализа по месяцам: {str(e)}</p>"



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    map_html = generate_map()
    time_plots = generate_time_plots()
    cluster_map_html = kmeans_cluster_map()
    cluster_dist_html = kmeans_cluster_distribution()
    agglo_map_html = agglo_cluster_map()
    agglo_dist_html = agglo_cluster_distribution()
    birch_map_html = birch_cluster_map()
    birch_dist_html = birch_cluster_distribution()
    gauss_map_html = gauss_cluster_map()
    gauss_dist_html = gauss_cluster_distribution()


    has_clusters = 'KMeans_clusters' in data.columns
    has_agglo = 'Agglo_clusters' in data.columns
    has_birch = 'Birch_clusters' in data.columns
    has_gauss = 'Gauss_clusters' in data.columns

    context = {
        "request": request,
        "map": map_html,
        "cluster_map": cluster_map_html,
        "cluster_dist": cluster_dist_html,
        "agglo_map": agglo_map_html,
        "agglo_dist": agglo_dist_html,
        "birch_map": birch_map_html,
        "birch_dist": birch_dist_html,
        "gauss_map": gauss_map_html,
        "gauss_dist": gauss_dist_html,
        "has_clusters": has_clusters,  # Добавляем флаг
        "has_agglo_clusters": has_agglo,
        "has_birch_clusters": has_birch,
        "has_gauss_clusters": has_gauss,
        "year_plot": time_plots.get('Year', ''),
        "month_plot": time_plots.get('Month', ''),
        "day_plot": time_plots.get('Day', ''),
        "hour_plot": time_plots.get('Hour', ''),
        "primary_factors": generate_primary_factors(),
        "weekend_pie": generate_weekend_pie(),
        "injury_pie": generate_injury_pie()
    }

    return templates.TemplateResponse("index.html", context)



@app.get("/cluster_analysis", response_class=HTMLResponse)
async def cluster_analysis(
    request: Request,
    cluster_type: str = "KMeans_clusters",
    cluster_num: int = 0,
    year: int = None
):
    # Устанавливаем текущий год по умолчанию
    if year is None:
        year = 2010
    
    # Получаем данные для выпадающих списков
    available_years = sorted(data['Year'].unique()) if 'Year' in data.columns else []
    
    # Генерируем аналитику для выбранного кластера
    factors_html = cluster_factor_analysis(cluster_type, cluster_num)
    injuries_html = cluster_injury_analysis(cluster_type, cluster_num)
    months_html = cluster_month_analysis(cluster_type, cluster_num, year)
    
    # Проверяем, какие методы кластеризации доступны
    available_methods = {
        "KMeans_clusters": "K-Means",
        "Agglo_clusters": "Agglomerative",
        "Birch_clusters": "BIRCH",
        "Gauss_clusters": "Gaussian Mixture"
    }
    
    # Оставляем только те методы, которые есть в данных
    available_methods = {k: v for k, v in available_methods.items() if k in data.columns}
    
    context = {
        "request": request,
        "cluster_factors": factors_html,
        "cluster_injuries": injuries_html,
        "cluster_months": months_html,
        "available_methods": available_methods,
        "available_years": available_years,
        "selected_method": cluster_type,
        "selected_cluster": cluster_num,
        "selected_year": year,
        "available_clusters": get_available_clusters(cluster_type)
    }
    
    return templates.TemplateResponse("cluster_analysis.html", context)

def get_available_clusters(cluster_type: str) -> List[int]:
    """Возвращает список доступных номеров кластеров для выбранного метода"""
    if cluster_type not in data.columns:
        return []
    return sorted(data[cluster_type].unique())



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


