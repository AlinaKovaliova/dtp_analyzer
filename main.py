from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

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


def generate_cluster_map():
    if data.empty:
        return "<p>Нет данных для кластеризации</p>"

    try:
        # Работаем с копией данных
        cluster_data = data.copy()

        # Подготовка данных для кластеризации
        le = LabelEncoder()
        cluster_data['Primary Factor'] = le.fit_transform(cluster_data['Primary Factor'])
        cluster_data['Weekend?'] = cluster_data['Weekend?'].map({'Weekday': 0, 'Weekend': 1})
        cluster_data['Collision Type'] = le.fit_transform(cluster_data['Collision Type'])

        injury_ohe = pd.get_dummies(cluster_data['Injury Type'], prefix='Injury', dtype=int)
        cluster_data = pd.concat([cluster_data, injury_ohe], axis=1)
        cluster_data.drop(['Injury Type', 'Reported_Location'], axis=1, inplace=True)

        cluster_data = cluster_data.fillna(-1)
        cluster_data = cluster_data[
            (cluster_data['Latitude'].between(35, 45)) &
            (cluster_data['Longitude'].between(-90, -80))
            ]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=10, random_state=42))
        ])

        # Создаем кластеры
        cluster_data['Cluster'] = pipe.fit_predict(cluster_data[['Latitude', 'Longitude']])

        # Сохраняем кластеры в основной DataFrame
        data['Cluster'] = cluster_data['Cluster']

        # Визуализация
        fig = px.scatter_map(
            data_frame=cluster_data,
            lat="Latitude",
            lon="Longitude",
            color="Cluster",
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


def generate_cluster_distribution():
    if 'Cluster' not in data.columns:
        return "<p>Кластеры не были созданы</p>"

    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'count']

    fig = px.pie(
        cluster_counts,
        values='count',
        names='Cluster',
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


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    map_html = generate_map()
    time_plots = generate_time_plots()
    cluster_map_html = generate_cluster_map()
    cluster_dist_html = generate_cluster_distribution()

    has_clusters = 'Cluster' in data.columns

    context = {
        "request": request,
        "map": map_html,
        "cluster_map": cluster_map_html,
        "cluster_dist": cluster_dist_html,
        "has_clusters": has_clusters,  # Добавляем флаг
        "year_plot": time_plots.get('Year', ''),
        "month_plot": time_plots.get('Month', ''),
        "day_plot": time_plots.get('Day', ''),
        "hour_plot": time_plots.get('Hour', ''),
        "primary_factors": generate_primary_factors(),
        "weekend_pie": generate_weekend_pie(),
        "injury_pie": generate_injury_pie()
    }

    return templates.TemplateResponse("index.html", context)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


