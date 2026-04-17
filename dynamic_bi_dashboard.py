from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="BI Dashboard | Sales Analytics", layout="wide")

# Подключаем иконки для интерфейса.
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# Функции подготовки данных. Кэш нужен, чтобы не читать и не обрабатывать файл заново при каждом изменении фильтров.
@st.cache_data
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Поддерживаются только CSV и Excel файлы.")

@st.cache_data
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    return df

@st.cache_data
def optimize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.strip()
            cleaned = cleaned.str.replace(r'[$€£]', '', regex=True)
            cleaned = cleaned.str.replace(r'%$', '', regex=True)
            cleaned = cleaned.str.replace(',', '.', regex=False)
            cleaned = cleaned.str.replace(r'\(([\d\.]+)\)', r'-\1', regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() > 0.85:
                df[col] = numeric
    return df

@st.cache_data
def try_parse_dates(df: pd.DataFrame, dayfirst: bool = False) -> pd.DataFrame:
    df = df.copy()
    date_keywords = ["date", "time", "day", "month", "year", "created", "updated", "period", "order_date"]
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if any(kw in col.lower() for kw in date_keywords):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
            if parsed.notna().mean() > 0.5:
                df[col] = parsed
                continue
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
    return df

@st.cache_data
def infer_schema(df: pd.DataFrame) -> dict:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = []
    high_cardinality_cols = []
    id_like_cols = []

    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if col in date_cols or col in numeric_cols:
            if ("id" in col.lower() or col.lower().endswith("_id")) and nunique >= len(df) * 0.8:
                id_like_cols.append(col)
            continue
        if nunique <= 30:
            categorical_cols.append(col)
        else:
            high_cardinality_cols.append(col)
        if "id" in col.lower() or col.lower().endswith("_id"):
            id_like_cols.append(col)

    metric_candidates = [c for c in numeric_cols if c not in id_like_cols]
    preferred_metric_order = ["revenue", "sales", "amount", "profit", "income", "price", "cost", "quantity", "count"]
    def metric_priority(col_name: str) -> tuple:
        lower = col_name.lower()
        matches = next((i for i, key in enumerate(preferred_metric_order) if key in lower), 999)
        return (matches, df[col_name].isna().mean(), -df[col_name].nunique(dropna=True))
    metric_candidates = sorted(metric_candidates, key=metric_priority)

    return {
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "metric_cols": metric_candidates,
        "category_cols": categorical_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "id_cols": id_like_cols,
    }

def detect_target_metric(metric_cols: list[str]) -> str | None:
    if not metric_cols:
        return None
    target_keywords = ["revenue", "sales", "amount", "profit", "income"]
    for kw in target_keywords:
        for col in metric_cols:
            if kw in col.lower():
                return col
    return metric_cols[0]

# Левая панель: загрузка файла и базовые настройки.
with st.sidebar:
    st.header("Источник данных")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Загрузите файл, чтобы начать анализ.")
        st.stop()
    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

    st.divider()
    dayfirst = st.checkbox("Формат даты: день первый (DD/MM/YYYY)", value=False)

# Приводим таблицу к рабочему виду: выравниваем имена колонок, пробуем распознать числа и даты.
raw_shape = df.shape
df = normalize_columns(df)
df = optimize_numeric_strings(df)
df = try_parse_dates(df, dayfirst=dayfirst)
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

if df.empty:
    st.error("После очистки датасет пуст.")
    st.stop()

schema = infer_schema(df)
metric_cols = schema["metric_cols"]
main_metric = detect_target_metric(metric_cols)

# Применяем фильтры из боковой панели.
with st.sidebar:
    st.header("Фильтры")
    filtered_df = df.copy()

    if schema["date_cols"]:
        date_col = st.selectbox("Колонка даты", schema["date_cols"])
        min_date = filtered_df[date_col].min()
        max_date = filtered_df[date_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input("Период", value=(min_date.date(), max_date.date()))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                filtered_df = filtered_df[filtered_df[date_col].between(start_date, end_date)]
    else:
        date_col = None

    for col in schema["category_cols"][:5]:
        unique_vals = filtered_df[col].dropna().unique()
        if len(unique_vals) > 100:
            top_vals = filtered_df[col].value_counts().head(100).index.tolist()
            selected = st.multiselect(col, sorted(top_vals), default=top_vals)
        else:
            selected = st.multiselect(col, sorted(unique_vals), default=sorted(unique_vals))
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

if filtered_df.empty:
    st.warning("После применения фильтров не осталось данных.")
    st.stop()

# Если данных слишком много, для графиков берем выборку. Это ускоряет интерфейс и не влияет на таблицу с результатом.
MAX_ROWS_FOR_PLOTS = 50000
if len(filtered_df) > MAX_ROWS_FOR_PLOTS:
    st.warning(f"Графики построены на случайной выборке ({MAX_ROWS_FOR_PLOTS} строк из {len(filtered_df)}).")
    plot_df = filtered_df.sample(n=MAX_ROWS_FOR_PLOTS, random_state=42)
else:
    plot_df = filtered_df

# Короткая сводка по текущему срезу данных.
st.title("BI Dashboard for Sales Analysis")
st.subheader("Обзор")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Строк после фильтров", f"{len(filtered_df):,}")
with col2:
    st.metric("Колонок", len(filtered_df.columns))
with col3:
    if main_metric:
        st.metric(f"Сумма по {main_metric}", f"{filtered_df[main_metric].sum():,.2f}")
    else:
        st.metric("Числовых метрик", 0)
with col4:
    if main_metric:
        st.metric(f"Среднее по {main_metric}", f"{filtered_df[main_metric].mean():,.2f}")
    else:
        st.metric("Категориальных полей", len(schema["category_cols"]))

with st.expander("Структура датасета"):
    st.write(f"**Исходный размер:** {raw_shape[0]} строк × {raw_shape[1]} колонок")
    st.write("**Определённая схема:**")
    st.json(schema)
    st.write("**Основная метрика:**", main_metric if main_metric else "не определена")
    st.write("**Колонки даты:**", schema["date_cols"] if schema["date_cols"] else "не найдены")

# Основные разделы дашборда.
tab1, tab2, tab3, tab4 = st.tabs([
    "Динамика и категории",
    "Распределения",
    "Прогнозирование",
    "Данные и статистика"
])

# Вкладка с динамикой и сравнением категорий.
with tab1:
    if date_col and metric_cols:
        st.subheader("Динамика по времени")
        metric_for_time = st.selectbox("Метрика", metric_cols, key="time_metric")
        agg_func = st.selectbox("Агрегация", ["sum", "mean", "median"], key="agg")
        freq_label = st.selectbox("Период агрегации", ["День", "Месяц", "Квартал"], index=1)
        freq_map = {"День": "D", "Месяц": "MS", "Квартал": "QS"}
        selected_freq = freq_map[freq_label]

        ts = (plot_df.dropna(subset=[date_col, metric_for_time])
              .groupby(pd.Grouper(key=date_col, freq=selected_freq))[metric_for_time]
              .agg(agg_func).reset_index())
        if not ts.empty:
            fig_ts = px.line(ts, x=date_col, y=metric_for_time, markers=True,
                             title=f"{agg_func.upper()}({metric_for_time}) по времени")
            fig_ts.update_traces(line=dict(color="#2563EB", width=3), marker=dict(color="#F59E0B", size=8))
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Для динамики нужна колонка даты и числовая метрика.")

    if schema["category_cols"] and metric_cols:
        st.subheader("Сравнение по категориям")
        col_left, col_right = st.columns(2)
        with col_left:
            cat_col = st.selectbox("Категория", schema["category_cols"], key="cat")
            metric_cat = st.selectbox("Метрика", metric_cols, key="cat_metric")
            cat_df = (plot_df.groupby(cat_col, dropna=False)[metric_cat].sum()
                      .reset_index().sort_values(metric_cat, ascending=False).head(15))
            fig_bar = px.bar(cat_df, x=cat_col, y=metric_cat, color=metric_cat,
                             title=f"SUM({metric_cat}) по {cat_col}",
                             color_continuous_scale="Blues")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_right:
            fig_pie = px.pie(cat_df, names=cat_col, values=metric_cat,
                             title=f"Структура {metric_cat} по {cat_col}",
                             color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6"])
            st.plotly_chart(fig_pie, use_container_width=True)

# Вкладка с распределением числовых полей.
with tab2:
    if schema["numeric_cols"]:
        st.subheader("Распределение числовых показателей")
        num_col = st.selectbox("Числовая колонка", schema["numeric_cols"], key="hist")
        fig_hist = px.histogram(plot_df, x=num_col, nbins=30, title=f"Распределение {num_col}")
        fig_hist.update_traces(marker_color="#10B981")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Нет числовых колонок для отображения распределения.")

# Вкладка с прогнозом по временному ряду.
with tab3:
    if date_col and main_metric:
        st.subheader(f"Прогнозирование: {main_metric}")

        with st.expander("Настройки прогноза", expanded=True):
            forecast_freq = st.selectbox("Частота агрегации", ["День", "Месяц", "Квартал"], index=1, key="forecast_freq")
            freq_map = {"День": "D", "Месяц": "MS", "Квартал": "QS"}
            freq = freq_map[forecast_freq]
            horizon = st.slider("Горизонт прогноза (количество периодов)", 1, 12, 3)
            use_seasonal = st.checkbox("Учитывать сезонность (SARIMA)", value=True)
            seasonal_period = 12 if forecast_freq == "Месяц" else 4 if forecast_freq == "Квартал" else 7

        # Собираем временной ряд по выбранной частоте.
        target_ts = (filtered_df.dropna(subset=[date_col, main_metric])
                     .groupby(pd.Grouper(key=date_col, freq=freq))[main_metric].sum()
                     .reset_index().sort_values(date_col).dropna().reset_index(drop=True))

        if len(target_ts) < 8:
            st.info("Недостаточно временных точек для прогнозирования (нужно минимум 8).")
        else:
            # Дополнительная обработка ряда перед моделью.
            with st.expander("Предобработка временного ряда (экспериментально)"):
                smooth_window = st.selectbox("Сглаживание (скользящее среднее)", [1, 3, 5, 7], index=0)
                remove_outliers = st.checkbox("Удалять выбросы (IQR)", value=False)
                log_transform = st.checkbox("Логарифмирование целевой метрики", value=False)

            series = target_ts[main_metric].copy()
            if remove_outliers:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                series = series.clip(lower=lower, upper=upper)
                st.info(f"Выбросы ограничены диапазоном [{lower:.0f}, {upper:.0f}]")
            if smooth_window > 1:
                series = series.rolling(window=smooth_window, min_periods=1).mean()
                st.info(f"Применено скользящее среднее с окном {smooth_window}")
            if log_transform:
                series = np.log1p(series)
                st.info("Применено логарифмирование (после прогноза будет обратное преобразование)")
            target_ts[main_metric] = series

            # Для долгих рядов оставляем последние точки, чтобы модель считалась быстрее.
            if len(target_ts) > 200:
                target_ts = target_ts.iloc[-200:].reset_index(drop=True)
                st.info("Для ускорения расчётов использованы последние 200 точек.")

            split = max(int(len(target_ts) * 0.8), 2)
            train = target_ts.iloc[:split]
            test = target_ts.iloc[split:]
            y_true = test[main_metric].values

            # Ошибка наивного прогноза нужна только для метрики MASE.
            naive_pred = train[main_metric].iloc[-1]
            naive_errors_train = np.abs(np.diff(train[main_metric].values))

            try:
                if use_seasonal and len(train) >= 2 * seasonal_period:
                    model = SARIMAX(train[main_metric],
                                    order=(1, 1, 1),
                                    seasonal_order=(0, 1, 1, seasonal_period),
                                    simple_differencing=False)
                else:
                    model = ARIMA(train[main_metric], order=(1, 1, 1))
                fitted = model.fit()
                pred = fitted.forecast(steps=len(test))

                # Возвращаем значения в исходный масштаб после логарифмирования.
                if log_transform:
                    pred = np.expm1(pred)
                    y_true_original = np.expm1(y_true)
                else:
                    y_true_original = y_true

                # Считаем метрики на исходной шкале, чтобы они были понятнее.
                mae = mean_absolute_error(y_true_original, pred)
                rmse = np.sqrt(mean_squared_error(y_true_original, pred))
                r2 = r2_score(y_true_original, pred) if len(y_true_original) > 1 else np.nan
                mean_actual = np.mean(y_true_original)

                # Дополнительно переводим MAE и RMSE в проценты от среднего уровня ряда.
                mae_pct = (mae / mean_actual) * 100 if mean_actual != 0 else np.nan
                rmse_pct = (rmse / mean_actual) * 100 if mean_actual != 0 else np.nan

                # Отдельно считаем процентные и сравнительные метрики качества.
                nonzero_mask = y_true_original != 0
                if np.any(nonzero_mask):
                    mape = np.mean(np.abs((y_true_original[nonzero_mask] - pred[nonzero_mask]) / y_true_original[nonzero_mask])) * 100
                else:
                    mape = np.nan

                denominator = (np.abs(y_true_original) + np.abs(pred)) / 2
                denominator = np.where(denominator == 0, 1e-6, denominator)
                smape = np.mean(100 * np.abs(y_true_original - pred) / denominator)

                if len(naive_errors_train) > 0:
                    mae_naive_train = np.mean(naive_errors_train)
                    mase = mae / mae_naive_train if mae_naive_train > 0 else np.nan
                else:
                    mase = np.nan

                # Пользователь сам выбирает, смотреть ошибки в абсолютных значениях или в процентах.
                st.markdown("**Оценка точности прогноза на тестовом периоде**")

                show_pct = st.checkbox("Показывать ошибки в % от среднего значения", value=True, key="show_pct_errors")

                if show_pct:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("MAE%", f"{mae_pct:.2f}%" if not np.isnan(mae_pct) else "—", help="MAE относительно среднего")
                    with col_m2:
                        st.metric("RMSE%", f"{rmse_pct:.2f}%" if not np.isnan(rmse_pct) else "—", help="RMSE относительно среднего")
                    with col_m3:
                        st.metric("R²", f"{r2:.3f}" if not np.isnan(r2) else "—")
                    with col_m4:
                        st.metric("MAPE", f"{mape:.1f}%" if not np.isnan(mape) else "—")

                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.metric("sMAPE", f"{smape:.1f}%")
                    with col_a2:
                        st.metric("MASE", f"{mase:.2f}" if not np.isnan(mase) else "—")
                else:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("MAE", f"{mae:,.2f}")
                    with col_m2:
                        st.metric("RMSE", f"{rmse:,.2f}")
                    with col_m3:
                        st.metric("R²", f"{r2:.3f}" if not np.isnan(r2) else "—")
                    with col_m4:
                        st.metric("MAPE", f"{mape:.1f}%" if not np.isnan(mape) else "—")

                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.metric("sMAPE", f"{smape:.1f}%")
                    with col_a2:
                        st.metric("MASE", f"{mase:.2f}" if not np.isnan(mase) else "—")

                st.markdown("---")
                if not np.isnan(mape):
                    if mape < 10:
                        st.success(f"Отличная точность (MAPE = {mape:.1f}%) — модель хорошо прогнозирует.")
                    elif mape < 30:
                        st.warning(f"Удовлетворительная точность (MAPE = {mape:.1f}%) — прогноз можно использовать с осторожностью.")
                    else:
                        st.error(f"Низкая точность (MAPE = {mape:.1f}%) — модель неадекватна.")
                if r2 < 0:
                    st.info("Совет: R² отрицательный — модель хуже, чем простое среднее. Попробуйте другую агрегацию или удалите выбросы.")

                # Показываем, как распределилась ошибка прогноза по тестовому периоду.
                relative_errors = np.abs(y_true_original - pred) / y_true_original * 100
                relative_errors = relative_errors[~np.isnan(relative_errors) & np.isfinite(relative_errors)]
                if len(relative_errors) > 0:
                    fig_err = px.histogram(relative_errors, nbins=30, title="Распределение относительной ошибки прогноза (%)")
                    fig_err.update_traces(marker_color="#EF4444")
                    st.plotly_chart(fig_err, use_container_width=True)

                # Сравниваем реальные значения и прогноз на тестовой части ряда.
                compare = pd.DataFrame({date_col: test[date_col], "Факт": y_true_original, "Прогноз": pred})
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=compare[date_col], y=compare["Факт"], mode="lines+markers",
                                              name="Факт", line=dict(color="#2563EB", width=3)))
                fig_test.add_trace(go.Scatter(x=compare[date_col], y=compare["Прогноз"], mode="lines+markers",
                                              name="Прогноз", line=dict(color="#F59E0B", width=3, dash="dash")))
                fig_test.update_layout(title="Факт vs прогноз на тестовой выборке")
                st.plotly_chart(fig_test, use_container_width=True)

                # Строим прогноз на будущие периоды уже по всему доступному ряду.
                full_model = ARIMA(target_ts[main_metric], order=(1, 1, 1)).fit()
                future_forecast = full_model.forecast(steps=horizon)
                if log_transform:
                    future_forecast = np.expm1(future_forecast)

                last_date = target_ts[date_col].iloc[-1]
                future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
                future_df = pd.DataFrame({date_col: future_dates, "Прогноз": future_forecast})

                hist_plot = target_ts.tail(12).copy().rename(columns={main_metric: "Факт"})
                if log_transform:
                    hist_plot["Факт"] = np.expm1(hist_plot["Факт"])
                hist_plot["Тип"] = "История"
                fut_plot = future_df.rename(columns={"Прогноз": "Факт"})
                fut_plot["Тип"] = "Прогноз"
                combined = pd.concat([hist_plot, fut_plot], ignore_index=True)
                fig_future = px.line(combined, x=date_col, y="Факт", color="Тип", markers=True,
                                     title=f"Прогноз {main_metric} на {horizon} периодов вперёд",
                                     color_discrete_map={"История": "#10B981", "Прогноз": "#EF4444"})
                st.plotly_chart(fig_future, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при построении модели: {e}. Попробуйте другую частоту агрегации или отключите сезонность.")
    else:
        st.info("Для прогнозирования необходимы колонка даты и числовая метрика.")

# Вкладка с таблицей и описательной статистикой.
with tab4:
    st.subheader("Таблица данных")
    MAX_DISPLAY_ROWS = 1000
    if len(filtered_df) > MAX_DISPLAY_ROWS:
        st.warning(f"Показаны первые {MAX_DISPLAY_ROWS} строк из {len(filtered_df)}. Скачайте CSV для полного просмотра.")
        display_df = filtered_df.head(MAX_DISPLAY_ROWS)
    else:
        display_df = filtered_df
    st.dataframe(display_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать отфильтрованные данные (CSV)", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

    if schema["numeric_cols"]:
        st.subheader("Описательная статистика (числовые колонки)")
        stats_df = filtered_df[schema["numeric_cols"]].describe().T
        st.dataframe(stats_df, use_container_width=True)