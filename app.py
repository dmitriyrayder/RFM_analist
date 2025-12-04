import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import gspread
from google.oauth2.service_account import Credentials

# ==================== КОНФИГУРАЦИЯ ====================

st.set_page_config(
    page_title="Аналитический отчет RFM - Оптика",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def load_excel(file):
    """Загрузка данных из Excel"""
    try:
        df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, f"Ошибка загрузки Excel: {str(e)}"

def load_google_sheet(sheet_url, credentials_json):
    """Загрузка данных из Google Sheets"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        creds = Credentials.from_service_account_info(credentials_json, scopes=scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.get_worksheet(0)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        return df, None
    except Exception as e:
        return None, f"Ошибка загрузки Google Sheets: {str(e)}"

def validate_data(df, required_fields):
    """Валидация обязательных полей"""
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        return False, f"Отсутствуют обязательные поля: {', '.join(missing)}"
    return True, "OK"

def calculate_rfm(df, analysis_date=None):
    """Расчет RFM метрик"""
    if analysis_date is None:
        analysis_date = df['transaction_date'].max()
    
    rfm = df.groupby('client_id').agg({
        'transaction_date': lambda x: (analysis_date - x.max()).days,
        'transaction_id': 'count',
        'transaction_amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['client_id', 'recency', 'frequency', 'monetary']
    
    return rfm

def create_rfm_scores(rfm_df):
    """Создание RFM скоров (1-5) с правильной обработкой дубликатов"""
    rfm_scored = rfm_df.copy()

    # Для Recency: меньше = лучше (5 баллов)
    try:
        rfm_scored['R_score'] = pd.qcut(rfm_scored['recency'], q=5, labels=False, duplicates='drop')
        # Инвертируем шкалу для Recency (меньше значение = выше балл)
        max_r = rfm_scored['R_score'].max()
        rfm_scored['R_score'] = max_r - rfm_scored['R_score'] + 1
    except ValueError:
        # Если невозможно создать квантили, используем процентили
        rfm_scored['R_score'] = pd.cut(rfm_scored['recency'].rank(pct=True), bins=5, labels=False) + 1
        max_r = rfm_scored['R_score'].max()
        rfm_scored['R_score'] = max_r - rfm_scored['R_score'] + 1

    # Для Frequency: больше = лучше (5 баллов)
    try:
        rfm_scored['F_score'] = pd.qcut(rfm_scored['frequency'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm_scored['F_score'] = pd.cut(rfm_scored['frequency'].rank(pct=True), bins=5, labels=False) + 1

    # Для Monetary: больше = лучше (5 баллов)
    try:
        rfm_scored['M_score'] = pd.qcut(rfm_scored['monetary'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm_scored['M_score'] = pd.cut(rfm_scored['monetary'].rank(pct=True), bins=5, labels=False) + 1

    rfm_scored['RFM_score'] = (rfm_scored['R_score'].astype(int) * 100 +
                                rfm_scored['F_score'].astype(int) * 10 +
                                rfm_scored['M_score'].astype(int))

    return rfm_scored

def segment_customers_rfm(rfm_scored):
    """Сегментация клиентов по RFM"""
    def assign_segment(row):
        r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
        
        # Специфика для оптики
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP Клиенты"
        elif r >= 4 and f >= 3 and m >= 3:
            return "Лояльные"
        elif r >= 4 and f <= 2 and m >= 3:
            return "Новые Перспективные"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Спящие VIP"
        elif r <= 2 and f >= 3 and m >= 3:
            return "В Зоне Риска"
        elif r >= 3 and f == 2 and m <= 3:
            return "Нуждаются в Внимании"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Потерянные"
        elif r >= 4 and f <= 2 and m <= 2:
            return "Новички"
        else:
            return "Потенциальные"
    
    rfm_scored['segment'] = rfm_scored.apply(assign_segment, axis=1)
    return rfm_scored

def kmeans_segmentation(rfm_df, n_clusters=5):
    """K-means кластеризация"""
    features = rfm_df[['recency', 'frequency', 'monetary']].copy()
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    silhouette = silhouette_score(features_scaled, rfm_df['cluster'])
    
    return rfm_df, silhouette, kmeans

def calculate_clv(rfm_df, avg_margin=0.3, discount_rate=0.1, years=3):
    """Расчет Customer Lifetime Value (исправленная формула)"""
    # Средний чек
    avg_order = rfm_df['monetary'] / rfm_df['frequency']

    # Годовая частота покупок (более корректный расчет)
    # Если recency < 365, экстраполируем; если > 365, используем фактическую частоту
    days_period = rfm_df['recency'].clip(upper=365)
    annual_frequency = (rfm_df['frequency'] / days_period.clip(lower=1)) * 365
    annual_frequency = annual_frequency.clip(upper=365)  # Не больше 1 раза в день

    # CLV = (avg_order * annual_frequency * margin) * NPV за N лет
    # Используем формулу NPV для дисконтирования будущих потоков
    clv = 0
    for year in range(1, years + 1):
        clv += (avg_order * annual_frequency * avg_margin) / ((1 + discount_rate) ** year)

    return clv

def generate_segment_insights(rfm_segmented, raw_data=None):
    """Автоматическая генерация инсайтов для каждого сегмента"""
    insights = {}
    
    for segment in rfm_segmented['segment'].unique():
        segment_data = rfm_segmented[rfm_segmented['segment'] == segment]
        
        avg_recency = segment_data['recency'].mean()
        avg_frequency = segment_data['frequency'].mean()
        avg_monetary = segment_data['monetary'].mean()
        count = len(segment_data)
        
        insight = {
            'count': count,
            'avg_recency': avg_recency,
            'avg_frequency': avg_frequency,
            'avg_monetary': avg_monetary,
            'events': [],
            'recommendations': [],
            'priority': ''
        }
        
        # События и рекомендации специфичные для оптики
        if segment == "VIP Клиенты":
            insight['events'] = [
                f"✅ {count} клиентов приносят {segment_data['monetary'].sum():.0f} грн дохода",
                f"⏱️ Средняя давность покупки: {avg_recency:.0f} дней",
                f"🔄 Покупают в среднем {avg_frequency:.1f} раз"
            ]
            insight['recommendations'] = [
                "🎁 VIP-карты с эксклюзивными скидками 15-20%",
                "📱 Персональные напоминания о проверке зрения (раз в 6 месяцев)",
                "👔 Приглашения на закрытые презентации премиум-коллекций",
                "🎯 Программа раннего доступа к новинкам",
                "💎 Персональный менеджер и приоритетное обслуживание"
            ]
            insight['priority'] = "🔥 ВЫСОКИЙ - максимальное удержание"
            
        elif segment == "Лояльные":
            insight['events'] = [
                f"✅ Стабильные {count} клиентов со средним чеком {avg_monetary:.0f} грн",
                f"📊 Регулярность покупок: каждые {avg_recency:.0f} дней"
            ]
            insight['recommendations'] = [
                "🎯 Программа лояльности: 1 грн = 1 бонус",
                "👓 Предложить вторую пару очков со скидкой 30%",
                "☀️ Акция на солнцезащитные очки в сезон",
                "👨‍👩‍👧 Семейные предложения: скидка при покупке от 3-х пар",
                "📧 Email-рассылка с новинками раз в месяц"
            ]
            insight['priority'] = "🔥 ВЫСОКИЙ - развитие"
            
        elif segment == "Новые Перспективные":
            insight['events'] = [
                f"🆕 {count} новых клиентов с высоким первым чеком ({avg_monetary:.0f} грн)",
                f"⏳ Недавняя первая покупка: {avg_recency:.0f} дней назад"
            ]
            insight['recommendations'] = [
                "🎁 Welcome-бонус 500 грн на вторую покупку",
                "📱 SMS через 3 месяца: 'Как Ваши очки? Проверка зрения бесплатно'",
                "👓 Предложение линз с защитой от синего света",
                "💳 Оформление карты лояльности с приветственной скидкой",
                "📞 Обратная связь через неделю после покупки"
            ]
            insight['priority'] = "⚡ СРЕДНИЙ - быстрая активация"
            
        elif segment == "Спящие VIP":
            days_since = avg_recency
            insight['events'] = [
                f"⚠️ {count} ценных клиентов не покупают {days_since:.0f} дней!",
                f"💰 Потенциальная потеря дохода: {segment_data['monetary'].sum():.0f} грн",
                f"📉 Риск оттока высокоценных клиентов"
            ]
            insight['recommendations'] = [
                "🚨 СРОЧНО: Персональный звонок с предложением VIP-скидки 25%",
                "🔬 Бесплатная проверка зрения + диагностика в подарок",
                "🎁 Эксклюзивное предложение: новая оправа + линзы -30%",
                "👨‍⚕️ Напоминание: 'Прошло больше года, рекомендуем проверку'",
                "💎 Возврат VIP-статуса при покупке в течение 30 дней"
            ]
            insight['priority'] = "🔴 КРИТИЧЕСКИЙ - реактивация"
            
        elif segment == "В Зоне Риска":
            insight['events'] = [
                f"⚠️ {count} клиентов давно не покупали ({avg_recency:.0f} дней)",
                f"💸 Средний чек был {avg_monetary:.0f} грн"
            ]
            insight['recommendations'] = [
                "📧 Email-кампания: 'Мы скучаем! Скидка 20% на любую покупку'",
                "👓 Акция trade-in: сдай старые очки, получи скидку 15%",
                "🎯 Ремаркетинг в соцсетях с персональными предложениями",
                "📱 SMS: 'Время обновить очки? Специальная цена для Вас'",
                "🔬 Бесплатная проверка зрения как повод вернуться"
            ]
            insight['priority'] = "🔴 КРИТИЧЕСКИЙ - срочная реактивация"
            
        elif segment == "Нуждаются в Внимании":
            insight['events'] = [
                f"📊 {count} клиентов с потенциалом роста",
                f"💡 Низкая частота: {avg_frequency:.1f} покупок"
            ]
            insight['recommendations'] = [
                "🎁 Программа стимулирования: купи 2 - получи скидку 15% на 3-ю",
                "👓 Предложение аксессуаров: футляры, салфетки, цепочки",
                "📧 Образовательный контент: 'Как выбрать очки для компьютера'",
                "☀️ Допродажа: солнцезащитные очки со скидкой 25%",
                "💳 Бонусы за каждую покупку: 10% возврат баллами"
            ]
            insight['priority'] = "⚡ СРЕДНИЙ - развитие частоты"
            
        elif segment == "Потерянные":
            insight['events'] = [
                f"❌ {count} клиентов давно ушли (>{avg_recency:.0f} дней)",
                f"💸 Низкий LTV: {avg_monetary:.0f} грн"
            ]
            insight['recommendations'] = [
                "🔄 Win-back кампания: скидка 30% на возврат",
                "📞 Холодный обзвон с опросом: 'Почему перестали покупать?'",
                "🎁 Последний шанс: промокод на 40% скидку (срок 14 дней)",
                "📊 Анализ причин оттока - улучшение сервиса",
                "⚠️ Если нет реакции - исключить из активной базы"
            ]
            insight['priority'] = "🟡 НИЗКИЙ - оценка целесообразности"
            
        elif segment == "Новички":
            insight['events'] = [
                f"🆕 {count} новых клиентов с низким первым чеком",
                f"💡 Средний чек: {avg_monetary:.0f} грн (низкий)"
            ]
            insight['recommendations'] = [
                "📚 Обучение: 'Как выбрать качественные очки'",
                "🎁 Купон на скидку 15% на следующую покупку",
                "👓 Предложение апгрейда линз со скидкой",
                "📱 Подписка на полезный контент о здоровье глаз",
                "💳 Карта лояльности + стартовые 200 бонусов"
            ]
            insight['priority'] = "⚡ СРЕДНИЙ - вовлечение"
            
        else:  # Потенциальные
            insight['events'] = [
                f"📊 {count} клиентов со средними показателями",
                f"💡 Возможность роста в другие сегменты"
            ]
            insight['recommendations'] = [
                "🎯 А/B тестирование различных предложений",
                "📧 Регулярные email с персональными предложениями",
                "🎁 Сезонные акции и распродажи",
                "👓 Кросс-селл: аксессуары, растворы для линз",
                "📊 Мониторинг поведения для перехода в активный сегмент"
            ]
            insight['priority'] = "🟢 НОРМАЛЬНЫЙ - стандартный подход"
        
        insights[segment] = insight
    
    return insights

def generate_business_conclusions(rfm_segmented, insights):
    """Общие бизнес-выводы"""
    total_clients = len(rfm_segmented)
    total_revenue = rfm_segmented['monetary'].sum()
    
    vip_count = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клиенты', 'Лояльные'])])
    vip_revenue = rfm_segmented[rfm_segmented['segment'].isin(['VIP Клиенты', 'Лояльные'])]['monetary'].sum()
    
    at_risk_count = len(rfm_segmented[rfm_segmented['segment'].isin(['Спящие VIP', 'В Зоне Риска', 'Потерянные'])])
    at_risk_revenue = rfm_segmented[rfm_segmented['segment'].isin(['Спящие VIP', 'В Зоне Риска'])]['monetary'].sum()
    
    conclusions = {
        'summary': [
            f"📊 Всего клиентов: {total_clients}",
            f"💰 Общая выручка: {total_revenue:,.0f} грн",
            f"💳 Средний чек: {total_revenue/total_clients:,.0f} грн"
        ],
        'key_findings': [
            f"🌟 Топ-клиенты: {vip_count} ({vip_count/total_clients*100:.1f}%) приносят {vip_revenue:,.0f} грн ({vip_revenue/total_revenue*100:.1f}% выручки)",
            f"⚠️ В зоне риска: {at_risk_count} клиентов ({at_risk_count/total_clients*100:.1f}%) с потенциальной потерей {at_risk_revenue:,.0f} грн",
            f"🎯 Приоритет #1: Удержание VIP и реактивация спящих VIP-клиентов"
        ],
        'strategic_actions': [
            "🔥 НЕМЕДЛЕННО: Запустить программу реактивации для 'Спящие VIP' (персональные звонки)",
            "💎 Создать VIP-клуб с эксклюзивными условиями для топ-20% клиентов",
            "🎯 Разработать welcome-сценарий для новых клиентов (первые 90 дней)",
            "📊 Внедрить систему предиктивной аналитики оттока",
            "🔄 Автоматизировать триггерные рассылки по сегментам"
        ],
        'expected_impact': [
            f"📈 Ожидаемое увеличение выручки: +8-12% при правильной работе с сегментами",
            f"🎯 Снижение оттока VIP на 3-5% = экономия ~{at_risk_revenue*0.04:,.0f} грн/год",
            f"💰 Рост среднего чека на 5-7% = дополнительно ~{total_revenue*0.06:,.0f} грн/год"
        ]
    }
    
    return conclusions

# ==================== ОСНОВНОЙ КОД ====================

def main():
    st.title("📊 Аналитический отчет RFM: Оптика")
    st.markdown("#### Стратегический анализ клиентской базы и рекомендации по управлению")

    # Добавляем информационный блок
    st.info("""
    **Назначение отчета:** Анализ клиентской базы по методологии RFM (Recency, Frequency, Monetary)
    для принятия стратегических решений по управлению взаимоотношениями с клиентами и увеличению прибыльности бизнеса.
    """)
    
    # Sidebar - загрузка данных
    st.sidebar.header("📥 Загрузка данных")
    
    data_source = st.sidebar.radio(
        "Источник данных:",
        ["Excel файл", "Google Sheets"]
    )
    
    df = None
    
    if data_source == "Excel файл":
        uploaded_file = st.sidebar.file_uploader(
            "Загрузите Excel файл",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file:
            df, error = load_excel(uploaded_file)
            if error:
                st.sidebar.error(error)
    
    else:  # Google Sheets
        st.sidebar.markdown("**Для подключения Google Sheets:**")
        st.sidebar.info(
            "1. Создайте Service Account в Google Cloud\n"
            "2. Скачайте JSON ключ\n"
            "3. Дайте доступ к таблице для email из JSON"
        )
        
        sheet_url = st.sidebar.text_input("URL Google Sheets:")
        credentials_file = st.sidebar.file_uploader(
            "JSON ключ от Service Account",
            type=['json']
        )
        
        if sheet_url and credentials_file:
            try:
                credentials_json = json.load(credentials_file)
                df, error = load_google_sheet(sheet_url, credentials_json)
                if error:
                    st.sidebar.error(error)
            except Exception as e:
                st.sidebar.error(f"Ошибка чтения credentials: {str(e)}")
    
    # Информация о полях
    with st.sidebar.expander("ℹ️ Структура данных"):
        st.markdown("""
        **Обязательные поля:**
        - `client_id` - ID клиента
        - `transaction_id` - ID транзакции
        - `transaction_date` - Дата покупки
        - `transaction_amount` - Сумма покупки

        **Опциональные поля:**
        - `client_name` - ФИО клиента (рекомендуется)
        - `product_category` - Категория (оправы/линзы/солнцезащитные/аксессуары)
        - `sales_channel` - Канал (онлайн/офлайн)
        - `store_id` - ID магазина
        - `loyalty_points` - Баллы лояльности
        - `age` - Возраст
        - `gender` - Пол
        - `city` - Город

        **Примечание:** При наличии поля `client_name` оно будет отображаться во всех таблицах для удобной идентификации клиентов.
        """)
    
    if df is not None:
        # Валидация данных
        required_fields = ['client_id', 'transaction_id', 'transaction_date', 'transaction_amount']
        is_valid, message = validate_data(df, required_fields)
        
        if not is_valid:
            st.error(message)
            st.stop()
        
        # Конвертация даты
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Показываем превью данных
        st.success(f"✅ Загружено {len(df)} транзакций от {df['client_id'].nunique()} клиентов")
        
        with st.expander("👀 Превью данных"):
            st.dataframe(df.head(10))
        
        # Параметры анализа
        st.sidebar.header("⚙️ Параметры анализа")
        
        analysis_date = st.sidebar.date_input(
            "Дата анализа:",
            value=df['transaction_date'].max().date()
        )
        analysis_date = pd.Timestamp(analysis_date)
        
        n_clusters = st.sidebar.slider(
            "Количество кластеров K-means:",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Расчет RFM
        with st.spinner("🔄 Расчет RFM метрик..."):
            rfm = calculate_rfm(df, analysis_date)
            rfm_scored = create_rfm_scores(rfm)
            rfm_segmented = segment_customers_rfm(rfm_scored)
            rfm_clustered, silhouette, kmeans = kmeans_segmentation(rfm.copy(), n_clusters)
            
            # CLV
            rfm_segmented['clv'] = calculate_clv(rfm_segmented)
        
        # Генерация инсайтов
        with st.spinner("🤖 Генерация автоматических инсайтов..."):
            insights = generate_segment_insights(rfm_segmented, df)
            conclusions = generate_business_conclusions(rfm_segmented, insights)
        
        # ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Executive Summary",
            "🎯 Сегменты RFM",
            "👥 Детальный анализ клиентов",
            "🔬 K-means Кластеры",
            "💎 CLV Анализ",
            "📋 Стратегические рекомендации"
        ])
        
        # TAB 1: Executive Summary
        with tab1:
            st.header("📊 Executive Summary")
            st.markdown("### Ключевые показатели бизнеса")

            # Ключевые метрики
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всего клиентов", f"{len(rfm):,}")
                st.caption("Уникальных клиентов в базе")
            with col2:
                total_revenue = rfm['monetary'].sum()
                st.metric("Общая выручка", f"{total_revenue:,.0f} грн")
                st.caption("Совокупный доход")
            with col3:
                avg_revenue = rfm['monetary'].mean()
                st.metric("Средний LTV", f"{avg_revenue:,.0f} грн")
                st.caption("На одного клиента")
            with col4:
                avg_freq = rfm['frequency'].mean()
                st.metric("Средняя частота", f"{avg_freq:.1f}")
                st.caption("Покупок на клиента")

            st.markdown("---")

            # Сегментация клиентов
            st.markdown("### Сегментация клиентской базы")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Таблица по сегментам
                segment_summary = rfm_segmented.groupby('segment').agg({
                    'client_id': 'count',
                    'monetary': 'sum'
                }).round(0)
                segment_summary.columns = ['Количество клиентов', 'Общая выручка (грн)']
                segment_summary['Доля клиентов (%)'] = (segment_summary['Количество клиентов'] / len(rfm_segmented) * 100).round(1)
                segment_summary['Доля выручки (%)'] = (segment_summary['Общая выручка (грн)'] / total_revenue * 100).round(1)
                segment_summary = segment_summary.sort_values('Общая выручка (грн)', ascending=False)

                st.dataframe(segment_summary, use_container_width=True)

            with col2:
                # График распределения выручки
                fig = px.pie(
                    segment_summary.reset_index(),
                    values='Общая выручка (грн)',
                    names='segment',
                    title='Распределение выручки по сегментам',
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Критические инсайты
            st.markdown("### 🎯 Критические инсайты")

            vip_count = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клиенты', 'Лояльные'])])
            vip_revenue = rfm_segmented[rfm_segmented['segment'].isin(['VIP Клиенты', 'Лояльные'])]['monetary'].sum()
            at_risk_count = len(rfm_segmented[rfm_segmented['segment'].isin(['Спящие VIP', 'В Зоне Риска'])])
            at_risk_revenue = rfm_segmented[rfm_segmented['segment'].isin(['Спящие VIP', 'В Зоне Риска'])]['monetary'].sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"**✅ VIP клиенты**")
                st.metric("Количество", vip_count, f"{vip_count/len(rfm_segmented)*100:.1f}% базы")
                st.metric("Выручка", f"{vip_revenue:,.0f} грн", f"{vip_revenue/total_revenue*100:.1f}% доли")

            with col2:
                st.warning(f"**⚠️ В зоне риска**")
                st.metric("Количество", at_risk_count, f"{at_risk_count/len(rfm_segmented)*100:.1f}% базы")
                st.metric("Потенциальная потеря", f"{at_risk_revenue:,.0f} грн")

            with col3:
                st.info(f"**📈 Потенциал роста**")
                potential_increase = total_revenue * 0.10  # 10% рост при правильной работе
                st.metric("Прогноз при оптимизации", f"+{potential_increase:,.0f} грн")
                st.metric("Рост выручки", "+8-12%")

            st.markdown("---")

            # Динамика активности
            st.markdown("### 📈 Распределение метрик")

            col1, col2, col3 = st.columns(3)

            with col1:
                fig = px.histogram(rfm, x='recency', nbins=50,
                                 title='Recency (давность покупки)',
                                 labels={'recency': 'Дней с последней покупки'},
                                 color_discrete_sequence=['#636EFA'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(rfm, x='frequency', nbins=30,
                                 title='Frequency (частота покупок)',
                                 labels={'frequency': 'Количество покупок'},
                                 color_discrete_sequence=['#EF553B'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                fig = px.histogram(rfm, x='monetary', nbins=50,
                                 title='Monetary (сумма покупок)',
                                 labels={'monetary': 'Выручка (грн)'},
                                 color_discrete_sequence=['#00CC96'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Сегменты RFM
        with tab2:
            st.header("🎯 RFM Сегменты")
            
            # Статистика по сегментам
            segment_stats = rfm_segmented.groupby('segment').agg({
                'client_id': 'count',
                'monetary': 'sum',
                'recency': 'mean',
                'frequency': 'mean',
                'clv': 'mean'
            }).round(0)
            segment_stats.columns = ['Количество', 'Выручка', 'Avg Recency', 'Avg Frequency', 'Avg CLV']
            segment_stats = segment_stats.sort_values('Выручка', ascending=False)
            
            st.dataframe(segment_stats, use_container_width=True)
            
            # Визуализация сегментов
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    rfm_segmented,
                    names='segment',
                    title='Распределение клиентов по сегментам',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                segment_revenue = rfm_segmented.groupby('segment')['monetary'].sum().reset_index()
                fig = px.bar(
                    segment_revenue,
                    x='segment',
                    y='monetary',
                    title='Выручка по сегментам',
                    labels={'monetary': 'Выручка (грн)', 'segment': 'Сегмент'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            # 2D визуализация RFM сегментов
            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    rfm_segmented,
                    x='recency',
                    y='monetary',
                    color='segment',
                    size='frequency',
                    title='Сегменты: Recency vs Monetary',
                    labels={
                        'recency': 'Recency (дни)',
                        'monetary': 'Monetary (грн)',
                        'segment': 'Сегмент'
                    },
                    hover_data=['client_id', 'frequency', 'RFM_score']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    rfm_segmented,
                    x='frequency',
                    y='monetary',
                    color='segment',
                    title='Сегменты: Frequency vs Monetary',
                    labels={
                        'frequency': 'Frequency',
                        'monetary': 'Monetary (грн)',
                        'segment': 'Сегмент'
                    },
                    hover_data=['client_id', 'recency', 'RFM_score']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Дополнительная аналитика: heat map RFM
            st.markdown("---")
            st.markdown("### 🔥 Heat Map: RFM Score Distribution")

            # Создаем pivot таблицу для heat map
            heatmap_data = rfm_segmented.groupby(['R_score', 'F_score']).agg({
                'client_id': 'count',
                'monetary': 'sum'
            }).reset_index()
            heatmap_pivot = heatmap_data.pivot(index='R_score', columns='F_score', values='client_id').fillna(0)

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[f'F{i}' for i in heatmap_pivot.columns],
                y=[f'R{i}' for i in heatmap_pivot.index],
                colorscale='Viridis',
                text=heatmap_pivot.values,
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Количество клиентов")
            ))
            fig.update_layout(
                title='Распределение клиентов по R и F Score',
                xaxis_title='Frequency Score',
                yaxis_title='Recency Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Box plot по сегментам
            st.markdown("---")
            st.markdown("### 📦 Распределение метрик по сегментам")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.box(
                    rfm_segmented,
                    x='segment',
                    y='monetary',
                    title='Распределение Monetary по сегментам',
                    labels={'monetary': 'Monetary (грн)', 'segment': 'Сегмент'},
                    color='segment'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(
                    rfm_segmented,
                    x='segment',
                    y='frequency',
                    title='Распределение Frequency по сегментам',
                    labels={'frequency': 'Frequency', 'segment': 'Сегмент'},
                    color='segment'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # АВТОМАТИЧЕСКИЕ ИНСАЙТЫ ПО СЕГМЕНТАМ
            st.markdown("---")
            st.header("🤖 Автоматические инсайты и рекомендации")

            # Таблица распределения клиентов по сегментам
            st.subheader("📋 Распределение клиентов по сегментам")

            # Подготовка данных для таблицы
            client_segments = rfm_segmented[['client_id', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'clv']].copy()

            # Если есть поле ФИО клиента в исходных данных, добавляем его
            if 'client_name' in df.columns:
                client_names = df[['client_id', 'client_name']].drop_duplicates()
                client_segments = client_segments.merge(client_names, on='client_id', how='left')
                client_segments = client_segments[['client_id', 'client_name', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'clv']]

            # Форматирование для отображения
            client_segments_display = client_segments.copy()
            client_segments_display['monetary'] = client_segments_display['monetary'].round(0)
            client_segments_display['clv'] = client_segments_display['clv'].round(0)

            # Показываем сводку по сегментам
            col1, col2, col3 = st.columns(3)
            with col1:
                top_segment = rfm_segmented['segment'].value_counts().index[0]
                st.metric("Крупнейший сегмент", top_segment, f"{rfm_segmented['segment'].value_counts().values[0]} клиентов")
            with col2:
                high_priority = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клиенты', 'Лояльные'])])
                st.metric("Приоритетные клиенты", high_priority, f"{high_priority/len(rfm_segmented)*100:.1f}%")
            with col3:
                at_risk = len(rfm_segmented[rfm_segmented['segment'].isin(['Спящие VIP', 'В Зоне Риска'])])
                st.metric("В зоне риска", at_risk, f"{at_risk/len(rfm_segmented)*100:.1f}%")

            st.markdown("**Полная таблица клиентов:**")
            st.dataframe(
                client_segments_display,
                use_container_width=True,
                height=400
            )

            # Сортируем сегменты по приоритету
            priority_order = {
                "🔴 КРИТИЧЕСКИЙ - реактивация": 1,
                "🔴 КРИТИЧЕСКИЙ - срочная реактивация": 2,
                "🔥 ВЫСОКИЙ - максимальное удержание": 3,
                "🔥 ВЫСОКИЙ - развитие": 4,
                "⚡ СРЕДНИЙ - быстрая активация": 5,
                "⚡ СРЕДНИЙ - развитие частоты": 6,
                "⚡ СРЕДНИЙ - вовлечение": 7,
                "🟢 НОРМАЛЬНЫЙ - стандартный подход": 8,
                "🟡 НИЗКИЙ - оценка целесообразности": 9
            }
            
            sorted_segments = sorted(
                insights.items(),
                key=lambda x: priority_order.get(x[1]['priority'], 99)
            )
            
            for segment, insight in sorted_segments:
                with st.expander(f"**{segment}** - {insight['count']} клиентов | {insight['priority']}"):
                    
                    # Метрики сегмента
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Клиентов", f"{insight['count']}")
                    with col2:
                        st.metric("Avg Recency", f"{insight['avg_recency']:.0f} дней")
                    with col3:
                        st.metric("Avg Frequency", f"{insight['avg_frequency']:.1f}")
                    with col4:
                        st.metric("Avg Monetary", f"{insight['avg_monetary']:,.0f} грн")
                    
                    # События
                    st.markdown("**📌 Ключевые события:**")
                    for event in insight['events']:
                        st.markdown(f"- {event}")
                    
                    st.markdown("")
                    
                    # Рекомендации
                    st.markdown("**💡 Рекомендации:**")
                    for rec in insight['recommendations']:
                        st.markdown(f"- {rec}")

        # TAB 3: Детальный анализ клиентов (с фильтрами)
        with tab3:
            st.header("👥 Детальный анализ клиентов")
            st.markdown("Интерактивная таблица с возможностью фильтрации по различным параметрам")

            # Подготовка данных
            detailed_df = rfm_segmented.copy()

            # Добавляем ФИО если есть
            if 'client_name' in df.columns:
                client_names = df[['client_id', 'client_name']].drop_duplicates()
                detailed_df = detailed_df.merge(client_names, on='client_id', how='left')

            # Фильтры
            st.markdown("### Фильтры")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                segment_filter = st.multiselect(
                    "Сегмент:",
                    options=sorted(detailed_df['segment'].unique()),
                    default=None,
                    placeholder="Все сегменты"
                )

            with col2:
                recency_range = st.slider(
                    "Recency (дни):",
                    min_value=int(detailed_df['recency'].min()),
                    max_value=int(detailed_df['recency'].max()),
                    value=(int(detailed_df['recency'].min()), int(detailed_df['recency'].max()))
                )

            with col3:
                frequency_range = st.slider(
                    "Frequency (покупки):",
                    min_value=int(detailed_df['frequency'].min()),
                    max_value=int(detailed_df['frequency'].max()),
                    value=(int(detailed_df['frequency'].min()), int(detailed_df['frequency'].max()))
                )

            with col4:
                monetary_range = st.slider(
                    "Monetary (грн):",
                    min_value=float(detailed_df['monetary'].min()),
                    max_value=float(detailed_df['monetary'].max()),
                    value=(float(detailed_df['monetary'].min()), float(detailed_df['monetary'].max())),
                    format="%.0f"
                )

            # Применяем фильтры
            filtered_df = detailed_df.copy()

            if segment_filter:
                filtered_df = filtered_df[filtered_df['segment'].isin(segment_filter)]

            filtered_df = filtered_df[
                (filtered_df['recency'] >= recency_range[0]) &
                (filtered_df['recency'] <= recency_range[1]) &
                (filtered_df['frequency'] >= frequency_range[0]) &
                (filtered_df['frequency'] <= frequency_range[1]) &
                (filtered_df['monetary'] >= monetary_range[0]) &
                (filtered_df['monetary'] <= monetary_range[1])
            ]

            # Статистика после фильтрации
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Клиентов после фильтрации", f"{len(filtered_df):,}")
            with col2:
                st.metric("Общая выручка", f"{filtered_df['monetary'].sum():,.0f} грн")
            with col3:
                st.metric("Средний чек", f"{filtered_df['monetary'].mean():,.0f} грн")
            with col4:
                st.metric("Средний CLV", f"{filtered_df['clv'].mean():,.0f} грн")

            st.markdown("---")

            # Сортировка
            sort_options = {
                'CLV (убывание)': ('clv', False),
                'CLV (возрастание)': ('clv', True),
                'Monetary (убывание)': ('monetary', False),
                'Monetary (возрастание)': ('monetary', True),
                'Recency (убывание)': ('recency', False),
                'Recency (возрастание)': ('recency', True),
                'Frequency (убывание)': ('frequency', False),
                'Frequency (возрастание)': ('frequency', True),
                'RFM Score (убывание)': ('RFM_score', False),
                'RFM Score (возрастание)': ('RFM_score', True)
            }

            sort_by = st.selectbox("Сортировать по:", list(sort_options.keys()), index=0)
            sort_col, sort_asc = sort_options[sort_by]
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

            # Отображение таблицы
            st.markdown("### Таблица клиентов")

            # Выбор колонок для отображения
            display_columns = ['client_id', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'R_score', 'F_score', 'M_score', 'clv']
            if 'client_name' in filtered_df.columns:
                display_columns = ['client_id', 'client_name'] + display_columns[1:]

            # Форматирование
            display_df = filtered_df[display_columns].copy()
            display_df['monetary'] = display_df['monetary'].round(0)
            display_df['clv'] = display_df['clv'].round(0)

            # Отображаем с возможностью выбора строк
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )

            # Экспорт отфильтрованных данных
            st.markdown("---")
            st.markdown("### Экспорт данных")

            col1, col2 = st.columns(2)

            with col1:
                # CSV экспорт
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать CSV",
                    data=csv,
                    file_name=f"filtered_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

            with col2:
                # Excel экспорт
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    display_df.to_excel(writer, sheet_name='Filtered_Clients', index=False)

                st.download_button(
                    label="📥 Скачать Excel",
                    data=buffer.getvalue(),
                    file_name=f"filtered_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

        # TAB 4: K-means кластеры
        with tab4:
            st.header("🔬 K-means Кластеризация")
            
            st.info(f"Silhouette Score: {silhouette:.3f}")
            
            # Статистика по кластерам
            cluster_stats = rfm_clustered.groupby('cluster').agg({
                'client_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean'
            }).round(0)
            cluster_stats.columns = ['Количество', 'Avg Recency', 'Avg Frequency', 'Avg Monetary']
            
            st.dataframe(cluster_stats, use_container_width=True)

            # 2D визуализации кластеров
            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    rfm_clustered,
                    x='recency',
                    y='monetary',
                    color='cluster',
                    size='frequency',
                    title='Кластеры: Recency vs Monetary',
                    labels={
                        'recency': 'Recency (дни)',
                        'monetary': 'Monetary (грн)',
                        'cluster': 'Кластер'
                    },
                    hover_data=['client_id', 'frequency']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    rfm_clustered,
                    x='frequency',
                    y='monetary',
                    color='cluster',
                    size='recency',
                    title='Кластеры: Frequency vs Monetary',
                    labels={
                        'frequency': 'Frequency',
                        'monetary': 'Monetary (грн)',
                        'cluster': 'Кластер'
                    },
                    hover_data=['client_id', 'recency']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: CLV Анализ
        with tab5:
            st.header("💎 Customer Lifetime Value (CLV)")
            
            # Топ клиенты по CLV
            top_clv = rfm_segmented.nlargest(20, 'clv')[['client_id', 'segment', 'monetary', 'frequency', 'clv']]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🏆 Топ-20 клиентов по CLV")
                st.dataframe(
                    top_clv.style.format({
                        'monetary': '{:,.0f} грн',
                        'clv': '{:,.0f} грн'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Метрики CLV
                st.metric("Средний CLV", f"{rfm_segmented['clv'].mean():,.0f} грн")
                st.metric("Медианный CLV", f"{rfm_segmented['clv'].median():,.0f} грн")
                st.metric("Топ-10% CLV", f"{rfm_segmented['clv'].quantile(0.9):,.0f} грн")
            
            # CLV по сегментам
            fig = px.box(
                rfm_segmented,
                x='segment',
                y='clv',
                title='Распределение CLV по сегментам',
                labels={'clv': 'CLV (грн)', 'segment': 'Сегмент'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Парето анализ
            rfm_sorted = rfm_segmented.sort_values('clv', ascending=False).reset_index(drop=True)
            rfm_sorted['cumulative_clv'] = rfm_sorted['clv'].cumsum()
            rfm_sorted['cumulative_clv_pct'] = rfm_sorted['cumulative_clv'] / rfm_sorted['clv'].sum() * 100
            rfm_sorted['client_pct'] = (rfm_sorted.index + 1) / len(rfm_sorted) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rfm_sorted['client_pct'],
                y=rfm_sorted['cumulative_clv_pct'],
                mode='lines',
                name='Кумулятивный CLV'
            ))
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=100, y1=100,
                line=dict(dash='dash', color='gray')
            )
            fig.update_layout(
                title='Парето анализ CLV (правило 80/20)',
                xaxis_title='% клиентов',
                yaxis_title='% кумулятивного CLV'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Сколько клиентов дают 80% выручки
            clients_80 = rfm_sorted[rfm_sorted['cumulative_clv_pct'] <= 80]
            st.info(f"📊 **{len(clients_80)} клиентов ({len(clients_80)/len(rfm_sorted)*100:.1f}%) генерируют 80% прогнозируемой выручки**")
        
        # TAB 6: Стратегические рекомендации
        with tab6:
            st.header("📋 Стратегические рекомендации")
            
            # Summary
            st.subheader("📊 Итоговая статистика")
            for item in conclusions['summary']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Key Findings
            st.subheader("🔍 Ключевые находки")
            for item in conclusions['key_findings']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Strategic Actions
            st.subheader("🎯 Стратегические действия")
            for item in conclusions['strategic_actions']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Expected Impact
            st.subheader("📈 Ожидаемый эффект")
            for item in conclusions['expected_impact']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Action Plan
            st.subheader("📅 План действий на ближайшие 30 дней")
            
            action_plan = pd.DataFrame({
                'Неделя': ['1', '1-2', '2-3', '3-4', '4+'],
                'Действие': [
                    'Реактивация Спящих VIP: персональные звонки + VIP-предложения',
                    'Запуск welcome-программы для Новых Перспективных клиентов',
                    'Email-кампании для сегмента "В Зоне Риска"',
                    'Тестирование программы лояльности для Лояльных клиентов',
                    'Анализ результатов, корректировка стратегии, масштабирование'
                ],
                'Ответственный': [
                    'Менеджер по работе с VIP',
                    'CRM-маркетолог',
                    'Email-маркетолог',
                    'Руководитель отдела маркетинга',
                    'Вся команда'
                ],
                'KPI': [
                    'Конверсия звонков >15%',
                    'Повторная покупка >25%',
                    'Open rate >30%, конверсия >5%',
                    'Участие в программе >40%',
                    'Общий рост выручки >8%'
                ]
            })
            
            st.dataframe(action_plan, use_container_width=True)
            
            # Скачивание отчета
            st.markdown("---")
            st.subheader("💾 Экспорт данных")
            
            # Подготовка данных для экспорта
            export_df = rfm_segmented.merge(
                rfm_clustered[['client_id', 'cluster']],
                on='client_id'
            )
            
            # Excel экспорт
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='RFM_Segments', index=False)
                segment_stats.to_excel(writer, sheet_name='Segment_Stats')
                cluster_stats.to_excel(writer, sheet_name='Cluster_Stats')
            
            st.download_button(
                label="📥 Скачать полный отчет (Excel)",
                data=buffer.getvalue(),
                file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    else:
        st.info("👆 Загрузите данные в боковой панели для начала анализа")
        
        # Пример структуры данных
        st.subheader("📋 Пример структуры данных")

        st.markdown("""
        Загрузите Excel файл со следующей структурой. Обязательные поля выделены **жирным**.
        """)

        example_data = pd.DataFrame({
            'client_id': [1001, 1001, 1002, 1003, 1003],
            'client_name': ['Иванов И.И.', 'Иванов И.И.', 'Петрова А.С.', 'Сидоров П.К.', 'Сидоров П.К.'],
            'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'transaction_date': ['2024-01-15', '2024-06-20', '2024-03-10', '2024-02-05', '2024-11-12'],
            'transaction_amount': [2500, 1800, 3200, 4500, 2200],
            'product_category': ['Оправы', 'Солнцезащитные', 'Оправы + Линзы', 'Премиум оправы', 'Линзы'],
            'sales_channel': ['Офлайн', 'Онлайн', 'Офлайн', 'Офлайн', 'Онлайн']
        })

        st.dataframe(example_data)

        st.markdown("""
        **Примечание:**
        - Поле `client_name` опционально, но рекомендуется для удобства работы с отчетами
        - Дата должна быть в формате YYYY-MM-DD или DD.MM.YYYY
        - Сумма транзакции - число без валюты
        """)

if __name__ == "__main__":
    main()