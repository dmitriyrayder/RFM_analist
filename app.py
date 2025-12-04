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
    page_title="Аналітичний звіт RFM - Оптика",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def load_excel(file):
    """Завантаження даних з Excel"""
    try:
        df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, f"Помилка завантаження Excel: {str(e)}"

def load_google_sheet(sheet_url, credentials_json):
    """Завантаження даних з Google Sheets"""
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
        return None, f"Помилка завантаження Google Sheets: {str(e)}"

def validate_data(df, required_fields):
    """Валідація обов'язкових полів"""
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        return False, f"Відсутні обов'язкові поля: {', '.join(missing)}"
    return True, "OK"

def calculate_rfm(df, analysis_date=None):
    """Розрахунок RFM метрик"""
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
    """Створення RFM скорів (1-5) з правильною обробкою дублікатів"""
    rfm_scored = rfm_df.copy()

    # Для Recency: менше = краще (5 балів)
    try:
        rfm_scored['R_score'] = pd.qcut(rfm_scored['recency'], q=5, labels=False, duplicates='drop')
        # Інвертуємо шкалу для Recency (менше значення = вищий бал)
        max_r = rfm_scored['R_score'].max()
        rfm_scored['R_score'] = max_r - rfm_scored['R_score'] + 1
    except ValueError:
        # Якщо неможливо створити квантилі, використовуємо процентилі
        rfm_scored['R_score'] = pd.cut(rfm_scored['recency'].rank(pct=True), bins=5, labels=False) + 1
        max_r = rfm_scored['R_score'].max()
        rfm_scored['R_score'] = max_r - rfm_scored['R_score'] + 1

    # Для Frequency: більше = краще (5 балів)
    try:
        rfm_scored['F_score'] = pd.qcut(rfm_scored['frequency'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm_scored['F_score'] = pd.cut(rfm_scored['frequency'].rank(pct=True), bins=5, labels=False) + 1

    # Для Monetary: більше = краще (5 балів)
    try:
        rfm_scored['M_score'] = pd.qcut(rfm_scored['monetary'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm_scored['M_score'] = pd.cut(rfm_scored['monetary'].rank(pct=True), bins=5, labels=False) + 1

    rfm_scored['RFM_score'] = (rfm_scored['R_score'].astype(int) * 100 +
                                rfm_scored['F_score'].astype(int) * 10 +
                                rfm_scored['M_score'].astype(int))

    return rfm_scored

def segment_customers_rfm(rfm_scored):
    """Сегментація клієнтів за RFM"""
    def assign_segment(row):
        r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])

        # Специфіка для оптики
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP Клієнти"
        elif r >= 4 and f >= 3 and m >= 3:
            return "Лояльні"
        elif r >= 4 and f <= 2 and m >= 3:
            return "Нові Перспективні"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Сплячі VIP"
        elif r <= 2 and f >= 3 and m >= 3:
            return "В Зоні Ризику"
        elif r >= 3 and f == 2 and m <= 3:
            return "Потребують Уваги"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Втрачені"
        elif r >= 4 and f <= 2 and m <= 2:
            return "Новачки"
        else:
            return "Потенційні"

    rfm_scored['segment'] = rfm_scored.apply(assign_segment, axis=1)
    return rfm_scored

def kmeans_segmentation(rfm_df, n_clusters=5):
    """K-means кластеризація"""
    features = rfm_df[['recency', 'frequency', 'monetary']].copy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['cluster'] = kmeans.fit_predict(features_scaled)

    silhouette = silhouette_score(features_scaled, rfm_df['cluster'])

    return rfm_df, silhouette, kmeans

def calculate_clv(rfm_df, avg_margin=0.3, discount_rate=0.1, years=3):
    """Розрахунок Customer Lifetime Value (виправлена формула)"""
    # Середній чек
    avg_order = rfm_df['monetary'] / rfm_df['frequency']

    # Річна частота покупок (більш коректний розрахунок)
    # Якщо recency < 365, екстраполюємо; якщо > 365, використовуємо фактичну частоту
    days_period = rfm_df['recency'].clip(upper=365)
    annual_frequency = (rfm_df['frequency'] / days_period.clip(lower=1)) * 365
    annual_frequency = annual_frequency.clip(upper=365)  # Не більше 1 разу на день

    # CLV = (avg_order * annual_frequency * margin) * NPV за N років
    # Використовуємо формулу NPV для дисконтування майбутніх потоків
    clv = 0
    for year in range(1, years + 1):
        clv += (avg_order * annual_frequency * avg_margin) / ((1 + discount_rate) ** year)

    return clv

def generate_segment_insights(rfm_segmented, raw_data=None):
    """Автоматична генерація інсайтів для кожного сегменту"""
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

        # Події та рекомендації специфічні для оптики
        if segment == "VIP Клієнти":
            insight['events'] = [
                f"✅ {count} клієнтів приносять {segment_data['monetary'].sum():.0f} грн доходу",
                f"⏱️ Середня давність покупки: {avg_recency:.0f} днів",
                f"🔄 Купують в середньому {avg_frequency:.1f} раз"
            ]
            insight['recommendations'] = [
                "🎁 VIP-картки з ексклюзивними знижками 15-20%",
                "📱 Персональні нагадування про перевірку зору (раз на 6 місяців)",
                "👔 Запрошення на закриті презентації преміум-колекцій",
                "🎯 Програма раннього доступу до новинок",
                "💎 Персональний менеджер та пріоритетне обслуговування"
            ]
            insight['priority'] = "🔥 ВИСОКИЙ - максимальне утримання"
            
        elif segment == "Лояльні":
            insight['events'] = [
                f"✅ Стабільні {count} клієнтів з середнім чеком {avg_monetary:.0f} грн",
                f"📊 Регулярність покупок: кожні {avg_recency:.0f} днів"
            ]
            insight['recommendations'] = [
                "🎯 Програма лояльності: 1 грн = 1 бонус",
                "👓 Запропонувати другу пару окулярів зі знижкою 30%",
                "☀️ Акція на сонцезахисні окуляри в сезон",
                "👨‍👩‍👧 Сімейні пропозиції: знижка при купівлі від 3-х пар",
                "📧 Email-розсилка з новинками раз на місяць"
            ]
            insight['priority'] = "🔥 ВИСОКИЙ - розвиток"

        elif segment == "Нові Перспективні":
            insight['events'] = [
                f"🆕 {count} нових клієнтів з високим першим чеком ({avg_monetary:.0f} грн)",
                f"⏳ Недавня перша покупка: {avg_recency:.0f} днів тому"
            ]
            insight['recommendations'] = [
                "🎁 Welcome-бонус 500 грн на другу покупку",
                "📱 SMS через 3 місяці: 'Як Ваші окуляри? Перевірка зору безкоштовно'",
                "👓 Пропозиція лінз із захистом від синього світла",
                "💳 Оформлення картки лояльності з вітальною знижкою",
                "📞 Зворотний зв'язок через тиждень після покупки"
            ]
            insight['priority'] = "⚡ СЕРЕДНІЙ - швидка активація"

        elif segment == "Сплячі VIP":
            days_since = avg_recency
            insight['events'] = [
                f"⚠️ {count} цінних клієнтів не купують {days_since:.0f} днів!",
                f"💰 Потенційна втрата доходу: {segment_data['monetary'].sum():.0f} грн",
                f"📉 Ризик відтоку високоцінних клієнтів"
            ]
            insight['recommendations'] = [
                "🚨 ТЕРМІНОВО: Персональний дзвінок з пропозицією VIP-знижки 25%",
                "🔬 Безкоштовна перевірка зору + діагностика у подарунок",
                "🎁 Ексклюзивна пропозиція: нова оправа + лінзи -30%",
                "👨‍⚕️ Нагадування: 'Минув більше року, рекомендуємо перевірку'",
                "💎 Повернення VIP-статусу при покупці протягом 30 днів"
            ]
            insight['priority'] = "🔴 КРИТИЧНИЙ - реактивація"

        elif segment == "В Зоні Ризику":
            insight['events'] = [
                f"⚠️ {count} клієнтів давно не купували ({avg_recency:.0f} днів)",
                f"💸 Середній чек був {avg_monetary:.0f} грн"
            ]
            insight['recommendations'] = [
                "📧 Email-кампанія: 'Ми сумуємо! Знижка 20% на будь-яку покупку'",
                "👓 Акція trade-in: здай старі окуляри, отримай знижку 15%",
                "🎯 Ремаркетинг у соцмережах з персональними пропозиціями",
                "📱 SMS: 'Час оновити окуляри? Спеціальна ціна для Вас'",
                "🔬 Безкоштовна перевірка зору як привід повернутись"
            ]
            insight['priority'] = "🔴 КРИТИЧНИЙ - термінова реактивація"
            
        elif segment == "Потребують Уваги":
            insight['events'] = [
                f"📊 {count} клієнтів з потенціалом зростання",
                f"💡 Низька частота: {avg_frequency:.1f} покупок"
            ]
            insight['recommendations'] = [
                "🎁 Програма стимулювання: купи 2 - отримай знижку 15% на 3-ю",
                "👓 Пропозиція аксесуарів: футляри, серветки, ланцюжки",
                "📧 Освітній контент: 'Як вибрати окуляри для комп'ютера'",
                "☀️ Допродаж: сонцезахисні окуляри зі знижкою 25%",
                "💳 Бонуси за кожну покупку: 10% повернення балами"
            ]
            insight['priority'] = "⚡ СЕРЕДНІЙ - розвиток частоти"

        elif segment == "Втрачені":
            insight['events'] = [
                f"❌ {count} клієнтів давно пішли (>{avg_recency:.0f} днів)",
                f"💸 Низький LTV: {avg_monetary:.0f} грн"
            ]
            insight['recommendations'] = [
                "🔄 Win-back кампанія: знижка 30% на повернення",
                "📞 Холодний обдзвон з опитуванням: 'Чому перестали купувати?'",
                "🎁 Останній шанс: промокод на 40% знижку (термін 14 днів)",
                "📊 Аналіз причин відтоку - покращення сервісу",
                "⚠️ Якщо немає реакції - виключити з активної бази"
            ]
            insight['priority'] = "🟡 НИЗЬКИЙ - оцінка доцільності"

        elif segment == "Новачки":
            insight['events'] = [
                f"🆕 {count} нових клієнтів з низьким першим чеком",
                f"💡 Середній чек: {avg_monetary:.0f} грн (низький)"
            ]
            insight['recommendations'] = [
                "📚 Навчання: 'Як вибрати якісні окуляри'",
                "🎁 Купон на знижку 15% на наступну покупку",
                "👓 Пропозиція апгрейду лінз зі знижкою",
                "📱 Підписка на корисний контент про здоров'я очей",
                "💳 Картка лояльності + стартові 200 бонусів"
            ]
            insight['priority'] = "⚡ СЕРЕДНІЙ - залучення"

        else:  # Потенційні
            insight['events'] = [
                f"📊 {count} клієнтів з середніми показниками",
                f"💡 Можливість зростання в інші сегменти"
            ]
            insight['recommendations'] = [
                "🎯 А/B тестування різних пропозицій",
                "📧 Регулярні email з персональними пропозиціями",
                "🎁 Сезонні акції та розпродажі",
                "👓 Крос-сел: аксесуари, розчини для лінз",
                "📊 Моніторинг поведінки для переходу в активний сегмент"
            ]
            insight['priority'] = "🟢 НОРМАЛЬНИЙ - стандартний підхід"
        
        insights[segment] = insight
    
    return insights

def generate_business_conclusions(rfm_segmented, insights):
    """Загальні бізнес-висновки"""
    total_clients = len(rfm_segmented)
    total_revenue = rfm_segmented['monetary'].sum()

    vip_count = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клієнти', 'Лояльні'])])
    vip_revenue = rfm_segmented[rfm_segmented['segment'].isin(['VIP Клієнти', 'Лояльні'])]['monetary'].sum()

    at_risk_count = len(rfm_segmented[rfm_segmented['segment'].isin(['Сплячі VIP', 'В Зоні Ризику', 'Втрачені'])])
    at_risk_revenue = rfm_segmented[rfm_segmented['segment'].isin(['Сплячі VIP', 'В Зоні Ризику'])]['monetary'].sum()

    conclusions = {
        'summary': [
            f"📊 Всього клієнтів: {total_clients}",
            f"💰 Загальна виручка: {total_revenue:,.0f} грн",
            f"💳 Середній чек: {total_revenue/total_clients:,.0f} грн"
        ],
        'key_findings': [
            f"🌟 Топ-клієнти: {vip_count} ({vip_count/total_clients*100:.1f}%) приносять {vip_revenue:,.0f} грн ({vip_revenue/total_revenue*100:.1f}% виручки)",
            f"⚠️ В зоні ризику: {at_risk_count} клієнтів ({at_risk_count/total_clients*100:.1f}%) з потенційною втратою {at_risk_revenue:,.0f} грн",
            f"🎯 Пріоритет #1: Утримання VIP та реактивація сплячих VIP-клієнтів"
        ],
        'strategic_actions': [
            "🔥 НЕГАЙНО: Запустити програму реактивації для 'Сплячі VIP' (персональні дзвінки)",
            "💎 Створити VIP-клуб з ексклюзивними умовами для топ-20% клієнтів",
            "🎯 Розробити welcome-сценарій для нових клієнтів (перші 90 днів)",
            "📊 Впровадити систему предиктивної аналітики відтоку",
            "🔄 Автоматизувати тригерні розсилки по сегментах"
        ],
        'expected_impact': [
            f"📈 Очікуване збільшення виручки: +8-12% при правильній роботі з сегментами",
            f"🎯 Зниження відтоку VIP на 3-5% = економія ~{at_risk_revenue*0.04:,.0f} грн/рік",
            f"💰 Зростання середнього чеку на 5-7% = додатково ~{total_revenue*0.06:,.0f} грн/рік"
        ]
    }

    return conclusions

# ==================== ОСНОВНОЙ КОД ====================

def main():
    st.title("📊 Аналітичний звіт RFM: Оптика")
    st.markdown("#### Стратегічний аналіз клієнтської бази та рекомендації з управління")

    # Додаємо інформаційний блок
    st.info("""
    **Призначення звіту:** Аналіз клієнтської бази за методологією RFM (Recency, Frequency, Monetary)
    для прийняття стратегічних рішень з управління взаємовідносинами з клієнтами та збільшення прибутковості бізнесу.
    """)

    # Sidebar - завантаження даних
    st.sidebar.header("📥 Завантаження даних")

    data_source = st.sidebar.radio(
        "Джерело даних:",
        ["Excel файл", "Google Sheets"]
    )
    
    df = None
    
    if data_source == "Excel файл":
        uploaded_file = st.sidebar.file_uploader(
            "Завантажте Excel файл",
            type=['xlsx', 'xls']
        )

        if uploaded_file:
            df, error = load_excel(uploaded_file)
            if error:
                st.sidebar.error(error)

    else:  # Google Sheets
        st.sidebar.markdown("**Для підключення Google Sheets:**")
        st.sidebar.info(
            "1. Створіть Service Account в Google Cloud\n"
            "2. Завантажте JSON ключ\n"
            "3. Надайте доступ до таблиці для email з JSON"
        )

        sheet_url = st.sidebar.text_input("URL Google Sheets:")
        credentials_file = st.sidebar.file_uploader(
            "JSON ключ від Service Account",
            type=['json']
        )

        if sheet_url and credentials_file:
            try:
                credentials_json = json.load(credentials_file)
                df, error = load_google_sheet(sheet_url, credentials_json)
                if error:
                    st.sidebar.error(error)
            except Exception as e:
                st.sidebar.error(f"Помилка читання credentials: {str(e)}")
    
    # Інформація про поля
    with st.sidebar.expander("ℹ️ Структура даних"):
        st.markdown("""
        **Обов'язкові поля:**
        - `client_id` - ID клієнта
        - `transaction_id` - ID транзакції
        - `transaction_date` - Дата купівлі
        - `transaction_amount` - Сума купівлі

        **Опціональні поля:**
        - `client_name` - ПІБ клієнта (рекомендується)
        - `product_category` - Категорія (оправи/лінзи/сонцезахисні/аксесуари)
        - `sales_channel` - Канал (онлайн/офлайн)
        - `store_id` - ID магазину
        - `loyalty_points` - Бали лояльності
        - `age` - Вік
        - `gender` - Стать
        - `city` - Місто

        **Примітка:** При наявності поля `client_name` воно буде відображатись у всіх таблицях для зручної ідентифікації клієнтів.
        """)
    
    if df is not None:
        # Валідація даних
        required_fields = ['client_id', 'transaction_id', 'transaction_date', 'transaction_amount']
        is_valid, message = validate_data(df, required_fields)
        
        if not is_valid:
            st.error(message)
            st.stop()
        
        # Конвертация даты
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Показуємо превʼю даних
        st.success(f"✅ Завантажено {len(df)} транзакцій від {df['client_id'].nunique()} клієнтів")
        
        with st.expander("👀 Превʼю даних"):
            st.dataframe(df.head(10))
        
        # Параметри аналізу
        st.sidebar.header("⚙️ Параметри аналізу")
        
        analysis_date = st.sidebar.date_input(
            "Дата аналізу:",
            value=df['transaction_date'].max().date()
        )
        analysis_date = pd.Timestamp(analysis_date)
        
        n_clusters = st.sidebar.slider(
            "Кількість кластерів K-means:",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Расчет RFM
        with st.spinner("🔄 Розрахунок RFM метрик..."):
            rfm = calculate_rfm(df, analysis_date)
            rfm_scored = create_rfm_scores(rfm)
            rfm_segmented = segment_customers_rfm(rfm_scored)
            rfm_clustered, silhouette, kmeans = kmeans_segmentation(rfm.copy(), n_clusters)
            
            # CLV
            rfm_segmented['clv'] = calculate_clv(rfm_segmented)
        
        # Генерация инсайтов
        with st.spinner("🤖 Генерація автоматичних інсайтів..."):
            insights = generate_segment_insights(rfm_segmented, df)
            conclusions = generate_business_conclusions(rfm_segmented, insights)
        
        # ==================== ВИВЕДЕННЯ РЕЗУЛЬТАТІВ ====================
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Executive Summary",
            "🎯 Сегменти RFM",
            "👥 Детальний аналіз клієнтів",
            "🔬 K-means Кластери",
            "💎 CLV Аналіз",
            "📋 Стратегічні рекомендації"
        ])
        
        # TAB 1: Executive Summary
        with tab1:
            st.header("📊 Executive Summary")
            st.markdown("### Ключові показники бізнесу")

            # Ключові метрики
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всього клієнтів", f"{len(rfm):,}")
                st.caption("Унікальних клієнтів у базі")
            with col2:
                total_revenue = rfm['monetary'].sum()
                st.metric("Загальна виручка", f"{total_revenue:,.0f} грн")
                st.caption("Сукупний дохід")
            with col3:
                avg_revenue = rfm['monetary'].mean()
                st.metric("Середній LTV", f"{avg_revenue:,.0f} грн")
                st.caption("На одного клієнта")
            with col4:
                avg_freq = rfm['frequency'].mean()
                st.metric("Середня частота", f"{avg_freq:.1f}")
                st.caption("Покупок на клієнта")

            st.markdown("---")

            # Сегментация клієнтів
            st.markdown("### Сегментація клієнтської бази")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Таблиця по сегментах
                segment_summary = rfm_segmented.groupby('segment').agg({
                    'client_id': 'count',
                    'monetary': 'sum'
                }).round(0)
                segment_summary.columns = ['Кількість клієнтів', 'Загальна виручка (грн)']
                segment_summary['Частка клієнтів (%)'] = (segment_summary['Кількість клієнтів'] / len(rfm_segmented) * 100).round(1)
                segment_summary['Частка виручки (%)'] = (segment_summary['Загальна виручка (грн)'] / total_revenue * 100).round(1)
                segment_summary = segment_summary.sort_values('Загальна виручка (грн)', ascending=False)

                st.dataframe(segment_summary, use_container_width=True)

            with col2:
                # Графік розподілу виручки
                fig = px.pie(
                    segment_summary.reset_index(),
                    values='Загальна виручка (грн)',
                    names='segment',
                    title='Розподіл виручки по сегментах',
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Критичні інсайти
            st.markdown("### 🎯 Критичні інсайти")

            vip_count = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клієнти', 'Лояльні'])])
            vip_revenue = rfm_segmented[rfm_segmented['segment'].isin(['VIP Клієнти', 'Лояльні'])]['monetary'].sum()
            at_risk_count = len(rfm_segmented[rfm_segmented['segment'].isin(['Сплячі VIP', 'В Зоні Ризику'])])
            at_risk_revenue = rfm_segmented[rfm_segmented['segment'].isin(['Сплячі VIP', 'В Зоні Ризику'])]['monetary'].sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"**✅ VIP клієнти**")
                st.metric("Кількість", vip_count, f"{vip_count/len(rfm_segmented)*100:.1f}% бази")
                st.metric("Виручка", f"{vip_revenue:,.0f} грн", f"{vip_revenue/total_revenue*100:.1f}% частки")

            with col2:
                st.warning(f"**⚠️ В зоні ризику**")
                st.metric("Кількість", at_risk_count, f"{at_risk_count/len(rfm_segmented)*100:.1f}% бази")
                st.metric("Потенційна втрата", f"{at_risk_revenue:,.0f} грн")

            with col3:
                st.info(f"**📈 Потенціал зростання**")
                potential_increase = total_revenue * 0.10  # 10% рост при правильной работе
                st.metric("Прогноз при оптимізації", f"+{potential_increase:,.0f} грн")
                st.metric("Зростання виручки", "+8-12%")

            st.markdown("---")

            # Динамика активности
            st.markdown("### 📈 Розподіл метрик")

            col1, col2, col3 = st.columns(3)

            with col1:
                fig = px.histogram(rfm, x='recency', nbins=50,
                                 title='Recency (давність покупки)',
                                 labels={'recency': 'Днів з останньої покупки'},
                                 color_discrete_sequence=['#636EFA'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(rfm, x='frequency', nbins=30,
                                 title='Frequency (частота покупок)',
                                 labels={'frequency': 'Кількість покупок'},
                                 color_discrete_sequence=['#EF553B'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                fig = px.histogram(rfm, x='monetary', nbins=50,
                                 title='Monetary (сума покупок)',
                                 labels={'monetary': 'Виручка (грн)'},
                                 color_discrete_sequence=['#00CC96'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Сегменти RFM
        with tab2:
            st.header("🎯 RFM Сегменты")
            
            # Статистика по сегментах
            segment_stats = rfm_segmented.groupby('segment').agg({
                'client_id': 'count',
                'monetary': 'sum',
                'recency': 'mean',
                'frequency': 'mean',
                'clv': 'mean'
            }).round(0)
            segment_stats.columns = ['Кількість', 'Виручка', 'Серед. Recency', 'Серед. Frequency', 'Серед. CLV']
            segment_stats = segment_stats.sort_values('Виручка', ascending=False)
            
            st.dataframe(segment_stats, use_container_width=True)
            
            # Візуалізація сегментів
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    rfm_segmented,
                    names='segment',
                    title='Розподіл клієнтів по сегментах',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                segment_revenue = rfm_segmented.groupby('segment')['monetary'].sum().reset_index()
                fig = px.bar(
                    segment_revenue,
                    x='segment',
                    y='monetary',
                    title='Виручка по сегментам',
                    labels={'monetary': 'Виручка (грн)', 'segment': 'Сегмент'}
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
                    title='Сегменти: Recency vs Monetary',
                    labels={
                        'recency': 'Recency (дні)',
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
                    title='Сегменти: Frequency vs Monetary',
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
            st.markdown("### 🔥 Heat Map: розподіл RFM Score")

            # Створюємо pivot таблицю для heat map
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
                colorbar=dict(title="Кількість клієнтів")
            ))
            fig.update_layout(
                title='Розподіл клієнтів по R та F Score',
                xaxis_title='Frequency Score',
                yaxis_title='Recency Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Box plot по сегментах
            st.markdown("---")
            st.markdown("### 📦 Розподіл метрик по сегментам")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.box(
                    rfm_segmented,
                    x='segment',
                    y='monetary',
                    title='Розподіл Monetary по сегментах',
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
                    title='Розподіл Frequency по сегментах',
                    labels={'frequency': 'Frequency', 'segment': 'Сегмент'},
                    color='segment'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # АВТОМАТИЧНІ ІНСАЙТИ ПО СЕГМЕНТАХ
            st.markdown("---")
            st.header("🤖 Автоматичні інсайти та рекомендації")

            # Таблица распределения клієнтів по сегментам
            st.subheader("📋 Розподіл клієнтів по сегментах")

            # Підготовка даних для таблиці
            client_segments = rfm_segmented[['client_id', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'clv']].copy()

            # Якщо є поле ПІБ клієнта в вихідних даних, додаємо його
            if 'client_name' in df.columns:
                client_names = df[['client_id', 'client_name']].drop_duplicates()
                client_segments = client_segments.merge(client_names, on='client_id', how='left')
                client_segments = client_segments[['client_id', 'client_name', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'clv']]

            # Форматування для відображення
            client_segments_display = client_segments.copy()
            client_segments_display['monetary'] = client_segments_display['monetary'].round(0)
            client_segments_display['clv'] = client_segments_display['clv'].round(0)

            # Показуємо зведення по сегментах
            col1, col2, col3 = st.columns(3)
            with col1:
                top_segment = rfm_segmented['segment'].value_counts().index[0]
                st.metric("Найбільший сегмент", top_segment, f"{rfm_segmented['segment'].value_counts().values[0]} клієнтів")
            with col2:
                high_priority = len(rfm_segmented[rfm_segmented['segment'].isin(['VIP Клієнти', 'Лояльні'])])
                st.metric("Пріоритетні клієнти", high_priority, f"{high_priority/len(rfm_segmented)*100:.1f}%")
            with col3:
                at_risk = len(rfm_segmented[rfm_segmented['segment'].isin(['Сплячі VIP', 'В Зоні Ризику'])])
                st.metric("В зоні ризику", at_risk, f"{at_risk/len(rfm_segmented)*100:.1f}%")

            st.markdown("**Повна таблиця клієнтів:**")
            st.dataframe(
                client_segments_display,
                use_container_width=True,
                height=400
            )

            # Сортуємо сегменти за пріоритетом
            priority_order = {
                "🔴 КРИТИЧНИЙ - реактивація": 1,
                "🔴 КРИТИЧНИЙ - термінова реактивація": 2,
                "🔥 ВИСОКИЙ - максимальне утримання": 3,
                "🔥 ВИСОКИЙ - розвиток": 4,
                "⚡ СЕРЕДНІЙ - швидка активація": 5,
                "⚡ СЕРЕДНІЙ - розвиток частоти": 6,
                "⚡ СЕРЕДНІЙ - залучення": 7,
                "🟢 НОРМАЛЬНИЙ - стандартний підхід": 8,
                "🟡 НИЗЬКИЙ - оцінка доцільності": 9
            }
            
            sorted_segments = sorted(
                insights.items(),
                key=lambda x: priority_order.get(x[1]['priority'], 99)
            )
            
            for segment, insight in sorted_segments:
                with st.expander(f"**{segment}** - {insight['count']} клієнтів | {insight['priority']}"):
                    
                    # Метрики сегменту
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Клієнтів", f"{insight['count']}")
                    with col2:
                        st.metric("Серед. Recency", f"{insight['avg_recency']:.0f} днів")
                    with col3:
                        st.metric("Серед. Frequency", f"{insight['avg_frequency']:.1f}")
                    with col4:
                        st.metric("Серед. Monetary", f"{insight['avg_monetary']:,.0f} грн")
                    
                    # События
                    st.markdown("**📌 Ключові події:**")
                    for event in insight['events']:
                        st.markdown(f"- {event}")
                    
                    st.markdown("")
                    
                    # Рекомендации
                    st.markdown("**💡 Рекомендації:**")
                    for rec in insight['recommendations']:
                        st.markdown(f"- {rec}")

        # TAB 3: Детальний аналіз клієнтів (с фильтрами)
        with tab3:
            st.header("👥 Детальний аналіз клієнтів")
            st.markdown("Інтерактивна таблиця з можливістю фільтрації за різними параметрами")

            # Підготовка даних
            detailed_df = rfm_segmented.copy()

            # Додаємо ПІБ якщо є
            if 'client_name' in df.columns:
                client_names = df[['client_id', 'client_name']].drop_duplicates()
                detailed_df = detailed_df.merge(client_names, on='client_id', how='left')

            # Фільтри
            st.markdown("### Фільтри")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                segment_filter = st.multiselect(
                    "Сегмент:",
                    options=sorted(detailed_df['segment'].unique()),
                    default=None,
                    placeholder="Всі сегменти"
                )

            with col2:
                recency_range = st.slider(
                    "Recency (дні):",
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

            # Застосовуємо фільтри
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

            # Статистика після фільтрації
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Клієнтів после фильтрации", f"{len(filtered_df):,}")
            with col2:
                st.metric("Загальна виручка", f"{filtered_df['monetary'].sum():,.0f} грн")
            with col3:
                st.metric("Середній чек", f"{filtered_df['monetary'].mean():,.0f} грн")
            with col4:
                st.metric("Середній CLV", f"{filtered_df['clv'].mean():,.0f} грн")

            st.markdown("---")

            # Сортування
            sort_options = {
                'CLV (спадання)': ('clv', False),
                'CLV (зростання)': ('clv', True),
                'Monetary (спадання)': ('monetary', False),
                'Monetary (зростання)': ('monetary', True),
                'Recency (спадання)': ('recency', False),
                'Recency (зростання)': ('recency', True),
                'Frequency (спадання)': ('frequency', False),
                'Frequency (зростання)': ('frequency', True),
                'RFM Score (спадання)': ('RFM_score', False),
                'RFM Score (зростання)': ('RFM_score', True)
            }

            sort_by = st.selectbox("Сортувати за:", list(sort_options.keys()), index=0)
            sort_col, sort_asc = sort_options[sort_by]
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

            # Відображення таблиці
            st.markdown("### Таблиця клієнтів")

            # Вибір колонок для відображення
            display_columns = ['client_id', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score', 'R_score', 'F_score', 'M_score', 'clv']
            if 'client_name' in filtered_df.columns:
                display_columns = ['client_id', 'client_name'] + display_columns[1:]

            # Форматування
            display_df = filtered_df[display_columns].copy()
            display_df['monetary'] = display_df['monetary'].round(0)
            display_df['clv'] = display_df['clv'].round(0)

            # Відображаємо з можливістю вибору рядків
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )

            # Експорт відфільтрованих даних
            st.markdown("---")
            st.markdown("### Експорт даних")

            col1, col2 = st.columns(2)

            with col1:
                # CSV экспорт
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Завантажити CSV",
                    data=csv,
                    file_name=f"filtered_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

            with col2:
                # Excel експорт
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    display_df.to_excel(writer, sheet_name='Filtered_Clients', index=False)

                st.download_button(
                    label="📥 Завантажити Excel",
                    data=buffer.getvalue(),
                    file_name=f"filtered_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

        # TAB 4: K-means кластеры
        with tab4:
            st.header("🔬 K-means Кластеризація")
            
            st.info(f"Silhouette Score: {silhouette:.3f}")
            
            # Статистика по кластерах
            cluster_stats = rfm_clustered.groupby('cluster').agg({
                'client_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean'
            }).round(0)
            cluster_stats.columns = ['Кількість', 'Серед. Recency', 'Серед. Frequency', 'Серед. Monetary']
            
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
                    title='Кластери: Recency vs Monetary',
                    labels={
                        'recency': 'Recency (дні)',
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
                    title='Кластери: Frequency vs Monetary',
                    labels={
                        'frequency': 'Frequency',
                        'monetary': 'Monetary (грн)',
                        'cluster': 'Кластер'
                    },
                    hover_data=['client_id', 'recency']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: CLV Аналіз
        with tab5:
            st.header("💎 Customer Lifetime Value (CLV)")
            
            # Топ клієнти за CLV
            top_clv = rfm_segmented.nlargest(20, 'clv')[['client_id', 'segment', 'monetary', 'frequency', 'clv']]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🏆 Топ-20 клієнтів за CLV")
                st.dataframe(
                    top_clv.style.format({
                        'monetary': '{:,.0f} грн',
                        'clv': '{:,.0f} грн'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Метрики CLV
                st.metric("Середній CLV", f"{rfm_segmented['clv'].mean():,.0f} грн")
                st.metric("Медіанний CLV", f"{rfm_segmented['clv'].median():,.0f} грн")
                st.metric("Топ-10% CLV", f"{rfm_segmented['clv'].quantile(0.9):,.0f} грн")
            
            # CLV по сегментам
            fig = px.box(
                rfm_segmented,
                x='segment',
                y='clv',
                title='Розподіл CLV по сегментах',
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
                title='Парето аналіз CLV (правило 80/20)',
                xaxis_title='% клієнтів',
                yaxis_title='% кумулятивного CLV'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Скільки клієнтів дають 80% виручки
            clients_80 = rfm_sorted[rfm_sorted['cumulative_clv_pct'] <= 80]
            st.info(f"📊 **{len(clients_80)} клієнтів ({len(clients_80)/len(rfm_sorted)*100:.1f}%) генерують 80% прогнозованої виручки**")
        
        # TAB 6: Стратегічні рекомендації
        with tab6:
            st.header("📋 Стратегічні рекомендації")
            
            # Summary
            st.subheader("📊 Підсумкова статистика")
            for item in conclusions['summary']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Key Findings
            st.subheader("🔍 Ключові знахідки")
            for item in conclusions['key_findings']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Strategic Actions
            st.subheader("🎯 Стратегічні дії")
            for item in conclusions['strategic_actions']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Expected Impact
            st.subheader("📈 Очікуваний ефект")
            for item in conclusions['expected_impact']:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # Action Plan
            st.subheader("📅 План действий на ближайшие 30 днів")
            
            action_plan = pd.DataFrame({
                'Тиждень': ['1', '1-2', '2-3', '3-4', '4+'],
                'Дія': [
                    'Реактивація Сплячих VIP: персональні дзвінки + VIP-предложения',
                    'Запуск welcome-програми для Нових Перспективних клієнтів',
                    'Email-кампанії для сегменту "В Зоні Ризику"',
                    'Тестування програми лояльності для Лояльних клієнтів',
                    'Аналіз результатів, коригування стратегії, масштабування'
                ],
                'Відповідальний': [
                    'Менеджер з роботи з VIP',
                    'CRM-маркетолог',
                    'Email-маркетолог',
                    'Керівник відділу маркетингу',
                    'Вся команда'
                ],
                'KPI': [
                    'Конверсія дзвінків >15%',
                    'Повторна покупка >25%',
                    'Open rate >30%, конверсія >5%',
                    'Участь у програмі >40%',
                    'Загальне зростання виручки >8%'
                ]
            })
            
            st.dataframe(action_plan, use_container_width=True)
            
            # Завантаження звіту
            st.markdown("---")
            st.subheader("💾 Експорт даних")
            
            # Підготовка даних для экспорта
            export_df = rfm_segmented.merge(
                rfm_clustered[['client_id', 'cluster']],
                on='client_id'
            )
            
            # Excel експорт
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='RFM_Segments', index=False)
                segment_stats.to_excel(writer, sheet_name='Segment_Stats')
                cluster_stats.to_excel(writer, sheet_name='Cluster_Stats')
            
            st.download_button(
                label="📥 Завантажити повний звіт (Excel)",
                data=buffer.getvalue(),
                file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    else:
        st.info("👆 Завантажте дані в бічній панелі для початку аналізу")
        
        # Приклад структури даних
        st.subheader("📋 Приклад структури даних")

        st.markdown("""
        Завантажте Excel файл з наступною структурою. Обовʼязкові поля виділені **жирным**.
        """)

        example_data = pd.DataFrame({
            'client_id': [1001, 1001, 1002, 1003, 1003],
            'client_name': ['Иванов И.И.', 'Иванов И.И.', 'Петрова А.С.', 'Сидоров П.К.', 'Сидоров П.К.'],
            'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'transaction_date': ['2024-01-15', '2024-06-20', '2024-03-10', '2024-02-05', '2024-11-12'],
            'transaction_amount': [2500, 1800, 3200, 4500, 2200],
            'product_category': ['Оправи', 'Сонцезахисні', 'Оправи + Лінзи', 'Преміум оправи', 'Лінзи'],
            'sales_channel': ['Офлайн', 'Онлайн', 'Офлайн', 'Офлайн', 'Онлайн']
        })

        st.dataframe(example_data)

        st.markdown("""
        **Примітка:**
        - Поле `client_name` опціонально, але рекомендується для зручності роботи зі звітами
        - Дата повинна бути у форматі YYYY-MM-DD або DD.MM.YYYY
        - Сума транзакції - число без валюти
        """)

if __name__ == "__main__":
    main()