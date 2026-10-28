import json
from pathlib import Path
from typing import List
import streamlit as st
from pydantic import BaseModel, Field

from src.config import settings, LEGAL_DISCLAIMER, PROCESSED_DATA_DIR
from src.nlp.grls_resolver import DrugResolver


# Схема валидации входных данных пациента
class PatientProfile(BaseModel):
    diagnoses: List[str] = Field(..., description="Список выбранных диагнозов по МКБ-10")
    medications: List[str] = Field(..., description="Список выбранных лекарственных препаратов")


st.set_page_config(
    page_title="Medical Assistant | Clinical Verification",
    layout="wide"
)


# Загрузка справочника МКБ-10 из локального хранилища
@st.cache_data
def load_icd10_options() -> List[str]:
    file_path = PROCESSED_DATA_DIR / "icd10.json"
    options = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                nodes = data.get("nodes", [])
                for node in nodes:
                    if node.get("level") in ["category", "subcategory"]:
                        code = node.get("code", "")
                        name = node.get("name", "")
                        options.append(f"{code} — {name}")
        except Exception as e:
            print(f"ICD-10 load error: {e}")
    
    if not options:
        options = [
            "E10 — Инсулинзависимый сахарный диабет",
            "I10 — Эссенциальная (первичная) гипертензия"
        ]
    return options


# Инициализация DrugResolver с обработкой ошибок доступа к файлу
@st.cache_resource
def load_drug_resolver() -> DrugResolver | None:
    parquet_path = PROCESSED_DATA_DIR / "grls_normalized.parquet"
    if not parquet_path.exists():
        print(f"GRLS parquet not found at: {parquet_path.resolve()}")
        return None
    try:
        return DrugResolver(parquet_path)
    except Exception as e:
        print(f"DrugResolver init error: {e}")
        return None


# Инициализация справочных данных
icd10_options = load_icd10_options()
drug_resolver = load_drug_resolver()

# Генерация опций для селекта: объединяем торговое название и МНН
drug_options = []
if drug_resolver and hasattr(drug_resolver, "df"):
    try:
        unique_drugs = drug_resolver.df.unique(subset=["trade_name_norm", "inn_norm"])
        for row in unique_drugs.iter_rows(named=True):
            trade = row.get("trade_name_norm", "")
            inn = row.get("inn_norm", "")
            if trade:
                if inn and inn != trade:
                    drug_options.append(f"{trade} (МНН: {inn})")
                else:
                    drug_options.append(str(trade))
        drug_options = sorted(list(set(drug_options)))
    except Exception as e:
        print(f"Drug options extraction error: {e}")

if not drug_options:
    drug_options = [
        "берлиприл 10 (МНН: эналаприл)",
        "аторвастатин",
        "бисопролол"
    ]

# UI стилизация кнопок
st.markdown("""
    <style>
    .stButton button {
        background-color: #0071e3;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #0077ed;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель: конфигурация и статус бэкенда
with st.sidebar:
    st.markdown("### Системный монитор")
    st.markdown("---")
    st.caption("Параметры окружения:")
    st.code(f"URI: {settings.neo4j_uri}\nUser: {settings.neo4j_user}", language="text")
    
    db_status = st.toggle("Имитация соединения с Neo4j", value=True)
    if db_status:
        st.success("Neo4j: Подключено (Bolt)")
    else:
        st.error("Neo4j: Отключено")
        
    st.markdown("---")
    st.caption("Интерфейс верификации клинических данных.")

# Основной макет страницы
st.title("Система верификации клинических данных")
st.caption("Модуль ввода диагнозов, связывания с ГРЛС и графового анализа.")

with st.container(border=True):
    st.markdown(f"**Юридическое уведомление:** {LEGAL_DISCLAIMER}")

st.markdown("")

col_form, col_agents = st.columns([1, 1.2], gap="large")

with col_form:
    with st.container(border=True):
        st.subheader("Клинический ввод")
        st.caption("Ввод параметров пациента для проверки.")
        
        with st.form("patient_form"):
            selected_diagnoses = st.multiselect(
                "Диагнозы (МКБ-10)",
                options=icd10_options,
                placeholder="Введите код или название нозологии",
                help="Данные из icd10.json"
            )
            
            selected_medications = st.multiselect(
                "Лекарственные препараты (ГРЛС)",
                options=drug_options,
                placeholder="Выберите или начните ввод препарата",
                help="Данные из grls_normalized.parquet"
            )
            
            st.markdown("")
            submitted = st.form_submit_button("Проверить назначения")

    # Интерактивный вывод результатов резолвера в реальном времени
    if selected_medications and drug_resolver:
        with st.container(border=True):
            st.markdown("**Разбор DrugResolver:**")
            for med in selected_medications:
                clean_med = med.split(" (МНН:")[0]
                res = drug_resolver.search(clean_med)
                
                with st.expander(f"Результат для: {clean_med} (Найдено: {len(res.matches)})"):
                    if res.matches:
                        for m in res.matches[:5]:
                            st.markdown(
                                f"- **Торговое название:** {m.trade_name}  \n"
                                f"  **МНН:** `{m.inn or 'н/д'}`  \n"
                                f"  **АТХ:** `{', '.join(m.atx_codes) if m.atx_codes else 'нет'}`"
                            )
                    else:
                        st.warning("Совпадений в базе не найдено.")

    # Валидация и сериализация формы при сабмите
    if submitted:
        try:
            profile = PatientProfile(
                diagnoses=selected_diagnoses,
                medications=selected_medications
            )
            with st.container(border=True):
                st.success("Валидация Pydantic успешна.")
                st.markdown("**JSON профиля:**")
                st.code(profile.model_dump_json(indent=4), language="json")
        except Exception as e:
            st.error(f"Ошибка валидации: {e}")

with col_agents:
    with st.container(border=True):
        st.subheader("Граф связей базы знаний")
        st.caption("Визуализация нозологий и фармакологических связей (Neo4j).")
        
        st.info("Ожидание данных графа.")
        st.code(
            "[\n"
            "  { Диагноз: МКБ-10 } ─── (противопоказание) ─── { Препарат },\n"
            "  { Препарат } ─── (синергия / метаболизм) ─── { Препарат }\n"
            "]", 
            language="json"
        )
        st.markdown("")

    with st.container(border=True):
        st.subheader("Вердикт агентов")
        st.caption("Анализ безопасности и межлекарственных взаимодействий.")
        
        st.markdown(
            "> **Статус:** Ожидание входных данных.\n\n"
            "* Мультиагентный модуль проверит конфликты и дублирование терапии после подтверждения формы."
        )