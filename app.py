import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from transformers import pipeline


# 모델 불러오기
@st.cache_resource
def load_ml_model():
    return joblib.load("sleep_model.pkl")


model = load_ml_model()


# 로컬 LLM 불러오기
@st.cache_resource
def load_llm():
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct"
    )
    return generator


# 페이지 설정
st.set_page_config(
    page_title="수면의 질 예측",
    page_icon="🌙"
)

st.title("🌙 AI 수면의 질 예측")

st.write(
    "생활습관 정보를 입력하면 머신러닝 모델이 "
    "수면의 질을 예측하고 주요 영향 요인을 분석합니다."
)


# 사용자 입력
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)

sleep_duration_hrs = st.slider("수면 시간 (시간)", 1.0, 12.0, 7.0, 0.1)

sleep_latency_mins = st.slider("잠드는 데 걸리는 시간 (분)", 0, 120, 20)

wake_episodes_per_night = st.slider("밤에 깨는 횟수", 0, 10, 1)

caffeine_mg_before_bed = st.number_input(
    "취침 전 카페인 섭취량 (mg)",
    min_value=0,
    max_value=1000,
    value=0
)

alcohol_units_before_bed = st.number_input(
    "취침 전 음주량",
    min_value=0.0,
    max_value=10.0,
    value=0.0
)

screen_time_before_bed_mins = st.slider("취침 전 화면 사용 시간 (분)", 0, 300, 60)

stress_score = st.slider("스트레스 점수", 0, 10, 5)

work_hours_that_day = st.slider("오늘 근무/학업 시간", 0.0, 16.0, 8.0, 0.5)

chronotype = st.selectbox(
    "생활 유형",
    ["Morning", "Neutral", "Evening"]
)

mental_health_condition = st.selectbox(
    "정신건강 관련 상태",
    ["Healthy", "Anxiety", "Depression", "Both"]
)

heart_rate_resting_bpm = st.number_input(
    "안정 시 심박수",
    min_value=40,
    max_value=150,
    value=70
)

sleep_aid_used = st.selectbox(
    "수면 보조제 사용",
    [0, 1],
    format_func=lambda x: "사용" if x == 1 else "사용 안 함"
)

shift_work = st.selectbox(
    "교대근무 여부",
    [0, 1],
    format_func=lambda x: "예" if x == 1 else "아니오"
)


# 입력 데이터
user_data = pd.DataFrame([{
    "bmi": bmi,
    "sleep_duration_hrs": sleep_duration_hrs,
    "sleep_latency_mins": sleep_latency_mins,
    "wake_episodes_per_night": wake_episodes_per_night,
    "caffeine_mg_before_bed": caffeine_mg_before_bed,
    "alcohol_units_before_bed": alcohol_units_before_bed,
    "screen_time_before_bed_mins": screen_time_before_bed_mins,
    "stress_score": stress_score,
    "work_hours_that_day": work_hours_that_day,
    "chronotype": chronotype,
    "mental_health_condition": mental_health_condition,
    "heart_rate_resting_bpm": heart_rate_resting_bpm,
    "sleep_aid_used": sleep_aid_used,
    "shift_work": shift_work
}])


# 범주형 컬럼
categorical_cols = [
    "chronotype",
    "mental_health_condition"
]


# 원래 변수명 찾기
def get_original_feature(feature_name):
    if feature_name.startswith("remainder__"):
        return feature_name.replace("remainder__", "")

    if feature_name.startswith("cat__"):
        for col in categorical_cols:
            prefix = f"cat__{col}_"

            if feature_name.startswith(prefix):
                return col

    return feature_name


# SHAP 계산
def calculate_shap(user_data):
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    transformed = preprocessor.transform(user_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    shap_df["original_feature"] = shap_df["feature"].apply(get_original_feature)

    grouped = (
        shap_df.groupby("original_feature")["shap_value"]
        .sum()
        .reset_index()
    )

    grouped["abs_shap"] = grouped["shap_value"].abs()

    top3 = grouped.sort_values("abs_shap", ascending=False).head(3)

    return top3


# 예측
if st.button("🌙 수면의 질 분석하기"):
    predicted_score = model.predict(user_data)[0]
    predicted_score = np.clip(predicted_score, 1, 10)

    st.subheader("📊 예측 결과")
    st.metric("예측 수면의 질", f"{predicted_score:.2f} / 10")

    if predicted_score >= 8:
        st.success("전반적으로 높은 수면의 질로 예측되었습니다.")

    elif predicted_score >= 6:
        st.info("비교적 양호한 수면 상태로 예측되었습니다.")

    elif predicted_score >= 4:
        st.warning("수면 습관을 일부 개선해볼 필요가 있습니다.")

    else:
        st.warning("현재 입력된 생활습관에서 수면의 질이 낮게 예측되었습니다.")


    # 주요 영향 요인
    top3 = calculate_shap(user_data)

    st.subheader("🔎 주요 영향 요인")

    korean_names = {
        "stress_score": "스트레스",
        "wake_episodes_per_night": "야간 각성 횟수",
        "mental_health_condition": "정신건강 관련 상태",
        "sleep_duration_hrs": "수면 시간",
        "sleep_latency_mins": "잠드는 데 걸리는 시간",
        "screen_time_before_bed_mins": "취침 전 화면 사용",
        "work_hours_that_day": "근무/학업 시간",
        "heart_rate_resting_bpm": "안정 시 심박수",
        "shift_work": "교대근무",
        "bmi": "BMI",
        "caffeine_mg_before_bed": "카페인 섭취",
        "alcohol_units_before_bed": "음주량",
        "chronotype": "생활 유형",
        "sleep_aid_used": "수면 보조제"
    }

    factors = []

    for _, row in top3.iterrows():
        feature = row["original_feature"]
        shap_value = row["shap_value"]

        name = korean_names.get(feature, feature)

        direction = (
            "점수를 높이는 방향"
            if shap_value > 0
            else "점수를 낮추는 방향"
        )

        st.write(f"• **{name}** → {direction} ({shap_value:.3f})")

        factors.append({
            "name": name,
            "value": shap_value
        })

    st.caption(
        "SHAP 값은 모델의 예측 결과를 설명하는 값이며 "
        "의학적 인과관계를 의미하지 않습니다."
    )


    # AI 맞춤 수면 코칭
    st.subheader("🤖 AI 맞춤 수면 코칭")

    with st.spinner("AI가 수면 습관을 분석하고 있습니다..."):
        try:
            generator = load_llm()

            factor_text = "\n".join([
                f"- {x['name']}"
                for x in factors
            ])

            prompt = f"""
다음은 사용자의 수면 분석 결과입니다.

예측 수면의 질 점수:
{predicted_score:.2f} / 10

주요 영향 요인:
{factor_text}

사용자의 생활습관:
- 수면 시간: {sleep_duration_hrs}시간
- 스트레스 점수: {stress_score}점
- 잠드는 데 걸리는 시간: {sleep_latency_mins}분
- 취침 전 화면 사용: {screen_time_before_bed_mins}분
- 야간 각성: {wake_episodes_per_night}회
- 취침 전 카페인: {caffeine_mg_before_bed}mg
- 취침 전 음주량: {alcohol_units_before_bed}

위 결과를 참고해서 사용자에게 수면 코칭을 해주세요.

조건:
1. 현재 수면 상태를 간단히 설명해주세요.
2. 주요 영향 요인을 쉽게 설명해주세요.
3. 오늘부터 실천할 수 있는 개선 방법을 3가지 알려주세요.
4. 의학적 진단이나 치료는 하지 마세요.
5. SHAP 결과를 의학적인 인과관계처럼 단정하지 마세요.
6. 한국어로 자연스럽고 이해하기 쉽게 작성해주세요.
"""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "당신은 생활습관 정보를 바탕으로 "
                        "일반적인 수면 습관 개선 방법을 알려주는 "
                        "한국어 수면 코칭 도우미입니다. "
                        "질병을 진단하거나 치료하지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            result = generator(
                messages,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.5,
                repetition_penalty=1.15
            )

            advice = result[0]["generated_text"][-1]["content"]

            st.write(advice)

        except Exception as e:
            st.error("AI 코칭 생성 중 오류가 발생했습니다.")
            st.write(e)