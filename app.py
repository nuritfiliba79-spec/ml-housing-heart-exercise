"""
פתרון תרגיל למידת מכונה - חלק א (רגרסיה לינארית) וחלק ב (רגרסיה לוגיסטית)
הרצה:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
HOUSING_CSV = BASE / "Project housing data .csv"
HEART_CSV = BASE / "heart_data.csv"
SEED = 42

st.set_page_config(page_title="תרגיל למידת מכונה", page_icon="🏠", layout="wide")
st.markdown(
    """
    <style>
    .main .block-container, [data-testid="stSidebar"] {direction: rtl; text-align: right;}
    [data-testid="stMarkdownContainer"] {direction: rtl; text-align: right;}
    [data-testid="stMetric"] {direction: rtl;}
    table {direction: ltr;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# חלק א - מחירי דיור
# =====================================================================
@st.cache_data
def load_housing():
    return pd.read_csv(HOUSING_CSV)


def encode_housing(df):
    """קידוד משתנים:
    - view (0-4) ו-condition (1-5) הם משתנים קטגוריאליים-אורדינליים -> One-Hot (drop_first)
    - yr_built -> גיל הבית (house_age), פרשנות נוחה יותר
    - sqft_living = sqft_above + sqft_basement בדיוק -> מוריד את sqft_above כדי למנוע מולטיקולינאריות מלאה
    """
    X = df.drop(columns=["price"]).copy()
    X["house_age"] = 2015 - X.pop("yr_built")
    X = X.drop(columns=["sqft_above"])
    X["view"] = pd.Categorical(X["view"], categories=[0, 1, 2, 3, 4])
    X["condition"] = pd.Categorical(X["condition"], categories=[1, 2, 3, 4, 5])
    X = pd.get_dummies(X, columns=["view", "condition"], drop_first=True, dtype=int)
    return X


@st.cache_resource
def train_housing(df):
    X = encode_housing(df)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(X_train, y_train)
    return model, X.columns.tolist(), X_train, X_test, y_train, y_test


def part_a():
    st.header("חלק א: אנליזה וניבוי מחירי דיור")
    df = load_housing()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 חקירת נתונים", "🛠️ בניית המודל", "📈 הערכת ביצועים", "🔮 פרדיקציה"])

    # ---------- חקירת נתונים ----------
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("מספר דירות", f"{len(df):,}")
        c2.metric("מחיר ממוצע", f"${df.price.mean():,.0f}")
        c3.metric("ערכים חסרים", int(df.isna().sum().sum()))
        st.dataframe(df.head(10), width="stretch")
        st.subheader("סטטיסטיקה תיאורית")
        st.dataframe(df.describe().T.round(2), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df, x="price", nbins=60, title="התפלגות מחירים"))
        with col2:
            st.plotly_chart(
                px.scatter(df, x="sqft_living", y="price", opacity=0.4,
                           title="מחיר מול שטח מגורים"),
                width="stretch",
            )
        corr = df.corr()["price"].drop("price").sort_values()
        st.plotly_chart(px.bar(corr, orientation="h", title="מתאם של כל משתנה עם המחיר"))

    model, cols, X_train, X_test, y_train, y_test = train_housing(df)

    # ---------- בניית המודל ----------
    with tab2:
        st.markdown(
            """
            **שלבי הכנת הנתונים:**
            1. **בדיקת ערכים חסרים** – אין ערכים חסרים בקובץ.
            2. **קידוד משתנים קטגוריאליים** – `view` (0–4) ו-`condition` (1–5) הם מדדים בדידים ולא כמויות רציפות
               (המעבר ממצב 1 ל-2 לא בהכרח "שווה" למעבר מ-4 ל-5), ולכן קודדו ב-**One-Hot Encoding** (עם `drop_first` למניעת מלכודת הדמי).
               `waterfront` כבר משתנה דמי (0/1).
            3. **הנדסת משתנים** – `yr_built` הומר ל-`house_age` (גיל הבית).
            4. **מולטיקולינאריות** – בנתונים מתקיים בדיוק `sqft_living = sqft_above + sqft_basement`, ולכן `sqft_above` הוסר.
            5. **חלוקה** ל-80% אימון / 20% מבחן (`random_state=42`).
            6. **אימון** `LinearRegression`.
            """
        )
        st.write(f"גודל סט אימון: **{len(X_train)}** | גודל סט מבחן: **{len(X_test)}**")
        coef = pd.DataFrame({"משתנה": cols, "מקדם": model.coef_}).sort_values("מקדם", key=abs, ascending=False)
        st.subheader("מקדמי המודל")
        st.write(f"חותך (Intercept): **{model.intercept_:,.0f}**")
        st.dataframe(coef.style.format({"מקדם": "{:,.1f}"}), width="stretch", hide_index=True)
        st.info(
            "פרשנות: כל מקדם מראה בכמה דולר משתנה המחיר החזוי כשהמשתנה עולה ביחידה אחת, בהנחה ששאר המשתנים קבועים. "
            "למשל המקדם של sqft_living = תוספת מחיר לכל רגל רבועה נוספת."
        )

    # ---------- הערכת ביצועים ----------
    with tab3:
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        r2 = r2_score(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        mape = mean_absolute_percentage_error(y_test, pred_test)

        st.subheader("מדדים על סט המבחן")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"${mae:,.0f}")
        c3.metric("RMSE", f"${rmse:,.0f}")
        c4.metric("MAPE", f"{mape:.1%}")
        st.caption(f"R² על סט האימון: {r2_score(y_train, pred_train):.3f} (להשוואה – בדיקת התאמת יתר)")

        st.markdown(
            f"""
            **הסבר המדדים:**
            - **R² = {r2:.3f}** – המודל מסביר כ-{r2:.0%} מהשונות במחירי הדירות.
            - **MAE = \\${mae:,.0f}** – בממוצע, החיזוי טועה בכ-\\${mae:,.0f} (ערך מוחלט).
            - **RMSE = \\${rmse:,.0f}** – גבוה מ-MAE כי הוא "מעניש" טעויות גדולות (בעיקר בדירות יוקרה יקרות במיוחד).
            - **MAPE = {mape:.1%}** – שגיאה יחסית ממוצעת.

            **מסקנה:** מודל לינארי פשוט נותן הערכה סבירה, אך הוא אינו מתחשב במיקום (שכונה/מיקוד) –
            הגורם החשוב ביותר במחירי נדל"ן – ולכן חלק ניכר מהשונות נשאר בלתי מוסבר.
            R² דומה באימון ובמבחן מעיד שאין התאמת יתר.
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=y_test, y=pred_test, opacity=0.5, labels={"x": "מחיר אמיתי", "y": "מחיר חזוי"},
                             title="מחיר אמיתי מול חזוי (סט מבחן)")
            lim = [0, max(y_test.max(), pred_test.max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="חיזוי מושלם", line=dict(dash="dash")))
            st.plotly_chart(fig)
        with col2:
            resid = y_test - pred_test
            st.plotly_chart(px.histogram(resid, nbins=60, title="התפלגות השאריות (Residuals)"))

    # ---------- פרדיקציה ----------
    with tab4:
        st.markdown("הזינו מאפייני דירה חדשה (ברירת המחדל היא דירה ריאליסטית שהמצאנו):")
        c1, c2, c3 = st.columns(3)
        with c1:
            bedrooms = st.number_input("חדרי שינה", 1, 8, 3)
            bathrooms = st.number_input("חדרי רחצה", 1, 6, 2)
            floors = st.selectbox("קומות", [1, 2, 3], index=1)
            waterfront = st.radio("נוף לחוף", [0, 1], format_func=lambda v: "כן" if v else "לא", horizontal=True)
        with c2:
            sqft_living = st.number_input("שטח מגורים (sqft)", 400, 8000, 2000, step=50)
            sqft_basement = st.number_input("מתוכו מרתף (sqft)", 0, 2600, 400, step=50)
            sqft_lot = st.number_input("שטח מגרש (sqft)", 500, 200000, 7500, step=100)
        with c3:
            view = st.slider("איכות נוף (0-4)", 0, 4, 0)
            condition = st.slider("מצב הדירה (1-5)", 1, 5, 3)
            yr_built = st.number_input("שנת בנייה", 1900, 2015, 1985)

        if sqft_basement >= sqft_living:
            st.error("שטח המרתף חייב להיות קטן משטח המגורים הכולל.")
            return

        new = pd.DataFrame([{
            "price": 0, "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
            "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront, "view": view,
            "condition": condition, "sqft_above": sqft_living - sqft_basement,
            "sqft_basement": sqft_basement, "yr_built": yr_built,
        }])
        x_new = encode_housing(new)[cols]
        price = model.predict(x_new)[0]
        st.dataframe(new.drop(columns="price"), width="stretch", hide_index=True)
        st.success(f"### מחיר חזוי: \\${price:,.0f}")

        # בדיקת סבירות מול דירות דומות בנתונים
        similar = df[(df.bedrooms == bedrooms) & (df.sqft_living.between(sqft_living * 0.85, sqft_living * 1.15))]
        if len(similar):
            lo, med, hi = similar.price.quantile([0.25, 0.5, 0.75])
            in_range = lo <= price <= hi
            st.markdown(
                f"""
                **האם התוצאה סבירה?**
                בנתונים נמצאו **{len(similar)}** דירות דומות ({bedrooms} חדרי שינה, שטח ±15%).
                המחיר החציוני שלהן **\\${med:,.0f}**, והטווח הבין-רבעוני **\\${lo:,.0f} – \\${hi:,.0f}**.

                {"✅ המחיר החזוי נמצא בתוך הטווח הטיפוסי של דירות דומות – התוצאה נראית סבירה." if in_range else
                 "⚠️ המחיר החזוי מחוץ לטווח הטיפוסי של דירות דומות – ייתכן שהשילוב שבחרתם חריג (נוף, מצב, חוף) או שהמודל הלינארי פחות מדויק בקצוות."}

                המודל מתמחר בעיקר לפי שטח המגורים, הנוף, קרבה לחוף ומצב הדירה. יש לזכור שטעות טיפוסית
                של המודל היא כ-\\${mae:,.0f}, ולכן כדאי להתייחס לחיזוי כהערכה בטווח ולא כמחיר מדויק.
                """
            )
            fig = px.histogram(similar, x="price", nbins=30, title="מחירי דירות דומות ומיקום החיזוי")
            fig.add_vline(x=price, line_dash="dash", line_color="red", annotation_text="חיזוי")
            st.plotly_chart(fig)


# =====================================================================
# חלק ב - התקפי לב
# =====================================================================
HEART_FEATURES = ["age", "sex", "exang", "ca", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach"]
CP_LABELS = {0: "0 - אסימפטומטי", 1: "1 - תעוקה טיפוסית", 2: "2 - תעוקה לא טיפוסית", 3: "3 - כאב לא תעוקתי"}
ECG_LABELS = {0: "0 - תקין", 1: "1 - חריגת ST-T", 2: "2 - היפרטרופיה של חדר שמאל"}


@st.cache_data
def load_heart():
    df = pd.read_csv(HEART_CSV)
    # ca מוגדר 0-3; הערך 4 הוא שגיאה ידועה בנתונים -> מוסר
    return df[df["ca"] <= 3].reset_index(drop=True)


def encode_heart(df):
    """cp ו-rest_ecg הם קטגוריות נומינליות (המספרים הם רק קודים) -> One-Hot.
    sex, exang, fbs כבר בינאריים. ca הוא ספירה (0-3) -> נשאר מספרי."""
    X = df[HEART_FEATURES].copy()
    X["cp"] = pd.Categorical(X["cp"], categories=[0, 1, 2, 3])
    X["rest_ecg"] = pd.Categorical(X["rest_ecg"], categories=[0, 1, 2])
    return pd.get_dummies(X, columns=["cp", "rest_ecg"], drop_first=True, dtype=int)


@st.cache_resource
def train_heart(df):
    X = encode_heart(df)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    return model, scaler, X.columns.tolist(), X_train, X_test, y_train, y_test


def part_b():
    st.header("חלק ב: אנליזה וניבוי של התקפי לב")
    df = load_heart()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 חקירת נתונים", "🛠️ בניית המודל", "📈 הערכת ביצועים", "🔮 פרדיקציה"])

    with tab1:
        st.caption("מקור הנתונים: UCI Heart Disease (Cleveland), 303 מטופלים – הקובץ heart_data.csv. "
                   "נשמרו רק המשתנים שמופיעים בתרגיל. הוסרו 5 שורות עם ca=4 (ערך לא חוקי).")
        c1, c2, c3 = st.columns(3)
        c1.metric("מטופלים", len(df))
        c2.metric("סיכון מוגבר (target=1)", f"{df.target.mean():.0%}")
        c3.metric("ערכים חסרים", int(df.isna().sum().sum()))
        st.dataframe(df.head(10), width="stretch")
        st.markdown(
            """
            **מה מסמל כל משתנה:**
            | משתנה | משמעות |
            |---|---|
            | age | גיל המטופל |
            | sex | מין (1=זכר, 0=נקבה) |
            | exang | תעוקת חזה במאמץ (1=כן) |
            | ca | מספר כלי דם עיקריים שנצבעו בצנתור (0-3) |
            | cp | סוג כאב בחזה (0-3) |
            | trtbps | לחץ דם במנוחה (mm Hg) |
            | chol | כולסטרול (mg/dl) |
            | fbs | סוכר בצום > 120 (1=כן) |
            | rest_ecg | תוצאת אק"ג במנוחה (0-2) |
            | thalach | דופק מקסימלי שהושג במאמץ |
            | target | 1 = סיכוי גבוה להתקף לב |
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.box(df, x="target", y="thalach", color="target", title="דופק מקסימלי לפי target"),
                            width="stretch")
        with col2:
            ct = pd.crosstab(df.cp, df.target, normalize="index").reset_index()
            st.plotly_chart(px.bar(ct, x="cp", y=1, title="שיעור target=1 לפי סוג כאב בחזה (cp)"),
                            width="stretch")

    model, scaler, cols, X_train, X_test, y_train, y_test = train_heart(df)

    with tab2:
        st.markdown(
            """
            **שלבי בניית המודל:**
            1. **קידוד קטגוריאלי** – `cp` (סוג כאב בחזה) ו-`rest_ecg` הם משתנים **נומינליים**: הערכים 0-3 הם קודים ולא כמות,
               ולכן קודדו ב-**One-Hot Encoding** (`drop_first`). `sex`, `exang`, `fbs` כבר בינאריים (0/1).
               `ca` הוא ספירה של כלי דם ולכן נשאר מספרי.
            2. **ניקוי** – הוסרו רשומות עם `ca=4` שאינו ערך חוקי.
            3. **חלוקה** 80/20 עם `stratify` לשמירת יחס המחלקות.
            4. **נרמול** `StandardScaler` (חשוב לרגרסיה לוגיסטית ומאפשר להשוות מקדמים).
            5. **אימון** `LogisticRegression`.
            """
        )
        coef = pd.DataFrame({"משתנה": cols, "מקדם": model.coef_[0], "Odds Ratio": np.exp(model.coef_[0])})
        coef = coef.sort_values("מקדם")
        st.plotly_chart(px.bar(coef, x="מקדם", y="משתנה", orientation="h",
                               title="מקדמי המודל (משתנים מנורמלים) – חיובי = מעלה סיכון"),
                        width="stretch")
        st.info(
            "פרשנות: מקדם חיובי מעלה את הסיכוי ל-target=1. למשל דופק מקסימלי גבוה (thalach) וכאב חזה מסוג 1-3 קשורים לסיכון גבוה, "
            "ואילו תעוקה במאמץ (exang), יותר כלי דם חסומים (ca) ומין זכר קשורים בנתונים אלה ל-target נמוך יותר."
        )

    with tab3:
        Xs_test = scaler.transform(X_test)
        pred = model.predict(Xs_test)
        proba = model.predict_proba(Xs_test)[:, 1]
        acc, prec = accuracy_score(y_test, pred), precision_score(y_test, pred)
        rec, f1 = recall_score(y_test, pred), f1_score(y_test, pred)

        st.subheader("מדדים על סט המבחן")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

        cm = confusion_matrix(y_test, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["חזוי 0", "חזוי 1"], y=["אמיתי 0", "אמיתי 1"], title="מטריצת בלבול")
        col1, col2 = st.columns([1, 1.3])
        col1.plotly_chart(fig)
        tn, fp, fn, tp = cm.ravel()
        col2.markdown(
            f"""
            - **Accuracy** – אחוז החיזויים הנכונים מכלל המטופלים: {acc:.1%}
            - **Precision** – מתוך מי שהמודל סימן "בסיכון", כמה באמת בסיכון: {prec:.1%}
            - **Recall** – מתוך כל מי שבאמת בסיכון, כמה המודל זיהה: {rec:.1%}
            - **F1** – ממוצע הרמוני של Precision ו-Recall: {f1:.1%}

            במבחן: **{fn}** מטופלים בסיכון שהמודל "פספס" (False Negative), ו-**{fp}** אזעקות שווא (False Positive).
            """
        )

        st.subheader("איזו מטריקה חשובה יותר בהקשר הרפואי?")
        st.markdown(
            """
            **Recall (רגישות) היא המטריקה החשובה ביותר.**

            - **False Negative** = מטופל שבאמת בסיכון להתקף לב, אבל המודל אמר שהוא "בסדר" → הוא נשלח הביתה בלי טיפול,
              והמחיר עלול להיות **חיי אדם**.
            - **False Positive** = מטופל בריא שסומן "בסיכון" → המחיר הוא בדיקות נוספות (אק"ג, מאמץ, צנתור), אי-נוחות ועלות כספית,
              אבל **לא סכנת חיים**.

            מכיוון שמחיר הפספוס גבוה בהרבה ממחיר אזעקת שווא, נעדיף מודל שמזהה כמה שיותר חולים אמיתיים – כלומר Recall גבוה,
            גם במחיר של ירידה ב-Precision. Accuracy לבדה מטעה, כי היא שוקלת את שני סוגי הטעויות באותה מידה.
            בפועל אפשר להוריד את סף ההחלטה (threshold) מ-0.5 כדי להעלות את ה-Recall – ראו בטבלה:
            """
        )
        rows = []
        for t in [0.3, 0.4, 0.5, 0.6]:
            p = (proba >= t).astype(int)
            rows.append({"Threshold": t, "Recall": recall_score(y_test, p), "Precision": precision_score(y_test, p),
                         "Accuracy": accuracy_score(y_test, p), "False Negatives": int(((p == 0) & (y_test == 1)).sum())})
        st.dataframe(pd.DataFrame(rows).style.format({"Recall": "{:.3f}", "Precision": "{:.3f}", "Accuracy": "{:.3f}"}),
                     width="stretch", hide_index=True)

    with tab4:
        st.markdown("הזינו פרטי מטופל חדש (ברירת המחדל: מטופל בדוי עם נתונים ריאליסטיים):")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("גיל", 20, 90, 58)
            sex = st.radio("מין", [1, 0], format_func=lambda v: "זכר" if v else "נקבה", horizontal=True)
            cp = st.selectbox("סוג כאב בחזה (cp)", list(CP_LABELS), index=2, format_func=CP_LABELS.get)
            exang = st.radio("תעוקה במאמץ (exang)", [0, 1], format_func=lambda v: "כן" if v else "לא", horizontal=True)
        with c2:
            trtbps = st.number_input("לחץ דם במנוחה (trtbps)", 80, 220, 140)
            chol = st.number_input("כולסטרול (chol)", 100, 600, 245)
            fbs = st.radio("סוכר בצום > 120 (fbs)", [0, 1], format_func=lambda v: "כן" if v else "לא", horizontal=True)
        with c3:
            rest_ecg = st.selectbox("אק\"ג במנוחה (rest_ecg)", list(ECG_LABELS), index=1, format_func=ECG_LABELS.get)
            thalach = st.number_input("דופק מקסימלי (thalach)", 60, 220, 155)
            ca = st.selectbox("כלי דם עיקריים (ca)", [0, 1, 2, 3], index=0)

        new = pd.DataFrame([{"age": age, "sex": sex, "exang": exang, "ca": ca, "cp": cp, "trtbps": trtbps,
                             "chol": chol, "fbs": fbs, "rest_ecg": rest_ecg, "thalach": thalach}])
        x_new = scaler.transform(encode_heart(new)[cols])
        p = model.predict_proba(x_new)[0, 1]
        st.dataframe(new, width="stretch", hide_index=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "הסתברות לסיכון מוגבר להתקף לב"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkred" if p >= 0.5 else "green"},
                   "steps": [{"range": [0, 30], "color": "#d4f4dd"}, {"range": [30, 50], "color": "#fff3c4"},
                             {"range": [50, 100], "color": "#f8d0d0"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}},
        ))
        st.plotly_chart(fig)

        # תרומת כל משתנה לחיזוי (מקדם * ערך מנורמל)
        contrib = pd.Series(model.coef_[0] * x_new[0], index=cols).sort_values()
        top_up = [c for c in contrib.index[::-1] if contrib[c] > 0][:3]
        top_down = [c for c in contrib.index if contrib[c] < 0][:3]

        if p >= 0.5:
            verdict = (f"🔴 **המודל מסווג את המטופל כבעל סיכון מוגבר (target=1) בהסתברות {p:.0%}.**\n\n"
                       "**משמעות קלינית:** מומלץ להפנות לבירור קרדיולוגי מעמיק – אק\"ג במאמץ, אקו לב, ובמידת הצורך צנתור – "
                       "ולטפל בגורמי הסיכון (לחץ דם, כולסטרול). מכיוון שהעדפנו Recall גבוה, עדיף לבדוק ולשלול מאשר לפספס.")
        elif p >= 0.3:
            verdict = (f"🟡 **הסתברות {p:.0%} – מתחת לסף 0.5 אך באזור אפור.**\n\n"
                       "**משמעות קלינית:** בהקשר רפואי, שבו פספוס מסוכן, כדאי בכל זאת לשקול מעקב/בירור נוסף ולא לשחרר באופן אוטומטי.")
        else:
            verdict = (f"🟢 **המודל מסווג את המטופל כבעל סיכון נמוך (target=0), הסתברות {p:.0%}.**\n\n"
                       "**משמעות קלינית:** אין אינדיקציה דחופה לפי המודל, אך יש להמשיך מעקב שגרתי ולטפל בגורמי סיכון קיימים.")
        st.markdown(verdict)
        st.markdown(f"**משתנים שהעלו את הסיכון אצל מטופל זה:** {', '.join(top_up) or '—'}  \n"
                    f"**משתנים שהורידו את הסיכון:** {', '.join(top_down) or '—'}")
        st.warning("המודל הוא כלי תומך החלטה המבוסס על 300 מטופלים בלבד, ואינו מחליף שיקול דעת של רופא.")


# =====================================================================
st.title("🤖 תרגיל למידת מכונה – פתרון")
part = st.sidebar.radio("בחרו חלק", ["חלק א – מחירי דיור 🏠", "חלק ב – התקפי לב ❤️"])
st.sidebar.markdown("---")
st.sidebar.markdown("**חלק א:** רגרסיה לינארית  \n**חלק ב:** רגרסיה לוגיסטית")

if part.startswith("חלק א"):
    part_a()
else:
    part_b()
