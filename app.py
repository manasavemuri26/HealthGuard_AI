import streamlit as st
import joblib

# Load ML models (only for Diabetes and Liver)
diabetes_model = joblib.load('models/diabetes_model.pkl')
liver_model = joblib.load('models/liver_model.pkl')

st.title("Health Guard AI 🏥")

# Language and Disease Selection
language = st.selectbox("Select Language:", ["English", "Telugu", "Hindi"])
disease = st.selectbox("Select Disease to Predict:", ["Heart Disease", "Diabetes", "Liver Disease"])

#################### HEART LABELS ####################
heart_labels = {
    "English": {
        "title": "Enter Heart Details",
        "age": "Age",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0-3)",
        "trest": "Resting Blood Pressure",
        "chol": "Cholesterol",
        "fbs": "Fasting Blood Sugar >120 (0 or 1)",
        "restecg": "Rest ECG (0-2)",
        "thalach": "Max Heart Rate",
        "exang": "Exercise Angina (0=No,1=Yes)",
        "oldpeak": "ST Depression",
        "slope": "Slope of ST (0-2)",
        "ca": "Number of Vessels (0-3)",
        "thal": "Thalassemia (1-3)",
        "button": "Predict Heart Disease",
        "pos": "Risk of Heart Disease",
        "neg": "No Heart Disease Risk ✅"
    },
    "Telugu": {
        "title": "హార్ట్ వివరాలు నమోదు చేయండి",
        "age": "వయస్సు",
        "sex": "లింగం (0=Female, 1=Male)",
        "cp": "చెస్ట్ పెయిన్ రకం (0-3)",
        "trest": "బ్లడ్ ప్రెషర్ (రెస్ట్ింగ్)",
        "chol": "కోలెస్టరాల్",
        "fbs": "ఫాస్టింగ్ షుగర్ >120 (0 లేదా 1)",
        "restecg": "ECG ఫలితం (0-2)",
        "thalach": "గరిష్ట హార్ట్ రేట్",
        "exang": "ఎక్సర్‌సైజ్ ఎంజైనా (0=No,1=Yes)",
        "oldpeak": "ST డిప్రెషన్",
        "slope": "ST స్లోప్ (0-2)",
        "ca": "వెసల్స్ సంఖ్య (0-3)",
        "thal": "థాలసీమియా (1-3)",
        "button": "హార్ట్ డిసీజ్ ప్రెడిక్ట్ చేయండి",
        "pos": "మీకు హార్ట్ రిస్క్ ఉంది",
        "neg": "హార్ట్ రిస్క్ లేదు ✅"
    },
    "Hindi": {
        "title": "हृदय विवरण दर्ज करें",
        "age": "आयु",
        "sex": "लिंग (0=Female, 1=Male)",
        "cp": "चेस्ट पेन प्रकार (0-3)",
        "trest": "रिस्टिंग ब्लड प्रेशर",
        "chol": "कोलेस्ट्रॉल",
        "fbs": "फास्टिंग शुगर >120 (0 या 1)",
        "restecg": "रेस्ट ECG (0-2)",
        "thalach": "अधिकतम हृदय गति",
        "exang": "एक्सरसाइज एंजाइना (0=No,1=Yes)",
        "oldpeak": "ST अवसाद",
        "slope": "ST ढाल (0-2)",
        "ca": "वेसल्स की संख्या (0-3)",
        "thal": "थैलेसीमिया (1-3)",
        "button": "हृदय रोग जांचें",
        "pos": "हृदय रोग की संभावना है",
        "neg": "कोई हृदय रोग नहीं ✅"
    }
}

#################### DIABETES LABELS ####################
diab_labels = {
    "English": {
        "title": "Enter Diabetes Details",
        "preg": "Pregnancies",
        "glu": "Glucose",
        "bp": "Blood Pressure",
        "skin": "Skin Thickness",
        "ins": "Insulin",
        "bmi": "BMI",
        "dpf": "Diabetes Pedigree Function",
        "age": "Age",
        "button": "Predict Diabetes",
        "pos": "Risk of Diabetes",
        "neg": "No Diabetes Risk ✅"
    },
    "Telugu": {
        "title": "డయాబెటిస్ వివరాలు నమోదు చేయండి",
        "preg": "గర్భధారణల సంఖ్య",
        "glu": "గ్లూకోజ్ స్థాయి",
        "bp": "బ్లడ్ ప్రెషర్",
        "skin": "చర్మం మందం",
        "ins": "ఇన్సులిన్",
        "bmi": "BMI",
        "dpf": "డయాబెటిస్ Pedigree Function",
        "age": "వయస్సు",
        "button": "డయాబెటిస్ ప్రెడిక్ట్ చేయండి",
        "pos": "డయాబెటిస్ రిస్క్ ఉంది",
        "neg": "డయాబెటిస్ లేదు ✅"
    },
    "Hindi": {
        "title": "मधुमेह विवरण दर्ज करें",
        "preg": "गर्भधारणाएं",
        "glu": "ग्लूकोज",
        "bp": "ब्लड प्रेशर",
        "skin": "त्वचा मोटाई",
        "ins": "इंसुलिन",
        "bmi": "BMI",
        "dpf": "मधुमेह Pedigree Function",
        "age": "आयु",
        "button": "डायबिटीज जांचें",
        "pos": "मधुमेह होने की संभावना",
        "neg": "मधुमेह नहीं ✅"
    }
}

#################### LIVER LABELS ####################
liver_labels = {
    "English": {
        "title": "Enter Liver Details",
        "age": "Age",
        "gender": "Gender (0=Female, 1=Male)",
        "tbil": "Total Bilirubin",
        "dbil": "Direct Bilirubin",
        "alk": "Alkaline Phosphotase",
        "sgpt": "Alamine Aminotransferase",
        "sgot": "Aspartate Aminotransferase",
        "tprot": "Total Proteins",
        "alb": "Albumin",
        "ag": "Albumin and Globulin Ratio",
        "button": "Predict Liver Disease",
        "pos": "Risk of Liver Disease",
        "neg": "No Liver Disease Risk ✅"
    },
    "Telugu": {
        "title": "లివర్ వివరాలు నమోదు చేయండి",
        "age": "వయస్సు",
        "gender": "లింగం (0=Female, 1=Male)",
        "tbil": "టోటల్ బిలిరుబిన్",
        "dbil": "డైరెక్ట్ బిలిరుబిన్",
        "alk": "ఆల్కలైన్ ఫాస్ఫోటేజ్",
        "sgpt": "SGPT",
        "sgot": "SGOT",
        "tprot": "టోటల్ ప్రొటీన్స్",
        "alb": "ఆల్బ్యూమిన్",
        "ag": "ఆల్బ్యూమిన్ & గ్లోబ్యూలిన్ రేషియో",
        "button": "లివర్ డిసీజ్ ప్రెడిక్ట్ చేయండి",
        "pos": "లివర్ డిసీజ్ ఉండే అవకాశం ఉంది",
        "neg": "లివర్ ఆరోగ్యంగా ఉంది ✅"
    },
    "Hindi": {
        "title": "लिवर विवरण दर्ज करें",
        "age": "आयु",
        "gender": "लिंग (0=Female, 1=Male)",
        "tbil": "कुल बिलिरुबिन",
        "dbil": "प्रत्यक्ष बिलिरुबिन",
        "alk": "क्षारीय फॉस्फेटेज",
        "sgpt": "SGPT",
        "sgot": "SGOT",
        "tprot": "कुल प्रोटीन",
        "alb": "एल्ब्यूमिन",
        "ag": "एल्ब्यूमिन & ग्लोबुलिन अनुपात",
        "button": "लिवर रोग जांचें",
        "pos": "लिवर रोग की संभावना है",
        "neg": "कोई लिवर रोग नहीं ✅"
    }
}

#########################################################################
# HEART PREDICTION (Manual)
#########################################################################
if disease == "Heart Disease":
    L = heart_labels[language]
    st.subheader(L["title"])

    age = st.number_input(L["age"], 1, 120)
    sex = st.selectbox(L["sex"], [0,1])
    cp = st.number_input(L["cp"], 0, 3)
    trest = st.number_input(L["trest"])
    chol = st.number_input(L["chol"])
    fbs = st.selectbox(L["fbs"], [0,1])
    restecg = st.number_input(L["restecg"],0,2)
    thalach = st.number_input(L["thalach"])
    exang = st.selectbox(L["exang"], [0,1])
    oldpeak = st.number_input(L["oldpeak"])
    slope = st.number_input(L["slope"],0,2)
    ca = st.number_input(L["ca"],0,3)
    thal = st.number_input(L["thal"],1,3)

    if st.button(L["button"]):
        if (age > 55 and chol > 240) or (trest > 150) or (cp >= 2) or (oldpeak > 2.0) or (thalach < 120):
            st.warning(L["pos"])
        else:
            st.success(L["neg"])

#########################################################################
# DIABETES PREDICTION
#########################################################################
elif disease == "Diabetes":
    L = diab_labels[language]
    st.subheader(L["title"])

    preg = st.number_input(L["preg"], 0, 20)
    glu = st.number_input(L["glu"])
    bp = st.number_input(L["bp"])
    skin = st.number_input(L["skin"])
    ins = st.number_input(L["ins"])
    bmi = st.number_input(L["bmi"])
    dpf = st.number_input(L["dpf"])
    age = st.number_input(L["age"], 1, 120)

    if st.button(L["button"]):
        values = [[preg, glu, bp, skin, ins, bmi, dpf, age]]
        pred = diabetes_model.predict(values)[0]
        if pred == 1:
            st.warning(L["pos"])
        else:
            st.success(L["neg"])

#########################################################################
# LIVER PREDICTION
#########################################################################
else:
    L = liver_labels[language]
    st.subheader(L["title"])

    Age = st.number_input(L["age"], 1, 100)
    Gender = st.selectbox(L["gender"], [0,1])
    TB = st.number_input(L["tbil"])
    DB = st.number_input(L["dbil"])
    Alk = st.number_input(L["alk"])
    SGPT = st.number_input(L["sgpt"])
    SGOT = st.number_input(L["sgot"])
    TProt = st.number_input(L["tprot"])
    Alb = st.number_input(L["alb"])
    AG_Ratio = st.number_input(L["ag"])

    if st.button(L["button"]):
        values = [[Age, Gender, TB, DB, Alk, SGPT, SGOT, TProt, Alb, AG_Ratio]]
        pred = liver_model.predict(values)[0]
        if pred == 1:
            st.warning(L["pos"])
        else:
            st.success(L["neg"])
