import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="GreenPulse AI 🌳", layout="wide")

# --------------------------------------------------
# CLEAN MINIMAL STYLE
# --------------------------------------------------
st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

/* Remove unwanted empty blocks */
div[data-testid="stVerticalBlock"] > div:empty {
    display:none;
}

/* Section Titles */
.section-title{
    text-align:center;
    font-size:22px;
    font-weight:600;
    margin-top:25px;
    margin-bottom:20px;
}

/* Output Info Boxes */
.output-box{
    background:#0f1b2a;
    padding:10px;
    border-radius:10px;
    margin-bottom:8px;
    font-size:14px;
}

/* Mobile Responsive */
@media (max-width:900px){
    .section-title{
        font-size:18px;
        margin-top:20px;
        margin-bottom:15px;
    }
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("tree_survival_ai.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🌳 GreenPulse AI — Smart Urban Tree Sustainability Dashboard")

# --------------------------------------------------
# DATA
# --------------------------------------------------
tree_list = ["Neem","Banyan","Mango","Peepal","Ashoka","Gulmohar"]

tree_presets = {
    "Neem":{"height":6,"canopy":4,"crown":5},
    "Banyan":{"height":10,"canopy":8,"crown":9},
    "Mango":{"height":7,"canopy":5,"crown":6},
}

city_presets = {
    "Pune":{"temp":30,"humidity":60,"pm25":55},
    "Mumbai":{"temp":33,"humidity":80,"pm25":65},
    "Delhi":{"temp":36,"humidity":40,"pm25":120},
}

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([1.2,1])

# ==================================================
# LEFT SIDE — INPUTS
# ==================================================
with left:

    st.markdown("<div class='section-title'>🌿 Environmental Inputs</div>", unsafe_allow_html=True)

    with st.form("form"):

        a,b,c = st.columns(3)
        with a:
            tree_name = st.selectbox("Tree Name",tree_list)
        with b:
            tree_choice = st.selectbox("Tree Preset",["Custom"]+list(tree_presets.keys()))
        with c:
            city_choice = st.selectbox("City",["Custom"]+list(city_presets.keys()))

        tp = tree_presets.get(tree_choice,{"height":5,"canopy":3,"crown":4})
        cp = city_presets.get(city_choice,{"temp":32,"humidity":55,"pm25":70})

        r1,r2,r3,r4 = st.columns(4)

        with r1:
            Height_m = st.number_input("Height",1.0,30.0,float(tp["height"]))
            Crown_Density = st.slider("Density",0,100,60)

        with r2:
            Canopy_Width_m = st.number_input("Canopy",1.0,20.0,float(tp["canopy"]))
            Ambient_Temperature_C = st.slider("Temp",10,50,int(cp["temp"]))

        with r3:
            Tree_Crown_Diameter_m = st.number_input("Crown",1.0,20.0,float(tp["crown"]))
            Humidity = st.slider("Humidity",10,100,int(cp["humidity"]))

        with r4:
            Soil_Moisture = st.slider("Soil",0,100,40)
            PM25 = st.slider("PM2.5",0,300,int(cp["pm25"]))

        s1,s2,s3,s4 = st.columns(4)
        with s1: NO2 = st.slider("NO2",0,150,20)
        with s2: CO2 = st.slider("CO2",300,800,400)
        with s3: Leaf_Color_Deviation = st.slider("Leaf",0,10,2)
        with s4: Bark_Damage_Score = st.slider("Bark",0,10,1)

        submit = st.form_submit_button("🌿 Predict")

# ==================================================
# RIGHT SIDE — OUTPUT
# ==================================================
with right:

    if submit:

        st.markdown("<div class='section-title'>📊 Survival Probability</div>", unsafe_allow_html=True)

        features = np.array([[Height_m,Canopy_Width_m,Tree_Crown_Diameter_m,
                              Crown_Density,Ambient_Temperature_C,
                              Humidity,Soil_Moisture,
                              PM25,NO2,CO2,
                              Leaf_Color_Deviation,Bark_Damage_Score]])

        prediction = round(float(model.predict(features)[0]),2)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix':"%"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,50],'color':"#ff4d4d"},
                    {'range':[50,75],'color':"#ffb020"},
                    {'range':[75,100],'color':"#1fdb6b"}
                ]
            }
        ))

        fig.update_layout(height=250,margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

        # Risk Label
        if prediction < 50:
            st.error("🔴 High Environmental Risk")
        elif prediction < 75:
            st.warning("🟡 Moderate Environmental Risk")
        else:
            st.success("🟢 Excellent Sustainability")

        # Two-column Output
        col1,col2 = st.columns(2)

        explanations=[]
        if PM25>100: explanations.append("High pollution detected.")
        if Soil_Moisture<30: explanations.append("Low soil moisture.")
        if Ambient_Temperature_C>38: explanations.append("Heat stress.")
        if not explanations: explanations.append("Environmental conditions stable.")

        suggestions=[]
        if prediction<50:
            suggestions.append("Increase irrigation.")
            suggestions.append("Improve soil health.")
        else:
            suggestions.append("Maintain current practices.")

        with col1:
            st.markdown("### 🧠 AI Explanation")
            for e in explanations:
                st.markdown(f"<div class='output-box'>⚡ {e}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 🌱 Smart Recommendations")
            for s in suggestions:
                st.markdown(f"<div class='output-box'>✅ {s}</div>", unsafe_allow_html=True)