import streamlit as st

st.set_page_config(
    page_title="Alzheimer's Analysis",
    page_icon="🧠",
)

welcome = st.Page("pages/welcome.py", title="Welcome", icon="👋")
about = st.Page("pages/about.py", title="About the Data", icon="💾")
modeling = st.Page("pages/modeling.py", title="Modeling", icon="📊")
documentation = st.Page("pages/documentation.py", title="Documentation", icon="📔")

pg = st.navigation([welcome, about, modeling, documentation])
pg.run()