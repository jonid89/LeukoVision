import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd


st.title('Data')

st.markdown(""" 
<div style="text-align: justify;">
In this project, we combined two publicly available datasets from independent research groups:

- **University of Barcelona**: ~17,000 single-cell images across **8 blood cell classes**. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578" target="_blank">(Acevedo et al. 2019)</a>  
- **Multi-university collaboration in Germany**: ~18,000 single-cell images across **17 blood cell classes**. <a href="https://www.nature.com/articles/s42256-019-0101-9" target="_blank">(Matek et al. 2019)</a> 

The table below summarizes the blood cell types and number of images per dataset.  
Additionally, we provide bar charts to visualize the class distributions and highlight dataset imbalances.  
</div>
""", unsafe_allow_html=True)


# Load data
cell = np.loadtxt('./Streamlit/pages/count_spanish_german_chinese.txt', usecols=0, dtype=str)
count = np.loadtxt('./Streamlit/pages/count_spanish_german_chinese.txt', usecols=[1,2,3], dtype=int)

# Create a stacked bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=cell,
    y=count[:,0],
    name='Spanish DS',
    marker_color='#3362b0'
))


fig.add_trace(go.Bar(
    x=cell,
    y=count[:,1],
    name='German DS',
    marker_color='#cc3164'
))

# Update layout for stacked bars
fig.update_layout(
    barmode='stack',
    xaxis_title='Cell type',
    yaxis_title='Population',
    xaxis_tickangle=-45,
    template='plotly_white'
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)


file_path = './Streamlit/pages/count_spanish_german_chinese.txt'
df = pd.read_csv(file_path, 
                 sep='\s+',          # whitespace separator
                 header=None,        # no header in file
                 names=['Abbreviation ', 'Spanish', 'German', 'Chinese'])
cell_name=['Basophil','Erythroblast','Eosinophil','Smudge cell','Lymphocyte (atypical)','Lymphocyte (typical)',
 'Metamyelocyte', 'Monoblast','Monocyte','Myelocyte','Myeloblast','Neutrophil (band)','Neutrophil (segmented)',
'Platelet','Promyelocyte (bilobled)','Promyelocyte', 'Not Assigned']
df.index = df.index + 1
df.insert(0, "Cell type", cell_name)
df=df.drop('Chinese',axis=1)
styled_df = df.style.set_properties(**{'text-align': 'center'}) \
                    .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
# Display in Streamlit
st.write("### Blood Cell Counts Across Datasets")
st.dataframe(styled_df)