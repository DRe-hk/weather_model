import streamlit as st
import numpy as np
import pickle
# Título de la aplicación
st.title('Predicción con K-Nearest Neighbors')

# Descripción de la aplicación
st.write('Ajusta los sliders para introducir las características.')

# Crear sliders para recibir valores de entrada
feature1 = st.slider('Feature 1', 0.0, 10.0, 5.0)
feature2 = st.slider('Feature 2', 0.0, 10.0, 5.0)
feature3 = st.slider('Feature 3', 0.0, 10.0, 5.0)
feature4 = st.slider('Feature 4', 0.0, 10.0, 5.0)

# Valores de entrada en un array
input_data = np.array([[feature1, feature2, feature3, feature4]])

knn = pickle.load(open('model.pkl', 'rb'))


# Hacer la predicción
prediction = knn.predict(input_data)

# Mostrar la predicción
st.write('Predicción:', [prediction][0])
