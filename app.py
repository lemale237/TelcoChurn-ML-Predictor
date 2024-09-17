import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Charger le modèle XGBoost optimisé
best_xgb_model = joblib.load('./models/best_xgb_model.joblib')

# Charger le scaler
scaler = joblib.load('./models/scaler.joblib')

# Charger le profit moyen par client
average_profit = 235.72740171938904

# Titre et description
st.title("Prédiction du Churn Client")
st.write("""
Cette application permet de prédire la probabilité qu'un client quitte l'entreprise (churn) en se basant sur diverses caractéristiques.
Vous pouvez soit entrer les données d'un seul client, soit importer un fichier CSV pour obtenir des prédictions en masse.
""")

# Sélection du mode d'utilisation
option = st.radio("Choisissez une option :", ('Prédiction pour un seul client', 'Prédictions en masse (import CSV)'))

# Prédiction pour un seul client
if option == 'Prédiction pour un seul client':
    st.subheader("Entrez les données du client :")
    
    LOG_DATA = st.number_input('LOG_DATA')
    LOG_TIME_CLIENT = st.number_input('LOG_TIME_CLIENT')
    INCOME = st.number_input('INCOME')
    OVERCHARGE = st.number_input('OVERCHARGE')
    OVERCHARGE_RATIO = st.number_input('OVERCHARGE_RATIO')
    LEFTOVER = st.number_input('LEFTOVER')
    HOUSE = st.number_input('HOUSE')
    HANDSET_PRICE = st.number_input('HANDSET_PRICE')
    OVER_15MINS_CALLS_PER_MONTH = st.number_input('OVER_15MINS_CALLS_PER_MONTH')
    HOUSE_INCOME_RATIO = st.number_input('HOUSE_INCOME_RATIO')
    DATA_TIME_RATIO = st.number_input('DATA_TIME_RATIO')
    
    # Bouton pour lancer la prédiction
    if st.button('Prédire'):
        # Préparation des données
        input_data = pd.DataFrame({
            'LOG_DATA': [LOG_DATA],
            'LOG_TIME_CLIENT': [LOG_TIME_CLIENT],
            'INCOME': [INCOME],
            'OVERCHARGE': [OVERCHARGE],
            'OVERCHARGE_RATIO': [OVERCHARGE_RATIO],
            'LEFTOVER': [LEFTOVER],
            'HOUSE': [HOUSE],
            'HANDSET_PRICE': [HANDSET_PRICE],
            'OVER_15MINS_CALLS_PER_MONTH': [OVER_15MINS_CALLS_PER_MONTH],
            'HOUSE_INCOME_RATIO': [HOUSE_INCOME_RATIO],
            'DATA_TIME_RATIO': [DATA_TIME_RATIO]
        })
        
        # Mise à l'échelle
        input_data_scaled = scaler.transform(input_data)
        
        # Prédiction
        churn_proba = best_xgb_model.predict_proba(input_data_scaled)[:, 1][0]
        
        # Affichage du résultat
        st.write(f"**Probabilité de churn : {churn_proba:.2f}**")
        
        # Détermination du label et du discount
        seuil_optimal = 0.2597881853580475  # Utilisez le seuil déterminé précédemment
        CHURN_LABEL = 'LEAVE' if churn_proba >= seuil_optimal else 'STAY'
        
        # Calcul du discount maximal
        max_discount = average_profit - 10
        
        # Calcul du discount basé sur la probabilité de churn
        DISCOUNT = min(round(10 * churn_proba, 2), max_discount)
        
        st.write(f"**Le client est prédit comme : {CHURN_LABEL}**")
        st.write(f"**Discount proposé : {DISCOUNT} €**")

# Prédictions en masse (import CSV)
elif option == 'Prédictions en masse (import CSV)':
    st.subheader("Importez votre fichier CSV :")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        # Lecture du fichier CSV
        data = pd.read_csv(uploaded_file)
        
        # Traitement des données
        data['LOG_DATA'] = np.log1p(data['DATA'])
        data['LOG_TIME_CLIENT'] = np.log1p(data['TIME_CLIENT'])
        data['HOUSE_INCOME_RATIO'] = data['HOUSE'] / data['INCOME']
        data['DATA_TIME_RATIO'] = data['DATA'] / data['TIME_CLIENT']
        data['OVERCHARGE_RATIO'] = data['OVERCHARGE'] / data['REVENUE']
        
        # Sélection des features
        features = ['LOG_DATA', 'LOG_TIME_CLIENT', 'INCOME', 'OVERCHARGE', 'OVERCHARGE_RATIO',
                    'LEFTOVER', 'HOUSE', 'HANDSET_PRICE', 'OVER_15MINS_CALLS_PER_MONTH',
                    'HOUSE_INCOME_RATIO', 'DATA_TIME_RATIO']
        X_test = data[features]
        
        # Normalisation
        X_test_scaled = scaler.transform(X_test)
        
        # Prédictions
        data['CHURN_PROBABILITY'] = best_xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Détermination du label et du discount
        seuil_optimal = 0.2597881853580475  
        data['CHURN_LABEL'] = data['CHURN_PROBABILITY'].apply(lambda x: 'LEAVE' if x >= seuil_optimal else 'STAY')
        
        max_discount = average_profit - 10
        data['DISCOUNT'] = data['CHURN_PROBABILITY'].apply(lambda x: min(round(10 * x, 2), max_discount))
        
        # Affichage des résultats en ordre décroissant
        st.subheader("Résultats des prédictions :")
        data_sorted = data[['CUSTOMER_ID', 'CHURN_PROBABILITY', 'CHURN_LABEL', 'DISCOUNT']].sort_values(by='CHURN_PROBABILITY', ascending=False)
        st.dataframe(data_sorted)
        
        # Option pour télécharger les résultats en format Excel
        def convert_df_to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            processed_data = output.getvalue()
            return processed_data
        
        excel_data = convert_df_to_excel(data_sorted)
        
        st.download_button(
            label="Télécharger les résultats en Excel",
            data=excel_data,
            file_name='predictions_churn.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )