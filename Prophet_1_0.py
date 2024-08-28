import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.express as px
import itertools
import numpy as np
from io import BytesIO

def perform_cross_validation(model, initial, period, horizon):
    df_cv = cross_validation(model, initial=f"{initial} days", period=f"{period} days", horizon=f"{horizon} days")
    return performance_metrics(df_cv)

def grid_search_prophet(df_prophet, param_grid, initial, period, horizon, metric):
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    best_params = None
    best_metric_value = float('inf')
    results = []

    valid_seasonality_modes = ['additive', 'multiplicative']

    for params in all_params:
        # Assurer que le mode de saisonnalité est valide
        mode = params.get('seasonality_mode', 'additive')
        if mode not in valid_seasonality_modes:
            st.warning(f"Mode de saisonnalité non valide détecté: {mode}")
            continue
        
        model_params = {**params, 'seasonality_mode': mode}
        model = Prophet(**model_params)
        model.add_country_holidays(country_name='FR')
        model.fit(df_prophet)

        df_cv = cross_validation(model, initial=f"{initial} days", period=f"{period} days", horizon=f"{horizon} days")
        df_performance = performance_metrics(df_cv)

        if metric not in df_performance.columns:
            st.warning(f"Métrique sélectionnée non disponible : {metric}")
            continue

        mean_metric = df_performance[metric].mean()

        if mean_metric < best_metric_value:
            best_metric_value = mean_metric
            best_params = model_params

        results.append({
            'params': model_params,
            f'mean_{metric}': mean_metric
        })

    return best_params, pd.DataFrame(results)

def cross_validation_cumulee(df_prophet, initial, period, horizon, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, seasonality_mode):
    results = []
    
    initial_start = df_prophet['ds'].min()
    initial_end = initial_start + pd.Timedelta(days=initial)
    
    while initial_end < df_prophet['ds'].max():
        train_df = df_prophet[(df_prophet['ds'] >= initial_start) & (df_prophet['ds'] < initial_end)]
        test_df = df_prophet[(df_prophet['ds'] >= initial_end) & (df_prophet['ds'] < initial_end + pd.Timedelta(days=horizon))]
        
        # Utilisation des paramètres passés à la fonction pour créer le modèle Prophet
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode
        )
        
        model.add_seasonality(name='mensuel', period=30.5, fourier_order=5)
        model.add_country_holidays(country_name='FR')
        model.fit(train_df)
        
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        forecast_test = forecast[forecast['ds'].isin(test_df['ds'])]
        merged = test_df.merge(forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
        
        merged['cumsum_y'] = merged['y'].cumsum()
        merged['cumsum_yhat'] = merged['yhat'].cumsum()
        merged['cumsum_error'] = merged['cumsum_y'] - merged['cumsum_yhat']
        merged['cumsum_abs_error'] = merged['cumsum_error'].abs()
        merged['cumsum_squared_error'] = merged['cumsum_error'] ** 2
        
        results.append({
            'Fin de période': initial_end,
            'period': period,
            'horizon': horizon,
            'cumulative_mse': merged['cumsum_squared_error'].mean(),
            'cumulative_rmse': np.sqrt(merged['cumsum_squared_error'].mean()),
            'cumulative_mae': merged['cumsum_abs_error'].mean(),
            'cumulative_mape': (merged['cumsum_abs_error'] / merged['cumsum_y']).mean(),
            'cumulative_mdape': (merged['cumsum_abs_error'] / merged['cumsum_y']).median(),
            'cumulative_smape': (200 * merged['cumsum_abs_error'] / (merged['cumsum_y'] + merged['cumsum_yhat'])).mean(),
            'coverage': (merged['yhat_lower'] <= merged['y']).mean() * 100
        })
        
        initial_end += pd.Timedelta(days=period)
        initial_start += pd.Timedelta(days=period)
    
    return pd.DataFrame(results)

def add_real_values_and_calculate_differences(df_forecast, forecast_periods):
    # Filtrer pour n'inclure que les dates dans la période de prédiction
    df_forecast = df_forecast.tail(forecast_periods)
    
    # Renommer les colonnes comme souhaité
    df_forecast = df_forecast.rename(columns={'ds': 'dates', 'yhat': 'Valeurs prédites'})
    
    # Garder uniquement les colonnes nécessaires
    df_forecast = df_forecast[['dates', 'Valeurs prédites']].copy()
    
    # Ajouter une colonne pour les valeurs réelles si elle n'existe pas déjà
    if 'Valeur Réelle' not in df_forecast.columns:
        df_forecast['Valeur Réelle'] = None

    # Permettre à l'utilisateur de remplir les valeurs réelles
    df_forecast = st.data_editor(df_forecast)

    # Convertir la colonne 'Valeur Réelle' en numérique, avec gestion des erreurs
    df_forecast['Valeur Réelle'] = pd.to_numeric(df_forecast['Valeur Réelle'], errors='coerce')

    # Calculer l'écart
    df_forecast['Écart'] = df_forecast['Valeur Réelle'] - df_forecast['Valeurs prédites']

    # Calculer les pourcentages d'écarts
    df_forecast['Pourcentage Écart'] = (df_forecast['Écart'] / df_forecast['Valeurs prédites']) * 100

    return df_forecast

def main():
    st.title("Application de prévision de séries temporelles avec Prophet - Composants supplémentaires")

    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        
        if file_extension == 'csv':
            data = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            data = pd.read_excel(uploaded_file)

        st.write("Aperçu des données:")
        st.write(data.head())

        st.write("Sélectionnez les colonnes pour Prophet:")
        date_column = st.selectbox("Colonne de date", data.columns)
        value_column = st.selectbox("Colonne de valeurs", data.columns)

        data[date_column] = pd.to_datetime(data[date_column])

        st.write("Visualisation des données:")
        fig_data = px.line(data, x=date_column, y=value_column, title='Visualisation des données',
                           labels={date_column: "Date", value_column: "Valeur"})
        fig_data.update_layout(xaxis_title="Date", yaxis_title="Valeur")
        st.plotly_chart(fig_data)

        st.write("Sélectionnez la plage de dates pour l'entraînement:")
        start_date = st.date_input("Date de début", value=data[date_column].min())
        end_date = st.date_input("Date de fin", value=data[date_column].max())

        df_prophet = data[[date_column, value_column]].rename(columns={date_column: 'ds', value_column: 'y'})
        mask = (df_prophet['ds'] >= pd.to_datetime(start_date)) & (df_prophet['ds'] <= pd.to_datetime(end_date))
        df_prophet = df_prophet.loc[mask]

        st.write("Données préparées pour Prophet:")
        st.write(df_prophet.head())

        forecast_periods = st.number_input("Nombre de jours de prévision :", min_value=1, max_value=3650, value=31)

        

        # Paramètres du modèle avec valeurs par défaut
        changepoint_prior_scale = st.number_input("changepoint_prior_scale", value=0.05, format="%.2f")
        seasonality_prior_scale = st.number_input("seasonality_prior_scale", value=10.0, format="%.2f")
        holidays_prior_scale = st.number_input("holidays_prior_scale", value=10.0, format="%.2f")
        seasonality_mode = st.selectbox("Sélectionnez le mode de saisonnalité :", ['additive', 'multiplicative'])


        # Sélectionner le meilleur modèle ou utiliser les paramètres par défaut
        #if st.button("Appliquer les paramètres du modèle"):
            # Utiliser les premiers paramètres valides ou par défaut
        model = Prophet(
               changepoint_prior_scale=changepoint_prior_scale,
               seasonality_prior_scale=seasonality_prior_scale,
               holidays_prior_scale=holidays_prior_scale,
               seasonality_mode=seasonality_mode
           )

        model.add_seasonality(name='mensuel', period=30.5, fourier_order=5)
        model.add_country_holidays(country_name='FR')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)

        total_forecast = forecast['yhat'].tail(forecast_periods).sum()
        st.markdown(f"<h2 style='text-align: center; color: black;'>Cumul des prévisions pour les {forecast_periods} prochains jours : {total_forecast:.2f}</h2>", unsafe_allow_html=True)
    
        st.write("Prévisions :")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))

        st.write("Graphique des prévisions :")
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast)

        st.write("Composants des prévisions:")
        fig_components = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_components)

        st.write("Cross-Validation")
        cross_val_type = st.selectbox("Type de cross-validation :", ["Standard", "Cumulée", "GridSearch", "Exploitation des écarts"])

        initial = st.number_input("Période initiale (en jours) :", min_value=30, value=365)
        period = st.number_input("Période de découpage (en jours) :", min_value=1, value=180)
        horizon = st.number_input("Horizon de prévision (en jours) :", min_value=1, value=365)

        if cross_val_type == "Standard":
            with st.form(key='cv_form'):
                if st.form_submit_button("Lancer la cross-validation"):
                    cv_results = perform_cross_validation(model, initial, period, horizon)
                    st.session_state.cv_results = cv_results  # Stocker les résultats dans le session_state

            # Afficher les résultats de la cross-validation
            if 'cv_results' in st.session_state and st.session_state.cv_results is not None:
                cv_results = st.session_state.cv_results
                st.write("Résultats de la cross-validation :")
                st.write(cv_results)

                # Afficher le bouton de téléchargement en dehors du formulaire
                csv2 = cv_results.to_csv(index=False)
                st.download_button(
                    label="Télécharger le diagnostique",
                    data=csv2,
                    file_name='diag.csv',
                    mime='text/csv'
                )

                # Sélection de la colonne à afficher
                column_to_plot = st.selectbox(
                    "Sélectionnez la colonne à afficher dans le graphique :", 
                    cv_results.columns,
                    index=cv_results.columns.get_loc('mape') if 'mape' in cv_results.columns else 0
                )
                fig_performance = px.line(cv_results, x='horizon', y=column_to_plot, title=f'{column_to_plot} sur la période de validation')
                st.plotly_chart(fig_performance)
        elif cross_val_type == "Cumulée":
            with st.form(key='cumulative_cv_form'):
                if st.form_submit_button("Lancer la cross-validation cumulée"):
                    metrics_df = cross_validation_cumulee(df_prophet, initial, period, horizon, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, seasonality_mode)
                    st.session_state.cumulative_cv_results = metrics_df  # Stocker les résultats dans le session_state

            # Afficher les résultats de la cross-validation cumulée
            if 'cumulative_cv_results' in st.session_state and st.session_state.cumulative_cv_results is not None:
                metrics_df = st.session_state.cumulative_cv_results
                st.write("Résultats de la cross-validation cumulée :")
                st.write(metrics_df)

                # Afficher le bouton de téléchargement en dehors du formulaire
                csv_cumulative = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats cumulés",
                    data=csv_cumulative,
                    file_name='cumulative_metrics.csv',
                    mime='text/csv'
                )

                # Sélection de la colonne à afficher
                column_to_plot = st.selectbox(
                    "Sélectionnez la colonne à afficher dans le graphique :", 
                    metrics_df.columns,
                    index=metrics_df.columns.get_loc('cumulative_mae') if 'cumulative_mae' in metrics_df.columns else 0
                )
                fig_cumulative_performance = px.line(metrics_df, x='Fin de période', y=column_to_plot, title=f'{column_to_plot} sur la période de validation')
                st.plotly_chart(fig_cumulative_performance)

        elif cross_val_type == "GridSearch":
            st.write("Paramètres de Grid Search :")
            changepoint_prior_scale = st.text_input("changepoint_prior_scale (séparés par des virgules)", "0.05")
            seasonality_prior_scale = st.text_input("seasonality_prior_scale (séparés par des virgules)", "10.0")
            holidays_prior_scale = st.text_input("holidays_prior_scale (séparés par des virgules)", "10.0")
            seasonality_mode = st.multiselect("Sélectionnez le(s) mode(s) de saisonnalité :", ['additive', 'multiplicative'], default=['additive'])
            
            metric = st.selectbox("Sélectionnez la métrique pour Grid Search :", ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'])

            if st.button("Lancer le Grid Search"):
                param_grid = {
                    'changepoint_prior_scale': [float(x) for x in changepoint_prior_scale.split(',')],
                    'seasonality_prior_scale': [float(x) for x in seasonality_prior_scale.split(',')],
                    'holidays_prior_scale': [float(x) for x in holidays_prior_scale.split(',')],
                    'seasonality_mode': seasonality_mode,
                }
                
                best_params, results = grid_search_prophet(df_prophet, param_grid, initial, period, horizon, metric)
                
                st.write("Meilleure combinaison d'hyperparamètres trouvée:")
                st.write(best_params)
                if best_params:
                    best_metric_value = results.loc[results['params'] == best_params, f'mean_{metric}'].values
                    if len(best_metric_value) > 0:
                        st.write(f"{metric.upper()} moyen pour cette combinaison : {best_metric_value[0]}")
                    else:
                        st.write(f"{metric.upper()} moyen pour cette combinaison : Non disponible")
                else:
                    st.write("Aucun hyperparamètre valide trouvé.")

                st.write("Tous les résultats du Grid Search :")
                st.write(results)

                csv4 = results.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats du Grid Search",
                    data=csv4,
                    file_name='grid_search_results.csv',
                    mime='text/csv')

        elif cross_val_type == "Exploitation des écarts":
            st.write("Exploitation des écarts entre les valeurs réelles et les prévisions:")
            df_with_real_values = add_real_values_and_calculate_differences(forecast, forecast_periods)
            st.write("Tableau des écarts :")
            st.write(df_with_real_values)
            # Affichage du graphique des écarts
            if 'Écart' in df_with_real_values.columns:
                fig_differences = px.line(df_with_real_values, x='dates', y='Écart', title='Écarts entre les valeurs réelles et les prévisions')
                st.write("Graphique des écarts :")
                st.plotly_chart(fig_differences)
            else:
                st.write("Aucun écart disponible pour l'affichage.")
           

            # Affichage du pourcentage pour sélectionner X%
            pourcentage = st.slider("Sélectionnez le pourcentage des plus gros écarts à afficher", min_value=1, max_value=50, value=15)

            # Calcul du nombre de lignes à afficher pour le pourcentage choisi
            top_n = max(1, round(len(df_with_real_values) * pourcentage / 100))

            # Filtrage des plus gros écarts positifs et négatifs
            plus_gros_positifs = df_with_real_values.nlargest(top_n, 'Écart')
            plus_gros_negatifs = df_with_real_values.nsmallest(top_n, 'Écart')

            st.subheader(f"Top {pourcentage}% des plus gros écarts positifs")
            st.dataframe(plus_gros_positifs)

            st.subheader(f"Top {pourcentage}% des plus gros écarts négatifs")
            st.dataframe(plus_gros_negatifs)


            # Téléchargement des résultats en Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                forecast.to_excel(writer, sheet_name='Prévisions', index=False)
                df_with_real_values.to_excel(writer, sheet_name='Valeurs Réelles et Écarts', index=False)

            st.download_button(
                label="Télécharger les résultats en Excel",
                data=buffer.getvalue(),
                file_name="résultats_prophet.xlsx",
                mime="application/vnd.ms-excel"
            )

if __name__ == "__main__":
    main()
