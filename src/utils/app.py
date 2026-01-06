# src/utils/app.py
import json
import os

import dash
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

# Importiere die Ladefunktion (relativer Import, da im selben Paket 'utils')
from .dashboard_data_loader import load_experiment_data

# --- Konfiguration ---
# Bestimme das Projekt-Root-Verzeichnis relativ zum aktuellen Skriptpfad.
# Annahme: app.py ist in src/utils/
current_script_path_app = os.path.abspath(__file__)
utils_dir_app = os.path.dirname(current_script_path_app)
src_dir_app = os.path.dirname(utils_dir_app)
project_root_app = os.path.dirname(src_dir_app)

# Passe diesen Ordnernamen an, falls dein Basis-Ergebnisordner anders heißt.
CONFIGURED_RESULTS_FOLDER_NAME_APP = "qat_runs" # Basierend auf deiner vorherigen Ausgabe
RESULTS_BASE_DIRECTORY = os.path.join(
    project_root_app, "results", CONFIGURED_RESULTS_FOLDER_NAME_APP
)


# --- Daten laden ---
df_experiments = pd.DataFrame()  # Initialisiere als leeren DataFrame
try:
    print(f"INFO [App]: Lade Experimentdaten aus: {RESULTS_BASE_DIRECTORY}")
    df_experiments = load_experiment_data(RESULTS_BASE_DIRECTORY)
    if df_experiments.empty:
        print(f"WARNUNG [App]: Keine Experimentdaten aus '{RESULTS_BASE_DIRECTORY}' geladen. Die Tabelle wird leer sein.")
    else:
        print(f"INFO [App]: Erfolgreich {len(df_experiments)} Experimente für das Dashboard geladen.")
except FileNotFoundError:
    print(f"FEHLER [App]: Ergebnisverzeichnis nicht gefunden unter {RESULTS_BASE_DIRECTORY}. Stelle sicher, dass der Pfad korrekt ist.")
except Exception as e:
    print(f"FEHLER [App]: Experimentdaten konnten nicht geladen werden: {e}")


# --- Dash App Initialisierung ---
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Der exakte URL-Basispfad, unter dem die App über den JupyterHub-Proxy erreichbar ist.
# Diesen verwenden wir NUR für routes_pathname_prefix (Asset-Pfade).
app_proxy_path_for_routes = '/user/paul.darius.fraunhofer.de/proxy/8051/'

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    # Die App soll intern auf dem Wurzelpfad "/" lauschen,
    # da wir annehmen, dass jupyter-server-proxy den app_proxy_path entfernt.
    requests_pathname_prefix='/',
    
    # routes_pathname_prefix setzen wir, damit Dash die URLs zu seinen
    # Assets (JS, CSS) und internen Dash-Seiten korrekt mit dem Proxy-Pfad generiert.
    routes_pathname_prefix=app_proxy_path_for_routes
)
app.title = "QAT Experiment Dashboard"

# --- Layout der App ---
app.layout = html.Div(style={'padding': '20px'}, children=[
    html.H1("QAT Experiment Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
    html.H2("Übersicht der Experimente"),
    dash_table.DataTable(
        id='experiments-table',
        columns=[{"name": i, "id": i, "hideable": True} for i in df_experiments.columns if i != 'json_log_path'],
        data=df_experiments.to_dict('records'),
        row_selectable='single', sort_action="native", filter_action="native", page_size=15,
        style_table={'overflowX': 'auto', 'marginTop': '20px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'minWidth': '100px', 'width': '150px', 'maxWidth': '250px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'textAlign': 'left', 'padding': '5px'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
    ),
    html.Hr(style={'marginTop': '30px', 'marginBottom': '30px'}),
    html.H2("Detailansicht für ausgewählten Lauf"),
    dcc.Loading(id="loading-details", type="circle", children=[
        html.Div(id='run-detail-info-config'),
        dcc.Graph(id='loss-plot-detail'),
        html.Div(id='run-detail-info-metrics'),
        dcc.Graph(id='lr-plot-detail'),
    ]),
])

# --- Callbacks für Interaktivität ---
@app.callback(
    [Output('run-detail-info-config', 'children'), Output('loss-plot-detail', 'figure'),
     Output('run-detail-info-metrics', 'children'), Output('lr-plot-detail', 'figure')],
    [Input('experiments-table', 'selected_rows')],
    prevent_initial_call=True
)
def display_run_details(selected_rows):
    ctx = dash.callback_context
    if not selected_rows or not ctx.triggered:
        empty_figure = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}}}
        return html.P("Bitte einen Lauf aus der Tabelle auswählen, um Details anzuzeigen."), empty_figure, "", empty_figure
    selected_row_index = selected_rows[0]
    selected_run_id = df_experiments.iloc[selected_row_index]['run_id']
    selected_run_json_path = df_experiments.iloc[selected_row_index]['json_log_path']
    if not selected_run_json_path or not isinstance(selected_run_json_path, str) or not os.path.exists(selected_run_json_path):
        return html.P(f"JSON-Logdatei für Lauf '{selected_run_id}' nicht gefunden: '{selected_run_json_path}'."), {}, "", {}
    try:
        with open(selected_run_json_path, 'r') as f: run_data = json.load(f)
    except Exception as e:
        return html.P(f"Fehler beim Laden der Detaildaten für '{selected_run_id}': {e}"), {}, "", {}
    config_details_children = [
        html.H3(f"Details für Lauf: {selected_run_id}"), html.H4("Konfiguration:"),
        html.Pre(json.dumps(run_data.get('input_configuration_from_yaml', {}), indent=2), style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'overflowX': 'auto'})
    ]
    epoch_history = run_data.get('epoch_history_data', {})
    train_loss, val_loss = epoch_history.get('train_loss_per_epoch', []), epoch_history.get('val_loss_per_epoch', [])
    epochs = list(range(1, len(train_loss) + 1))
    loss_figure = {}
    if epochs and train_loss and val_loss:
        df_loss_plot = pd.DataFrame({'Epoch': epochs, 'Train Loss': train_loss, 'Validation Loss': val_loss})
        loss_figure = px.line(df_loss_plot, x='Epoch', y=['Train Loss', 'Validation Loss'], title="Trainings- & Validierungsverlust", markers=True)
        loss_figure.update_layout(transition_duration=300)
    lr_history = epoch_history.get('lr_per_epoch', [])
    lr_figure = {}
    if epochs and lr_history:
        df_lr_plot = pd.DataFrame({'Epoch': epochs[:len(lr_history)], 'Learning Rate': lr_history})
        lr_figure = px.line(df_lr_plot, x='Epoch', y='Learning Rate', title="Lernratenverlauf", markers=True)
        lr_figure.update_layout(yaxis_type="log", transition_duration=300)
    metrics_children = [
        html.H4("Beste Validierungsmetriken:"),
        html.Pre(json.dumps(run_data.get('best_model_metrics_achieved_val', {}), indent=2), style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'overflowX': 'auto'}),
        html.H4("PyTorch Test-Eval:"),
        html.Pre(json.dumps(run_data.get('evaluation_execution_details', {}).get('pytorch_eval_summary', {}), indent=2), style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'overflowX': 'auto'}),
        html.H4("FHE Test-Eval:"),
        html.Pre(json.dumps(run_data.get('evaluation_execution_details', {}).get('fhe_eval_summary', {}), indent=2), style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'overflowX': 'auto'}),
    ]
    return config_details_children, loss_figure, metrics_children, lr_figure


if __name__ == '__main__':
    if not os.path.exists(RESULTS_BASE_DIRECTORY):
         print(f"KRITISCHER FEHLER [App]: Ergebnisverzeichnis '{RESULTS_BASE_DIRECTORY}' nicht gefunden.")
    elif df_experiments.empty:
        print(f"WARNUNG [App]: Keine Experimentdaten aus '{RESULTS_BASE_DIRECTORY}' geladen.")
    
    print(f"INFO [App]: Dash App wird gestartet.")
    print(f"INFO [App]: JUPYTERHUB_SERVICE_PREFIX (Umgebungsvariable): {os.environ.get('JUPYTERHUB_SERVICE_PREFIX')}")
    print(f"INFO [App]: Verwendeter app_proxy_path_for_routes für routes_pathname_prefix: '{app_proxy_path_for_routes}'")
    print(f"INFO [App]: app.config.requests_pathname_prefix (nach Dash-Init): {app.config.get('requests_pathname_prefix')}")
    print(f"INFO [App]: app.config.routes_pathname_prefix (nach Dash-Init): {app.config.get('routes_pathname_prefix')}")
    
    print("\nINFO [App]: Registrierte Routen im Flask Server (app.server.url_map) VOR app.run:")
    try:
        for rule in app.server.url_map.iter_rules(): # type: ignore
            print(f"  Endpoint: {rule.endpoint}, Path: {rule.rule}")
    except Exception as e:
        print(f"  Fehler beim initialen Iterieren der Routen: {e}")
    print("-----\n")
    
    app.run(debug=True, port=8051, host='0.0.0.0')