import base64
import io
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import mlflow
import logging
import os
mlflow.set_tracking_uri(uri="http://mlflow_server:5000")

# Set up logging
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

# Create a log file directory
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

# Create a file handler and set the formatter
log_file_path = os.path.join(log_directory, "example.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter(log_format))

# Add the file handler to the root logger
logging.getLogger().addHandler(file_handler)
credit_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = credit_app.server
credit_app.title = "Home Credit "

credit_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H2("Risk-O-Meter"), 
            width={'size': 12, 
                   'offset': 0, 
                   'order': 0}
        ), 
    style = {'textAlign': 'center', 
             'paddingBottom': '1%'}
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Client Files', 
                   style={'font-weight': '800'},
            )
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'width': '100%',
                                             'height': '80%'}),
])


def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['TARGET'])]
    Y = data['TARGET']
    logging.info("prepare_data")
    return X, Y


def make_predictions_with_model_registry_model(model_name, data):
    """ Function for retreiving a model from the Model Registry 
        and make predictions with it. 
    """

    # Prepare the data
    X, Y = prepare_data(data)

    # Load the model and make predictions
    # Raise an exception if there's no model in production stage
    
    model_version = 1
    logging.info('nom du model : '+model_name)
   
    # Load the model and make predictions
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        data['predictions'] = model.predict(X)
        #data['prediction_probs'] = model.predict_proba(X)[:, 1]
        logging.info(mlflow.pyfunc.get_model_dependencies(f"models:/{model_name}/{model_version}"))
        return data
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

        return data


def parse_content(content, filename):
    _, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            "There was an error processing this file. Make sure you're either uploading a CSV or an Excel file."
        ])
    
    # Make predictions with the model in Production stage
    predictions = make_predictions_with_model_registry_model(
        model_name="XGBClassifier", 
        data=df,
    )

    return html.Div([
        html.H5(filename, style={'margin': '2vh 0'}),

        dash_table.DataTable(
            data=predictions.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in predictions.columns],
            filter_action='native',
            sort_action="native",
            sort_mode="multi",
            page_current=0,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal'},
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{predicted_risk} eq "good"',
                        'column_id': 'predicted_risk'
                    },
                    'backgroundColor': '#F0FFF0',
                },
                {
                    'if': {
                        'filter_query': '{predicted_risk} eq "bad"',
                        'column_id': 'predicted_risk'
                    },
                    'backgroundColor': '#FFE4E1',
                },
                ],
        ),
    ])


@credit_app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(content, names):
    if content is not None:
        children = parse_content(content, names)
        return children




if __name__ == "__main__":
    credit_app.run_server(debug=True, host='0.0.0.0', port=8090)


