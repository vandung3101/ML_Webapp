The project had been deployed at https://share.streamlit.io/vandung3101/ml_webapp/main/webapp.py

Dataset (.csv):
    - Algorithm work best with dataset contains binary features (others run well but the prediction will bad) 
    - Record with NaN values would be removed by the algorithm
    - Dataset should following structure (see demo dataset /data.csv):
        feature_0   feature_1   ....    feature_n   Target
        --------    ---------   ....    ---------   -----
        --------    ---------   ....    ---------   -----
        --------    ---------   ....    ---------   -----

WebApp:
    - Make sure your system satisfied 'requirements.txt'
    - Run
        $ streamlit run webapp
    - Webapp now will open with your localhost
    - Follow Webapp UI