import os
import torch 
import pandas as pd
from epiweeks import Week
from itertools import product
import preprocess_data as prep
from models import LSTMLogNormalModel, train 
from dotenv import load_dotenv

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained with the regional health data before the year selected. 
'''

load_dotenv()

# Access the environment variables
data_path = os.getenv('DATA_PATH')

df_map = pd.read_csv(f'{data_path}/map_regional_health.csv')

if __name__ == '__main__':

    boxcox = False
    disease = 'dengue'
    min_year = 2015

    geocodes = [2931350, 2933307, 2302503, 3119401, 3549805,
           3541406, 1200401, 1200203, 1716109, 4113700, 4103701, 4104808,
            5201405, 5102637, 5215231]
    

    columns_to_normalize = ['casos','epiweek', 'enso']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = prep.load_cases_data(disease=disease)
    
    df = df.loc[df.index >= pd.to_datetime(Week(2015,41).startdate())]

    enso = prep.load_enso_data()

    df_map = pd.read_csv(f'{data_path}/map_regional_health.csv')
    
    for GEOCODE, TEST_YEAR in product(geocodes, [2023,2024,2025,2026]): 

        print(f'{GEOCODE} - {TEST_YEAR}')

        macro_reg = df_map.loc[df_map.geocode == GEOCODE].macroregional_geocode.values[0]

        df_f = df.loc[df.macroregional_geocode == macro_reg].copy()

        # generate the samples to train and test based on the regional data 
        X_train, y_train, norm  = prep.generate_regional_train_samples(df_f,
                                                                        enso,
                                                                        TEST_YEAR,
                                                                        columns_to_normalize=columns_to_normalize,
                                                                        boxcox = boxcox, min_year =  min_year)

        model = LSTMLogNormalModel(hidden=64, features=len(columns_to_normalize), 
                        predict_n=52, look_back=89)
            
        label = f'{GEOCODE}_{TEST_YEAR-1}_base'
        batch_size = 1
        epochs =500
        cross_val = False
        verbose = 0
        min_delta = 0
        patience= 25

        if TEST_YEAR > 2023:     
            model_path = f'./saved_models/trained_{disease}_{GEOCODE}_{TEST_YEAR-2}_base.pt'
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)  
    
        model = train(model, X_train, y_train, label=label, batch_size=batch_size, epochs=epochs,
                                                    overwrite=True, cross_val = cross_val, monitor='val_loss',
                                                    verbose=verbose, doenca=disease,
                                                    min_delta = min_delta, patience=patience)