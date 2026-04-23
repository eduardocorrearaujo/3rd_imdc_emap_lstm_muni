#import warnings
import pandas as pd
from epiweeks import Week
from itertools import product
import preprocess_data as prep
from models import LSTMLogNormalModel, apply_geocode_predictions
#warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import torch 


THR = 0.1

geocodes = [2931350, 2933307, 2302503, 3119401, 3549805,
           3541406, 1200401, 1200203, 1716109, 4113700, 4103701, 4104808,
            5201405, 5102637, 5215231]
    

if __name__ == '__main__': 
    model_name = 'base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxcox = False 
    disease = 'dengue'

    columns_to_normalize = ['casos','epiweek', 'enso']

    df = prep.load_cases_data(disease=disease)
    
    df = df.loc[df.index >= pd.to_datetime(Week(2015,41).startdate())]

    enso = prep.load_enso_data()


    for GEOCODE, TEST_YEAR in product(geocodes, [2023,2024,2025,2026]): \
    
        print(f'{GEOCODE} - {TEST_YEAR}')

        df_apply = df.loc[df.geocode == GEOCODE]        

        # base model 
        model = LSTMLogNormalModel(hidden=64, features=len(columns_to_normalize), 
                            predict_n=52, look_back=89)
                                        
        model_path = f'./saved_models/trained_{disease}_{GEOCODE}_{TEST_YEAR-1}_{model_name}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)  

        df_preds =  apply_geocode_predictions(model, df_apply, enso, TEST_YEAR, columns_to_normalize, n_passes = 500, boxcox=boxcox)

        df_preds.to_csv(f'predictions/preds_{disease}_{GEOCODE}_{TEST_YEAR}.csv', index = False)