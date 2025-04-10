from flask import Flask, request
import requests
import dill as pickle
import numpy as np
import pandas as pd
import os
import json
import torch
import re
from json import JSONEncoder
import shap

from xmlot.config import Configurator
from xmlot.data.load import load_descriptor
from xmlot.misc.lists import difference, intersection, union
from xmlot.misc.explanations import aggregate_shap_explanation, reformat_explanation

app = Flask(__name__)

# In terminal:
"""
cd ~/Documents/Projects/Xamelot/Python/xmlot/prototype
conda activate _xamelot_
export FLASK_ENV=development
export FLASK_APP=backend
flask run --port 5001
"""

# Security:
# https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https

FRONTEND = "http://localhost:5002"
MAIN_DIR = os.getcwd()
YEARS = ['1', '5', '10']
GRID_POINT = 20

#######################
#    MISCELLANEOUS    #
#######################

class SHAPModel:
    """
    As SHAP models are not preserved during serialisation,
    we need to rebuild a similar object that returns predictions as follows:
    prediction = model.f(x)
    """

    def __init__(self, model_, idx, accepted=True):
        self.m_model = model_
        self.m_idx = idx
        self.m_accepted = accepted

    def f(self, x):
        if self.m_accepted:
            return self.m_model.predict(x)[self.m_idx, :].detach().numpy()
        else:
            # Denied case
            return self.m_model.predict_CIF(x).detach().numpy()[:, self.m_idx, :].transpose() # GRIDPOINT

def get_original_features(ohe):
    return difference(ohe.columns, [
        "dcens",
        "dsurv",
        "gcens",
        "gsurv",
        "gclass",
        "pcens",
        "psurv",
        "pclass",
        "tcens",
        "tsurv",
        "tclass",
        "mclass"
    ])

###########################
#    PREDICTION ENGINE    #
###########################

class Engine:
    def __init__(self):
        # Main directory
        self.m_dir = MAIN_DIR
        print("MAIN_DIR={}".format(self.m_dir))

        # Features names and selection
        with open(self.data_dir + "/features.json") as json_object:
            self.m_features = json.load(json_object)

        config = Configurator(desc_dir=MAIN_DIR)
        self.m_descriptor = load_descriptor(config, csv_name="/data/description")

    @property
    def descriptor(self):
        return self.m_descriptor

    @property
    def features(self):
        return self.m_features

    @property
    def data_dir(self):
        return self.m_dir + "/data"

    def load(self, filename):
        """
        Load Python objects from serialised files.
        """
        file = open(self.data_dir + filename + ".pkl", 'rb')
        unpickled = pickle.load(file)
        file.close()
        return unpickled

    def format_result(self, predictions, explanations, ohe, scaler, i=None, scores=None):
        pred = predictions if i is None else predictions[i]

        formatted_explanations = [
            reformat_explanation(aggregate_shap_explanation({
                "base_values": explanation.base_values if i is None else explanation.base_values[i],
                "values": explanation.values.tolist() if i is None else explanation.values[:, i].tolist(),
                "data": explanation.data.to_dict()
            }),
                ohe, scaler, self.descriptor) for explanation in explanations
        ]

        results = {
            "predictions": pred,
            "uncertainties": ["Uncertainty is not implemented yet." for _ in YEARS],
            "explanations": formatted_explanations,
            "scores": scores
        }

        return results

    def predict_accepted(self, offer, outcome):
        def _build_score_(xcens_, xsurv_, offer_, prediction_, year_):
            """
            Compute L1 distance between prediction and ground-truth
            """
            if xcens_ in offer_.keys() and xsurv_ in offer_.keys():
                if float(year_) * 365 < float(offer_[xsurv_][0]):  # Alive
                    groundtruth = 0
                elif offer_[xcens_][0] != "Censored":  # Event
                    groundtruth = 1
                else:  # Censored
                    return "nan"
                return np.abs(prediction_ - groundtruth)
            # Else, if ground-truth is missing, do nothing
            return "nan"

        print("\t> PREDICT ACCEPTED ({})".format(outcome))

        # Load models
        ohe = self.load("/dump/accepted/ohe")
        scaler = self.load("/dump/accepted/scaler")
        imputer = self.load("/dump/accepted/_imputer")
        model = self.load("/dump/accepted/" + outcome + "/model")

        # Create DataFrame and select features
        features = intersection(offer.keys(), get_original_features(ohe))
        # Extract first element from each list value and handle missing values
        offer_scalar = {}
        for k, v in offer.items():
            try:
                offer_scalar[k] = v[0]
            except (IndexError, TypeError):
                offer_scalar[k] = np.nan
        
        df = pd.DataFrame([offer_scalar])[features]
        
        # Apply one-hot encoding BEFORE imputation
        df = ohe.encode(df, reboot_encoded_columns=False)

        # Get required features for imputation
        required_features = imputer.feature_names_in_.tolist()
        
        # Create DataFrame for imputation with only the features the imputer knows about
        imp_df = pd.DataFrame(0, index=df.index, columns=required_features)
        
        # Fill in the values we have
        for col in required_features:
            if col in df.columns:
                imp_df[col] = df[col]
                
        # Apply imputation
        imp_df = pd.DataFrame(imputer.transform(imp_df), columns=required_features)
        
        # Merge back the imputed features
        df_final = df.copy()
        
        # Update with imputed values where available
        for col in required_features:
            if col in df_final.columns:
                df_final[col] = imp_df[col]
                
        # Apply scaling
        df_final = scaler(df_final)
        
        # Fill any remaining NaN values with 0 (this should not happen with proper imputation)
        df_final = df_final.fillna(0)

        # Predict
        prediction = model.predict(df_final).flatten().tolist()

        # Explain
        explainers = self.load("/dump/accepted/" + outcome + "/explainers")
        explanations = []
        for idx, explainer in enumerate(explainers):
            explainer.model = SHAPModel(model, idx=idx, accepted=True)
            explanation = explainer(df_final.iloc[0])
            explanations.append(explanation)

        # Format results
        scores = None
        if all(k in offer for k in [outcome + "cens", outcome + "surv"]):
            scores = []
            for year in YEARS:
                scores.append(_build_score_(outcome + "cens", outcome + "surv", 
                                         offer, prediction, year))

        return self.format_result(prediction, explanations, ohe, scaler, scores=scores)

    def predict_denied(self, offer):
        print("\t> PREDICT DENIED")

        # Load models
        model = self.load("/dump/denied/model")
        ohe = self.load("/dump/denied/ohe")
        scaler = self.load("/dump/denied/scaler")
        imputer = self.load("/dump/denied/_imputer")
        discretiser = self.load("/dump/denied/discretiser")

        # Create DataFrame and select features
        features = intersection(offer.keys(), get_original_features(ohe))
        # Extract first element from each list value
        offer_scalar = {k: v[0] for k, v in offer.items()}
        df = pd.DataFrame([offer_scalar])[features]
        
        # Apply one-hot encoding BEFORE imputation
        df = ohe.encode(df, reboot_encoded_columns=False)

        # Get required features for imputation
        required_features = imputer.feature_names_in_.tolist()
        
        # Create DataFrame for imputation with only the features the imputer knows about
        imp_df = pd.DataFrame(0, index=df.index, columns=required_features)
        
        # Fill in the values we have
        for col in required_features:
            if col in df.columns:
                imp_df[col] = df[col]
                
        # Apply imputation
        imp_df = pd.DataFrame(imputer.transform(imp_df), columns=required_features)
        
        # Merge back the imputed features
        df_final = df.copy()
        
        # Update with imputed values where available
        for col in required_features:
            if col in df_final.columns:
                df_final[col] = imp_df[col]
                
        # Apply scaling
        df_final = scaler(df_final)

        # Derive Cumulative incidence function
        predictions = model.predict(df_final).squeeze().tolist()

        # Derive expected event times
        predicted_t = list()
        t = torch.Tensor(discretiser.grid) / 365
        pmf = model.predict_pmf(df_final)
        
        for risk in range(3):
            p = np.transpose(pmf[risk])[0]
            predicted_t.append(torch.dot(t, p) / p.sum())

        # Explain
        explainer = self.load("/dump/denied/explainer")
        explainer.model = SHAPModel(model, idx=GRID_POINT, accepted=False)
        explanations = [explainer(df_final.iloc[0])]

        results = {
            "grid": discretiser.grid,
            "outcomes": {
                "Transplant": self.format_result(predictions, explanations, ohe, scaler, i=0),
                "Removal from the waiting list": self.format_result(predictions, explanations, ohe, scaler, i=1),
                "Death": self.format_result(predictions, explanations, ohe, scaler, i=2)
            }
        }

        for i, outcome in enumerate(results["outcomes"].keys()):
            results["outcomes"][outcome]["time"] = float(predicted_t[i])

        # Ground-truth
        if "dcens" in offer.keys() and "dsurv" in offer.keys():
            key_to_idx = {
                "Transplant": 0,
                "Removal": 1,
                "Death": 2
            }

            i = np.argmin(list(map(lambda l: np.abs(offer["dsurv"][0]-l), discretiser.grid)))
            if discretiser.grid[i] < offer["dsurv"][0]:
                i = min(22, i+1)

            xcens = key_to_idx[offer["dcens"][0]]
            xsurv = offer["dsurv"][0]

            scores = {
                "error": float(np.abs(predicted_t[xcens] * 365 - xsurv)),
                "likelihood": float(pmf[xcens][i][0] / np.transpose(pmf[xcens])[0].sum())
            }

            results["scores"] = scores

        return results

def clean_nan_inf(obj):
    """Clean NaN and inf values from a nested structure"""
    if isinstance(obj, dict):
        return {k: clean_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_inf(x) for x in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_nan_inf(obj.tolist())
    return obj

def convert_to_json_safe(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj

@app.route('/predict', methods=['POST'])
def predict():
    print("PREDICT")

    engine = Engine()

    # Load feature list
    features_list = union(engine.features["DENIED"], engine.features["ACCEPTED"])
    features_list = sorted(list({re.match("^[^ #]*", feature)[0] for feature in features_list}))
    print("\n> Feature list: {}\n".format(features_list))

    # Load offer
    offer = request.json
    offer = {k: offer[k] for k in features_list}
    print("\t<- {0}".format(offer))
    results = {"offer": offer, "results": dict()}

    # Convert all values to lists
    offer = {k: [v] for k, v in offer.items()}

    # Accepted case
    results["results"]["accepted"] = dict()
    accepted_offer = {feature: offer[feature] for feature in engine.features["ACCEPTED"]}
    results["results"]["accepted"]["graft"] = engine.predict_accepted(accepted_offer, outcome="graft")
    results["results"]["accepted"]["patient"] = engine.predict_accepted(accepted_offer, outcome="patient")

    # Denied case
    denied_offer = {feature: offer[feature] for feature in engine.features["DENIED"]}
    results["results"]["denied"] = engine.predict_denied(denied_offer)

    print("\t-> {0}".format(results))

    return requests.post(FRONTEND + "/results", json=results).content
