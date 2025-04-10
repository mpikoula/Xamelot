from xmlot.data.build   import build_single_classification_from_survival
from xmlot.models.model import FromTheShelfModel

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline      import Pipeline
from sklearn.linear_model  import LinearRegression
from sklearn.calibration   import calibration_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#################
#   FUNCTIONS   #
#################

def survival_calibration(data, predictions, xcens, xsurv, events=[1.], censored=0., years=(1, 5, 10)):
    results = dict()
    
    for year_idx, year in enumerate(years):
        results[year] = dict()

        # Get event status with respect to year.
        df = pd.DataFrame(
            {"groundtruth": build_single_classification_from_survival(
                data, 
                365 * year, 
                xcens, 
                xsurv,
                events,
                censored=censored
            )}
        )

        # Add predictions
        df["prediction"] = predictions[year_idx]

        # Remove censored events
        df = df.drop(df[df["groundtruth"] == 2].index)

        # Get calibration curve
        prob_true, prob_pred = calibration_curve(
            df["groundtruth"],
            df["prediction"],
            n_bins=50,
            strategy="quantile",
            pos_label=1)

        # Store results
        results[year]["pred"] = prob_pred.tolist()
        results[year]["true"] = prob_true.tolist()
    return results


def plot_calibration_results(results):
    for year, subresults in results.items():
        plt.plot(subresults["pred"], subresults["true"], ".", label="Year {}".format(year))

    plt.plot([0, 1], [0, 1], ":k", )
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.legend()
    plt.axis('scaled')
    plt.show()


#############
#   MODEL   #
#############


class CalibratedSurvivalModel(FromTheShelfModel):
    def __init__(self, model, accessor_code=None, hyperparameters=None):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_model = model
        self.m_years = hyperparameters["years"]
        self.m_accessor_code = accessor_code

        if "degree" in self.m_hyperparameters.keys():
            degree = hyperparameters["degree"]
        else:
            degree = 1
        self.m_calibrator = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=False))]
        )

    def fit(self, data_train, parameters=None):
        
        # Handle the case where parameters might be None
        if parameters is None:
            parameters = {}
    
        accessor = getattr(data_train, self.m_accessor_code)
        predictions = self.m_model.predict(accessor.features)[0]
    
        results = survival_calibration(
            data_train,
            predictions,
            xcens=accessor.event,
            xsurv=accessor.duration,
            years=self.m_years
        )
    
        # Aggregate all results in the same container
        p = list()  # probabilities on x-axis
        t = list()  # true ratios on y-axis
        for year in self.m_years:
            p += results[year]["pred"]
            t += results[year]["true"]
    
        # Avoid division by zero by replacing zero values in t with a small epsilon value
        epsilon = parameters.get("epsilon", 1e-3)  # Use .get() to avoid key errors
    
        # Avoid division by zero in the transformation
        t = np.array(t)
        
        t = np.maximum(t, epsilon)  # Replace zeros with epsilon (small positive number)

        # Perform the transformation
        t = (1 / t) - 1
        t = np.maximum(t, epsilon)  # Ensure no negative or zero values
    
        p = np.array(p)

        # Fit sigmoid
        self.m_calibrator.fit(p.reshape(-1, 1), np.log(t).reshape(-1, 1))  # Log-transform t

    def predict(self, x, parameters=None):

        # Handle the case where parameters might be None
        if parameters is None:
            parameters = {}
            
        prediction = self.m_model.predict(x)[0]
        prediction = prediction.apply_(lambda p: self.m_calibrator.predict([[p]])[0, 0])
        return 1 / (1 + np.exp(prediction))