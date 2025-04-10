from flask import Flask, request
import json
import os
import requests

import matplotlib
matplotlib.use('Agg')  # Set a non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import shap

from xmlot.config     import Configurator
from xmlot.data.load  import load_descriptor
from xmlot.misc.lists import union, intersection
from xmlot.misc.explanations import build_waterfall_plot

import base64
from io import BytesIO

app = Flask(__name__)

# In terminal:
"""
cd ~/Documents/Projects/Xamelot/Python/xmlot/prototype
conda activate _xamelot_
export FLASK_ENV=development
export FLASK_APP=frontend
flask run --port 5000
"""

BACKEND = "http://localhost:5001"
MAIN_DIR = os.getcwd()
YEARS = ['1', '5', '10']
GRID_POINT = 20

#######################
#    MISCELLANEOUS    #
#######################

def _plot_to_html_(fig, style=""):
    """Helper function to convert matplotlib figure to HTML"""
    tmpfile = BytesIO()
    try:
        fig.savefig(tmpfile, format='png', bbox_inches='tight', dpi=100)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plt.close(fig)  # Close the figure to free memory
        return f'<img src=\'data:image/png;base64,{encoded}\' style=\'{style}\'>'
    except Exception as e:
        plt.close(fig)  # Make sure to close the figure even if saving fails
        return f'<div>Error generating plot: {str(e)}</div>'
    finally:
        tmpfile.close()

def _dict_to_html_table_(
        d,
        kprocessing= lambda k_: k_,
        vprocessing= lambda v_: v_,
        vertical_layout=False,
        width=100
):
    """
    Build a HTML table from a dictionary.

    Args:
        - d: the dictionary to process.
        - kprocessing: a function that indicates how to display the keys.
        - vprocessing: a function that indicates how to display the values.
        - vertical_layouts: tells if key-values are displayed in columns or in rows.
        - width: width of the table.

    Returns: HTML
    """

    html_table = """
<table style = "width:{0}%; margin:auto">""".format(width)

    if vertical_layout:
        for k, v in d.items():
            html_table += """
    <tr>
        <th style="width:50%">{0}</th>
        <td style="width:50%">{1}</td>
    </tr> """.format(kprocessing(k), vprocessing(v))
    else:
        html_table += """
    <tr>
    """
        for k in d.keys():
            html_table += """
        <th>{0}</th>""".format(kprocessing(k))
        html_table += """
    </tr>
    <tr>"""
        for v in d.values():
            html_table += """
        <td>{0}</td>""".format(vprocessing(v))
        html_table += """
    </tr>"""
    html_table += """
</table>"""

    return html_table


def _build_waterfall_plot_(explanation):
    """
    Build shap's waterfall plot from SHAP values,
    and encode it for HTML.

    Args:
        - explanation: SHAP values, presented as a dictionary:
            {
                "values"      : SHAP values regarding each feature,
                "base_values" : expected a priori outcomes,
                "data"        : kidney offer
            }
    Returns: an encoded figure (str)
    """
    fig = build_waterfall_plot(explanation, max_display=10)

    # Encode
    return _plot_to_html_(fig, style="width:100%;")

def _get_list_offers_():
    """
    List the files present in the offer directory.
    No particular check is performed.
    """
    list_offers = sorted(os.listdir(MAIN_DIR + "/data/offers/"))
    return list(map(lambda s: s[:-5], list_offers))


######################
#      FRONTEND      #
######################


@app.route('/')
def main():
    # Load
    config = Configurator(desc_dir=MAIN_DIR)
    descriptor = load_descriptor(config, csv_name="/data/description")

    list_offers = _get_list_offers_()

    offer_id = request.args.get("offer", default=None)
    if offer_id not in list_offers:
        offer_id = list_offers[0]
    with open(MAIN_DIR + "/data/offers/{0}.json".format(offer_id)) as json_object:
        json_offer = json.load(json_object)

    # Initialisation
    html_page = """
<html>
    <style>
        table, th, td {
            border:1px solid black;
        }
    </style>
    <body>"""

    # Offer selection
    html_page += """
        <h1>Offer</h1>"""

    html_page += """
        <label for="offers">Select an offer:</label>
        <select id="offers" name="offers" onchange="location = this.value;">"""
    for id_ in list_offers:
        selected = " selected" if id_ == offer_id else ""
        html_page += """
            <option value="/?offer={0}"{1}>{0}</option>""".format(id_, selected)
    html_page += """
        </select>"""

    # Display offer
    with open(MAIN_DIR + "/data/features.json") as json_object:
        features = json.load(json_object)
    features_list = union(features["ACCEPTED"], features["DENIED"])
    features_list = sorted(list({re.match("^[^ #]*", feature)[0] for feature in features_list}))

    offer = {k: json_offer[k] for k in features_list}

    html_page += _dict_to_html_table_(
        offer,
        kprocessing     = lambda k_: descriptor.get_entry(k_).description,
        vertical_layout = True,
        width           = 60
    )

    # Ground-truth
    targets_list = intersection(features["TARGETS"], json_offer.keys())
    groundtruth = {k: json_offer[k] for k in targets_list}

    if len(groundtruth) > 0:
        html_page += _dict_to_html_table_(
            groundtruth,
            kprocessing     = lambda k_: descriptor.get_entry(k_).description,
            vertical_layout = True,
            width           = 60
        )

    # Send offer button
    html_page += """
        <div style="text-align:center">
        <a href="http://localhost:5002/send?offer={0}">
            <button>
                Send offer
            </button>
        </a>
        </div>
    </body>
</html>
    """.format(offer_id)

    return html_page

@app.route('/send')
def send_offer():
    print("SEND_OFFER")

    list_offers = _get_list_offers_()

    offer_id = request.args.get("offer", default=None)

    print("<{0}>".format(offer_id))

    assert offer_id in list_offers, "The offer id must refer to a valid offer."
    with open(MAIN_DIR + "/data/offers/{0}.json".format(offer_id)) as json_object:
        offer = json.load(json_object)

    print("\t-> {0}".format(offer))

    return requests.post(BACKEND + "/predict", json=offer).content

@app.route('/results', methods=['POST'])
def display_results():
    print("\nDISPLAYING RESULTS\n")

    # Load
    config = Configurator(desc_dir=MAIN_DIR)
    descriptor = load_descriptor(config, csv_name="/data/description")

    results = request.json
    offer   = results["offer"]

    print("\t<- {0}".format(results))

    # Initialisation of the HTML page
    html_page = """
<html>
    <style>
    table, th, td {
    border:1px solid black;
    }
    </style>
    <body>
    <div style="width:80%; margin:auto">"""

    # Display offer
    html_page += """
        <h1>Offer</h1>
"""
    html_page += _dict_to_html_table_(
        offer,
        kprocessing     = lambda k_: descriptor.get_entry(k_).description,
        vertical_layout = True,
        width           = 70
    )

    # Display results
    html_page += """
        <h1>If the offer is accepted:</h1>"""

    print("\n\t> ACCEPTED:")

    results_graft   = results["results"]["accepted"]["graft"]
    results_patient = results["results"]["accepted"]["patient"]

    for year_idx, year in enumerate(YEARS):
        print("----- {} -----".format(year))
        html_page += """
        <h2>Before year {0}</h2>""".format(year)

        html_page += """
        <table style = "width:100%; margin:auto; table-layout:fixed">
            <tr>
                <th>Graft failure</th>
                <th>Patient death</th>
            </tr>
            <tr>
                <th>{0}</th>
                <th>{1}</th>
            </tr>
                <th>{2}</th>
                <th>{3}</th>
            </tr>
        </table>""".format(
            "{0:.2%} ({1})".format(
                results_graft["predictions"][year_idx],
                results_graft["uncertainties"][year_idx]
            ),
            "{0:.2%} ({1})".format(
                results_patient["predictions"][year_idx],
                results_patient["uncertainties"][year_idx]
            ),
            _build_waterfall_plot_(results_graft["explanations"][year_idx]),
            _build_waterfall_plot_(results_patient["explanations"][year_idx])
        )

        # Print validation score if possible
        if "score" in results_graft.keys() and "score" in results_patient.keys():
            html_page += """
            <div style="margin-top:1em">
            Validation:
            <ul>"""

            score = results_graft["score"]
            try:
                if 365 * float(year) < float(offer.get("gsurv", 0)):
                    status = "functioning"
                else:
                    if offer.get("gcens") == "Graft failure":
                        status = "failed"
                    else:
                        status = "unknown"
            except (ValueError, TypeError):
                status = "unknown"

            html_page += """
                <li>
                    {0}: at year {2}, the graft is {3}.
                    According to the model, the likelihood of such status is {1}
                    (the closer to 100% the better).
                </li>""".format(
                "Graft",
                "{0:.2%}".format(1 - score) if type(score) is float else "nan",
                year,
                status
            )

            score = results_patient["score"]
            try:
                if 365 * float(year) < float(offer.get("psurv", 0)):
                    status = "alive"
                else:
                    if offer.get("pcens") == "Death of recipient":
                        status = "dead"
                    else:
                        status = "unknown"
            except (ValueError, TypeError):
                status = "unknown"

            html_page += """
                <li>
                    {0}: at year {2}, the patient is {3}.
                    According to the model, the likelihood of such status is {1}
                    (the closer to 100% the better).
                </li>""".format(
                "Patient",
                "{0:.2%}".format(1 - score) if type(score) is float else "nan",
                year,
                status
            )

            html_page += """
            </ul>
            </div>"""

    html_page += """
        <h1>If the offer is instead declined:</h1>"""

    print("\n\t> DENIED")

    fig, ax = plt.subplots(figsize=(20, 10))
    grid = results["results"]["denied"]["grid"]
    for i, (outcome, predictions) in enumerate(results["results"]["denied"]["outcomes"].items()):

        print(predictions)

        x      = list(map(lambda x_: x_ / 365, grid))
        y      = predictions["predictions"]
        x_pred = predictions["time"]

        html_page += """
        Expected event time if '{0}' occurs first: {1:.1f} years  ({2:.1f} days).<br>""".format(
            outcome,
            x_pred,
            x_pred * 365
        )

        x[-1] = 5
        plt.step(x, y, label="{0}".format(outcome), color="C{0}".format(i))

        if x_pred <= x[-1]:
            plt.vlines(x_pred, 0, 1, color="C{0}".format(i), linestyles="dashed")

    plt.legend()

    html_page += _plot_to_html_(fig)

    html_page += "Explanations at {0:.1f} years:".format(grid[GRID_POINT] / 365)

    html_page += """
    <div style="overflow:hidden">"""

    results_denied = results["results"]["denied"]["outcomes"]
    width = int((1 / len(results_denied)) * 100)
    for outcome in results_denied.keys():
        html_page += """
        <div style="width:{}%;float:left">""".format(width)

        html_page += _build_waterfall_plot_(results_denied[outcome]["explanations"][0])
        html_page += """
        <div style="text-align:center">{}</div>""".format(outcome)

        html_page += """
        </div>"""

    html_page += """
    </div>"""

    if "scores" in results["results"]["denied"].keys():
        error = results["results"]["denied"]["scores"]["error"]
        likelihood = float(results["results"]["denied"]["scores"]["likelihood"])

        html_page += """
        <div style="margin-top:1em">
        Validation:
        <ul>
            <li>'{0}' occured after waiting {1} days: 
            according to the model, the likelihood of such event is {2:.2%}.</li>
            <li>Assuming '{0}' occured first, 
            the error between the predicted event time and the actual one is {3:.0f} days.</li>
        </ul>
        </div>
        """.format(offer["dcens"], offer["dsurv"], likelihood, error)

    html_page += """
        <div style="text-align:center">
            <a href="http://localhost:5000/">Return to offer selection.</a>
        </div>
    </div>
    </body>
</html>
    """

    return html_page
