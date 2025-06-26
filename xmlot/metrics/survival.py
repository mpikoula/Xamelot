# Provides a suite of metrics to evaluate performances of survival analysis models.

from lifelines.utils  import concordance_index
from pycox.evaluation import EvalSurv
from sksurv.datasets  import get_x_y

from xmlot.models.lifelines import LifelinesCoxModel
from xmlot.models.scikit    import ScikitCoxModel,       \
                                   RandomSurvivalForest, \
                                   XGBoost
from xmlot.models.pycox     import DeepSurv,             \
                                   DeepHit


############################
#      LIBRARY SCORES      #
############################
# Temporary: get concordance scores from each library.

def _lifelines_score_(model, df_test):
    a_test  = getattr(df_test, model.accessor_code)

    return concordance_index(
        a_test.durations,
        -model.model.predict_partial_hazard(a_test.features).to_numpy(),
        a_test.events
    )

def _skcox_score(model, df_test):
    a_test = getattr(df_test, model.accessor_code)

    x, _ = get_x_y(
        a_test.df,
        [a_test.event, a_test.duration],
        pos_label=1,
        survival=True
    )

    return concordance_index(
        a_test.durations,
        -model.model.predict(x),
        a_test.events
    )

def _sksurv_score_(model, df_test):
    a_test = getattr(df_test, model.accessor_code)

    x, y = get_x_y(
        a_test.df,
        [a_test.event, a_test.duration],
        pos_label=1,
        survival=True
    )
    return model.model.score(x, y)

def _deepsurv_score_(model, df_test):
    a_test = getattr(df_test, model.accessor_code)

    return concordance_index(
        a_test.durations,
        -model.predict(a_test.features),
        a_test.events
    )

def _deephit_score_(model, df_test):
    a_test  = getattr(df_test, model.accessor_code)
    interpolation = 10
    
    # Handle both standard and weighted DeepHit models
    try:
        # Try standard PyCox DeepHit approach
        surv = model.model.interpolate(interpolation).predict_surv_df(
            a_test.features.to_numpy()
        )
    except AttributeError:
        # Fallback for weighted DeepHit models
        surv = model.predict_surv(a_test.features.to_numpy())
    
    ev = EvalSurv(
        surv,
        a_test.durations.to_numpy(),
        a_test.events.to_numpy(),
        censor_surv='km'
    )
    return ev.concordance_td('antolini')

#########################
#      CONCORDANCE      #
#########################

def concordance(model, df_test, **_):
    """
    TODO: implement a concordance score that does not depend on the type of model!
    """
    if type(model) == LifelinesCoxModel:
        return _lifelines_score_(model, df_test)
    elif type(model) == ScikitCoxModel:
        return _skcox_score(model, df_test)
    elif type(model) == DeepSurv:
        return _deepsurv_score_(model, df_test)
    elif type(model) == DeepHit:
        return _deephit_score_(model, df_test)
    elif type(model) == RandomSurvivalForest:
        return _sksurv_score_(model, df_test)
    elif type(model) == XGBoost:
        return _sksurv_score_(model, df_test)
    else:
        a_test = getattr(df_test, model.accessor_code)

        return concordance_index(
                a_test.durations,
                model(a_test.features),
                a_test.events
            )

# def concordance_td(model, sdm, interpolation=10):
#     surv = model.m_model.interpolate(interpolation).predict_surv_df(sdm.covariates.to_numpy())
#     ev = EvalSurv(surv, sdm.durations.to_numpy(), sdm.events.to_numpy(), censor_surv='km')
#     return ev.concordance_td('antolini')
