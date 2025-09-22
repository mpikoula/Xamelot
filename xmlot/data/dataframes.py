# In the present library, DataFrames have been chosen to be the primary format for data.
# This dependency provides a suite of tools to ease their use.

import pandas as pd

from warnings   import catch_warnings, simplefilter
from contextlib import nullcontext

from xmlot.misc.misc           import get_var_name
from xmlot.misc.lists import difference, intersection, union


#####################
#     ACCESSORS     #
#####################
# Build pandas' accessors to easily access features, targets, events, durations, in a DataFrame
# More information about accessors at:
# https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors


def build_survival_accessor(event, duration, accessor_code="surv", exceptions=tuple(), disable_warning=True):
    """
    Build an accessor dedicated to the management of survival data.

    Args:
        - event           : event name;
        - duration        : duration name;
        - accessor_code   : code to access extended properties (for example: `df.surv.event`);
        - disable_warning : remove warnings linked to several uses of that function.

    Returns: the corresponding accessor class;
             instances are built when called from the DataFrame (for example: `df.surv`)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class SurvivalAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_event          = event
                self.m_duration       = duration
                self.m_stratification = event
                self.m_exceptions     = exceptions

            @staticmethod
            def _validate(obj):
                pass
                # # verify there is a column latitude and a column longitude
                # if event not in obj.columns or duration not in obj.columns:
                #     raise AttributeError("Must have {0} and {1}.".format(event, duration))

            @property
            def event(self):
                return self.m_event

            @property
            def events(self):
                return self._obj[self.event]

            @property
            def duration(self):
                return self.m_duration

            @property
            def durations(self):
                return self._obj[self.duration]

            @property
            def target(self):
                return [self.event, self.duration]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                return difference(self._obj.columns, union(self.target, self.m_exceptions))

            @property
            def features(self):
                return self._obj[self.features_list]

            @property
            def stratification_target(self):
                return self.m_stratification

            @property
            def df(self):
                return self._obj.drop(columns=self.m_exceptions, errors="ignore")

    return SurvivalAccessor


def build_weighted_survival_accessor(event, duration, weight_column="sample_weights", accessor_code="surv", exceptions=tuple(), disable_warning=True):
    """
    Build an accessor dedicated to the management of survival data with sample weights.

    Args:
        - event           : event name;
        - duration        : duration name;
        - weight_column   : name of the weight column;
        - accessor_code   : code to access extended properties (for example: `df.surv.event`);
        - exceptions      : columns to exclude from features;
        - disable_warning : remove warnings linked to several uses of that function.

    Returns: the corresponding accessor class;
             instances are built when called from the DataFrame (for example: `df.surv`)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class WeightedSurvivalAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_event          = event
                self.m_duration       = duration
                self.m_weight_column  = weight_column
                self.m_stratification = event
                self.m_exceptions     = exceptions

            @staticmethod
            def _validate(obj):
                pass
                # # verify there is a column latitude and a column longitude
                # if event not in obj.columns or duration not in obj.columns:
                #     raise AttributeError("Must have {0} and {1}.".format(event, duration))

            @property
            def event(self):
                return self.m_event

            @property
            def events(self):
                return self._obj[self.event]

            @property
            def duration(self):
                return self.m_duration

            @property
            def durations(self):
                return self._obj[self.duration]

            @property
            def weights(self):
                """Get sample weights if available."""
                if self.m_weight_column in self._obj.columns:
                    return self._obj[self.m_weight_column]
                else:
                    return None

            @property
            def target(self):
                return [self.event, self.duration]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                # Exclude weight column from features
                all_exceptions = list(self.m_exceptions) + [self.m_weight_column]
                return difference(self._obj.columns, union(self.target, all_exceptions))

            @property
            def features(self):
                return self._obj[self.features_list]

            @property
            def stratification_target(self):
                return self.m_stratification

            @property
            def df(self):
                # Exclude weight column from df, just like features
                all_exceptions = list(self.m_exceptions) + [self.m_weight_column]
                return self._obj.drop(columns=all_exceptions, errors="ignore")

    return WeightedSurvivalAccessor


def build_fairness_survival_accessor(event, duration, group_indicator_column, accessor_code="surv", exceptions=tuple(), disable_warning=True, exclude_group_from_features=True):    
    """
    Build a fairness-aware survival accessor with group indicators.
    
    This replaces the weighted accessor approach with group indicators for fairness training.
    
    Args:
        event: Column name for event indicators (0=censored, 1+=event type)
        duration: Column name for duration/times
        group_indicator_column: Column name for group membership (e.g., 'race')
        accessor_code: Name for the accessor (e.g., 'surv')
        exceptions: List of columns to exclude from features
        disable_warning: Remove warnings linked to several uses of that function
    
    Example:
        build_fairness_survival_accessor(
            event="pcens",
            duration="psurv", 
            group_indicator_column="race",
            accessor_code="surv",
            exceptions=["age", "sex"]
        )
        
        # Usage:
        # df.surv.features  # Feature columns
        # df.surv.durations  # Duration values
        # df.surv.events  # Event indicators
        # df.surv.group_indicators  # Group indicators (0=non-Black, 1=Black)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class FairnessSurvivalAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_event = event
                self.m_duration = duration
                self.m_group_column = group_indicator_column
                self.m_stratification = event
                self.m_exceptions = exceptions
                self.m_exclude_group_from_features = exclude_group_from_features

            @staticmethod
            def _validate(obj):
                pass

            @property
            def event(self):
                return self.m_event

            @property
            def events(self):
                return self._obj[self.event]

            @property
            def duration(self):
                return self.m_duration

            @property
            def durations(self):
                return self._obj[self.duration]

            @property
            def weights(self):
                """Return None for fairness training"""
                return None

            @property
            def group_indicators(self):
                """
                Get group indicators for fairness training.
                
                Returns:
                    Binary indicators where 0 = group1 (e.g., non-Black), 1 = group2 (e.g., Black)
                """
                # Convert group column to binary indicators
                # Assuming 'Black' is the minority group (group 1)
                # You can modify this mapping based on your data
                
                group_values = self._obj[self.m_group_column]
                
                # Handle different possible formats
                if group_values.dtype == 'object' or group_values.dtype == 'string':
                    # String/object column - map 'Black' to 1, others to 0
                    return (group_values == 'Black').astype(int)
                elif group_values.dtype in ['int64', 'int32', 'float64', 'float32']:
                    # Numeric column - assume 1 is Black, 0 is non-Black
                    return (group_values == 1).astype(int)
                else:
                    # Fallback - try to detect Black values
                    black_values = ['Black', 'black', 'BLACK', 'B', 'b', 1, 1.0]
                    return group_values.isin(black_values).astype(int)

            @property
            def target(self):
                return [self.event, self.duration]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                # Exclude group column from features if requested
                base_exclusions = union(self.target, self.m_exceptions)
                if self.m_exclude_group_from_features:
                    base_exclusions = union(base_exclusions, [self.m_group_column])
                return difference(self._obj.columns, base_exclusions)

            @property
            def features(self):
                return self._obj[self.features_list]

            @property
            def stratification_target(self):
                return self.m_stratification

            @property
            def df(self):
            # Exclude group column from df if requested, just like features
                base_exclusions = self.m_exceptions
                if self.m_exclude_group_from_features:
                    base_exclusions = union(base_exclusions, [self.m_group_column])
                return self._obj.drop(columns=base_exclusions, errors="ignore")

            def get_group_distribution(self):
                """Get distribution of groups for debugging."""
                group_indicators = self.group_indicators
                total = len(group_indicators)
                group0_count = (group_indicators == 0).sum()
                group1_count = (group_indicators == 1).sum()
                
                return {
                    'total_samples': total,
                    'group0_count': group0_count,
                    'group1_count': group1_count,
                    'group0_percentage': (group0_count / total) * 100,
                    'group1_percentage': (group1_count / total) * 100
                }

            def print_group_info(self):
                """Print group distribution information."""
                dist = self.get_group_distribution()
                print(f"📊 Group Distribution:")
                print(f"   Total samples: {dist['total_samples']}")
                print(f"   Group 0 (non-Black): {dist['group0_count']} ({dist['group0_percentage']:.1f}%)")
                print(f"   Group 1 (Black): {dist['group1_count']} ({dist['group1_percentage']:.1f}%)")
                
                # Show original group column values
                unique_values = self._obj[self.m_group_column].value_counts()
                print(f"   Original '{self.m_group_column}' values:")
                for value, count in unique_values.items():
                    print(f"     '{value}': {count} samples")

    return FairnessSurvivalAccessor


def build_classification_accessor(target, accessor_code="class", exceptions=(), disable_warning=True):
    """
    Build an accessor dedicated to the management of data for classification.

    Args:
        - target          : target name;
        - accessor_code   : code to access extended properties (for example: `df.class.event`);
        - disable_warning : remove warnings linked to several uses of that function.

    Returns: the corresponding accessor class;
             instances are built when called from the DataFrame (for example: `df.class`)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class ClassificationAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_target         = target
                self.m_stratification = target
                self.m_exceptions     = exceptions

            @staticmethod
            def _validate(obj):
                pass
                # if target not in obj.columns:
                #     raise AttributeError("Must have {0}.".format(target))

            @property
            def target(self):
                return [self.m_target]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                return difference(self._obj.columns, union(self.target, self.m_exceptions))

            @property
            def features(self):
                return self._obj[self.features_list]

            @property
            def stratification_target(self):
                return self.m_stratification

            @property
            def df(self):
                return self._obj.drop(columns=self.m_exceptions, errors="ignore")

    return ClassificationAccessor


#####################
#      COMPARE      #
#####################
# The following tools are not used in practice but can be helpful for debug.

class Comparison:
    def __init__(self, df1, df2, differences, depth=2, labels=("df1", "df2")):
        self.m_df1         = df1
        self.m_df2         = df2
        self.m_differences = differences
        self.m_depth       = depth
        self.m_labels      = labels

    @property
    def depth(self):
        return self.m_depth

    @depth.setter
    def depth(self, depth):
        self.m_depth = depth

    def __str__(self):
        name_df1 = self.m_labels[0]
        name_df2 = self.m_labels[1]

        string = "Comparing DataFrames df1={0} and df2={1}:\n".format(name_df1, name_df2)

        if self.m_differences == dict():
            string += "\n> They are equal."
        else:
            for k, v in self.m_differences.items():
                string += "\n> {0}:\n".format(k)
                for comment in v:
                    string += "\t- {0}\n".format(comment)

        return string

    def comments_count(self):
        count = dict()

        for comments in self.m_differences.values():
            for comment in comments:
                if comment not in count.keys():
                    count[comment] = 1
                else:
                    count[comment] += 1

        return count


def compare_dataframes(input_df1, input_df2, depth=2, labels=("df1", "df2")):
    """
    Compare two DataFrames.
    Args:
        - df1, df2  : the DataFrames to compare.

    Returns:
         - a dictionary: keys are mismatching columns, values are lists of comments about their differences.
    """
    df1 = input_df1.copy()
    df2 = input_df2.copy()

    diff = dict()
    if df1.equals(df2):
        return diff

    # Look for missing columns in one DataFrame or the other
    for column in (set(df1.columns) - set(df2.columns)):
        diff[column] = ["not in df2"]

    for column in (set(df2.columns) - set(df1.columns)):
        diff[column] = ["not in df1"]

    # Column-wise comparison
    def _append_message_(col, msg_):
        if col in diff.keys():
            diff[col].append(msg_)
        else:
            diff[col] = [msg_]

    columns = intersect_columns(df1.columns, df2)
    for column in columns:
        # Check position
        if df1.columns.get_loc(column) != df2.columns.get_loc(column):
            _append_message_(column, "different order")

        # Check typing
        if df1[column].dtypes != df2.dtypes[column]:
            _append_message_(column, "different type")

        # Tries to retype before to continue comparison
        df_ = pd.concat([
            df1[column].rename("df1"),
            df2[column].rename("df2").astype(df1[column].dtypes)
        ], axis=1)

        # compare content (first, ignore NA)
        if not df_.dropna()["df1"].equals(df_.dropna()["df2"]):

            df_ex   = df_.dropna()[df_.dropna()["df1"].ne(df_.dropna()["df2"])]
            example = str(df_ex.head(3))
            msg     = "different non-NaN values ({0} inconsistencies)\n\nExamples:\n".format(len(df_ex)) + example

            _append_message_(column, msg)

        # compare content (first, ignore NA)
        df_1 = df_.isna()["df1"]
        df_2 = df_.isna()["df2"]
        df_  = df_[(df_1 | df_2) & ~(df_1 & df_2)  ]
        if len(df_) > 0:
            example = str(df_.head(3))
            msg     = "NaN inconsistencies ({0})\n\nExamples:\n".format(len(df_)) + example

            _append_message_(column, msg)

    # Print

    return Comparison(df1, df2, diff, depth=depth, labels=labels)


#####################
#       MISC        #
#####################
# Various functions to make the code more readable.


def build_empty_mask(df):
    return pd.DataFrame(False, index=df.index, columns=df.columns)


def density(df, column):
    return df[column].count() / len(df)


def intersect_columns(l, df):
    """
    Ensure that elements of the list `l` are columns of the DataFrame `df`.
    """
    return intersection(l, df.columns)


def get_constant_columns(df):
    return df.columns[df.nunique() <= 1].to_list()

def get_sparse_columns(df, threshold):
    return [column for column in df.columns if density(df, column) < threshold]
# TEST COMMENT
