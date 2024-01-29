import shap
from abc import ABC, abstractmethod
import traceback
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import keras
from torch import nn
from collections import namedtuple

def get_shap_explainer(algorithm=None, model=None, data=None):
    """

    Args:
        algorithm: str, options are 'tree', 'kernel' or 'deep'
        model: a class that has a predict method.

    Returns:

    """


    if algorithm is None and model is None:
        raise ValueError('either algorithm or model should be specified')

    if algorithm is not None:
        if algorithm == 'tree':
            return ShapTreeExplainer(model)
        elif algorithm == 'kernel':
            return ShapKernelExplainer(model, data)
        elif algorithm == 'deep':
            return ShapDeepExplainer(model, data)
        else:
            raise ValueError('Unsupported algorithm')

    elif model is not None:
        tree_models = (CatBoostRegressor, CatBoostClassifier, RandomForestRegressor, RandomForestClassifier,
                       DecisionTreeRegressor, DecisionTreeClassifier, XGBClassifier, XGBRegressor,
                       LGBMRegressor, LGBMClassifier)
        nn_models = (nn.Module, keras.Model)
        if isinstance(model, tree_models):
            return ShapTreeExplainer(model)

        elif isinstance(model, nn_models):
            return ShapDeepExplainer(model, data)
        else:
            return ShapKernelExplainer(model, data)

class ShapExplainer(ABC):

    SHAP_OUTPUT = namedtuple('explanation','values data base_values')
    @staticmethod
    def abs_shap(df_shap, df):
            # import matplolib as plt
            # Make a copy of the input data
            shap_v = pd.DataFrame(df_shap)
            feature_list = df.columns
            shap_v.columns = feature_list
            df_v = df.copy().reset_index().drop('index', axis=1)

            # Determine the correlation in order to plot with different colors
            corr_list = list()
            for i in feature_list:
                b = np.corrcoef(shap_v[i], df_v[i])[1][0]
                corr_list.append(b)
            corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
            # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
            corr_df.columns = ['Variable', 'Corr']
            corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

            # Plot it
            shap_abs = np.abs(shap_v)
            k = pd.DataFrame(shap_abs.mean()).reset_index()
            k.columns = ['Variable', 'SHAP_abs']
            k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
            k2 = k2.sort_values(by='SHAP_abs', ascending=True)
            colorlist = k2['Sign']
            ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=(5, 6), legend=False)
            ax.set_xlabel("SHAP Value (Red = Positive Impact)")

    @abstractmethod
    def explain(self, X):
        raise NotImplementedError

    @abstractmethod
    def waterfall(self, shap_values):
        raise NotImplementedError

    @abstractmethod
    def beeswarm(self, shap_values):
        raise NotImplementedError

    @abstractmethod
    def bar(self, shap_values):
        raise NotImplementedError

    @abstractmethod
    def force(self, shap_values):
        raise NotImplementedError

    @abstractmethod
    def dependence_plot(self, variable_name, shap_values, data, interaction_index):
        raise NotImplementedError


class ShapTreeExplainer(ShapExplainer):

    def __init__(self, model, **kwargs):
        self.model = model
        self.explainer = shap.TreeExplainer(model, **kwargs)

    def explain(self, X, as_array=False):
        shap_values = self.explainer(X)
        if as_array:
            return ShapExplainer.SHAP_OUTPUT(shap_values.values, shap_values.data, shap_values.base_values)
        return shap_values


    def waterfall(self, shap_values, max_display=10, show=True, multiclass=False):
        """Plots an explanation of a single prediction as a waterfall plot."""
        if multiclass:
            if len(shap_values.shape) != 2:
                raise ValueError('In the multiclass, shap values for waterfall plot need to be 2d')
            for i in range(shap_values.shape[1]):
                shap.plots.waterfall(shap_values[:, i], max_display=max_display, show=show)
        else:
            if len(shap_values.shape) != 1:
                raise ValueError('Shap values for waterfall plot need to be 1d! did you mean to use multiclass? (multiclass=True)')
            shap.plots.waterfall(shap_values, max_display=max_display, show=show)

    def beeswarm(self, shap_values, multiclass=False, **kwargs):
        """Create a SHAP beeswarm plot, colored by feature values when they are provided."""
        if multiclass:
            if len(shap_values.shape) != 3:
                raise ValueError('In the multiclass, shap values for beeswarm plot need to be 3d')
            for i in range(shap_values.shape[-1]):
                shap.plots.beeswarm(shap_values[..., i],**kwargs)
        else:
            if  len(shap_values.shape) != 2:
                raise ValueError('Shap values for beeswarm plot need to be 2d! did you mean to use multiclass? (multiclass=True)')
            shap.plots.beeswarm(shap_values, **kwargs)

    def bar(self, shap_values, multiclass=False, **kwargs):
        """Create a SHAP bar plot."""
        if 'show_data' not in kwargs.keys() and len(shap_values.shape) < 2:
            kwargs['show_data'] = True
        if multiclass:
            for i in range(shap_values.shape[-1]):
                shap.plots.bar(shap_values[..., i],**kwargs)
        else:
            if  len(shap_values.shape) > 2:
                raise ValueError('Shap values for bar plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
            shap.plots.bar(shap_values, **kwargs)

    def force(self, shap_values, multiclass=False, **kwargs):
        """Create a SHAP force plot."""
        if multiclass:
            for i in range(shap_values.shape[-1]):
                shap.plots.force(shap_values[..., i], matplotlib=True, **kwargs)
        else:
            if len(shap_values.shape) > 2:
                raise ValueError(
                    'Shap values for force plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
            shap.plots.force(shap_values, matplotlib=True, **kwargs)

    def dependence_plot(self, variable_name, shap_values, data, interaction_index=None, multiclass=False, **kwargs):
        """Create a SHAP dependence plot to show the effect of a single feature value on the model output."""
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values.values
        if multiclass:
            for i in range(shap_values.shape[-1]):
                shap.dependence_plot(variable_name, shap_values[..., i], data, interaction_index=interaction_index, **kwargs)
        else:
            if len(shap_values.shape) > 2:
                raise ValueError(
                    'Shap values for dependence_plot plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
            shap.dependence_plot(variable_name, shap_values, data, interaction_index=interaction_index, **kwargs)

    def get_interaction_values(self, X):
        """ Estimate the SHAP interaction values for a set of samples."""
        if len(X.shape) > 2:
            raise ValueError('Interaction Values are not available for multi-output models.')
        try:
            return self.explainer.shap_interaction_values(X)
        except:
            traceback.print_exc()
    def plot_interaction_values(self, X, interaction_values=None):
        """ plot the SHAP interaction values for a set of samples."""

        if len(X.shape) > 2:
            raise ValueError('Interaction Values are not available for multi-output models.')
        if interaction_values is None:
            interaction_values = self.get_interaction_values(X)
            if interaction_values is None:
                print('could not compute interaction values')
                return
        shap.summary_plot(interaction_values, X)

class ShapKernelExplainer(ShapExplainer):

  def __init__(self, model, data, **kwargs):
      self.model = model
      k = kwargs.get('k')
      if k is not None:
          data = shap.kmeans(data, k)
      self.explainer = shap.KernelExplainer(model.predict, data, **kwargs)

  def explain(self, X, as_array=False):
      shap_values = self.explainer(X)
      if as_array:
          return ShapExplainer.SHAP_OUTPUT(shap_values.values, shap_values.data, shap_values.base_values)
      return shap_values

  def waterfall(self, shap_values, max_display=10, show=True, multiclass=False):
      """Plots an explanation of a single prediction as a waterfall plot."""
      if multiclass:
          if len(shap_values.shape) != 2:
              raise ValueError('In the multiclass, shap values for waterfall plot need to be 2d')
          for i in range(shap_values.shape[1]):
              shap.plots.waterfall(shap_values[:, i], max_display=max_display, show=show)
      else:
          if len(shap_values.shape) != 1:
              raise ValueError(
                  'Shap values for waterfall plot need to be 1d! did you mean to use multiclass? (multiclass=True)')
          shap.plots.waterfall(shap_values, max_display=max_display, show=show)

  def beeswarm(self, shap_values, multiclass=False, **kwargs):
      """Create a SHAP beeswarm plot, colored by feature values when they are provided."""
      if multiclass:
          if len(shap_values.shape) != 3:
              raise ValueError('In the multiclass, shap values for beeswarm plot need to be 3d')
          for i in range(shap_values.shape[-1]):
              shap.plots.beeswarm(shap_values[..., i], **kwargs)
      else:
          if len(shap_values.shape) != 2:
              raise ValueError(
                  'Shap values for beeswarm plot need to be 2d! did you mean to use multiclass? (multiclass=True)')
          shap.plots.beeswarm(shap_values, **kwargs)

  def bar(self, shap_values, multiclass=False, **kwargs):
      """Create a SHAP bar plot."""
      if 'show_data' not in kwargs.keys() and len(shap_values.shape) < 2:
          kwargs['show_data'] = True
      if multiclass:
          for i in range(shap_values.shape[-1]):
              shap.plots.bar(shap_values[..., i], **kwargs)
      else:
          if len(shap_values.shape) > 2:
              raise ValueError(
                  'Shap values for bar plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
          shap.plots.bar(shap_values, **kwargs)

  def force(self, shap_values, multiclass=False, **kwargs):
      """Create a SHAP force plot."""
      if multiclass:
          for i in range(shap_values.shape[-1]):
              shap.plots.force(shap_values[..., i], matplotlib=True, **kwargs)
      else:
          if len(shap_values.shape) > 2:
              raise ValueError(
                  'Shap values for force plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
          shap.plots.force(shap_values, matplotlib=True, **kwargs)

  def dependence_plot(self, variable_name, shap_values, data, interaction_index=None, multiclass=False, **kwargs):
      """Create a SHAP dependence plot to show the effect of a single feature value on the model output."""
      if isinstance(shap_values, shap.Explanation):
          shap_values = shap_values.values
      if multiclass:
          for i in range(shap_values.shape[-1]):
              shap.dependence_plot(variable_name, shap_values[..., i], data, interaction_index=interaction_index,
                                   **kwargs)
      else:
          if len(shap_values.shape) > 2:
              raise ValueError(
                  'Shap values for dependence_plot plot need to be at most 2d! did you mean to use multiclass? (multiclass=True)')
          shap.dependence_plot(variable_name, shap_values, data, interaction_index=interaction_index, **kwargs)

class ShapDeepExplainer(ShapExplainer):

  def __init__(self, model, data, feature_names=None, **kwargs):
      self.model = model
      self.explainer = shap.DeepExplainer(model, data, **kwargs)
      if feature_names is not None:
          self.feature_names = feature_names
      elif isinstance(data, pd.DataFrame):
          self.feature_names = data.columns
      else:
          self.feature_names = None

  def explain(self, X):
      shap_values = self.explainer.shap_values(X)
      return shap_values[1]

  def waterfall(self, shap_values, max_display=10, show=True, **kwargs):
      """Plots an explanation of a single prediction as a waterfall plot."""
      feature_names = kwargs.get('feature_names', self.feature_names)
      shap.plots._waterfall.waterfall_legacy(self.explainer.expected_value[0].numpy(), shap_values,
                                             feature_names=feature_names)

  def beeswarm(self, shap_values, test_data, **kwargs):
      """Create a SHAP beeswarm plot, colored by feature values when they are provided."""
      if self.feature_names is None and isinstance(test_data, pd.DataFrame):
          feature_names = test_data.columns
      else:
          feature_names = kwargs.get('feature_names')
      shap.summary_plot(shap_values, feature_names=feature_names, features=test_data)
  def bar(self, shap_values, **kwargs):
      """Create a SHAP bar plot."""
      if 'show_data' not in kwargs.keys() and len(shap_values.shape) < 2:
          kwargs['show_data'] = True
      shap.plots.bar(shap_values, **kwargs)

  def force(self, shap_values, **kwargs):
      feature_names = kwargs.get('feature_names', self.feature_names)
      """Create a SHAP force plot."""
      shap.plots.force(self.explainer.expected_value[0].numpy(), shap_values, feature_names=feature_names,
        matplotlib=True, **kwargs)

  def dependence_plot(self, variable_name, shap_values, data, interaction_index=None, **kwargs):
      """Create a SHAP dependence plot to show the effect of a single feature value on the model output."""
      if isinstance(shap_values, shap.Explanation):
          shap_values = shap_values.values
      shap.dependence_plot(variable_name, shap_values, data, interaction_index=interaction_index, **kwargs)



