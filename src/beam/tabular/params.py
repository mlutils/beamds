from ..config import BeamConfig, BeamParam


class TabularConfig(BeamConfig):

    defaults = dict(project_name='deep_tabular', algorithm='TabularNet', n_epochs=100, scheduler='one_cycle',
                    batch_size=512, lr_dense=2e-3, lr_sparse=2e-2, early_stopping_patience=16)

    parameters = [
        BeamParam('emb_dim', int, 128, 'latent embedding dimension', tune=True),
        BeamParam('n_transformer_head', int, 4, 'number of transformer heads', tune=True),
        BeamParam('n_encoder_layers', int, 4, 'number of encoder layers', tune=True),
        BeamParam('n_decoder_layers', int, 4, 'number of decoder layers', tune=True),
        BeamParam('transformer_hidden_dim', int, 256, 'transformer hidden dimension', tune=True),
        BeamParam('transformer_dropout', float, 0., 'transformer dropout', tune=True),
        BeamParam('mask_rate', float, 0.15, 'rate of masked features during training', tune=True),
        BeamParam('rule_mask_rate', float, 0., 'rate of masked rules during training', tune=True),
        BeamParam('maximal_mask_rate', float, 0.2, 'the maximal mask rate with dynamic masking', tune=True),
        BeamParam('minimal_mask_rate', float, 0.1, 'the minimal mask rate with dynamic masking', tune=True),
        BeamParam('dynamic_delta', float, 0.005, 'the incremental delta for dynamic masking', tune=True, model=False),
        BeamParam('n_rules', int, 64, 'number of transformers rules in the decoder', tune=True),
        BeamParam('activation', str, 'gelu', 'transformer activation', tune=True),
        BeamParam('n_quantiles', int, 10, 'number of quantiles for the quantile embeddings', tune=True),
        BeamParam('scaler', str, 'quantile', 'scaler for the preprocessing [robust, quantile]', tune=True),
        BeamParam('n_ensembles', int, 32, 'number of ensembles of the model for prediction in inference mode', tune=True),
        BeamParam('label_smoothing', float, 0., 'label smoothing for the cross entropy loss', tune=True),
        BeamParam('dropout', float, .0, 'Output layer dropout of the model', tune=True),
        BeamParam('oh_to_cat', bool, False, 'Try to convert one-hot encoded categorical features to categorical features', tune=True),
        BeamParam('dynamic_masking', bool, False, 'Use dynamic masking scheduling', tune=True, model=False),
        BeamParam('feature_bias', bool, True, 'Add bias to the features', tune=True),
        BeamParam('rules_bias', bool, True, 'Add bias to the rules', tune=True),
        BeamParam('lin_version', int, 1, 'version of the linear output layer', tune=True),

        BeamParam('dataset_name', str, 'covtype',
               'dataset name [year, california_housing, higgs_small, covtype, aloi, adult, epsilon, '
               'microsoft, yahoo, helena, jannis]'),
        BeamParam('catboost', bool, False, 'Train a catboost model on the data'),
        BeamParam('store_data_on_device', bool, True, 'Store the data on the device (GPU/CPU) in advance', model=False),
        BeamParam('rulenet', bool, True, 'Train our RuleNet model on the data'),
        BeamParam('sparse-embedding', bool, True, 'Use sparse embedding for the features embeddings'),
    ]

