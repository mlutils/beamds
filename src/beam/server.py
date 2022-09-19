from flask import Flask, jsonify, request
from .experiment import beam_algorithm_generator, Experiment

app = None


class BeamServer(object):

    def init(self, path, Alg, override_hparams=None, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, **argv):
        global app

        self.experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, **argv)
        self.alg = self.experiment.algorithm_generator(Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                             dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
        app = Flask(self.experiment.root)

    @app.route('/')
    def get_info(self):
        return jsonify(dict(self.experiment.hparams))

    @app.route('/predict', methods=['POST'])
    def predict(self):
        file = request.files['dataset']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

        return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})