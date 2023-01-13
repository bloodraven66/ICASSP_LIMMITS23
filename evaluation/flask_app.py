
import os
import numpy as np
import librosa
import flask
import fastspeech
from attrdict import AttrDict
import torch
import yaml
app = flask.Flask(__name__)

def load_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def load_model(args, token_path, fs_args):
    with open(token_path, 'rb') as f:
        all_symbols = np.load(f)
    model = fastspeech.FastSpeech(n_mel_channels=args.signal.n_mels, n_symbols=len(all_symbols)+1, **fs_args)
    return model, args

def tokenise(text):
    ids = {}
    tokens = []
    for t in text:
        if t not in ids: ids[t] = len(ids)+1
        tokens.append(ids[t])
    return torch.from_numpy(np.array(tokens))

def get_model():
    args = load_config('track1.yaml')
    fs_args = load_config('track1_model.yaml')
    token_path = 'track1_tokens.npy'
    model, args = load_model(args, token_path, fs_args)
    #load checkpoint here
    return model

@app.route("/predict", methods=["GET", "POST"])
def predict():
        spk = flask.request.form['spk']
        lang = flask.request.form['lang']
        text = flask.request.form['text']
        
        #tokenise
        tokens = tokenise(text)
        model = get_model()
        model.eval()
        mapping = {'te_m':0, 'te_f':1, 'hi_f':2, 'hi_m':3, 'mr_f':4, 'mr_m':5}
        spk = torch.tensor(mapping[spk]).unsqueeze(0)

        inputs = (tokens.unsqueeze(0), torch.from_numpy(np.array([len(tokens),])), None, None, spk, None, None)
        result = model(inputs, infer=True)[0]

        return flask.send_file("te_f_education_00005.wav",
                                mimetype="audio/wav",
                                as_attachment=True)
if __name__ == "__main__":
    print('starting server')
    app.run(host='0.0.0.0')
