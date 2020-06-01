# Strong paraphrase generation using universal transformer - 2020

[![License](http://img.shields.io/:license-MIT-blue.svg?style=flat-square)](LICENSE)

> Software implementation of research in the framework of graduate qualification work

Research of the generation of strong paraphrases and new metrics for the validation of strong paraphrases.

## Project structure

Top-level directory structure with a short description of the contents.

    .
    ├── paraphrase          
        |   ├── russian_model   # Scripts for creating and training the neural network Transformer architecture model for the Russian language
        |   └── english_model   # and for English (various training parameters, text corpora and tokenizer)
    ├── topic_model             # Scripts for creating and learning topic models
    ├── analys                  # Scripts for calculating various characteristics of text corpora
    ├── dataset                 # Scripts for collecting and manipulating data sets
    ├── download                # Raw files new dataset
    ├── processed               # Files with calculated characteristics of the text corpora and the new generated files
    ├── report                  # The files are generated reports (including visualizations of topic models)
    ├── log                     # Logs the process of downloading the new dataset
    ├── lib                     # Files of the Mallet library for creating latent dirichlet allocation
    ├── util                    # Util scripts
    ├── raw_data                # Files text corpora from old research
    ├── LICENSE                
    └── README.md

## Installation

Project dependencies:
```sh
pandas                   1.0.1 1.0.4
nltk                     3.4.5 +
numpy                    1.16.3 +
tensorboard              1.15.0 +
tensorflow-estimator     1.15.1 +
tensorflow-gpu           1.15.0 +
scikit-learn             0.23.1
scipy                    1.4.1 +
sentencepiece            0.1.86 +
Keras-Applications       1.0.8
Keras-Preprocessing      1.1.0 +
Werkzeug                 1.0.1
astor                    0.8.1
cachetools               4.1.0
chardet                  3.0.4
expressions              0.2.3
hparams                  0.3.0
idna                     2.8 +
joblib                   0.14.1 +
lxml                     4.5.0 +
mock                     4.0.2
opt-einsum               3.2.1 3.2.1
protobuf                 3.11.3 3.12.2
pyasn1                   0.4.8 0.4.8
python-dateutil          2.8.1 2.8.1
pytz                     2019.3 +
regex                    2017.4.5 +
requests                 2.21.0 +
tqdm                     4.31.1 +
typeguard                2.7.1
urllib3                  1.24.3 +
```

## Meta

Kirill Vakhrushev – [VK](https://vk.com/kirundia) – kirill.vakhrushev@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/krikyn/](https://github.com/krikyn)

## Contributing

1. Fork it (<https://github.com/krikyn/Strong-Paraphrase-Generation-2020/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
