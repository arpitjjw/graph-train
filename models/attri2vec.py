
from stellargraph import StellarGraph
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize

from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Lambda, Reshape, Embedding

from numpy import dot
from numpy.linalg import norm
import mlflow 

mlflow.tensorflow.autolog(every_n_iter=1, log_models=True, disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False, registered_model_name=None, log_input_examples=False, log_model_signatures=False)
    
def train_attri2vec(config,skills_graph):

    G = StellarGraph.from_networkx(skills_graph,node_features="x")
    nodes = list(G.nodes())
    number_of_walks = config['num_walks']
    length = config['len']

    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = config['batch_size']
    epochs = config['epochs']

    generator = Attri2VecLinkGenerator(G, batch_size)
    train_gen = generator.flow(unsupervised_samples)

    layer_sizes = [768]
    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
    )

    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )


    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        shuffle=True,
    )



def test_attri2vec(skills_graph,ground_truth):
    # Load test skill sets from config file
    test_skills_config = os.environ.get('TEST_SKILLS_CONFIG', 'test_skills.json')
    if os.path.exists(test_skills_config):
        with open(test_skills_config, 'r') as f:
            test_config = json.load(f)
            test_1 = set(test_config.get('test_1', []))
            test_2 = set(test_config.get('test_2', []))
    else:
        # Fallback to default values if config file doesn't exist
        test_1={1280,7909,1086,7728,604,5938}
        test_2={5690,976,4092,9843,2140,11129,11280,5739,515,11460,7878,26074,2434}

    G=StellarGraph.from_networkx(G,node_features="x")
    nodes = list(G.nodes())
    node_gen = Attri2VecNodeGenerator(G, 48).flow(list(G.nodes()))
    path=None
    model=mlflow.tensorflow.load_model(path, dst_path=None)
    dense_weights=model.get_layer('dense').get_weights()
    input=Input(shape=(768,))
    dense_layer=Dense(768, activation="sigmoid", use_bias=False,name='dense')(input)
    prediction_model=keras.Model(inputs=input,outputs=dense_layer)
    prediction_model.get_layer('dense').set_weights(dense_weights)

    node_embeddings = prediction_model.predict(node_gen, workers=2, verbose=1)

    def cos_sim(a,b):
        return dot(a, b)/(norm(a)*norm(b))

    test_score2_manual=0
    test_score2_auto_1=0
    test_score2_auto_2=0
    for ind,i in enumerate(ground_truth.keys()):
        embed=node_embeddings[i]
        
        scores_dict=dict()
        for j in node_embeddings.keys():
            id2=j
            score=cos_sim(embed,node_embeddings[id2])
            scores_dict[j]=score
        scores_dict = dict( sorted(scores_dict.items(),
                        key=lambda item: item[1],
                        reverse=True))
        
        

        final_pred=set()
        count=0
        for k,v in scores_dict.items():
            count+=1
            final_pred.add(k)
            if count==200:
                break

        common=final_pred.intersection(ground_truth[i])
        total=len(ground_truth[i])
        correct=len(common)
        if i in test_1:
            test_score2_auto_1+=correct/total
        elif i in test_2:
            test_score2_auto_2+=correct/total
        else:
            test_score2_manual+=correct/total
    test_score2_manual/=13
    test_score2_auto_1/=6
    test_score2_auto_2/=13
    print(f"Test scores - Manual: {test_score2_manual:.4f}, Auto-1: {test_score2_auto_1:.4f}, Auto-2: {test_score2_auto_2:.4f}")
    mlflow.log_metric(key='test_score2',value=test_score2_manual)
    mlflow.log_metric(key='test_score2_auto_1',value=test_score2_auto_1)
    mlflow.log_metric(key='test_score2_auto_2',value=test_score2_auto_2)
