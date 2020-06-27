#!/usr/bin/env python3

import json
import os
import random
import re

import encoder_sp
import fire
import model
import numpy as np
import tensorflow as tf
from gensim.models import LdaModel
from lxml import etree
from rouge import Rouge
from tqdm import tqdm
from gensim.matutils import cossim

LDA_MODEL_NAME = 'trained_lda_model'


def safe_list_get(l, idx):
    try:
        return l[idx][0]
    except IndexError:
        return -1
    except TypeError:
        return -2


def generate_with_model(
        model_name='117M',
        seed=None,
        nsamples=1,
        batch_size=1,
        length=None,
        temperature=1,
        top_k=0,
        top_p=0.0
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder_sp.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        root = etree.parse(
            r'C:\Users\kiva0319\IdeaProjects\hrdmd1803\Strong-Paraphrase-Generation-2020\raw_data\paraphrases.xml')

        root = root.getroot()
        corpus = etree.SubElement(root, "corpus")
        topic_save_percentage = 0.7
        number_generated_examples_at_stage = 10
        minimum_acceptable_similarity = 0.8
        max_intersection_similarity = 0.6
        indexs = list(range(len(root[1])))
        random.Random(3).shuffle(indexs)
        lda_model = LdaModel.load(LDA_MODEL_NAME)
        id2word = lda_model.id2word
        rouge = Rouge()

        for i in tqdm(indexs):
            element = root[1][i]
            id = element[0].text
            id_1 = element[1].text
            id_2 = element[2].text
            title_1 = element[3].text
            title_2 = element[4].text
            jaccard = element[5].text
            clas = element[6].text
            text_1 = "none"
            with open(
                    "C:/Users/kiva0319/IdeaProjects/hrdmd1803/Strong-Paraphrase-Generation-2020/download/v1/" + id_1 + ".txt",
                    'r', encoding="utf-8") as file:
                text = file.read()
                if len(text) < 50:
                    print("bad file id =", id_1)
                    continue
                text_1 = text
            paragraphs = text_1.split("\n\n")

            topic_to_paragraphs = {i: [] for i in range(200)}
            topic_met_num = {i: 0 for i in range(200)}

            for paragraph_num, paragraph in enumerate(paragraphs):
                corpus = [id2word.doc2bow(paragraph.split(" "))]
                row = lda_model[corpus][0]
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                for j in range(3):
                    topic_num = safe_list_get(row, j)
                    if topic_num >= 0:
                        topic_met_num[topic_num] += 1
                        topic_to_paragraphs[topic_num].append(paragraph_num)
            topic_count = 0
            main_topic = 0
            selected_paragraphs = set()
            paragraphs_grouped_by_topic = []

            topic_limit = int(float(len(topic_met_num)) * topic_save_percentage)
            for topic_num in sorted(topic_met_num, key=topic_met_num.get, reverse=True):
                if topic_count == 0:
                    main_topic = topic_num
                if topic_count > topic_limit:
                    break

                paragraphs_grouped_by_topic[topic_count] = []
                for tn in topic_to_paragraphs[topic_num]:
                    if tn not in selected_paragraphs:
                        paragraphs_grouped_by_topic[topic_count].append(paragraphs[tn])
                        selected_paragraphs.add(tn)
                topic_count += 1

            text_by_topic = []

            for topic_count, paragraphs in enumerate(paragraphs_grouped_by_topic):
                text_by_topic[topic_count] = " ".join(paragraphs)
            sentnces_group_by_topic = []

            for text in text_by_topic:
                ordinary_sentences = []
                sentnces_group = []
                group_count = 0
                sentnces = text.strip().split(". ")

                for sentence in sentnces:
                    if len(re.findall(r'\"(.+?)\"', sentence)) > 0:
                        sentnces_group[group_count] = [sentence]
                        group_count += 1
                    elif len(re.findall(r'[1-3][0-9]{3}', sentence)) > 0:
                        sentnces_group[group_count] = [sentence]
                        group_count += 1
                    elif len(re.findall(r"[A-Z][a-z]+", sentence[1:])) > 0:
                        sentnces_group[group_count] = [sentence]
                        group_count += 1
                    elif len(re.findall(r'[?]', text)) > 0:
                        sentnces_group[group_count] = [sentence]
                        group_count += 1
                    else:
                        ordinary_sentences.append(sentence)
                sentnces_group[group_count] = ordinary_sentences
                text_groups = []

                for group in sentnces_group:
                    text_groups.append(". ".join(group))
                sentnces_group_by_topic.append(text_groups)
            result = []

            for text_groups in sentnces_group_by_topic:
                for raw_text in text_groups:
                    context_tokens = enc.encode(raw_text)
                    samples = []
                    min_rl = 2
                    best_sample = raw_text
                    for s_num in range(number_generated_examples_at_stage):
                        vec_1 = lda_model[id2word.doc2bow(raw_text.split(" "))]
                        vec_2 = lda_model[id2word.doc2bow(best_sample.split(" "))]
                        if (cossim(vec_1, vec_2)) < minimum_acceptable_similarity:
                            break
                        for _ in range(nsamples // batch_size):
                            out = sess.run(output, feed_dict={
                                context: [context_tokens for _ in range(batch_size)]
                            })[:, len(context_tokens):]
                            for i in range(batch_size):
                                text = enc.decode(out[i])
                                samples.append(text)
                        for sample in samples:
                            sc = rouge.get_scores(raw_text, sample)[0]
                            r = sc['rouge-l']['f']
                            if r < min_rl:
                                min_rl = r
                                best_sample = sample
                    if rouge.get_scores(raw_text, best_sample)[0]['rouge-1']['f'] < max_intersection_similarity:
                        result.append(best_sample)
            random.shuffle(result)
            # print(" ".join(result))
            return " ".join(result)


if __name__ == '__main__':
    fire.Fire(generate_with_model)
