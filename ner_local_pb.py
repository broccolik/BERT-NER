#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time: 2020/10/17 13:02
# @Author: zhk
# @Description:加载本地保存的saved model模型进行预测--实体识别模型


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import collections
import pickle
from absl import logging
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import os
import requests
import numpy as np


class Info():
    def __init__(self):
        self.data_dir = './data'
        self.middle_output = './middle_data'


info = Info()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, type):
        """Read a BIO data!"""
        rf = open(input_file, 'r', encoding='utf-8')
        lines = [];
        words = [];
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            if type == "test":
                label = "O"
            else:
                label = line.strip().split(' ')[-1]

            if len(line.strip()) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt"), "train"), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt"), "dev"), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt"), "test"), "test"
        )

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ["[PAD]", "B-NUM", "I-NUM", "O", "B-TEXT", "I-TEXT", "B-DATE", "I-DATE", "B-ORG", "I-ORG", "B-COM",
                "I-COM", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    # # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(info.middle_output + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1] * len(input_ids)
    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    # return feature, ntokens, label_ids
    return feature


def tag2entity(text, tags):
    final_text = ''
    for word in text:
        final_text += word
    item = {"text": final_text, "entities": []}
    entity_name = ""
    flag = []
    visit = False
    for char, tag in zip(text, tags):
        if tag[0] == "B":
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                if [entity_name != item_['word'] for item_ in item["entities"]]:
                    item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
                entity_name = ""
            visit = True
            entity_name += char
            flag.append(tag[2:])
        elif tag[0] == "I" and visit:
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
            flag.clear()
            visit = False
            entity_name = ""

    if entity_name != "":
        x = dict((a, flag.count(a)) for a in flag)
        y = [k for k, v in x.items() if max(x.values()) == v]
        item["entities"].append({"word": entity_name, "type": y[0]})
    print(item)

    f2 = open('./result.json', 'a', encoding='utf-8')
    f2.write(str(item) + '\n')
    # jsonf = open('./test_result.json', 'a', encoding='utf-8')
    # json.dump(item,jsonf, ensure_ascii=False, indent=3)


def main():
    basedir = './'
    bert_config_file = basedir + 'chinese_L-12_H-768_A-12/bert_config.json'
    vocab_file = basedir + 'chinese_L-12_H-768_A-12/vocab.txt'
    init_checkpoint = basedir + 'bert_model.ckpt'
    do_lower_case = True
    max_seq_length = 200

    processor = NerProcessor()
    label_list = processor.get_labels()
    test_examples = processor.get_test_examples(info.data_dir)

    tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    f = open('result.txt', 'w', encoding='utf-8')

    with tf.Session(graph=tf.Graph()) as sess:
        for i, example in enumerate(test_examples):
            # print(example.text)
            # print(example.label)
            feature = convert_single_example(i, example, label_list, max_seq_length, tokenizer)
            label_ids1 = np.array(feature.label_ids).reshape(1, 200)
            input_ids1 = np.array(feature.input_ids).reshape(1, 200)
            mask1 = np.array(feature.mask).reshape(1, 200)
            segment_ids1 = np.array(feature.segment_ids).reshape(1, 200)

            pb_file_path = './output_ner/trans/saved_model'

            tf.saved_model.loader.load(sess, ["serve"], pb_file_path)
            graph = tf.get_default_graph()
            sess.run(tf.global_variables_initializer())

            # 需要先复原变量
            # print(sess.run('serve'))
            # 1

            # 输入
            input_ids = sess.graph.get_tensor_by_name('input_ids_1:0')
            input_mask = sess.graph.get_tensor_by_name('input_mask_1:0')
            segment_ids = sess.graph.get_tensor_by_name('segment_ids_1:0')
            label_ids = sess.graph.get_tensor_by_name('label_ids_1:0')
            mask = sess.graph.get_tensor_by_name('mask_1:0')

            op = sess.graph.get_tensor_by_name('ReverseSequence_1:0')

            ret = sess.run(op, feed_dict={input_ids: input_ids1, input_mask: mask1, segment_ids: segment_ids1,
                                          label_ids: label_ids1, mask: mask1})
            f.write(str(ret) + '\n')
            print(ret)
            tags = []
            #这里0、13、14是无效的标签
            for i in ret:
                if i not in [0, 13, 14]:
                    tags.append(label_list[i])
            print(tags)


if __name__ == '__main__':
    main()
