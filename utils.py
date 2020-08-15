#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

class Utils(object):

    def __init__(self, model_output_dir):
        self.model_output_dir = model_output_dir #模型预测输出路径

    def get_words_and_labels_for_docs(self):
        '''
        从预测结果中加载word和label,针对文档集合
        :return:words, labels
        '''
        with open(os.path.join(self.model_output_dir,'label_test.txt'), 'r', encoding='utf-8')as f1:
            words = []
            labels = []
            lines = []
            for line in f1:
                #print(line)
                word = line.strip().split('\t')[0]
                label = line.strip().split('\t')[-1]
                if len(line.strip()) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((l, w))
                    words = []
                    labels = []
                words.append(word)
                labels.append(label)
                #print(lines)
        return lines

    def get_words_and_labels_for_single_doc(self):
        '''
        从预测结果中加载word和label,针对单文档
        :return:words, labels
        '''
        with open(os.path.join(self.model_output_dir, 'label_test.txt'), 'r', encoding='utf-8')as f1:
            words = []
            labels = []
            lines = []
            for line in f1:
                word = line.strip().split('\t')[0]
                label = line.strip().split('\t')[-1]
                words.append(word)
                labels.append(label)
        return words, labels

    def get_entity_for_single_doc(self):
        '''
        从单文档预测结果words、labels集合中提取出实体，格式如下：
        处罚文书|杏环罚字[2015]10号
        处罚时间|2017-05-04
        处罚机关|杏花岭分局
        :return:
        '''
        words, labels = self.get_words_and_labels_for_single_doc()
        assert len(words) == len(labels)
        f2 = open(os.path.join(self.model_output_dir,'entity_test.txt'), 'w', encoding='utf-8')
        entity = ''
        type = ''

        for word, tag in zip(words, labels):
            item = tag.split('-')[-1]

            if str(tag) != 'O':
                entity += str(word)
                if str(item) == "TEXT":
                    type = '处罚文书'
                elif str(item) == "DATE":
                    type = '处罚时间'
                elif str(item) == "ORG":
                    type = '处罚机关'
                elif str(item) == "COM":
                    type = '被处罚公司'
                elif str(item) == "NUM":
                    #NUM
                    type = '处罚金额'
            if entity != '' and type != '' and str(tag) == 'O':
                f2.write(type + '|' + entity + '\n')
                entity = ''
                type = ''
        f2.close()

    def get_entity_for_docs(self):
        '''
        从文档集合中出实体
        :return:
        '''
        lines = self.get_words_and_labels_for_docs()
        f2 = open(os.path.join(self.model_output_dir, 'entity_test.txt'), 'w', encoding='utf-8')
        entity = ''
        type = ''
        for item in lines:
            tag_list = item[0].split(' ')
            word_list = item[1].split(' ')
            # print(word_list)
            # print(tag_list)
            assert len(word_list) == len(tag_list)
            for word, tag in zip(word_list, tag_list):
                tag_ = str(tag).split('-')[-1]
                #print(tag)
                if str(tag) != 'O':
                    entity += str(word)
                    if str(tag_) == "TEXT":
                        type = '处罚文书'
                    elif str(tag_) == "DATE":
                        type = '处罚时间'
                    elif str(tag_) == "ORG":
                        type = '处罚机关'
                    elif str(tag_) == "COM":
                        type = '被处罚公司'
                    elif str(tag_) == "NUM":
                        #NUM
                        type = '处罚金额'
                if entity != '' and type != '' and str(tag) == 'O':
                    f2.write(type + '|' + entity + '\n')
                    entity = ''
                    type = ''
            f2.write('\n')
        f2.close()




if __name__=='__main__':
   util = Utils('./output/result_dir')
   util.get_entity_for_single_doc()