# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import pandas as pd

import numpy as np
import re, random
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

rec_table = pd.read_csv("./actions/recommend_table_lifu.csv", encoding='utf-8')
syn_table = pd.read_csv("./actions/SYN_LIFU.csv", encoding='utf-8')
res_table = pd.read_csv("./actions/RESPONSE_EXP_LIFU.csv", encoding='utf-8')

city_all = list(set(rec_table['cities'].tolist()))
city_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'] == 'LOCATION-TYPE']['entity name']), list(syn_table[syn_table['position'] == 'LOCATION-TYPE']['norms'])):
    city_norm_dict[i] = j

food_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'] == 'FOOD-TYPE']['entity name']), list(syn_table[syn_table['position'] == 'FOOD-TYPE']['norms'])):
    food_norm_dict[i] = j

foodrestorant_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'] == 'FOOD-TYPE']['entity name']), list(syn_table[syn_table['position'] == 'FOOD-TYPE']['RESTAURANT-TYPE'])):
    foodrestorant_norm_dict[i] = j

ingredient_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'] == 'INGREDIENT-TYPE']['entity name']), list(syn_table[syn_table['position'] == 'INGREDIENT-TYPE']['norms'])):
    ingredient_norm_dict[i] = j

goodtogo_list = ['GOOD-TO-GO1','GOOD-TO-GO2','GOODTOGO-VERB']
goodtogo_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'].isin(goodtogo_list)]['entity name']),list(syn_table[syn_table['position'].isin(goodtogo_list)]['norms'])):
    goodtogo_norm_dict[i] = j

tastetype_list = ['FRESH','SWEET','HOT','SWEET-SALTY','CRUNCHY','GOSO','SPICY-SWEET','JJONDEUK','TTAKKEUN','CREAMY','CHEWY','FUDGY','COLD','SIWON','FIRE','KKALKKEUM','STRONG','NOT-SALTY','OILY','NOT-HOT','NOT-JAGEUKJEOK','NOT-OILY','NOT-SWEET','NOT-BLAND']
tastetype_norm_dict = {}
for i, j in zip(list(syn_table[syn_table['position'].isin(tastetype_list)]['entity name']),list(syn_table[syn_table['position'].isin(tastetype_list)]['norms'])):
    tastetype_norm_dict[i] = j

restaurant_norm_dict = {'KOREAN':'?????????', 'CHINESE':'?????????', 'JAPANESE':'?????????', 'WESTERN':'?????????', 'CAFE':'??????', 'WORLD':'?????? ?????????', 'PUB':'??????', 'SNACK':'?????????', 'BUFFET':'??????', 'RESTAURANT-GEN':'??????'}
provided_norm_dict = {'PLAYROOM':'?????????', 'PARKING':'?????????', 'KIOSK':'????????????', 'SOCKET':'?????? ?????????', 'PRIVATE-ROOM':'?????? ???', 'SALAD-BAR':'????????????', 'OPEN-SPACE':'??? ?????? ??????'}
view_norm_dict = {'MOUNTAIN-VIEW':'????????? ???', 'RIVER-VIEW':'?????? ???', 'CITY-VIEW':'?????? ???','HANOK-VIEW':'?????? ???'}
menu_ent_list = ['VEGAN','KIDS','COURSE','DESSERT','LUNCH']


column_all = list(rec_table.columns)


def Josa_Replace(pattern, sentence):
    if '](' in pattern:
        if len(pattern.split('](')) > 2:
            back_syl = pattern.split('](')[-2][-1]
        else:
            back_syl = pattern.split('](')[0][-1]
    else:
        back_syl = pattern[0]

    criteria = (ord(back_syl) - 44032) % 28
    if criteria == 0: #?????????
        repl_pattern = re.sub("<[??????]>", "???", pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "???", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "???", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "???", repl_pattern)
    else:
        repl_pattern = re.sub("<[??????]>", "???", pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<[??????]>", "???", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "??????", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "??????", repl_pattern)
        repl_pattern = re.sub("<(??????|???)>", "??????", repl_pattern)
    # print(pattern, '\t', back_syl , '\t', '%s' % repl_pattern)
    return sentence.replace(pattern, '%s' % repl_pattern)

class ActionRephraseResponse(Action):
    # ????????? ?????? ????????? ??????
    def name(self) -> Text:
        return "action_rephrase_restaurant"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        self.city_norm = ''
        self.city_entityname = ''
        self.food_norm = ''
        self.food_entityname = ''
        self.ingredient_norm = ''
        self.ingredient_entityname = ''
        self.tastetype_norm = ''
        self.tastetype_entityname = ''
        self.restype_norm = ''
        self.restype_entityname = ''
        self.goodtogo_norm = ''
        self.goodtogo_entityname = ''
        self.provided_norm = ''
        self.provided_entityname = ''
        self.view_norm = ''
        self.view_entityname = ''
        self.feature = []
        self.food_feature = []
        self.intent = tracker.get_intent_of_latest_message()
        entity_dicts = tracker.latest_message['entities']
        print(tracker.latest_message['entities'])
        # print(tracker.get_intent_of_latest_message())
        for entity in entity_dicts.copy():
            if '-LOC' in entity['entity']:
                if entity['entity'].replace('-LOC', '') in city_norm_dict.keys():
                    self.city_entityname = entity['entity'].replace('-LOC', '')
                    self.city_norm = city_norm_dict[self.city_entityname]
                    entity_dicts.remove(entity)

            elif entity['entity'] in restaurant_norm_dict.keys():
                self.restype_entityname = entity['entity']
                self.restype_norm = restaurant_norm_dict[self.restype_entityname]
                entity_dicts.remove(entity)

            elif entity['entity'] in food_norm_dict.keys():
                self.food_entityname = entity['entity']
                self.food_norm = food_norm_dict[self.food_entityname]
                if self.intent == 'RECOMMEND_TASTE-GOOD':
                    self.food_feature.append('TASTE-GOOD_%s' % self.food_entityname)
                elif self.intent == 'RECOMMEND_PRICE-FREE':
                    self.food_feature.append('PRICE-FREE_%s' % self.food_entityname)
                elif self.intent == 'RECOMMEND_MENU':
                    self.food_feature.append('MENU_%s' % self.food_entityname)
                else:
                    pass
                entity_dicts.remove(entity)

            elif entity['entity'] in ingredient_norm_dict.keys():
                if entity['entity'] == 'NOODLE' and '?????????' in entity['value']:
                    self.food_entityname = 'PASTA'
                    self.food_norm = food_norm_dict[self.food_entityname]
                elif entity['entity'] == 'MEAT' and '??????' in entity['value']:
                    self.food_entityname = 'PORK'
                    self.food_norm = food_norm_dict[self.food_entityname]
                else:
                    self.ingredient_entityname = entity['entity']
                    self.ingredient_norm = ingredient_norm_dict[self.ingredient_entityname]
                    if self.intent == 'RECOMMEND_TASTE-GOOD':
                        self.food_feature.append('TASTE-GOOD_%s' % self.food_entityname)
                    elif self.intent == 'RECOMMEND_PRICE-FREE':
                        self.food_feature.append('PRICE-FREE_%s' % self.food_entityname)
                    elif self.intent == 'RECOMMEND_MENU':
                        self.food_feature.append('MENU_%s' % self.food_entityname)
                    else:
                        pass
                entity_dicts.remove(entity)


        for entity in entity_dicts:
            if entity['entity'] in goodtogo_norm_dict.keys():
                self.goodtogo_entityname = entity['entity']
                self.goodtogo_norm = goodtogo_norm_dict[self.goodtogo_entityname]
                self.feature.append('GOODTOGO-%s' % self.goodtogo_entityname)

            elif entity['entity'] in tastetype_norm_dict.keys():
                self.tastetype_entityname= entity['entity']
                self.tastetype_norm = tastetype_norm_dict[self.tastetype_entityname]
                self.tastetype_feature = 'TASTE-TYPE-%s' % self.tastetype_entityname
                if self.food_entityname != '':
                    self.tastetype_feature += '_%s' % self.food_entityname
                    print(self.tastetype_feature)
                    self.food_feature.append(self.tastetype_feature)
                elif self.ingredient_entityname != '':
                    self.tastetype_feature += '_%s' % self.ingredient_entityname
                    self.food_feature.append(self.tastetype_feature)
                else:
                    pass

            elif entity['entity'] in provided_norm_dict.keys():
                self.provided_entityname = entity['entity']
                self.provided_norm = provided_norm_dict[self.provided_entityname]
                self.feature.append('PROVIDED-%s' % self.provided_entityname)

            elif entity['entity'] in view_norm_dict.keys():
                self.view_entityname = entity['entity']
                self.view_norm = view_norm_dict[self.view_entityname]
                self.feature.append(self.view_entityname)

            elif entity['entity'] in menu_ent_list:
                self.feature.append('MENU-%s' % entity['entity'])

            else:
                self.feature.append(entity['entity'])


        print('City: ', self.city_norm, '\tName :', self.city_entityname)
        print('Food: ', self.food_norm, '\tName :', self.food_entityname)
        print('Ingred: ', self.ingredient_norm, '\tName', self.ingredient_entityname)
        print('Restaurant Type: ', self.restype_norm, '\tName: ', self.restype_entityname)
        print('Other Features: ', self.feature)
        print('Intent: ', self.intent)

        data, output_info_list = self.DataSorting(df=rec_table)

        self.Bot_Messeging(data=data, dispatcher=dispatcher, output=output_info_list)

        return []

    def DataSorting(self, df):
        data = df
        temp = []

        if self.restype_entityname == '' or self.restype_entityname == 'RESTAURANT-GEN':
            if self.food_entityname != '':
                self.restype_entityname = foodrestorant_norm_dict[self.food_entityname]
                self.restype_norm = restaurant_norm_dict[self.restype_entityname]
            else:
                self.restype_entityname = 'RESTAURANT-GEN'
                self.restype_norm = restaurant_norm_dict[self.restype_entityname]

        if self.city_entityname != '' and self.city_entityname in city_all:
            data = data[data['cities'] == self.city_entityname]

        if self.restype_entityname != '' and self.restype_entityname != 'RESTAURANT-GEN':
            data = data[data['categories'] == self.restype_entityname]
            # data = data.sort_values(by=food ,ascending=False)

        if self.feature != []:
            for f in self.feature:
                if f in column_all:
                    temp.append(f)

        self.food_null = 0
        if self.food_feature != []:
            for f in self.food_feature:
                if f not in column_all:
                    self.food_null = 1
                else:
                    food_sort_head = list(df.sort_values(by=f, ascending=False)[f])[0]
                    if food_sort_head == 0:
                        self.food_null = 1
                    else:
                        temp.append(f)
        data = data.sort_values(by='scores', ascending=False)
        if temp != []:
            for f in temp:
                data = data.sort_values(by=f, ascending=False)
        output_info_list = []

        data_len = 3
        if data.empty != True:
            if len(data) < 3:
                data_len = len(data)
            output_info_list = list(zip(data['names'].to_list()[:data_len], data['location'].to_list()[:data_len],
                                        data['images'].to_list()[:data_len], data['links'].to_list()[:data_len]))

        return data, output_info_list

    def Bot_Messeging(self, data, dispatcher, output):
        utter_row = res_table[res_table['intent'] == self.intent]

        if self.restype_entityname == '':
            featureless_str = utter_row['featureless'].to_list()[0]
            return dispatcher.utter_message(text=featureless_str)
        elif data.empty == True:
            return dispatcher.utter_message(text='???????????????. ???????????? ?????? ??? %s ????????? ?????? ??? ????????????.' % self.restype_norm)
        else:
            first_response = utter_row['response'].to_list()[0].split(' / ')
            first_response = [s.replace('<RESTAURANT-TYPE_FEATURE>', self.restype_norm) for s in first_response]
            if self.city_entityname == '':
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' not in s]
            elif self.city_entityname not in city_all:
                dispatcher.utter_message(text='???????????? ????????? ?????? ????????? ????????? ?????????! ?????? ????????? ?????? ????????? ????????? ???????????????.')
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' not in s]
            else:
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' in s]
                first_response = [s.replace('<LOCATION-TYPE_FEATURE>', self.city_norm) for s in first_response]

            if self.food_entityname != '':
                first_response = [s for s in first_response if '<FOOD-TYPE_FEATURE>' in s]
                first_response = [s.replace('<FOOD-TYPE_FEATURE>', self.food_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<FOOD-TYPE_FEATURE>' not in s]

            if self.ingredient_entityname != '':
                first_response = [s for s in first_response if '<INGREDIENT-TYPE_FEATURE>' in s]
                first_response = [s.replace('<INGREDIENT-TYPE_FEATURE>', self.ingredient_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<FOOD-TYPE_FEATURE>' not in s]

            if self.intent == 'RECOMMEND_TASTE-TYPE' and self.tastetype_entityname != '':
                first_response = [s for s in first_response if '<TASTE-TYPE_FEATURE>' in s]
                first_response = [s.replace('<TASTE-TYPE_FEATURE>', self.tastetype_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<TASTE-TYPE_FEATURE>' not in s]

            if self.intent == 'RECOMMEND_GOODTOGO' and self.goodtogo_entityname != '':
                first_response = [s for s in first_response if '<GOODTOGO_FEATURE>' in s]
                first_response = [s.replace('<GOODTOGO_FEATURE>', self.goodtogo_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<GOODTOGO_FEATURE>' not in s]

            if self.intent == 'RECOMMEND_PROVIDED' and self.provided_entityname != '':
                first_response = [s for s in first_response if '<PROVIDED_FEATURE>' in s]
                first_response = [s.replace('<PROVIDED_FEATURE>', self.provided_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<PROVIDED_FEATURE>' not in s]

            if self.view_entityname != '':
                first_response = [s for s in first_response if '<VIEW_FEATURE>' in s]
                first_response = [s.replace('<VIEW_FEATURE>', self.view_norm) for s in first_response]
            else:
                first_response = [s for s in first_response if '<VIEW_FEATURE>' not in s]

            try:
                first_response = random.sample(first_response, 1)[0]
                undifine_josa = re.findall("\w<[???-???]+?>", first_response)
                if undifine_josa != []:
                    for pattern in undifine_josa:
                        first_response = Josa_Replace(pattern=pattern, sentence=first_response)

                dispatcher.utter_message(text=first_response)

                # ????????? ????????? ???????????? ?????? ????????? ????????? ?????? ????????? ?????????
                if self.food_null == 1:
                    null_sentence = "??? ??????! <LOCATION-TYPE_FEATURE>??? ?????? ?????? ????????? ???????????? ????????? ?????? <FOOD-TYPE_FEATURE><???> ???????????? ????????? ?????????. ??? ?????? <LOCATION-TYPE_FEATURE>?????? ?????? ?????? <RESTAURANT-TYPE_FEATURE><???> ?????????????????????."
                    null_sentence = null_sentence.replace('<FOOD-TYPE_FEATURE>',self.food_norm)
                    null_sentence = null_sentence.replace('<RESTAURANT-TYPE_FEATURE>', self.restype_norm)
                    if self.city_entityname == '':
                        null_sentence = null_sentence.replace('<LOCATION-TYPE_FEATURE>','??????')
                    else:
                        null_sentence = null_sentence.replace('<LOCATION-TYPE_FEATURE>', self.city_norm)
                    undifine_josa = re.findall("\w<[???-???]+?>", null_sentence)
                    if undifine_josa != []:
                        for pattern in undifine_josa:
                            null_sentence = Josa_Replace(pattern=pattern, sentence=null_sentence)
                    dispatcher.utter_message(text=null_sentence)

                dispatcher.utter_message(text=utter_row["utter_send_link"].values[0])

                num = 1
                for name, loc, img, link in output:
                    text = '%s??? ????????????: %s\n??????: %s\n??????: %s' % (str(num), name, loc, link)
                    dispatcher.utter_message(text=text)
                    dispatcher.utter_message(image=img)
                    num += 1

                return dispatcher.utter_message(text=utter_row["utter_ask_more"].values[0])
            except ValueError:
                return dispatcher.utter_message(text='????????? ????????? ????????? ???????????? ?????????! ?????? ??? ???????????? ????????? ????????? :)')


