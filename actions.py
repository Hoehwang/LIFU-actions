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

rec_table = pd.read_csv('./actions/recommend_table_lifu.csv')
syn = pd.read_csv("./actions/SYN_LIFU.csv")
res = pd.read_csv("./actions/RESPONSE_EXP_LIFU.csv")

city_all = list(set(rec_table['cities'].tolist()))
column_all = list(rec_table.columns)

restaurent_type_dict = {'KOREAN':'한식당', 'CHINESE':'중식집', 'JAPANESE':'일식집', 'WESTERN':'양식집', 'CAFE':'카페', 'WORLD':'세계 음식점', 'PUB':'술집', 'SNACK':'분식집', 'BUFFET':'뷔페', 'RESTAURANT-GEN':'식당'}
# taste_type_list = ['NOT-SWEET','NOT-HOT','NOT-JAGEUKJEOK','NOT-OILY','NOT-BLAND','CLEANESS','NOT-SALTY','KKALKKEUM','SWEET','HOT','SWEET-SALTY','SIWON','GOSO','FIRE','FRESH','JJONDEUK','CRUNCHY','OILY','TTAKKEUN','CHEWY','FUDGY','CREAMY','STRONG','SPICY-SWEET','COLD']
goodtogo_list = ["LOVER","FRIEND","FAMILY","ANNIVERSARY","ENGAGEMENT","FIRSTBIRTHDAY","COLD-WEATHER","HOT-WEATHER","LOVER-VERB","CLIMBING","SOCCER","BASKETBALL","ALCOHOL","PARTY","ALONE","COUPLE","TRIPLE","GROUP"]
taste_type_list = ['FRESH','SWEET','HOT','SWEET-SALTY','CRUNCHY','GOSO','SPICY-SWEET','JJONDEUK','TTAKKEUN','CREAMY','CHEWY','FUDGY','COLD','SIWON','FIRE','KKALKKEUM','STRONG','NOT-SALTY','OILY','NOT-HOT','NOT-JAGEUKJEOK','NOT-OILY','NOT-SWEET','NOT-BLAND']
# goodtogo_fea = ['LOVER','FRIEND','FAMILY','ANNIVERSARY','ENGAGEMENT','HOT-WEATHER','FIRSTBIRTHDAY','COLD-WEATHER','CLIMBING','SOCCER','BASKETBALL','ALCOHOL','PARTY','ALONE','COUPLE','TRIPLE','GROUP']
# provide_fea = ['PLAYROOM','PARKING','KIOSK','SOCKET','PRIVATE-ROOM','SALAD-BAR','OPENSPACE']

def Josa_Replace(pattern, sentence):
    if '](' in pattern:
        if len(pattern.split('](')) > 2:
            back_syl = pattern.split('](')[-2][-1]
        else:
            back_syl = pattern.split('](')[0][-1]
    else:
        back_syl = pattern[0]

    criteria = (ord(back_syl) - 44032) % 28
    if criteria == 0: #무종성
        repl_pattern = re.sub("<[이가]>", "가", pattern)
        repl_pattern = re.sub("<[은는]>", "는", repl_pattern)
        repl_pattern = re.sub("<[을를]>", "를", repl_pattern)
        repl_pattern = re.sub("<[와과]>", "와", repl_pattern)
        repl_pattern = re.sub("<(이랑|랑)>", "랑", repl_pattern)
        repl_pattern = re.sub("<(으로|로)>", "로", repl_pattern)
        repl_pattern = re.sub("<(이서|서)>", "서", repl_pattern)
    else:
        repl_pattern = re.sub("<[이가]>", "이", pattern)
        repl_pattern = re.sub("<[은는]>", "은", repl_pattern)
        repl_pattern = re.sub("<[을를]>", "을", repl_pattern)
        repl_pattern = re.sub("<[와과]>", "과", repl_pattern)
        repl_pattern = re.sub("<(이랑|랑)>", "이랑", repl_pattern)
        repl_pattern = re.sub("<(으로|로)>", "으로", repl_pattern)
        repl_pattern = re.sub("<(이서|서)>", "이서", repl_pattern)
    # print(pattern, '\t', back_syl , '\t', '%s' % repl_pattern)
    return sentence.replace(pattern, '%s' % repl_pattern)

class ActionRephraseResponse(Action):
    # 액션에 대한 이름을 설정
    def name(self) -> Text:
        return "action_rephrase_restaurant"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        self.city = ''
        self.city_name = ''
        self.food = ''
        self.food_name = ''
        self.ingredient = ''
        self.ingredient_name = ''
        self.taste = ''
        self.taste_name = ''
        self.res_type = ''
        self.res_type_name = ''
        self.goodtogo = ''
        self.feature = []
        self.feature_name = []
        entity_dicts = tracker.latest_message['entities']
        print(tracker.latest_message['entities'])
        # print(tracker.get_intent_of_latest_message())
        for entity in entity_dicts:
            if '-LOC' in entity['entity']:
                self.city = entity['entity'].replace('-LOC','')
                self.city_name = entity['value']
            elif 'FOOD' == entity['entity']:
                self.food_name = entity['value']
                try:
                    self.food = list(syn[(syn.examples == self.food_name) | (syn.examples.str.contains(self.food_name))]['norms'])[0]
                except IndexError:
                    for e, n in zip(list(syn['examples']), list(syn['norms'])):
                        if e in self.food_name:
                            self.food = n
                            break
                        else:
                            pass
            elif 'INGREDIENT' == entity['entity']:
                self.ingredient_name = entity['value']
                try:
                    self.ingredient = list(syn[(syn.examples == self.ingredient_name) | (syn.examples.str.contains(self.ingredient_name))]['norms'])[0]
                except IndexError:
                    for e, n in zip(list(syn['examples']), list(syn['norms'])):
                        if e in self.ingredient_name:
                            self.ingredient = n
                            break
                        else:
                            pass
            elif entity['entity'] in goodtogo_list:
                self.goodtogo_name = entity['entity']
                self.goodtogo = list(syn[syn['entity name'] == self.goodtogo_name]['norms'])[0]
                self.feature.append(entity['entity'])
            elif entity['entity'] in taste_type_list:
                self.taste_name = entity['entity']
                self.taste = list(syn[syn['entity name'] == self.taste_name]['norms'])[0]
                self.feature.append(entity['entity'])
            elif entity['entity'] in restaurent_type_dict.keys():
                self.res_type = entity['entity']
                self.res_type_name = restaurent_type_dict[self.res_type]
            else:
                self.feature.append(entity['entity'])
        self.intent = tracker.get_intent_of_latest_message()

        print('City: ', self.city, '\tName :', self.city_name)
        print('Food: ', self.food, '\tName :', self.food_name)
        print('Restaurent Type: ', self.res_type, '\tName: ', self.res_type_name)
        print('Other Features: ', self.feature)
        print('Intent: ', self.intent)

        data, output_info_list = self.DataSorting(df=rec_table, city=self.city, food=self.food, res_type=self.res_type, feature=self.feature)

        self.Bot_Messeging(intent=self.intent, data=data, dispatcher=dispatcher, output=output_info_list)

        return []

    def DataSorting(self, df, city, food, res_type, feature):
        data = df
        temp = []
        temp.append('scores')

        if city != '' and city in city_all:
            data = data[data['cities'] == city]
        if res_type != '' and res_type != 'RESTAURANT-GEN':
            data = data[data['categories'] == res_type]
        if food != '' and food in column_all:
            temp.append(food)
            # data = data.sort_values(by=food ,ascending=False)
        if feature != []:
            for f in feature:
                if f in column_all:
                    temp.append(f)

        data = data.sort_values(by=temp, ascending=False)
        output_info_list = []

        data_len = 3
        if data.empty != True:
            if len(data) < 3:
                data_len = len(data)
            output_info_list = list(zip(data['names'].to_list()[:data_len], data['location'].to_list()[:data_len],
                                        data['images'].to_list()[:data_len], data['links'].to_list()[:data_len]))

        return data, output_info_list

    def Bot_Messeging(self, intent, data, dispatcher, output):
        utter_row = res[res['intent'] == intent]

        if self.res_type == '':
            featureless_str = utter_row['featureless'].to_list()[0]
            return dispatcher.utter_message(text=featureless_str)
        elif data.empty == True:
            return dispatcher.utter_message(text='죄송합니다. 검색하신 지역 내 %s 정보를 찾을 수 없습니다.' % self.res_type_name)
        else:
            first_response = utter_row['response'].to_list()[0].split(' / ')
            if self.city == '':
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' not in s]
                first_response = [s.replace('<RESTAURANT-TYPE_FEATURE>', self.res_type_name) for s in first_response]
            elif self.city not in city_all:
                dispatcher.utter_message(text='입력하신 지역에 대한 음식점 정보가 없네요! 검색 범위를 지역 전체로 재설정 하겠습니다.')
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' not in s]
                first_response = [s.replace('<RESTAURANT-TYPE_FEATURE>', self.res_type_name) for s in first_response]
            else:
                first_response = [s for s in first_response if '<LOCATION-TYPE_FEATURE>' in s]
                first_response = [
                    s.replace('<LOCATION-TYPE_FEATURE>', self.city_name).replace('<RESTAURANT-TYPE_FEATURE>', self.res_type_name)
                    for s in first_response]

            if self.intent == 'RECOMMEND_TASTE-TYPE' and self.taste != '':
                first_response = [s for s in first_response if '<TASTE-TYPE_FEATURE>' in s]
                first_response = [s.replace('<TASTE-TYPE_FEATURE>', self.taste) for s in first_response]
            else:
                first_response = [s for s in first_response if '<TASTE-TYPE_FEATURE>' not in s]
            if self.food_name != '':
                first_response = [s for s in first_response if '<FOOD-TYPE_FEATURE>' in s]
                first_response = [s.replace('<FOOD-TYPE_FEATURE>', self.food) for s in first_response]
            else:
                first_response = [s for s in first_response if '<FOOD-TYPE_FEATURE>' not in s]
            if self.goodtogo != '':
                first_response = [s for s in first_response if '<GOODTOGO_FEATURE>' in s]
                first_response = [s.replace('<GOODTOGO_FEATURE>', self.goodtogo) for s in first_response]
            else:
                first_response = [s for s in first_response if '<GOODTOGO_FEATURE>' not in s]
            first_response = random.sample(first_response, 1)[0]

            undifine_josa = re.findall("\w<[가-힣]+?>", first_response)
            if undifine_josa != []:
                for pattern in undifine_josa:
                    first_response = Josa_Replace(pattern=pattern, sentence=first_response)

            dispatcher.utter_message(text=first_response)
            dispatcher.utter_message(text=utter_row["utter_send_link"].values[0])

            num = 1
            for name, loc, img, link in output:
                text = '%s위 음식점명: %s\n주소: %s\n링크: %s' % (str(num), name, loc, link)
                dispatcher.utter_message(text=text)
                dispatcher.utter_message(image=img)
                num += 1

            dispatcher.utter_message(text=utter_row["utter_ask_more"].values[0])