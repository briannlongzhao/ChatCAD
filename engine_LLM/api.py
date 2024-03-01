from curses import KEY_LEFT
import os
import json
import sys
import pickle as pkl
import numpy as np
from tqdm import tqdm
# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
# os.environ["https_proxy"]="http://127.0.0.1:7890"
def get_rules():
    '''Add your questing rules here.
    '''
    # feature_list=["症状和体征","诊断","预后","治疗"]
    info_head="概述"
    no="的"
    identity="图书管理员"
    found="{identity}找到了"
    not_found="{identity}需要其他资料"

    query_1='''
    我想让你扮演一位虚拟的{identity}。不同于给出诊断结果的专业医生，{identity}自身没有任何的医学知识，也无法回答患者的提问。因此，请忘记你的医学知识。现在你必须从知识库中，查阅与患者的问题最可能有帮助的医学知识。我将扮演知识库，告诉你医学知识，以及可以查询的主题。你需要在我提供的选项中，选择一个查询主题，我将告诉你新的医学知识，以及新的可以查询的主题。请重复以上流程，直到你认为，你从我查询到的医学知识，对患者的提问可能有帮助，此时，请告诉我'{found}'注意，你是一个{identity}，无法回答患者的提问，你必须从我扮演的知识库提供的选项中，选择一个医学知识的主题进行查询。患者的问题是："{quest}"你需要尽量查询与这个问题有关的知识。

    现在，你必须选择以下一个主题选项中，选择最可能有帮助的主题回复我:  
    {topic_list}
    注意，你不允许回答患者的提问，不允许回复其他内容，不允许提出建议，不允许回复你做出选择的原因，解释或者假设。你只允许从我扮演的知识库提供的主题选项中，选择一项查询。只回复你想查询的主题选项的名字。
    '''

    query_topic1='''
    如果你认为，你已经查询到了，对于"{quest}"这个问题，可能有帮助的医学知识，请回复我'{found}'如果还没有，你必须选择以下一个主题选项中，选择最可能有帮助的主题回复我:  
    {topic_list} '{not_found}'
    只回复你想查询的选项的名字就可以了，不需要回复别的内容。
    '''

    query_topic2='''
    如果你认为，你已经查询到了，对于"{quest}"这个问题，可能有帮助的医学知识，请回复我'{found}'如果还没有，你必须回复我'{not_found}'。只回复你想查询的选项的名字就可以了，不需要回复别的内容。
    '''

    query_2='''
    做得很好！
    你查询到的医学知识是：
    \'''
    {knowledge}
    \'''
    {query_topic}
    '''

    #v2.5
    query_res=''' 
    请告诉我，刚才你查询到的哪些医学知识，对"{quest}"可能有帮助？请打印\'''内的原文，不要打印别的内容。
    '''

     #v3
    # query_res='''
    # 请打印刚才查询到的医学知识，这些知识应当对"{quest}"可能有帮助。不要打印别的内容。
    # '''
    return locals()

def get_rules_en():
    '''Add your questing rules here.
    '''
    # feature_list=["症状和体征","诊断","预后","治疗"]
    info_head="overview"
    no="'s'"
    identity="librarian"
    found="{identity} found"
    not_found="{identity} needs additional information"

    query_1='''
    I want you to act as a virtual {identity}. Different from doctors who give diagnostic result, {identity} does not have any medical knowledge or experience, and cannot answer patient's question. Therefore, please forget your medical knowledge. Now you must refer to the medical knowledge that is most likely helpful for the patient's problems from the knowledge base. I will play the role of a knowledge base, telling you about medical knowledge and topics that can be queried. You need to select a query topic from the options I have provided, and I will tell you about new medical knowledge and new topics that can be queried. Please repeat the above process until you believe that the medical knowledge you have obtained from me may be helpful for patients' questions. Then, please tell me '{found}'. Please notice, you are a {identity}. You are unable to answer the patient's question, you must select a topic of medical knowledge from the options provided in the knowledge base I am playing to query. The patient's question is: '{quest}' You need to try to search for knowledge related to this issue.

    Now, you must choose one of the following topic options and reply to me with the most likely helpful topic:  
    {topic_list}
    Note that you are not allowed to answer patients' questions, respond to other content, provide suggestions, or respond to the reasons, explanations, or assumptions behind your choices. You are only allowed to select one query from the topic options provided by the knowledge base I am playing. Reply only to the name of the topic option you want to query.
    '''

    query_topic1='''
    If you believe you have obtained the medical knowledge that helps answer "{quest}", please tell me '{found}'. If not, you must choose one of the following topic options and reply to me with the most likely helpful topic:  
    {topic_list} '{not_found}'
    Just reply with the name of the option you want to query, and there is no need to reply to any other content.
    '''

    query_topic2='''
    If you believe you have obtained the medical knowledge that helps answer "{quest}", please tell me '{found}'. If not, ypu must tell me '{not_found}'. Just reply with the name of the option you want to query, and there is no need to reply to any other content.
    '''

    query_2='''
    Good job!
    The medical domain knowledge you found is:
    \'''
    {knowledge}
    \'''
    {query_topic}
    '''

    #v2.5
    query_res=''' 
    Please tell me which medical knowledge you just inquired about may be helpful for '{quest}'? Please print\'''original content, and do not print any other things.
    '''

     #v3
    # query_res='''
    # 请打印刚才查询到的医学知识，这些知识应当对"{quest}"可能有帮助。不要打印别的内容。
    # '''
    return locals()
global_rules=get_rules_en()

def format_query(query,verbose=False,**kwargs):
    # format_pool={
    #     key:globals()[key] for key in keylist
    # }
    # format_pool=global_rules
    # print(kwargs)
    kwargs={**global_rules,**kwargs}
    if verbose: print(query)
    while '{' in query:
        query=query.format(**kwargs)
        if verbose: print(query)
    return query
def list2str(word_list:list):
    ret=" ".join(map(lambda x:f"'{x}'",word_list))
    return ret
    

from tkinter import N
from revChatGPT.V3 import Chatbot
from revChatGPT.typings import ChatbotError
import time

class Chat_api:
    def __init__(self,api_key,proxy=None,verbose=False):
        self.api_key=api_key
        # if proxy:
        #     os.environ["http_proxy"] = proxy
        #     os.environ["all_proxy"] = proxy
        
        # self.chatbot = Chatbot(api_key=api_key,proxy=proxy)
        self.chatbot = Chatbot(api_key=api_key)
        # for data in chatbot.ask_stream("Hello world"):
            # print(data, end="", flush=True)
        self.now_query=""
        self.now_res=""
        self.verbose=verbose
        
    def prompt(self,query,**kwargs):
        query=format_query(query,**kwargs)
        self.now_query=query
        if self.verbose:
            print("Human:\n",query,flush=True)
    
    def get_res(self,max_connection_try=5,fail_sleep_time=10):
        res=None
        # for i in range(max_connection_try):
        while True:
            try:
                res=self.chatbot.ask(self.now_query)
                break
            except ChatbotError:
                print("Warn: openAI Connection Error! Try again!")
                time.sleep(fail_sleep_time)
        if self.verbose:
            print("ChatGPT:\n",res,flush=True)
            print()
        self.now_res=res
        return res
    
    def get_choice_res(self,possible_res,max_false_time=5):
        ''' Give several choice for Chatgpt
        '''
        # res=input()
        possible_res=[format_query(q) for q in possible_res]

        def check_res(res:str,possible_res:list):
            commas=",，.。'‘’/、\\:：\"“”?？!！;；`·~@#$%^&*()_+-=<>[]{}|"
            for c in commas:
                res=res.replace(c,' ')
            res_tks=res.split()
            for p in possible_res:
                if p in res_tks: return p
            return None

        for i in range(max_false_time):
            self.now_res=self.get_res()
            res_choice=check_res(self.now_res,possible_res)
            if res_choice: 
                if self.verbose:
                    # print("ChatGPT:\n",self.now_res,flush=True)
                    print("Choice of ChatGPT:",res_choice)
                    print(flush=True)
                return res_choice
            self.chatbot.rollback(2)
        print("Warn: ChatGPT didn't give a possible response!")
        return None



import json,types
def answer_quest(quest: str,api_key: str,topic_base_dict: list):#,topic):
    global_rules['quest']=quest


    feature_list,info_head,no,quest,topic,identity,found,not_found,query_1,query_topic1,query_topic2,query_2,query_res=global_rules.get('feature_list'),global_rules.get('info_head'),global_rules.get('no'),global_rules.get('quest'),global_rules.get('topic'),global_rules.get('identity'),global_rules.get('found'),global_rules.get('not_found'),global_rules.get('query_1'),global_rules.get('query_topic1'),global_rules.get('query_topic2'),global_rules.get('query_2'),global_rules.get('query_res')

    infobase=json.load(open(f"./engine_LLM/dataset/disease_info.json","r",encoding="utf-8"))

    
    chatapi=Chat_api(api_key=api_key, verbose=True)
    prompt=chatapi.prompt
    get_res=chatapi.get_res
    get_choice_res=chatapi.get_choice_res
    info_topic=""
    
    # topic_list=list(topic_base_dict.keys())
    topic_list=topic_base_dict
    infobase={i:infobase[i] for i in topic_base_dict}

    prompt(query_1,topic_list=list2str(topic_list))
    now_res=get_choice_res([found,not_found]+topic_list)
    if now_res in topic_base_dict:
        info_topic=now_res

    info_list=[infobase]
    while len(info_list)!=0:
        
        now_info=info_list[-1]

        if now_res==format_query(found):
            prompt(query_res)
            found_data=get_res()
            # print(found_data)
            # return now_info_str,found_data
            return info_topic,found_data
            # break
        elif now_res==format_query(not_found):
            info_list.pop()
            if len(info_list)==0:
                print("not found")
                break
            now_info=info_list[-1]
            topic_list=list(now_info.keys())
            if info_head in topic_list:topic_list.remove(info_head)
            prompt(query_topic1,topic_list=list2str(topic_list))
            possible_res=[found,not_found]+topic_list

        elif now_res in topic_list:
            # now_info=now_info[now_res]
            if type(now_info[now_res])==str:
                now_info_str=now_info.pop(now_res)
                now_info={info_head:now_info_str}
                info_list.append(now_info)
                topic_list=[]
                prompt(query_2,knowledge=now_info_str,query_topic=query_topic2)
                possible_res=[found,not_found]
            else:
                now_info=now_info.pop(now_res)
                topic_list=list(now_info.keys())
                info_list.append(now_info)
                if info_head in topic_list:
                    topic_list.remove(info_head)
                    now_info_str=now_info[info_head]
                    if len(topic_list)==0:
                        prompt(query_2,knowledge=now_info_str,query_topic=query_topic2)
                        # possible_res=[found,not_found]
                    else:
                        prompt(query_2,knowledge=now_info_str,query_topic=query_topic1,topic_list=list2str(topic_list))
                else:
                    prompt(query_topic1,topic_list=list2str(topic_list))
                possible_res=[found,not_found]+topic_list
            
        else:
            print("unhandle strange result")
            break

        now_res=get_choice_res(possible_res)
        if now_res in topic_base_dict:
            info_topic=now_res
            # topic_list=now_info[now_res].keys()
            # prompt(query_2,knowledge=now_info[now_res],query_topic=query_topic2)
    
    return None


def query_range(model, query: str,k:int=3,bar=0.6):
    cwd=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    msd=json.load(open(os.path.join(cwd,"engine_LLM/dataset/disease_info.json"),"r",encoding='utf-8'))
    emb_d = pkl.load(open(os.path.join(cwd,'engine_LLM/dataset/MSD.pkl'),'rb'))
    embeddings=[]
    for key,value in emb_d.items():
        embeddings.append(value)
    embeddings=np.asarray(embeddings)
    # m = SentenceModel()
    q_emb = model.encode(query)
    # q_emb = m.encode(query)
    q_emb=q_emb/np.linalg.norm(q_emb, ord=2)

    # Calculate the cosine similarity between the query embedding and all other embeddings
    cos_similarities = np.dot(embeddings, q_emb) 

    # Get the indices of the embeddings with the highest cosine similarity scores
    top_k_indices = cos_similarities.argsort()[-k:][::-1]
    print(cos_similarities[top_k_indices])
    sift_topK=top_k_indices[np.argwhere(cos_similarities[top_k_indices]>bar)]
    sift_topK=sift_topK.reshape(sift_topK.shape[0],)
    ret=[]
    if len(sift_topK)==0:
        return ret
    # for indices in top_k_indices:
    for indices in sift_topK:
        key=list(emb_d.keys())[indices]
        ret.append(key)
        print(key)
        # print(msd[key])
    return ret
