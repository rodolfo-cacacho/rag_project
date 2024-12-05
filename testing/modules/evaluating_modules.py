from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import pandas as pd

class RAGEvaluator:
    def __init__(self,sql_con,test_name,test_table_name,qas_table_name,chunks_eval_table_name,prompts_table_name):

        self.sql_con = sql_con
        self.TEST_NAME = test_name
        self.test_table_name = test_table_name
        self.qas_table_name = qas_table_name
        self.chunks_eval_table_name = chunks_eval_table_name
        self.prompts_table_name = prompts_table_name
        self.data_df = self.fetch_data()

    def fetch_data(self):
        """
        Fetch data from the test table based on the test name and other conditions.
        
        Returns:
            list of dict or pandas.DataFrame: Retrieved data.
        """
        test_prompts_cols = ['id_question','prompt_id','device']
        test_prompts_conditions = {
            'test_name':self.TEST_NAME,
            'status':'success'
        }

        test_prompts_data = self.sql_con.get_records(self.test_table_name,test_prompts_cols,test_prompts_conditions)

        df_test_prompts = pd.DataFrame(test_prompts_data)

        questions = df_test_prompts['id_question'].tolist()

        qas_cols = ['id_question','id_sample','type_question','question','expected_answer','clarity','specificity','relevance','clarity_q','specificity_q','relevance_q']
        qas_conditions = {
            'id_question':questions,
            'valid':1
        }

        qas_data = self.sql_con.get_records(self.qas_table_name,qas_cols,qas_conditions)

        df_questions = pd.DataFrame(qas_data)

        results_df = pd.merge(df_questions, df_test_prompts, on='id_question', how='inner')

        id_samples = df_questions['id_sample'].tolist()

        chunks_cols = ['id_sample','doc_type','source','content','id','metadata']
        chunks_conditions = {
            'id_sample':id_samples
        }
        chunks_data = self.sql_con.get_records(self.chunks_eval_table_name,chunks_cols,chunks_conditions)

        chunks_df = pd.DataFrame(chunks_data)

        results_df = pd.merge(results_df,chunks_df,on='id_sample',how='left')

        prompt_ids = results_df['prompt_id'].tolist()
        device_test = list(set(results_df['device'].tolist()))[0]

        prompts_cols = ['prompt_id','device','answer','begin_date','end_date','completion_tokens','prompt_tokens',
                        'context_used','context_ids','context_ids_total','alternate_prompts','improved_query','query_intent','keyterms']
        
        prompts_conditions = {
            'prompt_id':prompt_ids,
            'device':device_test
        }

        prompts_data = self.sql_con.get_records(self.prompts_table_name,prompts_cols,prompts_conditions)

        prompts_df = pd.DataFrame(prompts_data)
        prompts_df['device'] = prompts_df['device'].str.lower()

        results_df = pd.merge(results_df,prompts_df,on=['prompt_id','device'],how='inner')
        
        return results_df
    
    def generate_report(self):



        avg_prompt_tokens = self.data_df['prompt_tokens'].mean()
        avg_answer_tokens = self.data_df['completion_tokens'].mean()

        std_prompt_tokens = self.data_df['prompt_tokens'].std()
        std_answer_tokens = self.data_df['completion_tokens'].std()

        

        report = {
            'prompt_tokens':{
                'mean':avg_prompt_tokens,
                'std':std_prompt_tokens
            },
            'answer_tokens':{
                'mean':avg_answer_tokens,
                'std':std_answer_tokens
            }
        }

        return report