from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import json
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

        avg_proc_time = (self.data_df['end_date'] - self.data_df['begin_date']).mean()
        std_proc_time = (self.data_df['end_date'] - self.data_df['begin_date']).std()

        used_n_exp,used_exp,ret_n_exp,ret_exp,dif_ids_avg = self.evaluate_context()
        total_Qs = len(self.data_df)
        report = {
            'totalQs':total_Qs,
            'difRetChunks':dif_ids_avg,
            'prompt_tokens':{
                'mean':avg_prompt_tokens,
                'std':std_prompt_tokens
            },
            'answer_tokens':{
                'mean':avg_answer_tokens,
                'std':std_answer_tokens
            },
            'retrieval_accuracy':{
                'selected':used_n_exp,
                'retrieved':ret_n_exp
            },
            'retrieval_accuracy_exp':{
                'selected':used_exp,
                'retrieved':ret_exp
            },
            'processing_time':{
                'mean':avg_proc_time.total_seconds(),
                'std':std_proc_time.total_seconds()
            }
        }

        return report
    
    def get_adj_ids(self,id, last_id, dif=1):
        """
        Get all adjacent IDs within a given difference from the original ID.
        
        Args:
            id (str): The current ID in the format "base.cid".
            last_id (str): The last ID in the format "base.lid".
            dif (int): The difference to consider for adjacent IDs.

        Returns:
            list: A list of adjacent IDs within the specified range.
        """
        adjs_ids = []

        # Split and parse the base and numeric parts
        base, cid = id.split('.')
        _, lid = last_id.split('.')

        # Convert to integers
        cid = int(cid)
        lid = int(lid)

        # Generate all IDs within the range [cid-dif, cid+dif]
        for offset in range(-dif, dif + 1):
            adj_cid = cid + offset
            # Ensure IDs are within bounds and not the original ID
            if 0 <= adj_cid <= lid and adj_cid != cid:
                adjs_ids.append(f"{base}.{adj_cid}")

        return adjs_ids


    def evaluate_context(self):

        used_count = 0
        retrieved_count = 0

        used_exp_count = 0
        retrieved_exp_count = 0
        total_ids_retrieved_ct = 0

        for index,row in self.data_df.iterrows():
            metadata = json.loads(row['metadata'])
            end_chunk = metadata['last_id']
            context_ids = json.loads(row['context_ids'])
            total_context_ids = json.loads(row['context_ids_total'])
            total_ids_retrieved_ct += len(total_context_ids)
            original_chunk_id = row['id']
            adj_context = self.get_adj_ids(original_chunk_id,end_chunk,dif=1)
            adj_context.append(original_chunk_id)
            used_context = original_chunk_id in context_ids
            retrieved_context = original_chunk_id in total_context_ids
            
            used_context_ext = any(item in adj_context for item in context_ids) 
            retrieved_context_ext = any(item in adj_context for item in total_context_ids)

            used_count+=used_context
            retrieved_count+=retrieved_context
            used_exp_count+=used_context_ext
            retrieved_exp_count+=retrieved_context_ext

        used_n_exp = used_count/len(self.data_df)
        used_exp = used_exp_count/len(self.data_df)

        ret_n_exp = retrieved_count/len(self.data_df)
        ret_exp = retrieved_exp_count/len(self.data_df)

        total_ids_retrieved_avg = total_ids_retrieved_ct/len(self.data_df)

        return used_n_exp,used_exp,ret_n_exp,ret_exp,total_ids_retrieved_avg