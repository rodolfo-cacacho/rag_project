from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import json
import pandas as pd

class RAGEvaluator:
    def __init__(self,sql_con,test_table_name,qas_table_name,chunks_eval_table_name,prompts_table_name,eval_answers_table):

        self.sql_con = sql_con
        self.test_table_name = test_table_name
        self.qas_table_name = qas_table_name
        self.chunks_eval_table_name = chunks_eval_table_name
        self.prompts_table_name = prompts_table_name
        self.eval_answer_table_name = eval_answers_table
        self.data_df = self.calculate_custom_metrics(self.fetch_data())


    def fetch_data(self):
        """
        Fetch data from the test table based on the test name and other conditions.
        
        Returns:
            list of dict or pandas.DataFrame: Retrieved data.
        """
        test_prompts_cols = ['id_question','prompt_id','device','test_name','id']
        test_prompts_conditions = {
            'status':'success'
        }

        test_prompts_data = self.sql_con.get_records(self.test_table_name,test_prompts_cols,test_prompts_conditions)

        df_test_prompts = pd.DataFrame(test_prompts_data)
        df_test_prompts.rename(columns={'id': 'id_test_question'}, inplace=True)

        questions = df_test_prompts['id_question'].tolist()
        id_test_questions = df_test_prompts['id_test_question'].tolist()

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

        prompts_cols = ['prompt_id','device','answer','begin_date','end_date','completion_tokens','prompt_tokens','chunk_size','embedding_model','alpha_value',
                        'context_used','context_ids','context_ids_total','alternate_prompts','improved_query','query_intent','keyterms']
        
        prompts_conditions = {
            'prompt_id':prompt_ids,
        }

        prompts_data = self.sql_con.get_records(self.prompts_table_name,prompts_cols,prompts_conditions)

        prompts_df = pd.DataFrame(prompts_data)
        prompts_df['device'] = prompts_df['device'].str.lower()

        results_df = pd.merge(results_df,prompts_df,on=['prompt_id','device'],how='inner')
                # record={
                #     'id': id,
                #     'test_name': test_name,
                #     'score': score,
                #     'id_question':id_question,
                #     'comment':comment
                # },overwrite = overwrite

        eval_answer_cols = ['id','score','comment','precisionA','recallA','f1_scoreA']
        
        eval_answer_conditions = {
            'id':id_test_questions
        }

        answers_score = self.sql_con.get_records(self.eval_answer_table_name,eval_answer_cols,eval_answer_conditions)
        answers_score_df = pd.DataFrame(answers_score)
        answers_score_df.rename(columns={'id': 'id_test_question'}, inplace=True)

        results_df = pd.merge(results_df,answers_score_df,on=['id_test_question'],how='inner')
        
        return results_df
    
    def calculate_custom_metrics(self,df):
        """
        Preprocess the dataframe by adding custom metric columns.
        """
        def get_adj_ids(id, last_id, dif=1):
            adjs_ids = []
            base, cid = id.split('.')
            _, lid = last_id.split('.')
            cid = int(cid)
            lid = int(lid)
            for offset in range(-dif, dif + 1):
                adj_cid = cid + offset
                if 0 <= adj_cid <= lid and adj_cid != cid:
                    adjs_ids.append(f"{base}.{adj_cid}")
            return adjs_ids

        def row_metrics(row):
            metadata = json.loads(row['metadata'])
            type = metadata['type']
            type = 'Table' if 'table' in type.lower() else 'Text'
            end_chunk = metadata['last_id']
            doc_id,_ = row['id'].split('.')
            context_ids = json.loads(row['context_ids'])
            total_context_ids = json.loads(row['context_ids_total'])
            context_ids_docs = [i.split('.')[0] for i in context_ids]
            total_context_ids_docs = [i.split('.')[0] for i in total_context_ids]
            adj_context = get_adj_ids(row['id'], end_chunk, dif=1)
            adj_context.append(row['id']),
            binary_score = 1 if row['score'] > 3 else 0

            return {
                'type': type,
                'used_doc': doc_id in context_ids_docs,
                'retrieved_doc': (sum(1 for item in total_context_ids_docs if item == doc_id) / len(total_context_ids_docs)),
                'used_context': row['id'] in context_ids,
                'retrieved_context': row['id'] in total_context_ids,
                'used_context_ext': any(item in adj_context for item in context_ids),
                'retrieved_context_ext': any(item in adj_context for item in total_context_ids),
                'total_context_ids': len(total_context_ids),
                'binary_score' : binary_score,
                'score_n' : row['score']/5
            }

        # Apply the row-wise custom metric calculations
        metrics = df.apply(row_metrics, axis=1)
        metrics_df = pd.DataFrame(metrics.tolist())  # Convert list of dicts to a DataFrame
        return pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
