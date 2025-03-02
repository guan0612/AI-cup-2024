import pandas as pd
import json

ground_truths_example = pd.DataFrame(json.load(open('ground_truths_example.json'))['ground_truths']).set_index('qid')
pred_retrieve = pd.DataFrame(json.load(open('insurance_retrieve_ckiptagger.json'))['answers']).set_index('qid')
pred_retrieve.columns = ['pred_retrieve']

total_df = pd.concat([ground_truths_example, pred_retrieve], axis=1)

finance_df = total_df[total_df['category'] == 'finance']
print((finance_df['retrieve'] == finance_df['pred_retrieve']).mean())

insurance_df = total_df[total_df['category'] == 'insurance']
print((insurance_df['retrieve'] == insurance_df['pred_retrieve']).mean())

faq_df = total_df[total_df['category'] == 'faq']
print((faq_df['retrieve'] == faq_df['pred_retrieve']).mean())
