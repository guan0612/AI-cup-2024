import os
import json
import argparse

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from ckiptagger import data_utils, construct_dictionary, WS, POS

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理
    
    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        cleaned_text = text.replace('\n', '')
        if text:
            pdf_text += cleaned_text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本

    


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve_jieba(qs, source, corpus_dict):
    
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名

def filter_NV(tokennized, pos_driver) :

    # 進行詞性標註
    pos_results = pos_driver(tokennized)


    # 定義感興趣的詞性標註
    POS = {
        'Na': '普通名詞',
        'Nb': '專有名詞',
        'Nc': '地方名詞',
        'Nd': '時間名詞',
        'Nf': '量詞名詞',
        'VA': '表情動詞',
        'VB': '變化動詞',
        'VC': '動作動詞',
        'VD': '雙賓動詞',
        'VE': '存在動詞',
        'VF': '趨向動詞',
    }

    # 輸出結果，只顯示動詞和名詞
    filtered_results = []
    for sentence, pos_sentence in zip(tokennized, pos_results):
        filtered_sentence = []
        for word, pos in zip(sentence, pos_sentence):
            if pos in POS:
                filtered_sentence.append(word)
        filtered_results.append(filtered_sentence)
    
    return filtered_results

def BM25_retrieve_CKIP(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    ws_driver = WS('E:/AI_cup/data')
    pos_driver = POS('E:/AI_cup/data')

    insurance_terms = {
    "保險契約": 1,
    "保險金額": 1,
    "投保人": 1,
    "受益人": 1,
    "保單": 1,
    "理賠": 1,
    "保費": 1,
    "保險公司": 1,
    "延期間": 1,
    "身故保險金": 1,
    "完全失能保險金": 1,
    "保單價值準備金": 1,
    "營業費用": 1,
    "展期定期保險": 1,
    "繳清生存保險": 1,
    "墊繳保險費": 1,
    "借款本息": 1,
    "解約金": 1,
    "保險單借款": 1,
    "契約效力": 1,
    "不分紅保險單": 1,
    "投保年齡": 1,
    "保險事故": 1,
    "保險理賠": 1,
    "保險費率": 1,
    "保險標的": 1,
    "保險核保": 1,
    "保險承保": 1,
    "保險到期": 1,
    "保險續保": 1,
    "保險中介": 1,
    "保險產品": 1,
    "保險合同": 1,
    "保險評估": 1,
    "保險需求": 1,
    "保險服務": 1,
    "保險市場": 1,
    "保險行業": 1,
    "保險監管": 1,
    "保險政策": 1,
    "保險法規": 1,
    "風險管理": 1,
    "賠償責任": 1,
    "賠償金額": 1,
    "理賠流程": 1,
    "理賠服務": 1,
    "理賠專員": 1,
    "理賠審核": 1,
    "理賠報告": 1,
    "保險精算": 1,
    "精算師": 1,
    "保險基金": 1,
    "保險資產": 1,
    "保險負債": 1,
    "保險投資": 1,
    "保險收益": 1,
    "保險損失": 1,
    "保險風險": 1,
    "保險保障": 1,
    "保險利益": 1,
    "保險金支付": 1,
    "保險金申請": 1,
    "法定繼承人": 1,
    "住址變更": 1,
    "批註": 1,
    "管轄法院": 1,
    # 可以根據需要添加更多詞彙
    }

    # 使用 CKIP 進行斷詞
    print(len(filtered_corpus))
    filtered_corpus = [doc for doc in filtered_corpus if doc]
    insurance_dict = construct_dictionary(insurance_terms)# 對特定詞彙建立字典，增強斷詞效果
    tokenized_corpus = [ws_driver([doc], recommend_dictionary=insurance_dict)[0] for doc in filtered_corpus]
    tokenized_filtered_corpus = filter_NV(tokenized_corpus, pos_driver)
    bm25 = BM25Okapi(tokenized_filtered_corpus)
    tokenized_query = ws_driver([qs])[0]
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)
    a = ans[0]
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]

if __name__ == "__main__":
    
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            continue
        elif q_dict['category'] == 'insurance':
            retrieved = BM25_retrieve_CKIP(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            continue
        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
