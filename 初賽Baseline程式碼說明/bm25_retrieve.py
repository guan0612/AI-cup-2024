import os
import json
import argparse

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
from ckiptagger import WS, POS 
from transformers import BertTokenizer
from monpa import utils
import monpa
from itertools import chain
import pytesseract
import fitz
from PIL import Image, ImageEnhance, ImageFilter
import os
import io
import torch


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
    pdf_mupdf = fitz.open(pdf_loc)
    
    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
        else:
            # 如果沒有提取到文本，使用 PyMuPDF 提取圖像並進行 OCR
            mupdf_page = pdf_mupdf.load_page(_)
            pix = mupdf_page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            #轉為灰階
            img = img.convert('L')
            #加強對比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2)
            text = pytesseract.image_to_string(img, lang='chi_tra')
            text = text.replace('\n', '')
            text = text.replace(' ', '')
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本

def filter_NV(tokennized) :
    pos_driver = CkipPosTagger(model="bert-base")

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



# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve_jieba(qs, source, corpus_dict):
    
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    #使用jieba進行斷詞
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名

def BM25_retrieve_CKIP(qs, source, corpus_dict):

    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # 使用 CKIP 進行斷詞
    ws_driver = CkipWordSegmenter(model="bert-base")
    filtered_corpus = [doc for doc in filtered_corpus if doc]    
    tokenized_corpus = [ws_driver([doc])[0] for doc in filtered_corpus]
    tokenized_filtered_corpus = filter_NV(tokenized_corpus)#只針對特定詞性做檢索
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

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = BM25_retrieve_jieba(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = BM25_retrieve_jieba(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve_CKIP(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
