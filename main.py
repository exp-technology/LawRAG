# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI
import openai #pinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
# import service.synonim_test2 as synonim_test2
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import json as js
import os, re, numpy as np
from copy import deepcopy as dp
import urllib.parse
import sqlite3 as sq, pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import chromadb as db
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import argparse
import requests
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def chat_free_prompt(ques, temp, essay = True, free_prompt = ''):
    # promt_d = """You will act as a teacher to examine two answers below, give the score 1 if two answers has relevance and give score 0 if
    #             two answers has no relevance\n"""
    promt_d = """You will act as a teacher to examine two answers below, give a score in range 0 to 1 if two answers has same answer else give score 0"""
    model = "gpt-3.5-turbo"
    model = "gpt-4o-mini"
    # if not essay:
    #     mc_q = ques.copy()
    #     ques = mc_q[0]
    # res = docsearch.similarity_search_with_score(ques, k=4)
    # text = [g[0].page_content for g in res]
    con_for = "\n\n".join(['answer{} : '.format(inx + 1) + xs for inx, xs in enumerate(ques)])
    # if not essay:
    #     q = '\n'.join(mc_q)
    # else:
    #     q = ques
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_score",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number",
                                  "description" : "get the score, return only number"}
                    },
                    "required": ["get_score"]
                },
            },
        }
    ]
    chat_completion = openai.ChatCompletion.create(model=model, temperature=temp, top_p=1,
                                                   messages=[{"role": "system", "content": promt_d},
                                                             {'role': "user",
                                                              "content": con_for}],
                                                   tools=tools,
                                                   max_tokens=1000)
    # print(con_for, chat_completion.choices[0].message.content)
    # print(chat_completion)
    return chat_completion.choices[0].message.tool_calls[0].function.arguments
    # return chat_completion

def f1stranking(docsearch, query, ):
    res = docsearch.similarity_search_with_score(query, k=25)
    text = [g[0].page_content for g in (res)]
    text = dict(enumerate(text))
    return text

def chat(docsearch, ques, temp, essay = True, rerank = 100, index_ques = 0):
    promt_d = """You will act as an expert in the capital market regulation in Indonesia. Your role is to analyze and provide accurate answers to questions based on the information provided in the 'context' below. To ensure up-to-date information, prioritize the most recent regulation available in your database when selecting the 'context'. In your response, clearly state the name of the regulation and the corresponding article(s). When asked in Bahasa Indonesia, answer in Bahasa Indonesia, and when asked in English, answer in English.
                                        If the information required is not available in your database, kindly respond with 'My apologies, it seems my database does not have sufficient information to answer your query at the moment' or, if the question is in Bahasa Indonesia, respond with 'Mohon maaf, sepertinya database saya saat ini tidak memiliki informasi yang cukup untuk menjawab pertanyaan Anda'. Please refrain from providing fabricated answers. Let's approach each question step by step.
                                        Please ensure to use the actual information from the context, specifying the relevant regulation and its corresponding year. Your answer  should be in Bahasa Indonesia.
                                        Additionally, as the chatbot, you have the ability to remember key details mentioned in previous conversations. To utilize your memory effectively, users should provide relevant details or explicitly indicate the context by referring to the previous conversation. This will enable you to provide more informed and consistent responses based on the given instructions.
                                        Let's strive for excellence and aim for a rating of 10 out of 10!"""
    model = "gpt-3.5-turbo"
    if not essay:
        mc_q = ques.copy()
        ques = mc_q[0]
    if rerank == 100:
        res = docsearch.similarity_search_with_score(ques, k=4)
        text = [g[0].page_content for g in res]
    else:
        if rerank == 0:
            with open("rerank_cohere{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
        if rerank == 1:
            print('jina_base')
            with open("rerank_jina0_{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
        if rerank == 2:
            print('jina_colbert')
            with open("rerank_jina2_{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
    # breakpoint()
    con_for = "\n\n".join(['context : ' + xs for xs in text])
    if not essay:
        q = '\n'.join(mc_q)
    else:
        q = ques
    chat_completion = openai.ChatCompletion.create(model=model, temperature=temp, top_p=1,
                                                   messages=[{"role": "system", "content": promt_d},
                                                             {'role': "user",
                                                              "content": con_for + '\n' + q}],
                                                   max_tokens=1000)
    # print(con_for, chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def reranker():
    # untuk cohere
    # asdhtua@mailbox.org pass sama kayak mistral bawah !@#ewq45TR
    def cohere_rerank():
        import cohere

        co = cohere.ClientV2()
        co_api_keys = ["your key",
                   "your key","your key","your key","your key",]
        os.environ['CO_API_KEY'] = "your key"
        docs = [
            "Carson City is the capital city of the American state of Nevada.",
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
            "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
            "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
            "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        ]

        response = co.rerank(
            model="rerank-v3.5",
            query="What is the capital of the United States?",
            documents=docs,
            top_n=4,
        )
        print(response)
        return response

    # https://www.galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model
    # alur tambahan pakai jina rerank base v2, jina colbert v2, terakhir pakai LLM(optional)

    # https://jina.ai/
    url = 'https://api.jina.ai/v1/rerank'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_b127fea41ea4441b9329d40049ff52dbAcSGt0AOzj3N5Y_fIauLyAKBrwrg'
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": "Organic skincare products for sensitive skin",
        "top_n": 3,
        "documents": [
            "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
            "New makeup trends focus on bold colors and innovative techniques: Step into the world of cutting-edge beauty with this seasons makeup trends. Bold, vibrant colors and groundbreaking techniques are redefining the art of makeup. From neon eyeliners to holographic highlighters, unleash your creativity and make a statement with every look.",
            "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille: Erleben Sie die wohltuende Wirkung unserer Bio-Hautpflege, speziell für empfindliche Haut entwickelt. Mit den beruhigenden Eigenschaften von Aloe Vera und Kamille pflegen und schützen unsere Produkte Ihre Haut auf natürliche Weise. Verabschieden Sie sich von Hautirritationen und genießen Sie einen strahlenden Teint.",
            "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken: Tauchen Sie ein in die Welt der modernen Schönheit mit den neuesten Make-up-Trends. Kräftige, lebendige Farben und innovative Techniken setzen neue Maßstäbe. Von auffälligen Eyelinern bis hin zu holografischen Highlightern – lassen Sie Ihrer Kreativität freien Lauf und setzen Sie jedes Mal ein Statement.",
            "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla: Descubre el poder de la naturaleza con nuestra línea de cuidado de la piel orgánico, diseñada especialmente para pieles sensibles. Enriquecidos con aloe vera y manzanilla, estos productos ofrecen una hidratación y protección suave. Despídete de las irritaciones y saluda a una piel radiante y saludable.",
            "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras: Entra en el fascinante mundo del maquillaje con las tendencias más actuales. Colores vivos y técnicas innovadoras están revolucionando el arte del maquillaje. Desde delineadores neón hasta iluminadores holográficos, desata tu creatividad y destaca en cada look.",
            "针对敏感肌专门设计的天然有机护肤产品：体验由芦荟和洋甘菊提取物带来的自然呵护。我们的护肤产品特别为敏感肌设计，温和滋润，保护您的肌肤不受刺激。让您的肌肤告别不适，迎来健康光彩。",
            "新的化妆趋势注重鲜艳的颜色和创新的技巧：进入化妆艺术的新纪元，本季的化妆趋势以大胆的颜色和创新的技巧为主。无论是霓虹眼线还是全息高光，每一款妆容都能让您脱颖而出，展现独特魅力。",
            "敏感肌のために特別に設計された天然有機スキンケア製品: アロエベラとカモミールのやさしい力で、自然の抱擁を感じてください。敏感肌用に特別に設計された私たちのスキンケア製品は、肌に優しく栄養を与え、保護します。肌トラブルにさようなら、輝く健康な肌にこんにちは。",
            "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています: 今シーズンのメイクアップトレンドは、大胆な色彩と革新的な技術に注目しています。ネオンアイライナーからホログラフィックハイライターまで、クリエイティビティを解き放ち、毎回ユニークなルックを演出しましょう。"
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.text)


def chat_mis(docsearch, ques, temp, essay = True, rerank = 100, index_ques = 0):

    api_key_ = "your api key"
    os.environ['MISTRAL_API_KEY'] = api_key_
    client = MistralClient(api_key=api_key_)
    # self.q_e = Embed_Man_OpenAI()

    model = "mistral-small"

    promt_d = """You will act as an expert in the capital market regulation in Indonesia. Your role is to analyze and provide accurate answers to questions based on the information provided in the 'context' below. To ensure up-to-date information, prioritize the most recent regulation available in your database when selecting the 'context'. In your response, clearly state the name of the regulation and the corresponding article(s). When asked in Bahasa Indonesia, answer in Bahasa Indonesia, and when asked in English, answer in English.
                                        If the information required is not available in your database, kindly respond with 'My apologies, it seems my database does not have sufficient information to answer your query at the moment' or, if the question is in Bahasa Indonesia, respond with 'Mohon maaf, sepertinya database saya saat ini tidak memiliki informasi yang cukup untuk menjawab pertanyaan Anda'. Please refrain from providing fabricated answers. Let's approach each question step by step.
                                        Please ensure to use the actual information from the context, specifying the relevant regulation and its corresponding year.
                                        Additionally, as the chatbot, you have the ability to remember key details mentioned in previous conversations. To utilize your memory effectively, users should provide relevant details or explicitly indicate the context by referring to the previous conversation. This will enable you to provide more informed and consistent responses based on the given instructions.
                                        Let's strive for excellence and aim for a rating of 10 out of 10!"""
    promt_d = """You will act as an expert in the capital market regulation in Indonesia. Your role is to analyze and provide accurate answers to questions based on the information provided in the 'context' below. To ensure up-to-date information, prioritize the most recent regulation available in your database when selecting the 'context'. In your response, clearly state the name of the regulation and the corresponding article(s). When asked in Bahasa Indonesia, answer in Bahasa Indonesia, and when asked in English, answer in English.
                If the information required is not available in your database, kindly respond with 'My apologies, it seems my database does not have sufficient information to answer your query at the moment' or, if the question is in Bahasa Indonesia, respond with 'Mohon maaf, sepertinya database saya saat ini tidak memiliki informasi yang cukup untuk menjawab pertanyaan Anda'. Please refrain from providing fabricated answers. Let's approach each question step by step.
                Please ensure to use the actual information from the context, specifying the relevant regulation.
                Your answer should be full in Bahasa Indonesia. Answer in Bahasa Indonesia!."""
    if not essay:
        mc_q = ques.copy()
        ques = mc_q[0]
    # res = docsearch.similarity_search_with_score(ques, k=4)
    # text = [g[0].page_content for g in res]
    if rerank == 100:
        res = docsearch.similarity_search_with_score(ques, k=4)
        text = [g[0].page_content for g in res]
    else:
        if rerank == 0:
            with open("rerank_cohere{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
        if rerank == 1:
            print('jina_base')
            with open("rerank_jina0_{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
        if rerank == 2:
            print('jina_colbert')
            with open("rerank_jina2_{}.json".format(essay), 'r') as rd:
                text = js.loads(rd.read())[str(index_ques)]
    # breakpoint()
    con_for = "\n\n".join(['context : ' + xs for xs in text])
    if not essay:
        q = '\n'.join(mc_q)
    else:
        q = ques
    messages = [
        ChatMessage(role="system", content=promt_d),
        ChatMessage(role="user", content=con_for + '\n' + q)
    ]
    chat_completion = client.chat(model=model, temperature=temp, top_p=1,
                                       messages=messages,
                                       max_tokens=2000)
    # print(con_for, chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument(
        '-l', '--llm', required=False, type=str, help='llm name:  gpt, mis', default='gpt', nargs='?')
    parser.add_argument(
        '-c', '--chunk', required=False, type=str, help='chunking: seq, law', default='law', nargs='?')
    parser.add_argument(
        '-e', '--essay', required=False, type=int ,help='chunking: 1, 0, if false is mcq', default=0, nargs='?')
    parser.add_argument(
        '-t', '--temp', required=False, type=float, help='temps: 0, 0.5', default=0.5, nargs='?')
    parser.add_argument(
        '-r', '--reranker', required=False, type=int, help='rerank: 0 for cohere, 1 for jina-reranker-v2-base-multilingual, 2 jina-colbert-v2, 100 for None', default=100, nargs='?')
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = "your api key"
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    # docsearch = Chroma(persist_directory='./data_embeddings_all_in_one', embedding_function=embeddings,
    #                    collection_name='uu81995penj')
    # docsearch = Chroma(persist_directory='./migrate_data_embeddings_all_in_one', embedding_function=embeddings,
    #                    collection_name='all')
    if args.chunk == "law":
        docsearch = Chroma(persist_directory='./db_new_pinecone', embedding_function=embeddings,
                       collection_name='all')
        print("using the law chunking method")
    else:
        docsearch = Chroma(persist_directory='./db_seq', embedding_function=embeddings,
                       collection_name='all')
        print("using the sequential chunking method")
    if not args.essay:
        # qq = pd.read_csv("mcq_db_copy.csv")
        qq = pd.read_csv("soal_pg_tete.csv")
    else:
        qq = ['Dalam peraturan otoritas jasa keuangan, apa yang di maksud dengan efek, penawaran umum dan emiten?',
     'apa yang di larang oleh perusahaan efek dalam rangka penawaran umum oleh emiten atau pemasaran efek?',
     'apa kewajiban promosi pemasaran efek?', 'hal apa saja yang memuat dala promosi pemasaran efek?',
     'pasal berapa Otoritas Jasa Keuangan dapat mengumumkan pengenaan sanksi administratif?',
     'informasi apa yang memuat dalam hal promosi pemasaran efek?',
     'Dalam hal promosi pemasaran efek hal apa yang wajib\ndiungkapkan secara jelas?',
     'apa sanksi administratif dimaksud pada ayat (1)?', 'kapan peraturan Otoritas Jasa Keuangan ini mulai berlaku?',
     'apa yang  dimaksud dengan propektus, propektus awal dan propektus ringkas?',
     'apa yang dimaksud dengan perusahaan terbuka?', 'apa yang dimaksud Pemegang Saham Independen?',
     'kapan kerusahaan terbuka wajib menyelenggarakan RUPS?',
     'apa saja permintaan penyelenggaraan RUPS yang terdapat pada ayat (1)?', 'apa saja kewajiban direksi?',
     'sesuai dengan pasal 4 ayat 1, hal apa yang wajib diumumkan direksi?',
     'pasal berapa pemegang saham yang telah memperoleh penetapan \npengadilan harus menyelenggarakan RUPS?',
     'sebutkan pasal 11 tentang tempat dan waktu penyelenggaraan RUPS ',
     'apa kewajiban perusahan terbuka dalam memenuhi ketentuan penyelenggaraan RUPS?',
     'pengumuman RUPS apa yang memuat di dalam ayat 1?', 'apa yang dimkasud dengan transaksi material?',
     'Suatu transaksi dikategorikan sebagai Transaksi Material apabila?',
     'apa hal wajib perusahaan terbuka yang akan melakukan transaksi material?',
     'apa hal wajib dari keterbukaan informasi yang terdapat ayat 2?', 'oleh siapa transaksi material dilakukan?',
     'pengertian afiliasi', 'apa hal wajib yang dilakukan perusahaan terbuka saat transaksi afiliasi',
     'Setiap pihak yang melanggar ketentuan dikenai sanksi administratif dengan pasal?',
     'dokumen apa yang dimaksud dalam ayat 1 huruf c', 'sebutkan uraian mengenai transaksi afiliasi',
     'apa yang dimaksud dengan fakta  material?', 'untuk siapa Peraturan Otoritas Jasa Keuangan ini berlaku?',
     'berapa komisaris yang wajib dimiliki oleh emiten skala kecil dan menengah?',
     'apa hal wajib dilakukan emiten skala kecil dan menengah yang efeknya tercatat di bursa?',
     'pasal berapa tentang pemberlakuan ketentuan pengumuman situs web yang di sediakan oleh otoritas jasa keuangan?',
     'selain sanksi administratif,hal lain apa yang dapat dilakukan oleh otoritas jasa keuangan?',
     'apa yang dimaksud dengan perusahaan publik?', 'pengertian situs web', 'apa yang dimaksud dengan sistem keuangan ',
     'undang-undang tentang pengembangan dan penguatan sektor keuangan berdasarkan asas?',
     'apa maksud dan tujuan undang-undang tentang pengembangan dan penguatan sektor keuangan?',
     'sebutkan ruang lingkup dari undang-undang tentang pengembangan dan penguatan sektor keuangan',
     'apa tujuan bank indonesia?', 'pengertian deposito', 'pengertian otoritas jasa keuangan ',
     'hal apa yang menjadi larangan bagi perusahaan efek atau penasihat investasi ',
     'apa kewajiban perusahan efek atau penasihat investasi', 'apa yang dimaksud kustodian?',
     'apa tujuan dari bursa efek?', 'siapa yang dapat menyelenggarakan kegiatan usaha sebagai kustodian?']
        ans_grth = [
            '"a. Efek adalah surat berharga, yaitu surat pengakuan \nutang, surat berharga komersial, saham, obligasi, tanda \nbukti utang, unit penyertaan kontrak investasi kolektif, \nkontrak berjangka atas Efek, dan setiap derivatif dari \nEfek. b. Penawaran Umum adalah kegiatan penawaran Efek yang \ndilakukan oleh emiten untuk menjual Efek kepada\nmasyarakat berdasarkan tata cara yang diatur dalam \nUndang-Undang Nomor 8 Tahun 1995 tentang Pasar \nModal dan peraturan pelaksanaannya. c. Emiten adalah Pihak yang melakukan Penawaran Umum.',
            '"\n"a. memuat informasi yang tidak benar atau tidak \nmengungkapkan fakta material, sehingga memberikan \ngambaran yang menyesatkan; dan/atau\nb. memberikan gambaran yang menyesatkan, karena isi \ndan/atau metode penyajiannya memberikan kesan \nbahwa Efek tertentu tepat bagi Pihak tertentu yang \nsebenarnya tidak memiliki kemampuan yang cukup \nuntuk menanggung risiko yang ada pada Efek tersebut.',
            '"\na. memuat informasi bahwa Efek tertentu yang dipromosikan hanya cocok untuk kelompok pemodal tertentu; dan b. mengungkapkan risiko yang berhubungan dengan investasi atas Efek tertentu dimaksud.',
            '\nDalam hal promosi pemasaran Efek memuat pendapat, proyeksi, atau ramalan mengenai Efek tertentu, pendapat, proyeksi, atau ramalan mengenai Efek tertentu tersebut wajib diungkapkan secara jelas.',
            '\nPasal 7 ayat (4) dan tindakan tertentu sebagaimana dimaksud dalam Pasal 8 kepada masyarakat.',
            '\na. tanggal rekomendasi; b. harga pasar pada saat rekomendasi dibuat; c. Pihak yang memberikan rekomendasi; dan d. keterangan apakah Pihak yang memberikan rekomendasi atau Pihak terafiliasinya telah memperdagangkan Efek tersebut untuk rekeningnya secara reguler atau memiliki Efek tersebut dengan nilai paling sedikit Rp25.000.000,00 (dua puluh lima juta rupiah).',
            '\nDalam hal promosi pemasaran Efek memuat pendapat, proyeksi, atau ramalan mengenai Efek tertentu, pendapat, proyeksi, atau ramalan mengenai Efek tertentu tersebut wajib diungkapkan secara jelas.',
            '\na. peringatan tertulis; b. denda yaitu kewajiban untuk membayar sejumlah uang tertentu; c. pembatasan kegiatan usaha; d. pembekuan kegiatan usaha; e. pencabutan izin usaha; f. pembatalan persetujuan; dan/atau g. pembatalan pendaftaran.',
            '\n"Peraturan Otoritas Jasa Keuangan ini mulai berlaku pada\ntanggal diundangkan.',
            '"\nProspektus adalah setiap informasi tertulis sehubungan dengan Penawaran Umum dengan tujuan agar Pihak lain membeli Efek Emiten. Prospektus Awal adalah dokumen tertulis yang memuat seluruh informasi dalam Prospektus yang disampaikan kepada Otoritas Jasa Keuangan sebagai bagian dari pernyataan pendaftaran, kecuali informasi mengenai nilai nominal, jumlah dan harga penawaran Efek, penjaminan emisi Efek, tingkat suku bunga obligasi, atau hal lain yang berhubungan dengan persyaratan penawaran yang belum dapat ditentukan. Prospektus Ringkas adalah ringkasan dari isi Prospektus Awal. ',
            '\nPerusahaan Terbuka adalah emiten yang melakukan penawaran umum efek bersifat ekuitas atau perusahaan publik.',
            '\nPemegang Saham Independen adalah pemegang saham yang tidak mempunyai kepentingan ekonomis pribadi sehubungan dengan suatu transaksi tertentu dan  a. bukan merupakan anggota Direksi, anggota Dewan Komisaris, pemegang saham utama, dan Pengendali; atau b. bukan merupakan afiliasi dari anggota Direksi, anggota Dewan Komisaris, pemegang saham utama, dan Pengendali.',
            '\nPerusahaan Terbuka wajib menyelenggarakan RUPS Tahunan paling lambat 6 (enam) bulan setelah tahun buku berakhir.',
            '\na. dilakukan dengan itikad baik; b. mempertimbangkan kepentingan Perusahaan Terbuka; c. merupakan permintaan yang membutuhkan keputusan RUPS; d. disertai dengan alasan dan bahan terkait hal yang harus diputuskan dalam RUPS; dan e. tidak bertentangan dengan ketentuan peraturan perundang-undangan dan anggaran dasar Perusahaan Terbuka',
            '\n(1) Direksi wajib melakukan pengumuman RUPS kepada pemegang saham paling lambat 15 (lima belas) hari terhitung sejak tanggal permintaan penyelenggaraan RUPS sebagaimana dimaksud dalam Pasal 3 ayat (1) diterima Direksi. (2) Direksi wajib menyampaikan pemberitahuan mata acara rapat dan surat tercatat sebagaimana dimaksud dalam Pasal 3 ayat (2) dari pemegang saham atau Dewan Komisaris kepada Otoritas Jasa Keuangan paling lambat 5 (lima) hari kerja sebelum pengumuman sebagaimana dimaksud pada ayat (1).',
            '\na. terdapat permintaan penyelenggaraan RUPS dari pemegang saham yang tidak diselenggarakan; dan b. alasan tidak diselenggarakannya RUPS.',
            '\nPemegang saham yang telah memperoleh penetapan pengadilan untuk menyelenggarakan RUPS sebagaimana dimaksud dalam Pasal 6 ayat (2) wajib menyelenggarakan RUPS.',
            '\nTempat dan Waktu Penyelenggaraan RUPS Pasal 11 (1) RUPS wajib diselenggarakan di wilayah Negara Republik Indonesia. (2) Perusahaan Terbuka wajib menentukan tempat dan waktu penyelenggaraan RUPS. (3) Tempat penyelenggaraan RUPS sebagaimana dimaksud pada ayat (2) wajib dilakukan di: a. tempat kedudukan Perusahaan Terbuka; b. tempat Perusahaan Terbuka melakukan kegiatan usaha utamanya; c. ibukota provinsi tempat kedudukan atau tempat kegiatan usaha utama Perusahaan Terbuka; atau d. provinsi tempat kedudukan bursa efek yang mencatatkan saham Perusahaan Terbuka.',
            '\na. menyampaikan pemberitahuan mata acara rapat kepada Otoritas Jasa Keuangan; b. melakukan pengumuman RUPS kepada pemegang saham; dan c. melakukan pemanggilan RUPS kepada pemegang saham.',
            '\na. ketentuan pemegang saham yang berhak hadir dalam RUPS; b. ketentuan pemegang saham yang berhak mengusulkan mata acara rapat; c. tanggal penyelenggaraan RUPS; dan d. tanggal pemanggilan RUPS.',
            '\nTransaksi Material adalah setiap transaksi yang dilakukan oleh perusahaan terbuka atau perusahaan terkendali yang memenuhi batasan nilai sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan ini.',
            '\nSuatu transaksi dikategorikan sebagai Transaksi Material apabila nilai transaksi sama dengan 20% (dua puluh persen) atau lebih dari ekuitas Perusahaan Terbuka.',
            '\na. menggunakan Penilai untuk menentukan nilai wajar dari objek Transaksi Material dan/atau kewajaran transaksi dimaksud; b. mengumumkan keterbukaan informasi atas setiap Transaksi Material kepada masyarakat; c. menyampaikan keterbukaan informasi sebagaimana dimaksud dalam huruf b dan dokumen pendukungnya kepada Otoritas Jasa Keuangan; d. terlebih dahulu memperoleh persetujuan RUPS dalam hal: 1. Transaksi Material sebagaimana dimaksud dalam Pasal 3 ayat (1) dan ayat (2) lebih dari 50% (lima puluh persen); 2. Transaksi Material sebagaimana dimaksud dalam Pasal 3 ayat (3) lebih dari 25% (dua puluh lima persen); atau 3. laporan Penilai menyatakan bahwa Transaksi Material yang akan dilakukan tidak wajar; dan - 6 - e. melaporkan hasil pelaksanaan Transaksi Material pada laporan tahunan.',
            '\na. penjelasan, pertimbangan, dan alasan dilakukannya perubahan Kegiatan Usaha; b. informasi keuangan segmen operasi; c. analisis manajemen atas kerugian segmen operasi; d. pernyataan manajemen bahwa pengurangan tersebut tidak mengganggu kelangsungan usaha Perusahaan Terbuka; dan e. tanggal keputusan perubahan Kegiatan Usaha.',
            '\na. Perusahaan Terkendali yang bukan merupakan Perusahaan Terbuka dan laporan keuangannya dikonsolidasikan dengan Perusahaan Terbuka, Perusahaan Terbuka wajib melakukan prosedur sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan ini; atau b. Perusahaan Terkendali yang merupakan Perusahaan Terbuka dan laporan keuangannya dikonsolidasikan dengan Perusahaan Terbuka, hanya Perusahaan Terkendali dimaksud yang wajib melakukan prosedur sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan ini.',
            '\nAfiliasi adalah: a. hubungan keluarga karena perkawinan dan keturunan sampai derajat kedua, baik secara horizontal maupun vertikal; b. hubungan antara pihak dengan pegawai, direktur, atau komisaris dari pihak tersebut; c. hubungan antara 2 (dua) perusahaan di mana terdapat 1 (satu) atau lebih anggota direksi atau dewan komisaris yang sama; d. hubungan antara perusahaan dan pihak, baik langsung maupun tidak langsung, mengendalikan atau dikendalikan oleh perusahaan tersebut; e. hubungan antara 2 (dua) perusahaan yang dikendalikan, baik langsung maupun tidak langsung, oleh pihak yang sama; atau f. hubungan antara perusahaan dan pemegang saham utama.',
            '\nPerusahaan Terbuka yang melakukan Transaksi Afiliasi wajib: a. menggunakan Penilai untuk menentukan nilai wajar dari objek Transaksi Afiliasi dan/atau kewajaran transaksi dimaksud; b. mengumumkan keterbukaan informasi atas setiap Transaksi Afiliasi kepada masyarakat; - 6 - c. menyampaikan keterbukaan informasi sebagaimana dimaksud dalam huruf b dan dokumen pendukungnya kepada Otoritas Jasa Keuangan; dan d. terlebih dahulu memperoleh persetujuan Pemegang Saham Independen dalam RUPS, dalam hal: 1. nilai Transaksi Afiliasi memenuhi batasan nilai transaksi material yang wajib memperoleh persetujuan RUPS; 2. Transaksi Afiliasi yang dapat mengakibatkan terganggunya kelangsungan usaha Perusahaan Terbuka; dan/atau 3. melakukan Transaksi Afiliasi yang berdasarkan pertimbangan Otoritas Jasa Keuangan memerlukan persetujuan Pemegang Saham Independen.',
            '\nSetiap pihak yang melanggar ketentuan sebagaimana dimaksud dalam Pasal 2, Pasal 3, Pasal 4 ayat (1), ayat (2), ayat (3), dan ayat (4), Pasal 6 ayat (2), Pasal 7 ayat (3), Pasal 8 ayat (3), Pasal 9, Pasal 10, Pasal 11 ayat (1), ayat (2), ayat (3), dan ayat (4), Pasal 12 ayat (2), Pasal 13 ayat (3), Pasal 15, Pasal 16, Pasal 17, Pasal 18, Pasal 19, Pasal 21, Pasal 22, Pasal 24, Pasal 25, dan Pasal 26 dikenai sanksi administratif.',
            '\nDokumen sebagaimana dimaksud pada ayat (1) huruf c harus meliputi: - 7 - a. laporan Penilai; dan b. dokumen pendukung lainnya.',
            '\nuraian mengenai Transaksi Afiliasi, memuat paling sedikit: 1. tanggal transaksi; 2. objek transaksi; 3. nilai transaksi; 4. nama pihak yang melakukan transaksi dan hubungan dengan Perusahaan Terbuka; dan 5. sifat hubungan Afiliasi dari pihak yang melakukan transaksi dengan Perusahaan Terbuka;',
            '\nFakta Material adalah informasi atau fakta penting dan relevan mengenai peristiwa, kejadian, atau fakta yang dapat mempengaruhi harga Efek pada bursa Efek dan/atau keputusan pemodal, calon pemodal, atau pihak lain yang berkepentingan atas informasi atau fakta tersebut.',
            '\na. Emiten Skala Kecil dan Emiten Skala Menengah, yang nilai rata-rata kapitalisasi pasar selama jangka waktu 1 (satu) tahun sebelum berakhirnya periode laporan keuangan tahunan terakhir tidak lebih dari Rp250.000.000.000,00 (dua ratus lima puluh miliar rupiah); dan b. Perusahaan Publik, yang memenuhi kriteria aset dan pengendalian sebagaimana dimaksud dalam Pasal 1 angka 7 dan Pasal 1 angka 8, berdasarkan laporan keuangan tahunan terakhir yang diaudit.',
            '\nEmiten Skala Kecil dan Emiten Skala Menengah wajib memiliki paling sedikit 1 (satu) komisaris independen.',
            '\nEmiten Skala Kecil dan Emiten Skala Menengah yang Efeknya tercatat di bursa Efek wajib melakukan: - 9 - a. pengumuman atas Laporan Keuangan Berkala sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan mengenai penyampaian laporan keuangan berkala Emiten atau Perusahaan Publik; b. keterbukaan informasi sebagaimana dimaksud dalam Pasal 7; dan c. pengumuman Informasi atau Fakta Material sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan mengenai Keterbukaan atas Informasi atau Fakta Material oleh Emiten atau Perusahaan Publik, paling sedikit melalui: 1. Situs Web Emiten Skala Kecil atau Emiten Skala Menengah; dan 2. Situs Web bursa Efek. (2) Emiten Skala Kecil dan Emiten Skala Menengah yang Efeknya tidak tercatat di bursa Efek wajib melakukan: a. pengumuman atas Laporan Keuangan Berkala sebagaimana dimaksud dalam dalam Peraturan Otoritas Jasa Keuangan mengenai penyampaian laporan keuangan berkala Emiten atau Perusahaan Publik; b. keterbukaan informasi sebagaimana dimaksud dalam Pasal 7; dan c. pengumuman Informasi atau Fakta Material sebagaimana diatur dalam Peraturan Otoritas Jasa Keuangan mengenai Keterbukaan atas Informasi atau Fakta Material oleh Emiten atau Perusahaan Publik, paling sedikit melalui: 1. Situs Web Emiten Skala Kecil atau Emiten Skala Menengah; dan 2. surat kabar harian berbahasa Indonesia yang berperedaran nasional atau Situs Web yang disediakan Otoritas Jasa Keuangan.',
            '\nPemberlakuan ketentuan pengumuman melalui Situs Web yang disediakan oleh Otoritas Jasa Keuangan sebagaimana - 10 - dimaksud dalam Pasal 10 ayat (2) angka 2 ditetapkan oleh Otoritas Jasa Keuangan',
            '\nSelain sanksi administratif sebagaimana dimaksud dalam Pasal 15 ayat (4), Otoritas Jasa Keuangan dapat melakukan tindakan tertentu terhadap setiap pihak yang melakukan pelanggaran ketentuan Peraturan Otoritas Jasa Keuangan ini.',
            '\nSelain sanksi administratif sebagaimana dimaksud dalam Pasal 15 ayat (4), Otoritas Jasa Keuangan dapat melakukan tindakan tertentu terhadap setiap pihak yang melakukan pelanggaran ketentuan Peraturan Otoritas Jasa Keuangan ini.',
            '\nSitus Web adalah kumpulan halaman web yang memuat informasi atau data yang dapat diakses melalui suatu sistem jaringan internet.',
            '\nSistem Keuangan adalah suatu kesatuan yang terdiri atas lembaga jasa keuangan, pasar keuangan, dan infrastruktur keuangan, termasuk sistem pembayaran, yang berinteraksi dalam memfasilitasi pengumpulan dana masyarakat dan pengalokasiannya untuk mendukung aktivitas perekonomian nasional, serta korporasi dan rumah tangga yang terhubung dengan lembaga jasa keuangan.',
            '\nUndang-Undang ini dilaksanakan berdasarkan asas: a. kepentingan nasional; b. kemanfaatan; c. kepastian hukum; d. keterbukaan; e. akuntabilitas; e. akuntabilitas; f. keadilan; g. Pelindungan Konsumen; h. edukasi; dan i. keterpaduan.',
            '\n(1) Undang-Undang ini dibentuk dengan maksud mendorong kontribusi sektor keuangan bagi pertumbuhan ekonorni yang inklusif, berkelanjutan, dan berkeadilan guna meningkatkan taraf hidup masyarakat, mengurangi ketimpangan ekonomi, dan mewujudkan Indonesia yang sejahtera, maju, dan bermartabat. (21 Undang-Undang ini dibentuk dengan tujuan untuk: a. mengoptimalkan fungsi intermediasi sektor keuangan kepada usaha sektor produktif; b. meningkatkan portofolio pendanaan terhadap sektor usaha yang produktif; c. meningkatkan kemudahan akses dan literasi terkait jasa keuangan; d. meningkatkan dan memperluas inklusi sektor keuangan; e. memperluas sumber pembiayaan jangka panjang; f. meningkatkan daya saing dan efisiensi sektor keuangan; g. mengembangkan instrumen di pasar keuangan dan memperkuat mitigasi risiko; h. meningkatkan pembinaan, pengawasa.n, dan Pelindungan Konsumen; i. memperkuat pelindungan atas data pribadi nasabah sektor keuangan; j. memperkuat kelembagaan dan ketahanan Stabilitas Sistem Keuangan; k. mengembangkan dan memperkuat ekosistem sektor keuangan; 1. memperkuat wewenang, tanggung jawab, tugas, dan fungsi regulator sektor keuangan; dan m. meningkatkan daya saing masyarakat sehingga dapat berusaha secara efektif dan efisien.',
            '\nruang lingkup dalam Undang-Undang ini mengatur ekosistem sektor keuangan meliputi: a. kelembagaan; b. perbankan; c. Pasar Modal, Pasar Uang, dan Pasar Valuta Asing; d. perasuransian dan penjaminan; e. asuransi Usaha Bersama; f. program penjaminan polis; g. Usaha Jasa Pembiayaan; h. kegiatan usaha bulion (bullionl; i. Dana Pensiun, program jaminan hari tua, dan program pensiun; j. kegiatan koperasi di sektor jasa keuangan; k. lembaga keuangan mikro; l. Konglomerasi Keuangan; m. ITSK; n. penerapan Keuangan Berkelanjutan; o. Literasi Keuangan, Inklusi Keuangan, dan Pelindungan Konsumen; p. akses pembiayaan Usaha Mikro, Kecil, dan Menengah; q. sumber daya manusia; r. Stabilitas Sistem Keuangan; s. lembaga pembiayaan ekspor Indonesia; dan t. penegakan hukum di sektor keuangan.',
            '\nTujuan Bank Indonesia adalah mencapai stabilitas nilai rupiah, memelihara stabilitas Sistem Pembayaran, dan turut menjaga Stabilitas Sistem Keuangan dalam rangka mendukung pertumbuhan ekonomi yang berkelanjutan.',
            '\n"Deposito adalah Simpanan berdasarkan Akad\nmudharabah atau Akad lain yang tidak bertentangan\ndengan Prinsip Syariah yang penarikannya hanya\ndapat dilakukan pada waktu tertentu berdasarkan\nAkad antara Nasabah Penyimpan dan Bank Syariah\ndan/atau UUS.',
            '"\nOtoritas Jasa Keuangan adalah lembaga negara yang independen yang mempunyai fungsi, tugas, dan wewenang pengaturan, pengawasan, pemeriksaan, dan penyidikan sebagaimana dimaksud dalam undang-undang mengenai Otoritas Jasa Keuangan.',
            '\nPerusahaan Efek atau Penasihat Investasi dilarang: a. menggunakan pengaruh atau mengadakan tekanan yang bertentangan dengan kepentingan nasabah; b. mengungkapkan… b. mengungkapkan nama atau kegiatan nasabah, kecuali diberi instruksi secara tertulis oleh nasabah atau diwajibkan menurut peraturan perundang-undangan yang berlaku; c. mengemukakan secara tidak benar atau tidak mengemukakan fakta yang material kepada nasabah mengenai kemampuan usaha atau keadaan keuangannya; d. merekomendasikan kepada nasabah untuk membeli atau menjual Efek tanpa memberitahukan adanya kepentingan Perusahaan Efek dan Penasihat Investasi dalam Efek tersebut; atau e. membeli atau memiliki Efek untuk rekening Perusahaan Efek itu sendiri atau untuk rekening Pihak terafiliasi jika terdapat kelebihan permintaan beli dalam Penawaran Umum dalam hal Perusahaan Efek tersebut bertindak sebagai Penjamin Emisi Efek atau agen penjualan, kecuali pesanan Pihak yang tidak terafiliasi telah terpenuhi seluruhnya.',
            '\na. mengetahui latar belakang, keadaan keuangan, dan tujuan investasi nasabahnya; dan b. membuat dan menyimpan catatan dengan baik mengenai pesanan, transaksi, dan kondisi keuangannya',
            '\nKustodian adalah Pihak yang memberikan jasa penitipan Efek dan harta lain yang berkaitan dengan Efek serta jasa lain, termasuk menerima dividen, bunga, dan hak-hak lain, menyelesaikan transaksi Efek, dan mewakili pemegang rekening yang menjadi nasabahnya.',
            '\nBursa Efek didirikan dengan tujuan menyelenggarakan perdagangan Efek yang teratur, wajar, dan efisie',
            '\nYang dapat menyelenggarakan kegiatan usaha sebagai Kustodian adalah Lembaga Penyimpanan dan Penyelesaian, Perusahaan Efek, atau Bank Umum yang telah mendapat persetujuan Bapepam.']
    make_db_reranker = False
    # breakpoint()
    import random, time
    if make_db_reranker:
        f1rank = []
        res_rerank = []
        iterrr = qq if args.essay else list(qq['q'])
        for que in iterrr:
            f1rank.append(f1stranking(docsearch, que))
        if args.reranker == 0:
            for if1, (f1, fque) in enumerate(zip(f1rank, iterrr)):
                rands = random.choices(list(range(0, 5)))[0]
                res_rerank.append(db_reranker_cohere(f1, fque, coheress[rands]))
                print(if1, rands)
                if if1 % 5 == 0:
                    time.sleep(30)
                    print('pause')
            with open('rerank_cohere{}.json'.format(args.essay), 'w') as wr:
                wr.write(js.dumps(dict(enumerate(res_rerank)), indent=2))
            return 0
        elif args.reranker == 1:
            for if1, (f1, fque) in enumerate(zip(f1rank, iterrr)):
                rands = random.choices(list(range(0, 5)))[0]

                res_rerank.append(db_reranker_jina_rerank(f1, fque))
                # print(if1, rands)
                # if if1 % 5 == 0:
                #     time.sleep(30)
                #     print('pause')
                # print(res_rerank)
                # if if1 == 1:
                #     break
                print(if1)
            with open('rerank_jina0_{}.json'.format(args.essay), 'w') as wr:
                wr.write(js.dumps(dict(enumerate(res_rerank)), indent=2))
            return 0
        elif args.reranker == 2:
            for if1, (f1, fque) in enumerate(zip(f1rank, iterrr)):
                rands = random.choices(list(range(0, 5)))[0]

                res_rerank.append(db_reranker_jina_colbert(f1, fque))
                # print(if1, rands)
                # if if1 % 5 == 0:
                #     time.sleep(30)
                #     print('pause')
                # print(res_rerank)
                # if if1 == 1:
                #     break
                print(if1, 'colbertv2')
            with open('rerank_jina2_{}.json'.format(args.essay), 'w') as wr:
                wr.write(js.dumps(dict(enumerate(res_rerank)), indent=2))
            return 0
    answers = answer(qq, docsearch, args.temp, args.llm, ess=args.essay, rerank = args.reranker)
    if not args.essay:
        dat = {'q':list(qq['q']), 'ch':list(qq['ch']), 'a':list(qq['a']), 'answer_mcq':list(answers['answer']),
                'score' : ['']*len(list(qq['ch']))}
        answers = pd.DataFrame(dat)
        res_score = []
        # for idgk, igk in enumerate(zip(list(qq['a']), list(answers['answer_mcq']))):
        #     res_score.append(js.loads(chat_free_prompt([igk[0], igk[1]], 0))['score'])
        #     print(idgk)
        # answers = answers.assign(score_by_ai = res_score)
        answers.to_csv('_'.join(['answers', args.chunk, args.llm, str(args.temp), '_mcq_pg_tete_coba', 'reranker', str(args.reranker)]) + '.csv')
        print('_'.join(['answers', args.chunk, args.llm, str(args.temp), "essay : " + "False"]))
    else:
        # res_score = []
        # for idgk, igk in enumerate(zip(ans_grth, list(answers['answer']))):
        #     res_score.append(js.loads(chat_free_prompt([igk[0], igk[1]], 0))['score'])
        #     print(idgk)
        answers = answers.assign(ans_grth = ans_grth)
        # answers = answers.assign(score_by_ai = res_score)
        answers.to_csv('_'.join(['answers', args.chunk, args.llm, str(args.temp), '_essay', 'reranker', str(args.reranker)]) + '.csv')
        print('_'.join(['answers', args.chunk, args.llm, str(args.temp), "essay : " + "True"]))
    print("finish")

def runner():
    """python main.py -c "seq" -l "mis" -e 0 -t 0"""
    """python main.py -c "law" -l "mis" -e 0 -t 0"""
    """python main.py -c "seq" -l "gpt" -e 0 -t 0"""
    """python main.py -c "law" -l "gpt" -e 0 -t 0 -r 0,1,2,100"""
    import subprocess

    # This command could have multiple commands separated by a new line \n
    # some_command = "export PATH=$PATH://server.sample.mo/app/bin \n customupload abc.txt"


    import os
    cm = []
    for kjz in [1, 2]:
        for kj in ['seq', 'law']:
            for kjj in ['gpt', 'mis']:
                for kjjj in [0, 1]:
                    for kjjjj in [0, 0.5]:
                        cm.append("""python main.py -c "{}" -l "{}" -e {} -t {} -r {}""".format(kj, kjj, kjjj, kjjjj, kjz))
                        cmdd = """python main.py -c "{}" -l "{}" -e {} -t {} -r {}""".format(kj, kjj, kjjj, kjjjj, kjz)
                        print(cmdd)
                        # p = subprocess.Popen(cmdd, stdout=subprocess.PIPE, shell=True)
                        #
                        # (output, err) = p.communicate()
                        #
                        # # This makes the wait possible
                        # p_status = p.wait()
                        _ = os.system(cmdd)
    print(len(cm), cm)
    pass
def answer(qq, docsearch, t, l, ess = True, rerank=100):
    if not ess:
        new_g = qq.copy()
        qq = list(qq['q'])
    an = pd.DataFrame({'answer': []})
    for ind, g in enumerate(qq):
        if not ess:
            g = [new_g['q'][ind], new_g['ch'][ind]]
        if l == "gpt":
            ant = chat(docsearch, g, t, essay=ess, rerank=rerank, index_ques = ind)
        else:
            ant = chat_mis(docsearch, g, t, essay=ess, rerank=rerank, index_ques = ind)
        an.loc[len(an.index)] = [ant]
        # print(ind, "\t", g)
        # print(ant)
        print(ind)
        # if ind == 3:
        #     break
    return an

def add_to_chroma_per_100(t_text=[], pers=""):
    nm_id = [str(g) for g in range(len(t_text))]
    nm_met = [{'source': g} for g in t_text]
    str_nm = t_text
    em = Embed_Man_OpenAI()
    import chromadb
    os.environ['OPENAI_API_KEY'] = "your api key"
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    db = chromadb.PersistentClient(pers)
    db_col = db.get_or_create_collection('all', embedding_function=embeddings)
    # for change the metode search
    # collection.modify(name="new_name")
    # collection.modify(metadata={"hnsw:space": "cosine"})
    # collection = client.create_collection(
    #     name="collection_name",
    #     metadata={"hnsw:space": "cosine"}  # l2 is the default
    # )
    for g in range(0, len(nm_id), 100):
        adds = 100
        if g > int(len(nm_id)/100) * 100:
            adds = len(nm_id) - g
        id_t = nm_id[g:g + adds]
        nm_met_t = nm_met[g:g + adds]
        str_nm_t = str_nm[g:g + adds]
        em_t = em.get_embedding(str_nm_t)
        em_t = [em_t['data'][g]['embedding'] for g in range(len(em_t['data']))]
        db_col.add(ids=id_t, embeddings=em_t, documents=str_nm_t, metadatas=nm_met_t)
        print(g)

class Embed_Man_OpenAI:
    os.environ['OPENAI_API_KEY'] = 'your api key'
    api_key_ = "your api key"
    os.environ['OPENAI_API_KEY'] = api_key_
    openai.api_key = api_key_
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str, model="text-embedding-ada-002"):
        return openai.Embedding.create(input=text, model=model)

def create_seq_deb():
    os.environ['OPENAI_API_KEY'] = "your api key"
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    db_seq = db.PersistentClient('db_seq')
    db_seq_col = db_seq.get_or_create_collection('all', embedding_function=embeddings)
    return db_seq
# Press the green button in the gutter to run the script.

def reader_all_content(root='seq_pdf'):
    import fitz
    text = []
    for g in os.listdir(root):
        if g.endswith('pdf') or g.endswith('PDF'):
            with fitz.open(os.path.join(root, g)) as doc:  # open document
                textf = chr(12).join([page.get_text() for page in doc])
                Ov = Overlay()
                ov = Ov.overlay(textf)
                text.append(ov)
    return text

class Overlay:
    def overlay(self, var_txt, set_over=3, set_char = 1000):
        var_txt = re.sub("[\n\t]", " ", var_txt)
        var_txt = re.sub("[ ]+", " ", var_txt)
        var_txt_d = var_txt
        var_txt = var_txt.split(' ')
        var_txt_p = len(var_txt)
        if len(var_txt_d) < set_char:
            return [' '.join(var_txt)]
        else:
            var_txt_p_l = [len(g) for g in var_txt]
            var_txt_spl = []
            var_txt_temp = 0
            for ind, g in enumerate(var_txt_p_l):
                if var_txt_spl:
                    var_txt_temp = len(' '.join(var_txt[var_txt_spl[-1] : ind]))
                else:
                    var_txt_temp = len(' '.join(var_txt[:ind]))
                if var_txt_temp > set_char:
                    var_txt_spl.append(ind)
                    var_txt_temp = 0
            var_pl = len(var_txt_spl)
            if var_pl == 1:
                return [' '.join(var_txt[:var_txt_spl[0]]), ' '.join(var_txt[var_txt_spl[0] - set_over:])]
            else:
                res = []
                for ind, g in enumerate(var_txt_spl):
                    if ind+1 == len(var_txt_spl):
                        res.append(' '.join(var_txt[var_txt_spl[ind - 1] - set_over: var_txt_spl[ind]]))
                        res.append(' '.join(var_txt[var_txt_spl[ind] - set_over:]))
                    elif ind == 0:
                        res.append(' '.join(var_txt[:var_txt_spl[0]]))
                    else:
                        res.append(' '.join(var_txt[var_txt_spl[ind - 1] - set_over: var_txt_spl[ind]]))
        return res

class db_mcq():
    def __init__(self):
        db_mcq_ada_multi_line = [['1. Pasar uang merupakan pertemuan antara permintaan dan penawaran kredit … ', 'a.Jangka pendek', 'b.Jangka menengah', 'c.Jangka Panjang', 'd. jangka sedang', 'e.jangka tidak pasti', 'Jawaban : A '], ['2. Selain Bursa Efek Jakarta, Indonesia pernah mempunyai bursa efek di … ', 'a.Medan', 'b.Semarang', 'c.Surabaya ', 'd.Bandung', 'e. Makassar', 'Jawaban : C '], ['3. Di Bursa Efek diberlakukan system perdagangan otomatis yang dikenal dengan nama: ', 'a.The Jakarta Automated Trading System', 'b.capital market', 'c.Automated Teller Machine', 'd.capital gain', 'e.Invesment Company', 'Jawaban : A '], ['4. Surat tanda penyertaan atau kepemilikan seseorang dan atau badan usaha dalam suatu perusahaan dIsebut … ', 'a.Saham ', 'b.Obligasi ', 'c.Right ', 'd.Warrant', 'e.reksa dana ', 'Jawaban: A '], ['5. Surat yang berharga yang memberikan hak bagi pemodal untuk membeli saham baru yang dikeluarkan emiten adalah … ', 'a.Saham', 'b.Obligas', 'c.Right ', 'd.Warrant', 'e. reksa dana', 'Jawaban : C '], ['6. Sekuritas yang melekat pada penerbitan saham ataupun obligasi dan memberikan hak kepada pemiliknya untuk membeli saham perusahaan dengan ', 'harga dan pada jangka waktu tertentu disebut … ', 'a. Saham', 'b. Obligasi', 'c. Right', 'd. Warrant', 'e. reksa dana', 'Jawaban : D '], ['7. Profesi penunjang yang terkait dalam perdagangan efek adalah', '1.Kustodian ', '2.Wali amanat ', '3.Akuntan public ', '4.Notaris ', 'a. 1 dan 2', 'b. 1 dan 3', 'c. 3 dan 4', 'd. 2 dan 3', 'e. 1 dan 4', 'Jawaban : C '], ['8. Berikut ini adalah lembaga –lembaga yang terkait dengan pasar modal', '1.Biro Administrasi Efek (BAE) ', '2.Bank Kustodian', '3.Wali Amanat', '4.Penasehat Investasi', '5.Pemeringkat Efek (Rating Agencies) ', 'Lembaga penunjang pasar modal adalah', 'a. 1, 2, dan 3', 'b. semua benar', 'c. 2, 3, dan 4 ', 'd. 3, 4, dan 5', 'e. 1, 3, dan 4', 'Jawaban : A '], ['9. Berikut ini yang bukan pelaku di pasar modal …. ', 'a.Emiten ', 'b.Perusahaan efek ', 'c.Reksadana ', 'd.BUMN', 'e.perusahaan publik', 'Jawaban : D '], ['10. Lembaga yang bertugas melaksanakan penilaian saham atau obligasi yang beredar adalah …', 'a. Perusahaan Efek', 'b.Pemeringkat Efek', 'c.Biro Administrasi Efek', 'd.Bursa Efek', 'e.Kustodian ', 'Jawaban : B '], ['11. Berikut ini merupakan manfaat bursa efek, kecuali … ', 'a. Untuk memperoleh bunga yang tinggi ', 'b. Untuk memperoleh modal dari luar sector perbankan ', 'c. Agar masyarakat umum dapat keuntungan perusahaan melalui kepemilikan saham ', 'd. Untuk melakukan ekspansi perusahaan ', 'e. Meningkatkan produktivitas perusahaan ', 'Jawaban : A '], ['12. Surat bukti penyertaan modal dinamakan … ', 'a.Emiten', 'b.Saham', 'c.Penyertaan', 'd.Investasi', 'e.sertifikat', 'Jawaban: B '], ['13. Salah satu fungsi Bapepam adalah … ', 'a.Melakukan pemeriksaan atas laporan keuangan bagi perusahaan yang akan masuk ke bursa efek ', 'b.Menjamin emisi efek bagi perusahaan yang ingin menjual saham ', 'c.Sebagai agen (perantara) perdagangan efek ', 'd.Mengadakan pembinaan dan pengawasan terhadap bursa efek yang dikelola swasta ', 'e.Meningkatkan partisipasi mayarakat dalam pengumpulan dana di bursa efek ', 'Jawaban : D '], ['14. Lembaga penunjang pada pasar modal yang melaksanakan kegiatan pasar modal yang berfungsi sebagai pihak yang dipercaya untuk mewakili ', 'kepentingan seluruh pedagang obligasi atau sekuritas kredit adalah … ', 'a. Wali amanat ', 'b. Penanggung ', 'c. Biro administrasi efek ', 'd. Akuntan public ', 'e. Danareksa. ', 'Jawaban : A '], ['15. Pasar uang merupakan pertemuan antara permintaan dan penawaran kredit … ', 'a.Jangka pendek ', 'b.Jangka menengah', 'c. Jangka Panjang', 'd.jangka sedang', 'e.jangka tidak pasti', 'Jawaban : A '], ['16. Salah satu peran pasar modal adalah …', 'a. Membiayai pembangunan', 'b. Sebagai indikator perkembangan ekonomi suatu negara', 'c. Menutupi defisit APBN', 'd. Salah satu sumber penerimaan pemerintah', 'e. Harga saham stabil', 'Jawaban : A '], ['17. Peranan Bank dalam perdagangan efek adalah sebagai ... ', 'a. Perantara perdagangan efek', 'b. Penjual efek secara langsung di bursa', 'c. Penjamin emisi', 'd. Pembeli efek', 'e. Penyandang dana', 'Jawaban : E '], ['18. Tempat bertemunya permintaan dan penawaran modal untuk jangka waktu yang panjang adalah, kecuali … ', 'a. Pasar modal ', 'b. Bursa efek', 'c. Pasar dana ', 'd. capital market', 'e. stock exchange', 'Jawaban : C '], ['19. Berikut yang tidak termasuk perbedaan antara pasar perdana dan pasar sekunder adalah … ', 'a. pada pasar perdana harga tidak berubah, sedangkan pada pasar sekunder harga ditentukan oleh mekanisme pasar', 'b. transaksi perdagangan di pasar sekunder tidak dikenakan biaya komisi, sedangkanpasar perdana terdapat biaya komisi', 'c. pada pasar perdana hanya terjadi pembelian saham sedangkan di pasar sekunder dapat terjadi jual beli saham', 'd. Dari sudut pandang jangka waktu, pasar perdana memiliki batas waktu, sedangkan pasar sekunder tidak', 'e. Di pasar sekunder semua transaksi harus melalui pialang sedangkan pada pasar perdana tidak demikian', 'Jawaban : B '], ['20. Berikut istilah yang berarti perusahaan menerbitkan saham untuk pertama kali kepada masyarakat, kecuali … ', 'a. go public ', 'b. IPO', 'c. investasi modal', 'd. pasar perdana', 'e. primary market', 'Jawaban : C '], ['21. Bentuk perusahaan yang diperbolehkan untuk menerbitkan saham adalah … ', 'a. perusahaan perorangan', 'b. firma ', 'c. CV ', 'd. perseroan terbatas', 'e. perusahaan daerah', 'Jawaban : D '], ['22. Hal berikut yang bukan termasuk manfaat dari pasar modal adalah … ', 'a. menyediakan sumber pembiayaan jangka panjang bagi dunia usaha', 'b. memberikan keuntungan bagi para pemilik modal dengan tingkat resiko yang dapat diperhitungkan', 'c. mendorong peningkatan produksi dan memperluas lapangan kerja melalui upaya peningkatan modal d. meningkatkan pertumbuhan ekonomi', 'e.pengelompokkan kepemilikan perusahaan pada kalangan tertentu', 'Jawaban : E '], ['23. Dari beberapa fungsi pasar modal di bawah ini, yang secara langsung menguntungkan pemerintah adalah … ', 'a. sarana untuk mendapatkan tambahan modal ', 'b. sarana pemerataan pendapatan', 'c. memperbesar produksi nasional', 'd. meningkatkan pemasukan pajak', 'e. meminimalkan jumlah pengangguran', 'Jawaban : D '], ['24. Di bawah ini fungsi pasar modal di Indonesia yaitu … ', 'a. sarana untuk mendapatkan tambahan modal ', 'b. sarana pemusatan pendapatan masyarakat', 'c. meningkatkan upah buruh', 'd. mengurangi pemasukan pajak untuk pemerintah', 'e. memperoleh pengakuan Negara lain tentang keberhasilan pembangunan Indonesia ', 'Jawaban : A '], ['25. Berikut adalah keuntungan seseorang yang memiliki saham, kecuali … ', 'a. mendapat pembagian keuntungan', 'b. memiliki sebagian kecil perusahaan', 'c. gaji meningkat', 'd. memperoleh capital gain ', 'e. menerima deviden', 'Jawaban : C '], ['26. Pasar modal yang terdapat di Indonesia ada dua, yaitu … ', 'a. Bursa Efek Indonsia (BEI) dan Bursa Efek Makasar (BEM) ', 'b. Bursa Efek Indonesia (BEI) dan Bursa Efek Surabaya (BES) ', 'c. Bursa Efek Makasar (BEM) dan Bursa Efek Surabaya (BES) ', 'd. Bursa Efek Samarinda (BES) dan Bursa Efek Jakarta (BEJ) ', 'e. Bursa Efek Semarang (BES) dan Bursa Efek Yogyakarta ', 'Jawaban : B '], ['27. Resiko membeli saham suatu perusahaan adalah sebagai berikut, kecuali … ', 'a. Tidak mendapat deviden', 'b. Mendapat capital gain ', 'c. Perusahan bangkrutatu dilikuidasi', 'd. Saham dihpucatatkan dari bursa efek (delisting) ', 'e. Capital loss ', 'Jawaban: B '], ['28. Di bawah ini yang termasuk contoh effek, kecuali … ', 'a. obligasi', 'b. surat pengkuan utang', 'c. kuitansi', 'd. tanda bukti uang', 'e. saham', 'Jawaban : C '], ['29. Indeks yang menunjukkan harga 45 saham yang sangat sering diperdagangkan atau sangat likuid disebut … ', 'a. IHSG ', 'b.Indeks saham likuid', 'c.Indeks sektoral', 'd.Indeks LQ45 ', 'e.IHSG 45 ', 'Jawaban : D '], ['30. Indeks yang terdiri dari 39 jenis saham yang dipilih berdasarkan aturan syariah islam adalah :', 'a. Indeks LQ45 ', 'b. IHSG ', 'c. Indeks Sektoral ', 'd. Indeks Syariah ', 'e. Indeks Kompas 100 Pembahasan :', 'Jawaban : D '], ['31. Selembar kertas yang menyatakan bahwa pemilik kertas tersebut telah membeli utang perusahaan yang menerbitkannya, disebut…', 'a. saham ', 'b. obligasi ', 'c. reksa dana ', 'd. modal ', 'e. utang', 'Jawaban : B '], ['32. Resiko yang mungkin timbul saat berinvestasi pada obligasi adalah tidak pastinya perkembangan suku bunga. Pemilik obligasi akan mengalami ', 'kerugian pada saat …', 'a. harga obligasi turun dan suku bunga turun', 'b.harga obligasi naik dan suku bunga turun', 'c.harga obligasi turun dan suku bunga naik', 'd.harga obligasi naik dan suku bunga naik', 'e.harga obligasi turun dan suku bunga tetap', 'Jawaban : B '], ['33. Bapak Azhari pada tanggal 25 Agustus 2006, membeli saham PT Sepatu Bata tbk sebanyak 5 lot dengan harga per saham Rp. 12.500,00. Tiga bulan ', 'kemudian ternyata harga saham mengalami peningkatan. Pada saat harganya mencapai Rp. 13.200,00/lembar, Bapak Azhari menjual seluruh saham ', 'Sepatu Bata tersebut. Keuntungan yang diperoleh atas penjualan sahamnya adalah … ', 'a. Rp. 3.500,00', 'b. Rp. 1.750,00', 'c. Rp. 1.750.000,00', 'd. Rp. 3.500.000,00', 'e. Rp. 2.000.000,00', 'Jawaban : C '], ['34. Pihak yang kegiatan usahanya mengelola efek untuk para nasabah disebut … ', 'a. investor', 'b. debitur', 'c. broker', 'd. emiten sekuritas', 'e. manajer investasi', 'Jawaban : E '], ['35. Undang-undang yang mengatur pasar modal Indonesia adalah … ', 'a. UU No. 1 Tahun 1995 ', 'b. UU No. 8 Tahun 1995 ', 'c. UU No. 18 Tahun 1995 ', 'd. UU No. 8 Tahun 2000 ', 'e. UU No, 1 Tahun 2000 ', 'Jawaban : B '], ['36. Pasar di mana pertama kali suatu perusahaan menawarkan sahamnya kepada public disebut … ', 'a. pasar pertama ', 'b. pasar sekunder', 'c. pasar tunai', 'd. pasar perdana', 'e. pasar negosias', 'Jawaban : B '], ['37. Instrumen (jenis investasi) pasar modal pada umumnya berjangka waktu diatas … ', 'a. satu bulan', 'b. tiga bulan', 'c. enam bulan ', 'd. Sembilan bulan', 'e. satu tahun', 'Jawaban : E '], ['38. Berikut ini adalah ciri-ciri pasar perdana dan pasar sekunder ', '1.Harga yang telah ditentukan tidak dapat berubah ', '2. Dalam melakukan transaksi tidak kenakan komisi ', '3. Hanya berlangsung transaksi pembelian efek ', '4. Terdapat transaksi jual beli efek ', '5. Dalam melakukan transaksi terdapat biaya komisi ', '6. Harga berfluktuasi sesuai dengan kekuatan permintaan dan penawaran ', 'Yang merupakan ciri pasar perdana adalah : ', 'a. 1, 2, dan 3', 'b. 1, 2, dan 4', 'c. 1, 2, dan 5', 'd. 4, 5, dan 6', 'e. 3, 4, dan 5', 'Jawaban : A '], ['39. Lembaga atau otoritas tertinggi di pasar modal yang melakukan pengawasan dan pembinaan atas pasar modal adalah: ', 'a. Bursa Efek ', 'b. Kustodian', 'c. Bapepam', 'd. BAE ', 'e. Manajer Investasi ', 'Jawaban : C '], ['40. Menyebarluaskan informasi-informasi bursa ke masyarakat adalah tugas', 'a. Bapepam ', 'b. KSEI ', 'c. Emiten ', 'd. Bursa Efek ', 'e. Wali Amanat ', 'Jawaban : D '], ['41. Bank yang bertindak sebagai tempat penyimpanan dan penitipan uang, surat-surat berharga, maupun barang-barang berharga lainnya adalah', 'a. Bank Indonesia ', 'b. Bank Kustodian ', 'c. Wali Amanat ', 'd. Bursa Efek ', 'e. Manajer Investasi ', 'Jawaban : B'], ['42. Pihak yang membuat kontrak dengan emiten untuk melakukan penawaran umum bagi kepentingan emiten dengan atau tanpa kewajiban untuk membeli ', 'sisa efek yang tidak terjual, adalah … ', 'a. Penjamin Emisi Efek ', 'b. Investor ', 'c. Emiten', 'd. Kustodian', 'e. Manajer Investasi ', 'Jawban : A ', '43. Dalam saham preferen : ', 'a. Deviden dbayarkan sepanjang perusahaan memperoleh laba ', 'b. Memiliki hak suara ', 'c. Jika perusahaan dilikuidasi, hak memperoleh pembagian kekayaan dilakukan setelah semua kewajiban perusahaan dilunasi ', 'd. Tidak memiliki hak suara ', 'e. Tidak diberikan keuntungan perusahaan', 'Jawaban : D '], ['44. Sejak tahun 1995, BEJ telah menggunakan sistem perdagangan otomatis yang di kenal dengan istilah…', 'a. remote trading ', 'b. Scripless trading ', 'c. Jakarta automated trading system (JATS)', 'd. Automatic trading ', 'e. Halting system ', 'Jawaban : C '], ['45. Istilah yang menunjukan satuan perdagangan saham adalah', 'a.Fraksi harga', 'b.Unit penyertaan', 'c.Unit', 'd.Lot ', 'e.Nominal ', ' Jawaban : D '], ['46. Satu lot saham berjumlah … ', 'a. 100 ', 'b. 200 ', 'c. 500 ', 'd. 1000 ', 'e. 5000 ', 'Jawaban : C '], ['47. Kelipatan harga saham disebut ', 'a. Lot', 'b. Poin', 'c. Add lot', 'd. Gain', 'e. loss', 'Jawaban : B '], ['48. Perdagangan saham yang kurang dari 1 lot disebut … ', 'a. Lot', 'b. Poin', 'c. Add lot', 'd. Gain', 'e. loss', 'Jawaban : C '], ['49. Saham perusahaan dapat dicatatkan di beberapa bursa efek. Istilah yang menjelaskan suatu saham tercatat di lebih dari satu bursa di sebut?', 'a. Saveral listing ', 'b. Secondary offering ', 'c. Dual listing ', 'd. Many listing', 'e. Triple listing ', 'Jawaban : D '], ['50. Seorang investor membeli saham pada harga Rp.3400,-/saham.kerugian investor tersebut dikenal dengan istilah….', 'a. Capital gain ', 'b. Capital loss ', 'c. Depresiasi', 'd. Dilusi', 'e. Negative return ', 'Jawaban : B '], ['51. Sesi pertama perdangan saham di BEJ dimulai pukul…', 'a. 09.00 wib', 'b. 09.30 wib', 'c. 10.00 wib', 'd. 09.15 wib', 'e. 09.45 wib', 'Jawaban : B '], ['52. Sesi kedua perdangan saham di BEJ ditutup pada pukul…', 'a. 15.00 wib', 'b. 15.30 wib', 'c. 17.00 wib', 'd. 16.00 wib', 'e. 16.30 wib ', 'Jawaban : D'], ['53. Berikut ini adalah ciri-ciri pasar perdana dan pasar sekunder ', '1. Harga efek tetap ', '2. Tidak ada beban komisi ', '3. hanya untuk pembelian saham ', '4. setiap transaksi ada beban komisi ', '5. pemesanan dilakukan melalui anggota bursa ', '6. Jangka waktu perdagangan tidak terbatas ', 'Yang termasuk ciri pasar perdana adalah : ', 'a. 1, 2, dan 3', 'b.4, 5, dan 6', 'c.2, 4, dan 6', 'd.2, 3, dan 5', 'e.1, 4, dan 6', 'Jawaban : A '], ['54. Pihak yang didasarkan kontrak dengan Emiten melaksanakan pencatatan pemilikan Efek dan pembagian hak yang berkaitan dengan Efek adalah', 'a. BAE ', 'b.wali amanat ', 'c. custodian', 'd.manajer investasi ', 'e.emiten ', ' Jawaban : A '], ['55. Berikut ini adalah mekanisme perdagangan di Pasar Modal: ', '1. Calon penanam modal akan membuka opening account di perusahaan efek yang dipercaya untuk mengelola dana. ', '2. Perusahaan efek aktif mencatatnya dalam file customer perusahaan dan menyimpannya sebagai data perusahaan. ', '3. Saat pemilik modal ingin melakukan transaksi, ia harus menghubungi brokernya dan memberitahukan saham yang diinginkan beserta jumlah dan ', 'harga yang ingin dibeli atau dijual. ', '4. Broker akan bertindak sebagai sales person, dan akan meneruskan perintah tersebut pada dealer di perusahaan investasi. ', 'Urutan mekanisme perdagangan di pasar modal yang tepat adalah … ', 'a. 1, 2, 3, dan 4', 'b.2,1, 3, dan 4', 'c.4, 3, 2, dan 1', 'd.1, 3, 4, dan 2', 'e.3, 2, 4, dan 1', 'Jawaban : A ']]
        db_mcq_no_multiline = [['1. Pasar uang merupakan pertemuan antara permintaan dan penawaran kredit … ', 'a.Jangka pendek', 'b.Jangka menengah', 'c.Jangka Panjang', 'd. jangka sedang', 'e.jangka tidak pasti', 'Jawaban : A '], ['2. Selain Bursa Efek Jakarta, Indonesia pernah mempunyai bursa efek di … ', 'a.Medan', 'b.Semarang', 'c.Surabaya ', 'd.Bandung', 'e. Makassar', 'Jawaban : C '], ['3. Di Bursa Efek diberlakukan system perdagangan otomatis yang dikenal dengan nama: ', 'a.The Jakarta Automated Trading System', 'b.capital market', 'c.Automated Teller Machine', 'd.capital gain', 'e.Invesment Company', 'Jawaban : A '], ['4. Surat tanda penyertaan atau kepemilikan seseorang dan atau badan usaha dalam suatu perusahaan dIsebut … ', 'a.Saham ', 'b.Obligasi ', 'c.Right ', 'd.Warrant', 'e.reksa dana ', 'Jawaban: A '], ['5. Surat yang berharga yang memberikan hak bagi pemodal untuk membeli saham baru yang dikeluarkan emiten adalah … ', 'a.Saham', 'b.Obligas', 'c.Right ', 'd.Warrant', 'e. reksa dana', 'Jawaban : C '], ['9. Berikut ini yang bukan pelaku di pasar modal …. ', 'a.Emiten ', 'b.Perusahaan efek ', 'c.Reksadana ', 'd.BUMN', 'e.perusahaan publik', 'Jawaban : D '], ['10. Lembaga yang bertugas melaksanakan penilaian saham atau obligasi yang beredar adalah …', 'a. Perusahaan Efek', 'b.Pemeringkat Efek', 'c.Biro Administrasi Efek', 'd.Bursa Efek', 'e.Kustodian ', 'Jawaban : B '], ['11. Berikut ini merupakan manfaat bursa efek, kecuali … ', 'a. Untuk memperoleh bunga yang tinggi ', 'b. Untuk memperoleh modal dari luar sector perbankan ', 'c. Agar masyarakat umum dapat keuntungan perusahaan melalui kepemilikan saham ', 'd. Untuk melakukan ekspansi perusahaan ', 'e. Meningkatkan produktivitas perusahaan ', 'Jawaban : A '], ['12. Surat bukti penyertaan modal dinamakan … ', 'a.Emiten', 'b.Saham', 'c.Penyertaan', 'd.Investasi', 'e.sertifikat', 'Jawaban: B '], ['13. Salah satu fungsi Bapepam adalah … ', 'a.Melakukan pemeriksaan atas laporan keuangan bagi perusahaan yang akan masuk ke bursa efek ', 'b.Menjamin emisi efek bagi perusahaan yang ingin menjual saham ', 'c.Sebagai agen (perantara) perdagangan efek ', 'd.Mengadakan pembinaan dan pengawasan terhadap bursa efek yang dikelola swasta ', 'e.Meningkatkan partisipasi mayarakat dalam pengumpulan dana di bursa efek ', 'Jawaban : D '], ['15. Pasar uang merupakan pertemuan antara permintaan dan penawaran kredit … ', 'a.Jangka pendek ', 'b.Jangka menengah', 'c. Jangka Panjang', 'd.jangka sedang', 'e.jangka tidak pasti', 'Jawaban : A '], ['16. Salah satu peran pasar modal adalah …', 'a. Membiayai pembangunan', 'b. Sebagai indikator perkembangan ekonomi suatu negara', 'c. Menutupi defisit APBN', 'd. Salah satu sumber penerimaan pemerintah', 'e. Harga saham stabil', 'Jawaban : A '], ['17. Peranan Bank dalam perdagangan efek adalah sebagai ... ', 'a. Perantara perdagangan efek', 'b. Penjual efek secara langsung di bursa', 'c. Penjamin emisi', 'd. Pembeli efek', 'e. Penyandang dana', 'Jawaban : E '], ['18. Tempat bertemunya permintaan dan penawaran modal untuk jangka waktu yang panjang adalah, kecuali … ', 'a. Pasar modal ', 'b. Bursa efek', 'c. Pasar dana ', 'd. capital market', 'e. stock exchange', 'Jawaban : C '], ['19. Berikut yang tidak termasuk perbedaan antara pasar perdana dan pasar sekunder adalah … ', 'a. pada pasar perdana harga tidak berubah, sedangkan pada pasar sekunder harga ditentukan oleh mekanisme pasar', 'b. transaksi perdagangan di pasar sekunder tidak dikenakan biaya komisi, sedangkanpasar perdana terdapat biaya komisi', 'c. pada pasar perdana hanya terjadi pembelian saham sedangkan di pasar sekunder dapat terjadi jual beli saham', 'd. Dari sudut pandang jangka waktu, pasar perdana memiliki batas waktu, sedangkan pasar sekunder tidak', 'e. Di pasar sekunder semua transaksi harus melalui pialang sedangkan pada pasar perdana tidak demikian', 'Jawaban : B '], ['20. Berikut istilah yang berarti perusahaan menerbitkan saham untuk pertama kali kepada masyarakat, kecuali … ', 'a. go public ', 'b. IPO', 'c. investasi modal', 'd. pasar perdana', 'e. primary market', 'Jawaban : C '], ['21. Bentuk perusahaan yang diperbolehkan untuk menerbitkan saham adalah … ', 'a. perusahaan perorangan', 'b. firma ', 'c. CV ', 'd. perseroan terbatas', 'e. perusahaan daerah', 'Jawaban : D '], ['23. Dari beberapa fungsi pasar modal di bawah ini, yang secara langsung menguntungkan pemerintah adalah … ', 'a. sarana untuk mendapatkan tambahan modal ', 'b. sarana pemerataan pendapatan', 'c. memperbesar produksi nasional', 'd. meningkatkan pemasukan pajak', 'e. meminimalkan jumlah pengangguran', 'Jawaban : D '], ['24. Di bawah ini fungsi pasar modal di Indonesia yaitu … ', 'a. sarana untuk mendapatkan tambahan modal ', 'b. sarana pemusatan pendapatan masyarakat', 'c. meningkatkan upah buruh', 'd. mengurangi pemasukan pajak untuk pemerintah', 'e. memperoleh pengakuan Negara lain tentang keberhasilan pembangunan Indonesia ', 'Jawaban : A '], ['25. Berikut adalah keuntungan seseorang yang memiliki saham, kecuali … ', 'a. mendapat pembagian keuntungan', 'b. memiliki sebagian kecil perusahaan', 'c. gaji meningkat', 'd. memperoleh capital gain ', 'e. menerima deviden', 'Jawaban : C '], ['26. Pasar modal yang terdapat di Indonesia ada dua, yaitu … ', 'a. Bursa Efek Indonsia (BEI) dan Bursa Efek Makasar (BEM) ', 'b. Bursa Efek Indonesia (BEI) dan Bursa Efek Surabaya (BES) ', 'c. Bursa Efek Makasar (BEM) dan Bursa Efek Surabaya (BES) ', 'd. Bursa Efek Samarinda (BES) dan Bursa Efek Jakarta (BEJ) ', 'e. Bursa Efek Semarang (BES) dan Bursa Efek Yogyakarta ', 'Jawaban : B '], ['27. Resiko membeli saham suatu perusahaan adalah sebagai berikut, kecuali … ', 'a. Tidak mendapat deviden', 'b. Mendapat capital gain ', 'c. Perusahan bangkrutatu dilikuidasi', 'd. Saham dihpucatatkan dari bursa efek (delisting) ', 'e. Capital loss ', 'Jawaban: B '], ['28. Di bawah ini yang termasuk contoh effek, kecuali … ', 'a. obligasi', 'b. surat pengkuan utang', 'c. kuitansi', 'd. tanda bukti uang', 'e. saham', 'Jawaban : C '], ['29. Indeks yang menunjukkan harga 45 saham yang sangat sering diperdagangkan atau sangat likuid disebut … ', 'a. IHSG ', 'b.Indeks saham likuid', 'c.Indeks sektoral', 'd.Indeks LQ45 ', 'e.IHSG 45 ', 'Jawaban : D '], ['30. Indeks yang terdiri dari 39 jenis saham yang dipilih berdasarkan aturan syariah islam adalah :', 'a. Indeks LQ45 ', 'b. IHSG ', 'c. Indeks Sektoral ', 'd. Indeks Syariah ', 'e. Indeks Kompas 100 Pembahasan :', 'Jawaban : D '], ['31. Selembar kertas yang menyatakan bahwa pemilik kertas tersebut telah membeli utang perusahaan yang menerbitkannya, disebut…', 'a. saham ', 'b. obligasi ', 'c. reksa dana ', 'd. modal ', 'e. utang', 'Jawaban : B '], ['34. Pihak yang kegiatan usahanya mengelola efek untuk para nasabah disebut … ', 'a. investor', 'b. debitur', 'c. broker', 'd. emiten sekuritas', 'e. manajer investasi', 'Jawaban : E '], ['35. Undang-undang yang mengatur pasar modal Indonesia adalah … ', 'a. UU No. 1 Tahun 1995 ', 'b. UU No. 8 Tahun 1995 ', 'c. UU No. 18 Tahun 1995 ', 'd. UU No. 8 Tahun 2000 ', 'e. UU No, 1 Tahun 2000 ', 'Jawaban : B '], ['36. Pasar di mana pertama kali suatu perusahaan menawarkan sahamnya kepada public disebut … ', 'a. pasar pertama ', 'b. pasar sekunder', 'c. pasar tunai', 'd. pasar perdana', 'e. pasar negosias', 'Jawaban : B '], ['37. Instrumen (jenis investasi) pasar modal pada umumnya berjangka waktu diatas … ', 'a. satu bulan', 'b. tiga bulan', 'c. enam bulan ', 'd. Sembilan bulan', 'e. satu tahun', 'Jawaban : E '], ['39. Lembaga atau otoritas tertinggi di pasar modal yang melakukan pengawasan dan pembinaan atas pasar modal adalah: ', 'a. Bursa Efek ', 'b. Kustodian', 'c. Bapepam', 'd. BAE ', 'e. Manajer Investasi ', 'Jawaban : C '], ['40. Menyebarluaskan informasi-informasi bursa ke masyarakat adalah tugas', 'a. Bapepam ', 'b. KSEI ', 'c. Emiten ', 'd. Bursa Efek ', 'e. Wali Amanat ', 'Jawaban : D '], ['41. Bank yang bertindak sebagai tempat penyimpanan dan penitipan uang, surat-surat berharga, maupun barang-barang berharga lainnya adalah', 'a. Bank Indonesia ', 'b. Bank Kustodian ', 'c. Wali Amanat ', 'd. Bursa Efek ', 'e. Manajer Investasi ', 'Jawaban : B'], ['44. Sejak tahun 1995, BEJ telah menggunakan sistem perdagangan otomatis yang di kenal dengan istilah…', 'a. remote trading ', 'b. Scripless trading ', 'c. Jakarta automated trading system (JATS)', 'd. Automatic trading ', 'e. Halting system ', 'Jawaban : C '], ['45. Istilah yang menunjukan satuan perdagangan saham adalah', 'a.Fraksi harga', 'b.Unit penyertaan', 'c.Unit', 'd.Lot ', 'e.Nominal ', ' Jawaban : D '], ['46. Satu lot saham berjumlah … ', 'a. 100 ', 'b. 200 ', 'c. 500 ', 'd. 1000 ', 'e. 5000 ', 'Jawaban : C '], ['47. Kelipatan harga saham disebut ', 'a. Lot', 'b. Poin', 'c. Add lot', 'd. Gain', 'e. loss', 'Jawaban : B '], ['48. Perdagangan saham yang kurang dari 1 lot disebut … ', 'a. Lot', 'b. Poin', 'c. Add lot', 'd. Gain', 'e. loss', 'Jawaban : C '], ['49. Saham perusahaan dapat dicatatkan di beberapa bursa efek. Istilah yang menjelaskan suatu saham tercatat di lebih dari satu bursa di sebut?', 'a. Saveral listing ', 'b. Secondary offering ', 'c. Dual listing ', 'd. Many listing', 'e. Triple listing ', 'Jawaban : D '], ['50. Seorang investor membeli saham pada harga Rp.3400,-/saham.kerugian investor tersebut dikenal dengan istilah….', 'a. Capital gain ', 'b. Capital loss ', 'c. Depresiasi', 'd. Dilusi', 'e. Negative return ', 'Jawaban : B '], ['51. Sesi pertama perdangan saham di BEJ dimulai pukul…', 'a. 09.00 wib', 'b. 09.30 wib', 'c. 10.00 wib', 'd. 09.15 wib', 'e. 09.45 wib', 'Jawaban : B '], ['52. Sesi kedua perdangan saham di BEJ ditutup pada pukul…', 'a. 15.00 wib', 'b. 15.30 wib', 'c. 17.00 wib', 'd. 16.00 wib', 'e. 16.30 wib ', 'Jawaban : D'], ['54. Pihak yang didasarkan kontrak dengan Emiten melaksanakan pencatatan pemilikan Efek dan pembagian hak yang berkaitan dengan Efek adalah', 'a. BAE ', 'b.wali amanat ', 'c. custodian', 'd.manajer investasi ', 'e.emiten ', ' Jawaban : A ']]
        ja = [g[6] for g in db_mcq_no_multiline]
        qa = [g[0] for g in db_mcq_no_multiline]
        ca = ['\n'.join(g[1:6]) for g in db_mcq_no_multiline]

        # yang resmi bagus
        # mcq_db.csv
        mcq = pd.read_csv("mcq_db_copy.csv")

def db_reranker_cohere(docs1, fquery, cohe):

    docs = list(docs1.values())

    response = cohe.rerank(
        model="rerank-v3.5",
        query=fquery,
        documents=docs,
        top_n=4,
    )
    # print(response)
    indexes_res = [y.index for y in response.results]

    return [docs1[ir] for ir in indexes_res]

def db_reranker_jina_rerank(docs1, fquery):
    docs = list(docs1.values())
    url = 'https://api.jina.ai/v1/rerank'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_ff5bfa407dbb4795b1dfe799ecb20017pw2zBwQW98hA9B104T8co9C-PBag'
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": fquery,
        "top_n": "4",
        "documents": ""}
    data['documents'] = docs
    response = requests.post(url, headers=headers, json=data)

    # print(response.json())
    return [ir['document']['text'] for ir in response.json()['results']]

def db_reranker_jina_colbert(docs1, fquery):
    docs = list(docs1.values())
    url = 'https://api.jina.ai/v1/rerank'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_ff5bfa407dbb4795b1dfe799ecb20017pw2zBwQW98hA9B104T8co9C-PBag'
    }
    data = {
        "model": "jina-colbert-v2",
        "query": fquery,
        "top_n": "4",
        "documents": ""}
    data['documents'] = docs
    response = requests.post(url, headers=headers, json=data)

    # print(response.json())
    return [ir['document']['text'] for ir in response.json()['results']]

if __name__ == '__main__':
    import cohere

    co_api_keys = ["your key",
                   "your key","your key","your key","your key",]
    coheress = []
    for g in co_api_keys:
        os.environ['CO_API_KEY'] = g
        coheress.append(cohere.ClientV2())
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
