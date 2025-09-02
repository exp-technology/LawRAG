# LawRAG 

## Installation
You need to install several library for running this project:<br>
- Mistral library
- Openai
- ChromaDB
## runner : 
python main.py -c "law" -l "gpt" -e 0 -t 0 -r 0,1,2,100
> note
- "c : for chunking"<br>
set 'law' for our chunking_data<br>
set 'seq' for sequential data
- "l : LLM"<br>
set 'gpt' for chat_gpt_turbo_3.5<br>
set 'mis' for the mistral llm
- "e : mcq/essay"<br>
set 0 to MCQ<br>
set 1 to Essay
- "t : temperature" <br> set 0 for 0 temp <br>
set 0.5 for 0.5 temp
- "r : reranker"<br>
set 0 for cohere reranker <br>
set 1 for jina_reranker<br>
set 2 for jina_colbert_v2_reranker<br>
set 100 for no reranker



# Dataset
PDFs dataset can be found by this link : https://drive.google.com/drive/folders/10z_w246r-vkgI5VMT4DBc97YSntDPuKf?usp=drive_link
- Law chunking data<br>
Our chunking method dataset consisted 6,674 chunks can be check by this link:
<br>
https://drive.google.com/file/d/1YWNvMiZCQTPKvO02vzsi0QRnPhoBwmyI/view?usp=sharing
<br>
<!--you can ask me for the permission to show the file by clicking the url.---><br>
- Sequential data<br>
We provide the chroma_db version for the sequential data. You dont need to embed all datas anymore. 
<br> https://github.com/exp-technology/LawRAG/tree/main/db_seq
# Experiment data
Experiment show in experiment folder<br>
https://github.com/exp-technology/LawRAG/tree/main/experiment

