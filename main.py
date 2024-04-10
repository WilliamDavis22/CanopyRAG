import streamlit as st
import os
import re
import json
import fitz

from dotenv import load_dotenv
load_dotenv()

from canopy.knowledge_base import KnowledgeBase
from canopy.tokenizer import Tokenizer
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from canopy.models.data_models import UserMessage, SystemMessage
from canopy.knowledge_base.record_encoder import OpenAIRecordEncoder
from canopy.models.data_models import Document
from pinecone import Pinecone, PodSpec
from canopy.knowledge_base import list_canopy_indexes

Tokenizer.initialize()

st.title("RAG Demo")

uploaded_files = st.file_uploader(label="Upload your files", key='upload_file',accept_multiple_files=True)
if 'processed_files' not in st.session_state:
    processed_files = {}
else:
    processed_files = st.session_state['processed_files']
if uploaded_files is not None:
    if len(uploaded_files) != len(processed_files) or len(processed_files) == 0:
        processed_pages = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in processed_files.keys():
                with st.spinner(f"Reading {uploaded_file.name}..."):
                    pdf_file = None
                    processed_pages = None
                    formatted_json_fp = uploaded_file.name.replace('.pdf','.json')
                    if not os.path.exists(formatted_json_fp):
                        with open(uploaded_file.name,'wb') as f:
                            f.write(uploaded_file.getvalue())
                        doc = fitz.open(uploaded_file.name)
                        extracted_text = [page.get_text() for page in doc]
                        data = []
                        for i,page in enumerate(extracted_text):
                            data.append({
                                'id': str(i), 
                                'text': page, 
                                'source': f'{uploaded_file.name}: page {i+1}', 
                                'metadata': {'title': uploaded_file.name,
                                            'primary_category': 'Finance',
                                            'published': 2024
                                            },
                            })
                        with open(formatted_json_fp, 'w') as out_fp:
                            json.dump(data,out_fp,indent=4)
                    else:
                        with open(formatted_json_fp, 'r') as json_file:
                            data = json.load(json_file)

                processed_files[uploaded_file.name] = data
                st.session_state['processed_files'] = processed_files
            print("Reading Complete")

if len(list_canopy_indexes()) > 0:
    indexes = [item.replace('canopy--','') for item in list_canopy_indexes()]
    if 'processed_files' in st.session_state.keys():
        files = [re.sub('[^0-9a-zA-Z]+', '-', item.replace('.pdf','').lower()) for item in list(st.session_state['processed_files'].keys())]
        indexes += files
        indexes = list(set(indexes))
        
    selected_file = st.radio("Indexes", indexes, index=None,
                        key="file_lookup_key",help="Document to use for RAG",horizontal=False)
    if 'selected_file' not in st.session_state.keys():
        st.session_state['selected_file'] = selected_file

if 'selected_file' in st.session_state.keys() and st.session_state['selected_file'] is not None:
    fp = st.session_state['selected_file'].replace('.pdf','')
    fp = re.sub('[^0-9a-zA-Z]+', '-', fp).lower()
    idx_name = fp
    if f'canopy--{idx_name}' in list_canopy_indexes():
        kb = KnowledgeBase(index_name=idx_name)
        kb.connect()
        print(f'connected to {idx_name}')
    else:
        encoder = OpenAIRecordEncoder(model_name="text-embedding-3-small")
        kb = KnowledgeBase(index_name=idx_name, record_encoder=encoder)
        kb.create_canopy_index()
        kb.connect()
        print(f'connecting new index for {idx_name}...')
        documents = [Document(id=line['id'],
                        text=line['text'],
                        source=line['source'],
                        metadata=line['metadata']) for line in st.session_state['processed_files'][fp+'.pdf']]
        with st.spinner(f"Creating new index for {selected_file}... "):
            kb.upsert(documents,show_progress_bar=True)

    context_engine = ContextEngine(kb)
    chat_engine = ChatEngine(context_engine,allow_model_params_override=True)

st.markdown("Chat with your document below")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = []
        for m in st.session_state.messages:
            if m['role'] == 'user':
                messages.append(UserMessage(content=m['content']))
            else:
                messages.append(SystemMessage(content=m['content']))
        
        res = chat_engine.chat(
            messages=messages,
            stream=False, 
            model_params={'model':'gpt-4-0125-preview','temperature':0,'seed':42})
        print(res)
        response = st.write(res.choices[0].message.content)
        
    st.session_state.messages.append({"role": "assistant", "content": res.choices[0].message.content})