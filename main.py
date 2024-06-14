import streamlit as st
import os
import re
import json
import fitz
from streamlit_pdf_viewer import pdf_viewer
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
from streamlit_extras.stylable_container import stylable_container

Tokenizer.initialize()

st.set_page_config(layout="wide")
st.title("Chat with your Documents")

with st.columns([0.2,0.5])[0]:
    user = st.text_input("Username", "John Doe")

if user:
    with open('records.json','r') as f:
        records = json.load(f)

    if user not in records['demo']:
        records['demo'][user] = {'documents':[],'indexed':[]}

    #if st.button(label="🥚"):
    with st.columns([0.2,0.5])[0]:
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
                        fname = uploaded_file.name.replace(' ','-').replace('_','-').lower()
                        processed_files[fname] = data
                        st.session_state['processed_files'] = processed_files
                    print("Reading Complete")

            if 'processed_files' in st.session_state.keys():
                files = [item.replace('.pdf','').replace('_','-').replace(' ','-').lower() for item in list(st.session_state['processed_files'].keys())]
                records['demo'][user]['documents'].extend(files)
                records['demo'][user]['documents'] = list(set(records['demo'][user]['documents']))
                with open('records.json','w') as f:
                    json.dump(records,f,indent=4)
                if st.button('Submit'):
                    print('submitted')

    if len(records['demo'][user]['documents']) > 0:
        indexes = records['demo'][user]['documents']
        if 'processed_files' in st.session_state.keys():
            files = [item.replace('.pdf','').replace('_','-').replace(' ','-').lower() for item in list(st.session_state['processed_files'].keys())]
            indexes += files
            indexes = list(set(indexes))

        selected_files = [ind for selected, ind in zip([st.checkbox(item,value=True) for item in indexes],indexes) if selected]
        st.session_state['selected_files'] = selected_files

    if 'selected_files' in st.session_state.keys() and len(st.session_state['selected_files']) > 0:
        selected_file = st.session_state['selected_files'][0]
        print('selected',selected_file)
        fp = selected_file[0].replace('.pdf','')

        idx_name = "demo"
        namespace = user

        encoder = OpenAIRecordEncoder(model_name="text-embedding-3-small")
        kb = KnowledgeBase(index_name=idx_name, record_encoder=encoder)
        kb.connect()
        print('connected')
        for idx_fp in st.session_state['selected_files']:
            if idx_fp not in records['demo'][user]['indexed']:
                try:
                    documents = [Document(id=line['id'],
                                text=line['text'],
                                source=line['source'],
                                metadata=line['metadata']) for line in st.session_state['processed_files'][idx_fp+'.pdf']]
                    with st.spinner(f"Creating new index for {selected_file}... "):
                        kb.upsert(documents,namespace=namespace,show_progress_bar=True)

                    records['demo'][user]['indexed'].append(idx_fp)
                    with open('records.json','w') as f:
                        json.dump(records,f,indent=4)

                except Exception as e:
                    print(e)

        metadata_filter = {
            "title": {"$in":[idx_fp+'.pdf' for idx_fp in st.session_state['selected_files']]}
        }
        context_engine = ContextEngine(kb,global_metadata_filter=metadata_filter)
        chat_engine = ChatEngine(context_engine,allow_model_params_override=True)

    st.markdown("Chat with your document below")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1,col2 = st.columns([0.5,0.5],gap="small")
                
    with col1:
        pdf_list = [item for item in os.listdir('./') if '.pdf' in item]
        dct = {}
        for item in pdf_list:
            idx = item.replace('.pdf','')
            idx = idx.replace(' ','-').replace('_','-').lower()
            dct[idx] = item

        prompt = st.chat_input("What's on your mind?")
    
        with stylable_container(
            key="container_with_border",
            css_styles="""
                {
                    min-height: 800px;
                    max-height: 800px;
                    background-color: white;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    min-width: 600px;
                    max_width: 600px;
                    overflow-y: auto
                }
                """,
            ):
                for m in st.session_state.messages:
                    if m['role'] == 'user':
                        with st.chat_message("user"):
                            st.markdown(m['content'])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(m['content'])

                if prompt:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    messages = []
                    for m in st.session_state.messages:
                        if m['role'] == 'user':
                            messages.append(UserMessage(content=m['content']))
                        else:
                            messages.append(SystemMessage(content=m['content']))
                    res = chat_engine.chat(
                        messages=messages,
                        stream=False,
                        namespace=namespace,
                        model_params={'model':'gpt-4-0125-preview','temperature':0,'seed':42})
                    
                    ans = res.choices[0].message.content
                    sources = re.findall(f'Source: {dct[selected_file]}: page \d*',ans,re.IGNORECASE)
                    for s in sources:
                        ans = ans.replace(s,f'**{s}**')
                    with st.chat_message("assistant"):
                        st.markdown(ans) 
                    st.session_state.messages.append({"role": "assistant", "content": ans})
    with col2:
        pdf_list = [item for item in os.listdir('./') if '.pdf' in item]
        dct = {}
        for item in pdf_list:
            idx = item.replace('.pdf','')
            idx = idx.replace(' ','-').replace('_','-').lower()
            dct[idx] = item
        if 'selected_files' in st.session_state.keys() and len(st.session_state['selected_files']) > 0:
            selected_file = st.session_state['selected_files'][0]
            pdf_viewer(dct[selected_file],height=800,width=600)
