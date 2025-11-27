
import os
os.environ["GOOGLE_API_KEY"] = "API_KEY"

!pip install langchain chromadb google-generativeai tiktoken pypdf langchain_google_genai langchain-community langchain-chroma langchain_huggingface

!pip install sentence-transformers

from langchain_chroma import Chroma  #ab "from langchain.vectorstores import Chroma" nhi balki ye karna hai upyog
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document

#create Langchain documents for IPL players

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons. ",
    metadata={'team':"Royal Challangers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={'team':"Mumbai Indians"}
)

doc3= Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={'team':"Chennai Super Kings"}
)

doc4= Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1,doc2,doc3,doc4,doc5]

vector_store = Chroma(
    embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="meri_chroma_db",   #isse files me ek directory ya folder ban jayega iss naam ka jisse download kar sakte hai
    collection_name='sample' #iska kaam mughe nhi pata
)

#add documents
vector_store.add_documents(docs)

#view documents
vector_store.get(include=['embeddings','documents','metadatas'])

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler',
    k=2 #kitne jawab chahiye
)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={'team':"Royal Challangers Bangalore"}
)
vector_store.update_document(document_id='960cdfd8-7d23-4117-bd98-f560e7541991', document=updated_doc1)

#view ḍocuments
vector_store.get(include=['embeddings','documents','metadatas'])

# delete document
vector_store.delete(ids=['960cdfd8-7d23-4117-bd98-f560e7541991'])

#view
vector_store.get(include=['embeddings','documents','metadatas'])

