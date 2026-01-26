#This function provides the response in text for user query 
from langchain_classic.retrievers import MergerRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_openai import  ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate



def get_open_rag_chain(knn, embed_model, gpt_model, vector_dir, creativity):
    diseases=["Anaemia","Asthma","Covid-19","Dengue","Diabetes","GBS","HyperTension","Malaria","Nipah","Tuberculosis","Typhoid"]
    vectorstore=[]
    for disease in diseases:
        vector_dir_disease=vector_dir / disease
        try:
            embedding_function = OpenAIEmbeddings(model=embed_model)
            v_db = Chroma(collection_name=disease,persist_directory=vector_dir_disease.as_posix(), embedding_function=embedding_function)
            vectorstore.append(v_db)
        except Exception as e:
            print(f"\nSYSTEM ERROR: {e}")
    retriever_l=[]
    for vector in vectorstore:
        retriever_l.append(vector.as_retriever(search_kwargs={"k": knn}))
    lotr = MergerRetriever(retrievers=retriever_l)
    llm = ChatOpenAI(model=gpt_model,
                    temperature=creativity)
    system_prompt = (
        "You are an assistant. First answer the question using provided context. "
        "If no context found. Use your own internal knowledge."
        "\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    # This creates the logic to process documents
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # This connects the retriever (Chroma) to the logic
    rag_chain = create_retrieval_chain(lotr, combine_docs_chain)
    return rag_chain