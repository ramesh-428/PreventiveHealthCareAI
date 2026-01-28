#This function provides the response in text for user query 
from langchain_classic.retrievers import MergerRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_openai import  ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_strict_rag_chain(knn, embed_model, gpt_model, vector_dir, creativity,max_output):
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
        "You are a strict medical information assistant. "
    "Use ONLY the provided context to answer the user's question. "
    "\n\n"
    "### RULES:\n"
    "1. If the context contains the answer, provide it clearly using below rules.\n"
   f"1.Maximum output characters must not exceed {max_output}"
    "2.Always include preface with brief introduction about disease"
    "3.If answer contains bullet points then include include description of the bullet point."
    "4.Always provide follow up question to the provided answer"
    "2. ONLY if the context does not contain the answer at all, say exactly: "
    "'I am Sorry. I can't find the answer for this. However, I can provide answer for the following diseases: "
    "Anaemia, Asthma, Covid-19, Dengue, Diabetes, GBS, HyperTension, Malaria, Nipah, Tuberculosis, and Typhoid.'\n"
    "\n"
    "### CONTEXT:\n"
    "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    # This creates the logic to process documents
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # This connects the retriever (Chroma) to the logic
    rag_chain = create_retrieval_chain(lotr, combine_docs_chain)
    return rag_chain