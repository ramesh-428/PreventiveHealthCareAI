#This function provides the response in text for user query 
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import  ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.retrievers import MergerRetriever
def answer_question(vectorstore, query,knn, gpt_model,creativity):
    retriever_l=[]
    for vector in vectorstore:
        retriever_l.append(vector.as_retriever(search_kwargs={"k": knn}))
    lotr = MergerRetriever(retrievers=retriever_l)
    system_prompt = (
        "You are a strict assistant. Answer the user's question using ONLY "
        "the provided context below. Do not use your own internal knowledge. "
        "If the answer is not contained within the context, exactly say: "
        "'I am Sorry. I can't find the answer for this. I can provide answer for below diseases.\n Anaemia,Asthma,Covid-19,Dengue,Diabetes,HyperTension,Malaria,Tuberculosis and Typhoid' "
        "\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=gpt_model)
    # This creates the logic to process documents
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # This connects the retriever (Chroma) to the logic
    rag_chain = create_retrieval_chain(lotr, combine_docs_chain)

    # 4. Question & Answer
    response = rag_chain.invoke({"input": query})
    return response['answer']