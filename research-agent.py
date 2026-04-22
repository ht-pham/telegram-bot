"""
Mini-project B: RA Agent (smolagents version)

An local LLM agent that can answer questions about:
- search_papers(query: str) -> List of paper abstracts and returns top matches
- summarize_paper(abstract: str) -> a concise summary of the paper's main contributions and findings
- extract_keywords(abstract: str) -> a list of key keywords that capture the main topics and themes of the paper
- compare_papers(abstract1: str, abstract2: str) -> a comparison of the two papers, highlighting their similarities and differences 
in terms of contributions, methodologies, and findings

Run locally:
  python research-agent.py

Run via Telegram:
  export TELEGRAM_BOT_TOKEN=your-token-here
  python research-agent.py --telegram
"""

import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype="auto",
)

#------------ Mock data of papers ------------
PAPERS = {
    "paper1": {
        "title": "A Novel Approach to Natural Language Processing",
        "abstract": "This paper proposes a novel approach to natural language processing (NLP) that leverages deep learning techniques to achieve state-of-the-art performance on various NLP tasks. The proposed method utilizes a transformer-based architecture and incorporates a large-scale pretraining strategy to enhance the model's ability to understand and generate human language. Experimental results demonstrate that our approach outperforms existing methods on several benchmark datasets, including sentiment analysis, machine translation, and question answering. The paper also discusses the implications of our findings for future research in NLP and potential applications in real-world scenarios."
    },
    "paper2": {
        "title": "An Efficient Algorithm for Large-Scale Graph Processing",
        "abstract": "In this paper, we present an efficient algorithm for processing large-scale graphs that significantly reduces computational complexity while maintaining high accuracy. Our algorithm is based on a novel graph partitioning technique that allows for parallel processing of graph data across multiple computing nodes. We evaluate our method on several large graph datasets, including social networks and web graphs, and demonstrate that it achieves superior performance compared to existing graph processing algorithms. The results indicate that our approach can handle graphs with billions of edges while providing accurate results in a fraction of the time required by traditional methods."
    },
    "paper3":{
        "title": "A Comprehensive Survey of Deep Learning in Computer Vision",
        "abstract": "This survey paper provides a comprehensive overview of deep learning techniques applied to computer vision tasks. We review the evolution of deep learning architectures, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformer-based models, and their applications in various computer vision domains such as image classification, object detection, and image generation. The paper also discusses the challenges and future directions in deep learning for computer vision, including issues related to model interpretability, data efficiency, and real-time processing. We conclude by highlighting the potential impact of deep learning on the future of computer vision research and applications."
    },
    "paper4": {
        "title": "Attention is All You Need: A Transformer-Based Approach to NLP",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
    },
    "paper5": {
        "title": "Efficient Large Graph Processing with Chunk-Based Graph Representation Model",
        "abstract": "Existing external graph processing systems face challenges in terms of low I/O efficiency, expensive computation overhead, and high graph algorithm development costs when running on emerging NVMe SSDs, due to their reliance on complex loading and computing models that aim to convert numerous random I/Os into a few sequential I/Os. While in-memory graph systems working with memory-storage cache systems like OS page cache or TriCache, offer a promising solution for large graph processing with fine-grained I/Os and easy algorithm programming, they often overlook the specific characteristics of graph applications, resulting in inefficient graph processing. To address these challenges, we introduce ChunkGraph, an I/O-efficient graph system designed for processing large-scale graphs on NVMe SSDs. ChunkGraph introduces a novel chunk-based graph representation model, featuring classified and hierarchical vertex storage, and efficient chunk layout optimization. Evaluations show that ChunkGraph can outperform existing external graph systems, as well as in-memory graph systems relying on general cache systems, running several times faster."
    },
    "paper6":{
        "title": "Sequence to Sequence Learning with Neural Networks",
        "abstract": "Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT'14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM's performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier."
    },
    "paper7":{
        "title": "Small Language Models are the Future of Agentic AI",
        "abstract":"Here we lay out the position that small language models (SLMs) are sufficiently powerful, inherently more suitable, and necessarily more economical for many invocations in agentic systems, and are therefore the future of agentic AI. Our argumentation is grounded in the current level of capabilities exhibited by SLMs, the common architectures of agentic systems, and the economy of LM deployment. We further argue that in situations where general-purpose conversational abilities are essential, heterogeneous agentic systems (i.e., agents invoking multiple different models) are the natural choice. We discuss the potential barriers for the adoption of SLMs in agentic systems and outline a general LLM-to-SLM agent conversion algorithm."
    } 
}
#------------ End of mock data ------------

#------Tools for the RA agent------

def search_papers(query: str):
    """
    Search for papers relevant to the query and return their abstracts.
    
    Args:
        query: The search query string to find relevant papers
    """
    results = []
    for paper_id, paper in PAPERS.items():
        if re.search(query, paper["title"], re.IGNORECASE) or re.search(query, paper["abstract"], re.IGNORECASE):
            results.append(paper["abstract"])
    return json.dumps(results)

def summarize_paper(abstract: str):
    """
    Summarize the paper's main contributions and findings.
    
    Args:
        abstract: The paper abstract text to summarize
    """
    prompt = f"Summarize the following paper abstract:\n\n{abstract}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def extract_keywords(abstract: str):
    """
    Extract key keywords that capture the main topics and themes of the paper.
    
    Args:
        abstract: The paper abstract text to extract keywords from
    """
    prompt = f"Extract key keywords from the following paper abstract:\n\n{abstract}\n\nKeywords:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return keywords.split(", ")

def compare_papers(abstract1: str, abstract2: str):
    """
    Compare two papers, highlighting their similarities and differences in terms of contributions, methodologies, and findings.
    
    Args:
        abstract1: The first paper's abstract text
        abstract2: The second paper's abstract text
    """
    prompt = f"Compare the following two paper abstracts:\n\nPaper 1:\n{abstract1}\n\nPaper 2:\n{abstract2}\n\nComparison:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    comparison = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return comparison

TOOLS = {
    "search_papers": search_papers,
    "summarize_paper": summarize_paper,
    "extract_keywords": extract_keywords,
    "compare_papers": compare_papers
}

tool_functions = [search_papers, summarize_paper, extract_keywords, compare_papers]

SYSTEM_PROMPT = (
    "You are a research assistant agent that can help users with their research needs. You have access to the following tools:\n"
    "- search_papers(query: str) -> List of paper abstracts and returns top matches\n"
    "- summarize_paper(abstract: str) -> a concise summary of the paper's main contributions and findings\n"
    "- extract_keywords(abstract: str) -> a list of key keywords that capture the main topics and themes of the paper\n"
    "- compare_papers(abstract1: str, abstract2: str) -> a comparison of the two papers, highlighting their similarities and differences in terms of contributions, methodologies, and findings\n"
    "Use these tools to answer user queries about research papers. Always use the tools when appropriate to provide accurate and helpful responses."
)

#---- End of tools for the RA agent ----

#---- Agent loop for RA agent ----
def run_agent(user_query: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}]

    for _ in range(5):
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tool_functions,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)

        if not tool_calls:
            return response

        messages.append({"role": "assistant", "content": response})

        for tool_call in tool_calls:
            try:
                tool_call_data = json.loads(tool_call)
                tool_name = tool_call_data.get("name") or tool_call_data.get("tool_name", "")
                tool_args = tool_call_data.get("arguments",{})
                if tool_name in TOOLS:
                    tool_result = TOOLS[tool_name](**tool_args)
                    messages.append({"role": "tool", "content": f"Tool: {tool_name}\nResult: {tool_result}"})
                else:
                    messages.append({"role": "tool", "content": f"Tool: {tool_name}\nResult: Tool not found."})
            except json.JSONDecodeError:
                continue

            print(f"Tool call: {tool_name} with args {tool_args} returned result: {tool_result}")
            messages.append({"role": "assistant", "content": response})

    return response

# ------ CLI mode ------------

def run_cli():
    print("Welcome to the Research Assistant Agent! Type 'exit' to quit.")
    print("You can ask questions about: research papers, summaries, keywords, and comparisons.")
    print("For example:\n")
    print("- Search for papers on graph processing")
    print("- Summarize the paper with abstract: ...")
    print("- Extract keywords from the paper with abstract: ...")
    print("- Compare the following two papers with abstracts: ... and ...\n")
    while True:
        user_query = input("\nEnter your research query: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = run_agent(user_query)
        print(f"Agent response: {response}")

#------ End of CLI mode ------------

#------ Telegram bot mode ----------
def run_telegram_bot():
    import asyncio
    import os
    from telegram import Update
    from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        return
    
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        user_text = update.message.text
        # Display text message from user
        print(f"Received: {user_text}")
        print("Calling agent tool")

        try:
            #reply = run_agent(user_text)
            reply = await asyncio.to_thread(run_agent,user_text)
            print(f"Reply: {reply}")
            await update.message.reply_text(reply)
        except Exception as e:
            print(f"Error: {e}")
            await update.message.reply_text("Something went wrong")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Research Assistant Telegram bot is running.")
    app.run_polling(drop_pending_updates=True)

#------ End of Telegram bot mode ----------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--telegram":
        run_telegram_bot()
    else:
        run_cli()