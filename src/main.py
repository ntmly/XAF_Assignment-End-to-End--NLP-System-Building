import yaml
import argparse
import torch
import json
import os
from sentence_transformers import SentenceTransformer

from query_processing import preprocess_query
from indexer import build_index_from_chunks, DualIndexer
from retriever import HybridRetriever
from reranker import CPUReranker
# from reader import CPUReader
from generative_reader import GenerativeReader

def load_knowledge_base_fallback() -> list:
    """Fallback knowledge base nếu không có file chunks"""
    return [
        "Vietnam National University, Hanoi (VNU) was established in 1993.",
        "The University of Engineering and Technology (UET) is a member of VNU.",
        "Carnegie Mellon University was founded in 1900 by Andrew Carnegie in Pittsburgh.",
        "Pittsburgh is named after William Pitt, the 1st Earl of Chatham.",
        "The first ICML conference was held in Pittsburgh in 1980."
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="advanced_full",
                        choices=["baseline0", "dense_only", "advanced_full"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default="data/test/questions.txt")
    parser.add_argument("--output", type=str, default="system_outputs/system_output_1.txt")
    parser.add_argument("--debug", action="store_true", help="In ra các chunk được retrieve")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # CPU threads
    torch.set_num_threads(config['project_setup']['num_threads'])
    
    # Load knowledge base (từ chunks nếu có, hoặc fallback)
    chunks_path = "data/processed/chunks.json"
    if os.path.exists(chunks_path):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        knowledge_base = [f"{chunk.get('title', '')}\n{chunk['text']}" for chunk in chunks]
        print(f"Loaded {len(knowledge_base)} chunks from {chunks_path}")
    else:
        knowledge_base = load_knowledge_base_fallback()
        print(f"Using fallback knowledge base with {len(knowledge_base)} documents")
    
    # Load models
    print("Loading embedder...")
    embedder = SentenceTransformer(config['models']['dense_embedder'])
    
    print("Building dual index...")
    # Luôn tạo index từ knowledge_base (dù có chunks.json hay không)
    indexer = DualIndexer(knowledge_base, embedder)
    
    retriever = HybridRetriever(indexer, embedder, alpha=config['hyperparameters']['hybrid_alpha'])
    reranker = CPUReranker(config['models']['reranker'])
    reader = GenerativeReader(config['models']['reader_qa'])
    
    # Đọc câu hỏi
    with open(args.input, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    answers = []
    for q in questions:
        processed_q = preprocess_query(q)
        
        if args.mode == "baseline0":
            ans = reader.answer(processed_q, [])
        elif args.mode == "dense_only":
            q_emb = embedder.encode([processed_q], convert_to_tensor=False)[0].astype('float32')
            _, indices = indexer.search_faiss(q_emb, config['hyperparameters']['top_k_hybrid'])
            retrieved_chunks = [knowledge_base[i] for i in indices if i != -1]
            # Không cắt theo top_k_rerank nếu muốn giữ nhiều hơn
            retrieved_chunks = retrieved_chunks[:config['hyperparameters']['top_k_rerank']]
            ans = reader.answer(processed_q, retrieved_chunks)
        else:  # advanced_full
            # 1. Retrieve top-k từ hybrid
            retrieved_chunks = retriever.retrieve(processed_q, top_k=config['hyperparameters']['top_k_hybrid'])
            if args.debug:
                print(f"\n[DEBUG] Query: {processed_q}")
                print(f"[DEBUG] Top {len(retrieved_chunks)} chunks before rerank:")
                for i, c in enumerate(retrieved_chunks):
                    print(f"  [{i}] {c[:200]}...")
            
            # 2. Rerank (có thể bỏ qua nếu cần)
            if config['hyperparameters'].get('use_reranker', True):
                reranked = reranker.rerank(processed_q, retrieved_chunks, top_k=config['hyperparameters']['top_k_rerank'])
                if args.debug:
                    print(f"[DEBUG] Top {len(reranked)} chunks after rerank:")
                    for i, c in enumerate(reranked):
                        print(f"  [{i}] {c[:200]}...")
                final_chunks = reranked
            else:
                final_chunks = retrieved_chunks[:config['hyperparameters']['top_k_rerank']]
            
            # 3. Reader
            ans = reader.answer(processed_q, final_chunks)
        
        answers.append(ans)
        print(f"Q: {q}\nA: {ans}\n")
    
    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for ans in answers:
            f.write(ans + '\n')
    
    print(f"Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()