import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
import fitz
import numpy as np
import pandas as pd
import torch
import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from tqdm import tqdm

class RAGEngine:
    def __init__(self, pdf_path="RBIdoc.pdf", csv_cache_path="text_chunks_and_embeddings_df.csv", model_name="all-mpnet-base-v2"):
        self.pdf_path = pdf_path
        self.csv_cache_path = csv_cache_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load embedding model
        print(f"[INFO] Loading embedding model '{self.model_name}' on device: {self.device}...")
        self.embedding_model = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)
        
        # Internal state
        self.pages_and_chunks = []
        self.embeddings = None
        
        # Initialize database (load from cache or build)
        self.init_database()

    def init_database(self):
        if self.csv_cache_path and os.path.exists(self.csv_cache_path):
            print(f"[INFO] Found cached embeddings at '{self.csv_cache_path}'. Loading...")
            try:
                loaded_df = pd.read_csv(self.csv_cache_path)
                # Convert string representations of list back to numpy arrays
                loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=", "))
                self.pages_and_chunks = loaded_df.to_dict(orient="records")
                self.embeddings = torch.tensor(np.array(loaded_df["embedding"].tolist()), dtype=torch.float32).to(self.device)
                print(f"[INFO] Loaded {len(self.pages_and_chunks)} chunks with embeddings matrix shape: {self.embeddings.shape}")
                return
            except Exception as e:
                print(f"[ERROR] Failed to load cache: {e}. Rebuilding database from PDF...")

        print(f"[INFO] Cache not found or invalid. Processing PDF '{self.pdf_path}'...")
        self.build_database_from_pdf()

    def build_database_from_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF document not found at '{self.pdf_path}'. Please make sure it is in the workspace.")

        # 1. Read PDF page-by-page
        print("[INFO] Parsing PDF pages...")
        doc = fitz.open(self.pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc), total=len(doc)):
            text = page.get_text()
            # Basic formatting
            cleaned_text = text.replace("\n", " ").strip()
            pages_and_texts.append({
                "page_number": page_number,
                "text": cleaned_text
            })

        # 2. Sentence segmentation with spaCy English sentencizer
        print("[INFO] Segmenting text into sentences...")
        nlp = English()
        nlp.add_pipe("sentencizer")
        
        for item in tqdm(pages_and_texts):
            sentences = list(nlp(item["text"]).sents)
            item["sentences"] = [str(s) for s in sentences]

        # 3. Create Chunks (groups of 10 sentences)
        print("[INFO] Creating chunks...")
        num_sentence_chunk_size = 10
        raw_chunks = []
        for item in tqdm(pages_and_texts):
            sentences = item["sentences"]
            sentence_chunks = [sentences[i:i + num_sentence_chunk_size] for i in range(0, len(sentences), num_sentence_chunk_size)]
            
            for chunk in sentence_chunks:
                joined_chunk = "".join(chunk).replace("  ", " ").strip()
                # Clean up spacing around sentence endings
                joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
                
                # Compute token and char metrics
                char_count = len(joined_chunk)
                token_count = char_count / 4
                word_count = len(joined_chunk.split(" "))
                
                # Filter out very short chunks (e.g. <= 30 tokens)
                if token_count > 30:
                    raw_chunks.append({
                        "page_number": item["page_number"],
                        "sentence_chunk": joined_chunk,
                        "chunk_char_count": char_count,
                        "chunk_word_count": word_count,
                        "chunk_token_count": token_count
                    })

        # 4. Generate Embeddings
        print(f"[INFO] Generating embeddings for {len(raw_chunks)} chunks in batches (Device: {self.device})...")
        text_chunks = [c["sentence_chunk"] for c in raw_chunks]
        
        # Run encode
        embeddings_tensor = self.embedding_model.encode(
            text_chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Save to internal state
        self.embeddings = embeddings_tensor
        for i, item in enumerate(raw_chunks):
            # Store embedding list representation
            item["embedding"] = embeddings_tensor[i].cpu().tolist()
            
        self.pages_and_chunks = raw_chunks
        
        # 5. Write cache
        if self.csv_cache_path:
            print(f"[INFO] Saving processed chunks and embeddings to '{self.csv_cache_path}'...")
            df = pd.DataFrame(raw_chunks)
            df.to_csv(self.csv_cache_path, index=False)
            
            # Convert embeddings column in-place back to array for internal use
            df["embedding"] = df["embedding"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
            self.embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(self.device)
            print(f"[INFO] Database built successfully. Saved {len(raw_chunks)} chunks.")
        else:
            print(f"[INFO] Database built in-memory successfully. Loaded {len(raw_chunks)} chunks.")

    def retrieve(self, query: str, k=5):
        """
        Retrieves the top-k most semantically relevant chunks for a given query.
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        scores, indices = torch.topk(input=dot_scores, k=k)
        
        results = []
        for score, index in zip(scores, indices):
            idx = int(index.cpu().item())
            results.append({
                "page_number": self.pages_and_chunks[idx]["page_number"],
                "sentence_chunk": self.pages_and_chunks[idx]["sentence_chunk"],
                "score": float(score.cpu().item())
            })
        return results

    def format_prompt(self, query: str, context_items: list[dict]) -> str:
        """
        Builds the regulatory few-shot prompt.
        """
        context = "- " + "\n- ".join([f"[Page {item['page_number']}] {item['sentence_chunk']}" for item in context_items])
        
        base_prompt = """Based on the following regulatory context from the Reserve Bank of India (RBI) NBFC Master Directions, please answer the query.
Make sure your answers are explanatory, professional, and reference specific sections or terms if present in the context.
Do not hallucinate or guess. If the answer cannot be found in the provided context, state that the context does not contain sufficient information to answer the question, but try your best to answer based on the closest matched context.

Example 1:
Query: What is the regulatory minimum Capital Adequacy Ratio (CAR) for NBFCs?
Answer: According to RBI guidelines, non-banking financial companies (NBFCs) are required to maintain a minimum Capital to Risk-Weighted Assets Ratio (CRAR) or Capital Adequacy Ratio (CAR). For most systemically important non-deposit taking NBFCs and deposit-taking NBFCs, the regulatory minimum is set at 15%. This capital adequacy framework is designed to ensure that the NBFC retains a sufficient capital buffer against operational and credit risks, thereby preserving financial stability within the fintech and lending ecosystem.

Example 2:
Query: What is the Upper Layer classification criteria?
Answer: The Upper Layer (NBFC-UL) of non-banking financial companies comprises those entities which are specifically identified by the RBI as warranting enhanced regulatory discipline. This identification is based on a set of parameters including size, leverage, nature of operations, connectivity to the financial system, and group structure. Typically, the top 10 eligible NBFCs by asset size are automatically classified in the Upper Layer, while others are selected based on a multi-factor scoring methodology covering interconnectedness, complexity, and systemic risk.

Now use the following context items to answer the user query:
{context}

User query: {query}
Answer:"""
        return base_prompt.format(context=context, query=query)

    def generate_answer_local(self, query: str, context_items: list[dict], model_id="Qwen/Qwen2.5-0.5B-Instruct") -> str:
        """
        Generates an answer using a local causal language model running on CPU.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Lazily load local LLM model and tokenizer to save memory if not used
            if not hasattr(self, "local_tokenizer") or not hasattr(self, "local_llm"):
                print(f"[INFO] Lazily loading local LLM '{model_id}' on CPU...")
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.local_llm = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to("cpu")
            
            prompt = self.format_prompt(query, context_items)
            
            # Format prompt for the Qwen Chat model
            messages = [
                {"role": "system", "content": "You are a regulatory compliance assistant. Answer the user query using only the provided context."},
                {"role": "user", "content": prompt}
            ]
            text = self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.local_tokenizer([text], return_tensors="pt").to("cpu")
            
            print("[INFO] Generating response locally on CPU...")
            generated_ids = self.local_llm.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract generated response only
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            return f"Error during local model generation: {str(e)}"

    def generate_answer(self, query: str, context_items: list[dict], api_key: str = None) -> str:
        """
        Generates an answer using the formatted prompt and Gemini API.
        """
        # Determine API key to use
        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not effective_key:
            return "Error: Gemini API Key not set. Please set the GEMINI_API_KEY environment variable or enter it in the app sidebar to enable answers."

        try:
            genai.configure(api_key=effective_key)
            prompt = self.format_prompt(query, context_items)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during Gemini generation: {str(e)}"

    def query(self, query: str, k=5, api_key: str = None, use_local_llm: bool = False) -> tuple[str, list[dict]]:
        """
        Performs the complete RAG loop: Retrieve -> Augment -> Generate.
        """
        context_items = self.retrieve(query, k=k)
        if use_local_llm:
            answer = self.generate_answer_local(query, context_items)
        else:
            answer = self.generate_answer(query, context_items, api_key=api_key)
        return answer, context_items
