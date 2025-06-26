from sentence_transformers import SentenceTransformer, util
import numpy as np
import asyncio
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import torch
import os

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    """Get the best available device for computation with safe fallback"""
    try:
        if torch.cuda.is_available():
            # Test CUDA is actually working
            test_tensor = torch.tensor([1.0]).cuda()
            return "cuda"
    except Exception as e:
        print(f"⚠️ CUDA available but failed test: {e}, falling back to CPU")
    
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Test MPS is actually working
            test_tensor = torch.tensor([1.0]).to('mps')
            return "mps"
    except Exception as e:
        print(f"⚠️ MPS available but failed test: {e}, falling back to CPU")
    
    return "cpu"

# Global device and model instance - load once with GPU support
DEVICE = get_device()
print(f"🔧 RAG Metrics using device: {DEVICE}")

try:
    # Try to load model with specified device
    if DEVICE == "cpu":
        model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        print(f"✅ SentenceTransformer loaded successfully on CPU")
    else:
        # Load on CPU first, then move to target device to avoid meta tensor issues
        model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        try:
            # Use proper method to move model to target device
            if hasattr(model, 'to'):
                model = model.to(DEVICE)
            elif hasattr(model, '_modules') and hasattr(model, '_target_device'):
                model._target_device = DEVICE
                for module in model._modules.values():
                    if hasattr(module, 'to'):
                        module.to(DEVICE)
            print(f"✅ SentenceTransformer loaded and moved to {DEVICE}")
        except Exception as e:
            print(f"⚠️ Failed to move SentenceTransformer to {DEVICE}: {e}")
            print("🔄 Using CPU instead...")
            DEVICE = "cpu"  # Update global device to CPU
            # Model is already on CPU, so no need to reload
            print("✅ SentenceTransformer using CPU")
except Exception as e:
    print(f"⚠️ Failed to load SentenceTransformer: {e}")
    print("🔄 Falling back to basic CPU loading...")
    try:
        DEVICE = "cpu"
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Basic loading without device param
        print("✅ SentenceTransformer loaded with basic CPU fallback")
    except Exception as e2:
        print(f"❌ Critical error loading SentenceTransformer: {e2}")
        raise e2

def vectorized_semantic_similarity(retrieved_embs, gt_embs, threshold=0.8):
    """Vectorized semantic similarity calculation with GPU support and meta tensor handling"""
    if len(retrieved_embs) == 0 or len(gt_embs) == 0:
        return False
    
    try:
        # Convert to tensors if they're numpy arrays
        def to_tensor(emb):
            if isinstance(emb, np.ndarray):
                return torch.from_numpy(emb.copy())  # Ensure we have actual data, not meta
            elif isinstance(emb, torch.Tensor):
                if emb.is_meta:  # Handle meta tensors
                    return torch.zeros_like(emb, device='cpu')  # Create real tensor on CPU
                return emb.clone()  # Clone to avoid reference issues
            else:
                return torch.tensor(emb, dtype=torch.float32)
        
        # Stack embeddings on CPU first to avoid meta tensor issues
        ret_stack = torch.stack([to_tensor(emb) for emb in retrieved_embs]).cpu()
        gt_stack = torch.stack([to_tensor(emb) for emb in gt_embs]).cpu()
        
        # Try to move to target device if not CPU
        if DEVICE != "cpu":
            try:
                # Check if tensors are valid before moving
                if not ret_stack.is_meta and not gt_stack.is_meta:
                    ret_stack = ret_stack.to(DEVICE)
                    gt_stack = gt_stack.to(DEVICE)
                else:
                    print("⚠️ Meta tensors detected, using CPU for similarity calculation")
            except Exception as e:
                # If device transfer fails, stay on CPU
                ret_stack = ret_stack.cpu()
                gt_stack = gt_stack.cpu()
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(ret_stack, gt_stack)
        
        # Check threshold
        return (similarity_matrix >= threshold).any().item()
        
    except Exception as e:
        # Comprehensive fallback to original CPU-only implementation
        print(f"⚠️ GPU similarity calculation failed: {e}, using CPU fallback")
        try:
            # Safe CPU fallback
            def safe_to_tensor(emb):
                if isinstance(emb, np.ndarray):
                    return torch.from_numpy(emb.copy()).cpu()
                elif isinstance(emb, torch.Tensor):
                    return emb.cpu().clone()
                else:
                    return torch.tensor(emb, dtype=torch.float32).cpu()
            
            ret_stack = torch.stack([safe_to_tensor(emb) for emb in retrieved_embs])
            gt_stack = torch.stack([safe_to_tensor(emb) for emb in gt_embs])
            similarity_matrix = util.cos_sim(ret_stack, gt_stack)
            return (similarity_matrix >= threshold).any().item()
        except Exception as e2:
            print(f"❌ Even CPU fallback failed: {e2}")
            # Last resort: use original simple implementation
            for ret_emb in retrieved_embs:
                for gt_emb in gt_embs:
                    try:
                        if isinstance(ret_emb, np.ndarray) and isinstance(gt_emb, np.ndarray):
                            ret_tensor = torch.from_numpy(ret_emb.copy())
                            gt_tensor = torch.from_numpy(gt_emb.copy())
                            score = util.cos_sim(ret_tensor, gt_tensor).item()
                            if score >= threshold:
                                return True
                    except:
                        continue
            return False

# Global functions for multiprocessing
def process_chunk_global(chunk_data, threshold=0.8):
    """Global function for processing chunks in parallel"""
    results = []
    for retrieved_embs, gt_embs in chunk_data:
        hit = vectorized_semantic_similarity(retrieved_embs, gt_embs, threshold)
        results.append(hit)
    return results

def process_chunk_mrr_global(chunk_data, threshold=0.8):
    """Global function for processing MRR chunks in parallel"""
    results = []
    for retrieved_embs, gt_embs in chunk_data:
        mrr = vectorized_semantic_mrr(retrieved_embs, gt_embs, threshold)
        results.append(mrr)
    return results

def semantic_hit(retrieved_docs, ground_truth_docs, top_k=5, threshold=0.8):
    """Kiểm tra semantic similarity với cosine @top_k - Original function kept for compatibility"""
    retrieved_docs = retrieved_docs[:top_k]
    for ret in retrieved_docs:
        for gt in ground_truth_docs:
            score = util.cos_sim(model.encode(ret), model.encode(gt)).item()
            if score >= threshold:
                return True
    return False

async def async_semantic_hit_rate(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', top_k=5, threshold=0.8):
    """Async version của semantic hit rate calculation"""
    
    # 1. Collect all unique documents
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col][:top_k])
        all_docs.update(row[ground_truth_col])
    
    # 2. Async batch encoding
    async def batch_encode_documents():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            unique_docs = list(all_docs)
            # Run encoding in thread pool to not block event loop
            embeddings = await loop.run_in_executor(
                executor, 
                partial(model.encode, unique_docs, batch_size=32, show_progress_bar=True)
            )
            return dict(zip(unique_docs, embeddings))
    
    doc_embeddings = await batch_encode_documents()
    
    # 3. Async processing each row
    async def process_row_async(row):
        retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col][:top_k]]
        gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
        
        # Calculate similarity in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                partial(vectorized_semantic_similarity, retrieved_embs, gt_embs, threshold)
            )
    
    # 4. Process all rows in parallel
    tasks = [process_row_async(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    df['semantic_hit'] = results
    return df['semantic_hit'].mean()

def parallel_semantic_hit_rate(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', top_k=5, threshold=0.8, n_jobs=-1):
    """Parallel version với multiprocessing"""
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # 1. Pre-encode all documents
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col][:top_k])
        all_docs.update(row[ground_truth_col])
    
    unique_docs = list(all_docs)
    embeddings = model.encode(unique_docs, batch_size=32, show_progress_bar=True)
    doc_embeddings = dict(zip(unique_docs, embeddings))
    
    # 2. Prepare data for parallel processing
    def prepare_row_data(row):
        retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col][:top_k]]
        gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
        return retrieved_embs, gt_embs
    
    row_data = [prepare_row_data(row) for _, row in df.iterrows()]
    
    # 3. Chunk data and process in parallel using global function
    chunk_size = max(1, len(row_data) // n_jobs)
    chunks = [row_data[i:i+chunk_size] for i in range(0, len(row_data), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        chunk_results = list(executor.map(partial(process_chunk_global, threshold=threshold), chunks))
    
    # 4. Flatten results
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)
    
    df['semantic_hit'] = all_results
    return df['semantic_hit'].mean()

async def hybrid_semantic_hit_rate(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', top_k=5, threshold=0.8):
    """Hybrid approach combining async I/O and parallel processing"""
    
    # 1. Async document collection and encoding
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col][:top_k])
        all_docs.update(row[ground_truth_col])
    
    async def async_encode():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            unique_docs = list(all_docs)
            embeddings = await loop.run_in_executor(
                executor,
                lambda: model.encode(unique_docs, batch_size=64, show_progress_bar=True)
            )
            return dict(zip(unique_docs, embeddings))
    
    doc_embeddings = await async_encode()
    
    # 2. Prepare data for parallel processing
    def prepare_batch_data(batch_df):
        batch_data = []
        for _, row in batch_df.iterrows():
            retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col][:top_k]]
            gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
            batch_data.append((retrieved_embs, gt_embs))
        return batch_data
    
    # 3. Split DataFrame into batches
    n_jobs = mp.cpu_count()
    batch_size = max(1, len(df) // n_jobs)
    df_batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    # 4. Async parallel processing using global function
    async def process_batches_async():
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            tasks = []
            for batch_df in df_batches:
                batch_data = prepare_batch_data(batch_df)
                task = loop.run_in_executor(executor, partial(process_chunk_global, threshold=threshold), batch_data)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks)
            return batch_results
    
    batch_results = await process_batches_async()
    
    # 5. Flatten results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    df['semantic_hit'] = all_results
    return df['semantic_hit'].mean()

def calculate_semantic_hit_rate(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', top_k=5, threshold=0.8):
    """Tính semantic Hit Rate @ K bằng cosine similarity - Original function kept for compatibility"""
    df['semantic_hit'] = df.apply(
        lambda row: semantic_hit(row[retrieved_col], row[ground_truth_col], top_k, threshold),
        axis=1
    )
    return df['semantic_hit'].mean()

def semantic_rr(retrieved_docs, ground_truth_docs, threshold=0.8):
    """Tính Reciprocal Rank theo semantic similarity - Original function kept for compatibility"""
    for i, ret in enumerate(retrieved_docs):
        for gt in ground_truth_docs:
            score = util.cos_sim(model.encode(ret), model.encode(gt)).item()
            if score >= threshold:
                return 1.0 / (i + 1)
    return 0.0

async def async_semantic_mrr(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', threshold=0.8):
    """Async version của semantic MRR calculation"""
    
    # 1. Collect all unique documents
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col])
        all_docs.update(row[ground_truth_col])
    
    # 2. Async batch encoding
    async def batch_encode_documents():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            unique_docs = list(all_docs)
            embeddings = await loop.run_in_executor(
                executor, 
                partial(model.encode, unique_docs, batch_size=32, show_progress_bar=True)
            )
            return dict(zip(unique_docs, embeddings))
    
    doc_embeddings = await batch_encode_documents()
    
    # 3. Async processing each row for MRR
    async def process_row_mrr_async(row):
        retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col]]
        gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
        
        # Calculate MRR in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                partial(vectorized_semantic_mrr, retrieved_embs, gt_embs, threshold)
            )
    
    # 4. Process all rows in parallel
    tasks = [process_row_mrr_async(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    df['semantic_rr'] = results
    return df['semantic_rr'].mean()

def vectorized_semantic_mrr(retrieved_embs, gt_embs, threshold=0.8):
    """Vectorized MRR calculation with GPU support and meta tensor handling"""
    if len(retrieved_embs) == 0 or len(gt_embs) == 0:
        return 0.0
    
    try:
        # Convert to tensors if they're numpy arrays
        def to_tensor(emb):
            if isinstance(emb, np.ndarray):
                return torch.from_numpy(emb.copy())  # Ensure we have actual data, not meta
            elif isinstance(emb, torch.Tensor):
                if emb.is_meta:  # Handle meta tensors
                    return torch.zeros_like(emb, device='cpu')  # Create real tensor on CPU
                return emb.clone()  # Clone to avoid reference issues
            else:
                return torch.tensor(emb, dtype=torch.float32)
        
        # Stack embeddings on CPU first to avoid meta tensor issues
        ret_stack = torch.stack([to_tensor(emb) for emb in retrieved_embs]).cpu()
        gt_stack = torch.stack([to_tensor(emb) for emb in gt_embs]).cpu()
        
        # Try to move to target device if not CPU
        if DEVICE != "cpu":
            try:
                # Check if tensors are valid before moving
                if not ret_stack.is_meta and not gt_stack.is_meta:
                    ret_stack = ret_stack.to(DEVICE)
                    gt_stack = gt_stack.to(DEVICE)
                else:
                    print("⚠️ Meta tensors detected, using CPU for MRR calculation")
            except Exception as e:
                # If device transfer fails, stay on CPU
                ret_stack = ret_stack.cpu()
                gt_stack = gt_stack.cpu()
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(ret_stack, gt_stack)
        
        # Find first position where similarity >= threshold
        for i in range(len(retrieved_embs)):
            if (similarity_matrix[i] >= threshold).any():
                return 1.0 / (i + 1)
        
        return 0.0
        
    except Exception as e:
        # Comprehensive fallback to original CPU-only implementation
        print(f"⚠️ GPU MRR calculation failed: {e}, using CPU fallback")
        try:
            # Safe CPU fallback
            def safe_to_tensor(emb):
                if isinstance(emb, np.ndarray):
                    return torch.from_numpy(emb.copy()).cpu()
                elif isinstance(emb, torch.Tensor):
                    return emb.cpu().clone()
                else:
                    return torch.tensor(emb, dtype=torch.float32).cpu()
            
            ret_stack = torch.stack([safe_to_tensor(emb) for emb in retrieved_embs])
            gt_stack = torch.stack([safe_to_tensor(emb) for emb in gt_embs])
            similarity_matrix = util.cos_sim(ret_stack, gt_stack)
            for i in range(len(retrieved_embs)):
                if (similarity_matrix[i] >= threshold).any():
                    return 1.0 / (i + 1)
            return 0.0
        except Exception as e2:
            print(f"❌ Even CPU MRR fallback failed: {e2}")
            # Last resort: use original simple implementation
            for i, ret_emb in enumerate(retrieved_embs):
                for gt_emb in gt_embs:
                    try:
                        if isinstance(ret_emb, np.ndarray) and isinstance(gt_emb, np.ndarray):
                            ret_tensor = torch.from_numpy(ret_emb.copy())
                            gt_tensor = torch.from_numpy(gt_emb.copy())
                            score = util.cos_sim(ret_tensor, gt_tensor).item()
                            if score >= threshold:
                                return 1.0 / (i + 1)
                    except:
                        continue
            return 0.0

def parallel_semantic_mrr(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', threshold=0.8, n_jobs=-1):
    """Parallel version của semantic MRR calculation"""
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # 1. Pre-encode all documents
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col])
        all_docs.update(row[ground_truth_col])
    
    unique_docs = list(all_docs)
    embeddings = model.encode(unique_docs, batch_size=32, show_progress_bar=True)
    doc_embeddings = dict(zip(unique_docs, embeddings))
    
    # 2. Prepare data for parallel processing
    def prepare_row_data_mrr(row):
        retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col]]
        gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
        return retrieved_embs, gt_embs
    
    row_data = [prepare_row_data_mrr(row) for _, row in df.iterrows()]
    
    # 3. Chunk data and process in parallel using global function
    chunk_size = max(1, len(row_data) // n_jobs)
    chunks = [row_data[i:i+chunk_size] for i in range(0, len(row_data), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        chunk_results = list(executor.map(partial(process_chunk_mrr_global, threshold=threshold), chunks))
    
    # 4. Flatten results
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)
    
    df['semantic_rr'] = all_results
    return df['semantic_rr'].mean()

async def hybrid_semantic_mrr(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', threshold=0.8):
    """Hybrid approach cho semantic MRR calculation"""
    
    # 1. Async document collection and encoding
    all_docs = set()
    for _, row in df.iterrows():
        all_docs.update(row[retrieved_col])
        all_docs.update(row[ground_truth_col])
    
    async def async_encode():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            unique_docs = list(all_docs)
            embeddings = await loop.run_in_executor(
                executor,
                lambda: model.encode(unique_docs, batch_size=64, show_progress_bar=True)
            )
            return dict(zip(unique_docs, embeddings))
    
    doc_embeddings = await async_encode()
    
    # 2. Prepare data for parallel processing
    def prepare_batch_data_mrr(batch_df):
        batch_data = []
        for _, row in batch_df.iterrows():
            retrieved_embs = [doc_embeddings[doc] for doc in row[retrieved_col]]
            gt_embs = [doc_embeddings[doc] for doc in row[ground_truth_col]]
            batch_data.append((retrieved_embs, gt_embs))
        return batch_data
    
    # 3. Split DataFrame into batches
    n_jobs = mp.cpu_count()
    batch_size = max(1, len(df) // n_jobs)
    df_batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    # 4. Async parallel processing using global function
    async def process_batches_async():
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            tasks = []
            for batch_df in df_batches:
                batch_data = prepare_batch_data_mrr(batch_df)
                task = loop.run_in_executor(executor, partial(process_chunk_mrr_global, threshold=threshold), batch_data)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks)
            return batch_results
    
    batch_results = await process_batches_async()
    
    # 5. Flatten results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    df['semantic_rr'] = all_results
    return df['semantic_rr'].mean()

def calculate_semantic_mrr(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', threshold=0.8):
    """Tính Mean Reciprocal Rank theo semantic similarity - Original function kept for compatibility"""
    df['semantic_rr'] = df.apply(
        lambda row: semantic_rr(row[retrieved_col], row[ground_truth_col], threshold),
        axis=1
    )
    return df['semantic_rr'].mean()
