import asyncio
import atexit
import logging
import os
import time
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
import json
import uuid
import hashlib

from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse
from loguru import logger
from pydantic import BaseModel, Field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Token counting will use character approximation.")

from .WDMParser.WDMParser import WDMPDFParser, process_pdf_documents
from .prompts import GENERATE_PROMPT, QUERY_ANALYSIS_PROMPT
from .reranker import Reranker
from .setting import REANKER_MODEL_NAME, VECTORSTORE_CONFIG
from .vectorstore import QdrantClientManager, VectorStore


def cleanup_qdrant_clients():
    """Cleanup function to close all Qdrant clients on app shutdown."""
    try:
        QdrantClientManager.close_all_clients()
        logger.info("All Qdrant clients closed successfully")
    except Exception as e:
        logger.warning(f"Error during Qdrant cleanup: {e}")


# Register cleanup function
atexit.register(cleanup_qdrant_clients)


class ChatHistoryOptimizer:
    """Advanced conversation history optimization with token management and LLM-based summarization"""
    
    def __init__(self, 
                 max_recent_messages: int = 10,
                 max_total_tokens: int = 4000,
                 token_buffer: int = 500,
                 summary_ratio: float = 0.3,
                 model_name: str = "gemini-2.0-flash"
                ):
        self.max_recent_messages = max_recent_messages
        self.max_total_tokens = max_total_tokens
        self.token_buffer = token_buffer
        self.summary_ratio = summary_ratio
        self.model_name = model_name
        
        # Token counting setup
        if TIKTOKEN_AVAILABLE:
            try:
                # Use a generic encoding if model-specific is not available
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 compatible
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Initialize LLM for summarization
        self.summarization_llm = ChatVertexAI(
            model_name=model_name,
            temperature=0.1,  # Low temperature for consistent summaries
            max_tokens=512,   # Control summary length
        )
        
        # Create summarization chain
        self.summary_chain = self._create_summary_chain()
        
        # Metrics tracking
        self.optimization_count = 0
        self.total_tokens_saved = 0
        self.summarization_success_rate = 0.0
        self.successful_summarizations = 0
        self.total_summarization_attempts = 0
    
    def _create_summary_chain(self):
        """Create LangChain summarization chain"""
        template = """Hãy tóm tắt cuộc hội thoại sau một cách ngắn gọn và chính xác:

{conversation_history}

Yêu cầu tóm tắt:
- Chủ đề chính đã thảo luận
- Thông tin quan trọng người dùng đã cung cấp  
- Các quyết định hoặc kết luận quan trọng
- Context cần thiết cho câu hỏi tiếp theo
- Giữ lại tên và thông tin cá nhân người dùng

Tóm tắt (tối đa 200 từ):"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["conversation_history"]
        )
        
        return prompt | self.summarization_llm | StrOutputParser()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}, using character approximation")
        
        # Fallback: character-based approximation (roughly 4 chars = 1 token)
        return max(1, len(text) // 4)
    
    def should_optimize(self, messages: List["ConversationMessage"]) -> bool:
        """Check if optimization is needed"""
        if len(messages) <= self.max_recent_messages:
            return False
        
        # Calculate total tokens
        total_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        return total_tokens > (self.max_total_tokens - self.token_buffer)
    
    def optimize_conversation_history(self, messages: List["ConversationMessage"]) -> str:
        """Main optimization method"""
        if not messages:
            return ""
        
        # If optimization not needed, return simple format
        if not self.should_optimize(messages):
            return self._format_messages_simple(messages[-self.max_recent_messages:])
        
        self.optimization_count += 1
        
        try:
            # Calculate optimal split
            split_info = self._calculate_optimal_split(messages)
            
            # Get messages to summarize and keep
            messages_to_summarize = messages[:split_info['summarize_count']]
            recent_messages = messages[split_info['summarize_count']:]
            
            # Create summary if needed
            summary = ""
            if messages_to_summarize:
                summary = self._create_summary_with_fallback(messages_to_summarize)
            
            # Build optimized context
            context_parts = []
            
            if summary:
                context_parts.append(f"Tóm tắt cuộc hội thoại trước:\n{summary}")
            
            if recent_messages:
                context_parts.append("Cuộc hội thoại gần đây:")
                context_parts.extend([
                    f"{msg.role.capitalize()}: {msg.content}" 
                    for msg in recent_messages
                ])
            
            optimized_context = "\n".join(context_parts)
            
            # Track token savings
            original_tokens = sum(self.count_tokens(msg.content) for msg in messages)
            optimized_tokens = self.count_tokens(optimized_context)
            self.total_tokens_saved += max(0, original_tokens - optimized_tokens)
            
            return optimized_context
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, falling back to simple truncation")
            return self._simple_context_fallback(messages)
    
    def _calculate_optimal_split(self, messages: List["ConversationMessage"]) -> Dict[str, int]:
        """Calculate optimal token allocation between recent messages and summary"""
        available_tokens = self.max_total_tokens - self.token_buffer
        summary_tokens = int(available_tokens * self.summary_ratio)
        recent_tokens = available_tokens - summary_tokens
        
        # Work backwards from latest messages to fit in recent_tokens
        current_tokens = 0
        recent_count = 0
        
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = self.count_tokens(messages[i].content)
            if current_tokens + msg_tokens <= recent_tokens:
                current_tokens += msg_tokens
                recent_count += 1
            else:
                break
        
        # Ensure we keep at least some recent messages
        recent_count = max(min(self.max_recent_messages // 2, recent_count), min(3, len(messages)))
        summarize_count = max(0, len(messages) - recent_count)
        
        return {
            'summarize_count': summarize_count,
            'recent_count': recent_count,
            'estimated_summary_tokens': summary_tokens,
            'estimated_recent_tokens': recent_tokens
        }
    
    def _create_summary_with_fallback(self, messages: List["ConversationMessage"]) -> str:
        """Create LLM-based summary with fallback"""
        self.total_summarization_attempts += 1
        
        try:
            # Format messages for summarization
            conversation_text = self._format_messages_for_summary(messages)
            
            # Use LangChain chain for summarization
            summary = self.summary_chain.invoke({
                "conversation_history": conversation_text
            })
            
            if isinstance(summary, str) and len(summary.strip()) > 10:
                self.successful_summarizations += 1
                self._update_success_rate()
                return summary.strip()
            else:
                raise ValueError("Empty or invalid summary returned")
                
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using fallback")
            return self._fallback_summary(messages)
    
    async def create_summary_async(self, messages: List["ConversationMessage"]) -> str:
        """Create summary asynchronously using LangChain"""
        self.total_summarization_attempts += 1
        
        try:
            # Format messages for summarization
            conversation_text = self._format_messages_for_summary(messages)
            
            # Use async invoke with LangChain
            summary = await self.summary_chain.ainvoke({
                "conversation_history": conversation_text
            })
            
            if isinstance(summary, str) and len(summary.strip()) > 10:
                self.successful_summarizations += 1
                self._update_success_rate()
                return summary.strip()
            else:
                raise ValueError("Empty or invalid summary returned")
                
        except Exception as e:
            logger.warning(f"Async LLM summarization failed: {e}")
            return self._fallback_summary(messages)
    
    def _format_messages_for_summary(self, messages: List["ConversationMessage"]) -> str:
        """Format conversation messages for summarization"""
        if not messages:
            return "Không có nội dung hội thoại."
            
        formatted_messages = []
        
        for msg in messages:
            role = "Người dùng" if msg.role == "user" else "Trợ lý"
            timestamp = msg.timestamp.strftime("%H:%M")
            # Truncate very long messages for summarization
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            formatted_messages.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _fallback_summary(self, messages: List["ConversationMessage"]) -> str:
        """Fallback summary method if LLM fails"""
        if not messages:
            return "Không có nội dung hội thoại."
        
        # Simple extraction of key topics
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        if not user_messages:
            return "Cuộc hội thoại chưa có câu hỏi từ người dùng."
        
        # Get recent user topics
        recent_topics = user_messages[-3:] if len(user_messages) >= 3 else user_messages
        topics_text = ", ".join([topic[:50] + "..." if len(topic) > 50 else topic for topic in recent_topics])
        
        return f"Cuộc hội thoại trước bao gồm các chủ đề: {topics_text}"
    
    def _format_messages_simple(self, messages: List["ConversationMessage"]) -> str:
        """Simple message formatting without optimization"""
        if not messages:
            return ""
        
        formatted = []
        for msg in messages:
            formatted.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "Cuộc hội thoại gần đây:\n" + "\n".join(formatted)
    
    def _simple_context_fallback(self, messages: List["ConversationMessage"]) -> str:
        """Simple fallback when optimization fails"""
        # Just take the most recent messages
        recent_messages = messages[-self.max_recent_messages:]
        return self._format_messages_simple(recent_messages)
    
    def _update_success_rate(self):
        """Update summarization success rate"""
        if self.total_summarization_attempts > 0:
            self.summarization_success_rate = self.successful_summarizations / self.total_summarization_attempts
    
    def compress_similar_messages(self, messages: List["ConversationMessage"]) -> List["ConversationMessage"]:
        """Merge similar consecutive messages (future enhancement)"""
        # TODO: Implement semantic similarity compression
        return messages
    
    def identify_important_messages(self, messages: List["ConversationMessage"]) -> List[str]:
        """Identify messages that should be preserved (future enhancement)"""
        # TODO: Implement importance scoring
        # Criteria: contains questions, decisions, important information
        return [msg.message_id for msg in messages]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "total_optimizations": self.optimization_count,
            "total_tokens_saved": self.total_tokens_saved,
            "summarization_success_rate": self.summarization_success_rate,
            "successful_summarizations": self.successful_summarizations,
            "total_summarization_attempts": self.total_summarization_attempts,
            "tokenizer_available": self.tokenizer is not None,
            "max_total_tokens": self.max_total_tokens,
            "token_buffer": self.token_buffer
        }


class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationThread(BaseModel):
    """Complete conversation thread"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = Field(default=None)


class ConversationMemory:
    """Enhanced memory management for conversations with token optimization"""
    
    def __init__(self, 
                 max_messages_in_context: int = 10, 
                 summarize_after: int = 20,
                 max_tokens: int = 4000,
                 token_buffer: int = 500,
                 enable_optimization: bool = True):
        self.max_messages_in_context = max_messages_in_context
        self.summarize_after = summarize_after
        self.max_tokens = max_tokens
        self.token_buffer = token_buffer
        self.enable_optimization = enable_optimization
        self.conversations: Dict[str, ConversationThread] = {}
        
        # Initialize optimizer if enabled
        if enable_optimization:
            self.history_optimizer = ChatHistoryOptimizer(
                max_recent_messages=max_messages_in_context,
                max_total_tokens=max_tokens,
                token_buffer=token_buffer
            )
        else:
            self.history_optimizer = None
        
    def create_conversation(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Create a new conversation thread"""
        conversation = ConversationThread(
            user_id=user_id,
            session_id=session_id
        )
        self.conversations[conversation.conversation_id] = conversation
        return conversation.conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a message to conversation"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        conversation = self.conversations[conversation_id]
        conversation.messages.append(message)
        conversation.last_updated = datetime.now()
        
        # Auto-generate title from first user message
        if not conversation.title and role == "user" and len(conversation.messages) == 1:
            conversation.title = content[:50] + "..." if len(content) > 50 else content
        
        # Trigger summarization if needed
        if len(conversation.messages) > self.summarize_after:
            self._maybe_summarize_conversation(conversation_id)
        
        return message.message_id
    
    def get_conversation_context(self, conversation_id: str, include_summary: bool = True) -> str:
        """Enhanced conversation context with token optimization"""
        if conversation_id not in self.conversations:
            return ""
        
        conversation = self.conversations[conversation_id]
        
        # Use optimizer if available and enabled
        if self.enable_optimization and self.history_optimizer:
            try:
                optimized_context = self.history_optimizer.optimize_conversation_history(
                    conversation.messages
                )
                return optimized_context
            except Exception as e:
                logger.warning(f"Optimization failed for conversation {conversation_id}: {e}, falling back to simple method")
                return self._simple_context_fallback(conversation)
        else:
            # Original simple approach
            return self._simple_context_fallback(conversation)
    
    def _simple_context_fallback(self, conversation: ConversationThread) -> str:
        """Simple context fallback method"""
        context_parts = []
        
        # Add summary if available
        if conversation.summary:
            context_parts.append(f"Tóm tắt cuộc hội thoại trước: {conversation.summary}")
        
        # Get recent messages (limit to max_messages_in_context)
        recent_messages = conversation.messages[-self.max_messages_in_context:]
        
        if recent_messages:
            context_parts.append("Cuộc hội thoại gần đây:")
            for msg in recent_messages:
                context_parts.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationThread]:
        """Get full conversation thread"""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[ConversationThread]:
        """List conversations with optional filtering"""
        conversations = list(self.conversations.values())
        
        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]
        if session_id:
            conversations = [c for c in conversations if c.session_id == session_id]
        
        # Sort by last updated, newest first
        conversations.sort(key=lambda x: x.last_updated, reverse=True)
        return conversations
    
    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def _maybe_summarize_conversation(self, conversation_id: str):
        """Enhanced summarization with LLM integration"""
        conversation = self.conversations[conversation_id]
        
        # Only summarize if we have enough messages and no recent summary
        if len(conversation.messages) % self.summarize_after == 0:
            try:
                # Get messages to summarize (exclude recent ones to keep in context)
                messages_to_summarize = conversation.messages[:-self.max_messages_in_context]
                
                if not messages_to_summarize:
                    return
                
                # Use optimizer's LLM-based summarization if available
                if self.enable_optimization and self.history_optimizer:
                    try:
                        summary = self.history_optimizer._create_summary_with_fallback(messages_to_summarize)
                        conversation.summary = summary
                        logger.info(f"LLM-based summary created for conversation {conversation_id}")
                        return
                    except Exception as e:
                        logger.warning(f"LLM summarization failed for conversation {conversation_id}: {e}")
                
                # Fallback to simple summarization
                messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages_to_summarize])
                conversation.summary = f"Cuộc hội thoại trước bao gồm: {messages_text[:200]}..."
                logger.info(f"Simple summary created for conversation {conversation_id}")
                
            except Exception as e:
                logger.warning(f"Failed to summarize conversation {conversation_id}: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if self.enable_optimization and self.history_optimizer:
            return self.history_optimizer.get_optimization_stats()
        else:
            return {
                "optimization_enabled": False,
                "message": "Optimization is disabled"
            }


class ConversationManager:
    """High-level conversation management"""
    
    def __init__(self, memory: Optional[ConversationMemory] = None):
        self.memory = memory or ConversationMemory()
        self.current_conversation: Optional[str] = None
        
    def start_conversation(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Start a new conversation"""
        conversation_id = self.memory.create_conversation(user_id, session_id)
        self.current_conversation = conversation_id
        return conversation_id
    
    def use_conversation(self, conversation_id: str):
        """Switch to an existing conversation"""
        if conversation_id not in self.memory.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        self.current_conversation = conversation_id
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add user message to current conversation"""
        if not self.current_conversation:
            self.current_conversation = self.start_conversation()
        return self.memory.add_message(self.current_conversation, "user", content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add assistant message to current conversation"""
        if not self.current_conversation:
            raise ValueError("No active conversation")
        return self.memory.add_message(self.current_conversation, "assistant", content, metadata)
    
    def get_current_context(self) -> str:
        """Get context for current conversation"""
        if not self.current_conversation:
            return ""
        return self.memory.get_conversation_context(self.current_conversation)
    
    def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[Dict]:
        """Get conversation history in simple format"""
        conv_id = conversation_id or self.current_conversation
        if not conv_id:
            return []
        
        conversation = self.memory.get_conversation(conv_id)
        if not conversation:
            return []
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in conversation.messages
        ]


class RAG:
    def __init__(
        self,
        embedding_type: str,
        embedding_model: str,
        enable_hybrid_search: bool,
        chunk_type: str,
        use_memory: bool,
        collection_name: str,
        persist_dir: str,
        use_reranker: bool,
        langfuse_client: Optional[Langfuse] = None,
        enable_conversation_memory: bool = True,
        # New conversation optimization parameters
        max_conversation_tokens: int = 4000,
        conversation_token_buffer: int = 500,
        enable_conversation_optimization: bool = True,
    ):
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_type = chunk_type
        self.use_memory = use_memory
        self.collection_name = collection_name

        self.vectorstore_key = f"vs_{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_memory}_{collection_name}"
        self.vectorstore = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=use_memory,
        )
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = Reranker(method=REANKER_MODEL_NAME)
            self.reranker_name = REANKER_MODEL_NAME
        else:
            self.reranker = None
            self.reranker_name = "Disabled"
        self.langfuse = langfuse_client
        
        # Conversation management with optimization
        self.enable_conversation_memory = enable_conversation_memory
        self.max_conversation_tokens = max_conversation_tokens
        self.conversation_token_buffer = conversation_token_buffer
        self.enable_conversation_optimization = enable_conversation_optimization
        
        if enable_conversation_memory:
            # Create enhanced ConversationMemory with optimization
            memory = ConversationMemory(
                max_messages_in_context=10,
                summarize_after=20,
                max_tokens=max_conversation_tokens,
                token_buffer=conversation_token_buffer,
                enable_optimization=enable_conversation_optimization
            )
            self.conversation_manager = ConversationManager(memory=memory)
        else:
            self.conversation_manager = None

    def _format_time(self, seconds):
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"

    # ======================== CONVERSATION METHODS ========================
    
    def start_conversation(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Start a new conversation"""
        if not self.conversation_manager:
            raise ValueError("Conversation memory is disabled")
        return self.conversation_manager.start_conversation(user_id, session_id)
    
    def use_conversation(self, conversation_id: str):
        """Switch to an existing conversation"""
        if not self.conversation_manager:
            raise ValueError("Conversation memory is disabled")
        self.conversation_manager.use_conversation(conversation_id)
    
    def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[Dict]:
        """Get conversation history"""
        if not self.conversation_manager:
            return []
        return self.conversation_manager.get_conversation_history(conversation_id)
    
    def list_conversations(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """List all conversations"""
        if not self.conversation_manager:
            return []
        
        conversations = self.conversation_manager.memory.list_conversations(user_id, session_id)
        return [
            {
                "conversation_id": conv.conversation_id,
                "user_id": conv.user_id,
                "session_id": conv.session_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "last_updated": conv.last_updated.isoformat(),
                "message_count": len(conv.messages),
                "summary": conv.summary
            }
            for conv in conversations
        ]
    
    def clear_conversation(self, conversation_id: Optional[str] = None):
        """Clear a conversation"""
        if not self.conversation_manager:
            return
        
        if conversation_id:
            self.conversation_manager.memory.clear_conversation(conversation_id)
        elif self.conversation_manager.current_conversation:
            self.conversation_manager.memory.clear_conversation(self.conversation_manager.current_conversation)
            self.conversation_manager.current_conversation = None
    
    def get_current_conversation_id(self) -> Optional[str]:
        """Get current conversation ID"""
        if not self.conversation_manager:
            return None
        return self.conversation_manager.current_conversation

    def chat(self, query: str, filter: bool = True) -> dict:
        """
        Simple chat interface with automatic conversation management
        This is a convenience method that automatically handles conversation history
        """
        return self.__call__(query, filter=filter, use_conversation=True)
    
    def debug_conversation_context(self) -> Dict[str, Any]:
        """Debug method to see current conversation context"""
        if not self.conversation_manager:
            return {"error": "Conversation memory disabled"}
        
        if not self.conversation_manager.current_conversation:
            return {"error": "No active conversation"}
        
        try:
            context = self.conversation_manager.get_current_context()
            history = self.conversation_manager.get_conversation_history()
            
            return {
                "conversation_id": self.conversation_manager.current_conversation,
                "message_count": len(history),
                "conversation_context": context,
                "full_history": history,
                "context_length": len(context)
            }
        except Exception as e:
            return {"error": f"Failed to get conversation context: {e}"}
    
    def get_conversation_optimization_stats(self) -> Dict[str, Any]:
        """Get conversation optimization statistics"""
        if not self.conversation_manager or not self.conversation_manager.memory:
            return {"error": "Conversation memory not available"}
        
        try:
            stats = self.conversation_manager.memory.get_optimization_stats()
            stats.update({
                "max_conversation_tokens": self.max_conversation_tokens,
                "conversation_token_buffer": self.conversation_token_buffer,
                "optimization_enabled": self.enable_conversation_optimization,
                "tiktoken_available": TIKTOKEN_AVAILABLE
            })
            return stats
        except Exception as e:
            return {"error": f"Failed to get optimization stats: {e}"}

    async def process_pdfs_bytes(
        self,
        pdf_data_list: Union[List[str], List[bytes], List[Union[str, bytes]]],
        credential_path: Optional[str] = None,
        debug_mode: bool = False,
    ) -> tuple:
        """
        Process multiple PDF files/bytes using the new WDMParser with async support
        
        Args:
            pdf_data_list: List of PDF file paths or bytes data
            credential_path: Path to Google Cloud credentials
            debug_mode: Enable debug logging
            
        Returns:
            Tuple of (all_documents, processing_results, stats)
        """
        start_time = time.time()
        
        try:
            # Use the new WDMParser with bytes support
            settings = WDMPDFParser.create_settings(
                credential_path=credential_path,
                debug=debug_mode,
                debug_level=1,
                max_concurrent_files=3,
                max_memory_mb=8192,
                batch_size=5,
                cleanup_interval=2
            )
            
            parser = WDMPDFParser(settings=settings)
            
            # Process documents async
            result = await parser.process_documents(
                pdf_documents=pdf_data_list,
                merge_span_tables=True,
                enrich=False,
                extract_text=True,
                return_failed=True
            )
            
            # Handle tuple return type
            if isinstance(result, tuple):
                results, failed_files = result
            else:
                results = result
                failed_files = []
            
            # Combine all documents
            all_documents = []
            processing_results = []
            
            for identifier, documents in results.items():
                # Create result object similar to old format
                table_docs = [d for d in documents if d.metadata.get('type') == 'table']
                text_docs = [d for d in documents if d.metadata.get('type') == 'text']
                
                processing_results.append({
                    "file_name": identifier,
                    "success": True,
                    "splits": documents,
                    "count": len(documents),
                    "table_docs": len(table_docs),
                    "text_docs": len(text_docs),
                    "processing_time": 0,  # Not tracked per file in new system
                    "file_size_mb": 0,     # Not tracked per file in new system
                })
                
                all_documents.extend(documents)
            
            # Add failed files to results
            for failed_file in failed_files:
                processing_results.append({
                    "file_name": failed_file,
                    "success": False,
                    "error": "Processing failed",
                    "splits": [],
                    "count": 0,
                    "table_docs": 0,
                    "text_docs": 0,
                    "processing_time": 0,
                    "file_size_mb": 0,
                })
            
            # Calculate summary stats
            total_time = time.time() - start_time
            successful_files = len([r for r in processing_results if r["success"]])
            total_docs = len(all_documents)
            text_docs = len([doc for doc in all_documents if doc.metadata.get("type") == "text"])
            table_docs = len([doc for doc in all_documents if doc.metadata.get("type") == "table"])
            
            stats = {
                "successful_files": successful_files,
                "total_files": len(pdf_data_list),
                "total_docs": total_docs,
                "text_docs": text_docs,
                "table_docs": table_docs,
                "total_time": total_time,
            }
            
            logger.info(
                f"Processing completed: {successful_files}/{len(pdf_data_list)} files, "
                f"{total_docs} documents, {self._format_time(total_time)}"
            )
            
            return all_documents, processing_results, stats
            
        except Exception as e:
            logger.error(f"Error in process_pdfs_bytes: {e}")
            return [], [], {
                "successful_files": 0,
                "total_files": len(pdf_data_list),
                "total_docs": 0,
                "text_docs": 0,
                "table_docs": 0,
                "total_time": time.time() - start_time,
            }

    # Keep the old method for backward compatibility
    async def load_pdfs(
        self,
        pdf_files: List[str],
        credential_path: Optional[str] = None,
        temp_dir: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Process multiple PDF files asynchronously (backward compatibility method)
        Now uses the new WDMParser internally
        """
        return await self.process_pdfs_bytes(pdf_files, credential_path, debug_mode)

    def add_documents(
        self,
        documents: List[Document],
    ):
        if self.langfuse:
            with self.langfuse.start_as_current_span(
                name="document-ingestion",
                metadata={
                    "collection_name": self.collection_name,
                    "document_count": len(documents),
                },
                input={"document_sources": list(set(d.metadata.get("source", "Unknown") for d in documents))}
            ) as trace:
                with trace.start_as_current_span(
                    name="add-to-vectorstore",
                    input={"document_count": len(documents)}
                ) as span:
                    self.vectorstore.add_documents(documents)
                    span.update(output={"status": "success"})
        else:
            self.vectorstore.add_documents(documents)

    def retrieve_documents(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        filter_types: Optional[List[str]] = None,        
    ):
        
        if not self.use_reranker or self.reranker is None:
            return self.vectorstore.retrieve_documents(
                query=query, filter_sources=filter_sources, filter_types=filter_types
            )
        else:
            num_docs = VECTORSTORE_CONFIG["k"] * 3
            docs = self.vectorstore.retrieve_documents(
                query=query, filter_sources=filter_sources, filter_types=filter_types, num_docs=num_docs
            )
            
            # Safer approach: Use index-based mapping instead of content mapping
            contents = [doc.page_content for doc in docs]
            reranked_contents = self.reranker.rerank(query, contents, VECTORSTORE_CONFIG["k"])
            
            # Map back using indices to handle duplicates properly
            reranked_docs = []
            used_indices = set()
            
            for reranked_content in reranked_contents:
                # Find first unused occurrence of this content
                for i, original_content in enumerate(contents):
                    if original_content == reranked_content and i not in used_indices and i < len(docs):
                        reranked_docs.append(docs[i])
                        used_indices.add(i)
                        break
            
            # Add this for debugging
            if len(reranked_docs) != len(reranked_contents):
                logger.warning(f"Some documents lost in reranking: expected {len(reranked_contents)}, got {len(reranked_docs)}")
            
            return reranked_docs

    def clear_vectorstore(self):
        self.vectorstore.clear_vectorstore()

    def get_unique_sources(self):
        return self.vectorstore.get_unique_sources()

    def get_vectorstore(self):
        return self.vectorstore

    def query_analysis(
        self,
        query: str,
        available_sources: Optional[List[str]] = None,
        available_types: Optional[List[str]] = ["text", "table"],
        callbacks: Optional[list] = None
    ):
        """
        Analyze query and return relevant sources and types

        Args:
            query (str): User input query
            available_sources (Optional[List[str]]): List of available sources to filter against
            available_types (Optional[List[str]]): List of available types to filter against

        Returns:
            QueryAnalysis: Object containing analysis information

        Raises:
            ValueError: When unable to parse results
            Exception: Other errors from LLM
        """

        class QueryAnalysis(BaseModel):
            """Schema for query analysis results"""

            sources: List[str] = Field(
                default=[],
                description="List of sources relevant to the query"
            )
            types: List[str] = Field(
                default=[],
                description="List of types relevant to the query"
            )
            confidence_score: Optional[float] = Field(
                default=None,
                description="Confidence level of the analysis (0-1)",
                ge=0.0,
                le=1.0,
            )
            reasoning: Optional[str] = Field(
                default=None,
                description="Reasoning for why these sources/types were selected"
            )

        parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        try:
            llm = ChatVertexAI(
                model_name="gemini-2.0-flash",
                temperature=0.1,  # Lower temperature for more stable results
                max_tokens=1024,  # Limit tokens to avoid overly long responses
            )

            # Create context about available sources/types if provided
            context_info = ""
            if available_sources:
                context_info += f"\nAvailable sources: {', '.join(available_sources)}"
            if available_types:
                context_info += f"\nAvailable types: {', '.join(available_types)}"

            template = QUERY_ANALYSIS_PROMPT

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["query", "context_info"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            # Create chain
            chain = prompt_template | llm | parser

            # Invoke with error handling AND callbacks
            response = chain.invoke({"query": query, "context_info": ""}, config={"callbacks": callbacks})

            # Validate and filter results with improved source matching
            if available_sources:
                filtered_sources = []
                for suggested_source in response.sources:
                    # Check for exact match first
                    if suggested_source in available_sources:
                        filtered_sources.append(suggested_source)
                    else:
                        # Check for partial match (filename without extension)
                        for available_source in available_sources:
                            # Remove extension from available source for comparison
                            available_name = (
                                available_source.rsplit(".", 1)[0]
                                if "." in available_source
                                else available_source
                            )
                            suggested_name = (
                                suggested_source.rsplit(".", 1)[0]
                                if "." in suggested_source
                                else suggested_source
                            )

                            # Check if suggested name matches available name (case insensitive)
                            if suggested_name.lower() == available_name.lower():
                                filtered_sources.append(available_source)
                                break
                            # Also check if suggested name is contained in available name
                            elif (
                                suggested_name.lower() in available_name.lower()
                                or available_name.lower() in suggested_name.lower()
                            ):
                                filtered_sources.append(available_source)
                                break

                response.sources = list(set(filtered_sources))  # Remove duplicates

            if available_types:
                # Filter only available types
                response.types = [t for t in response.types if t in available_types]

            return response

        except Exception as e:
            logger.error(f"Error in query_analysis: {str(e)}")
            # Return default result when error occurs
            return QueryAnalysis(
                sources=[],
                types=[],
                confidence_score=0.0,
                reasoning=f"Error occurred during analysis: {str(e)}",
            )

    def prepare_context(self, docs: List[Document]) -> str:
        context_parts = ["<documents>"]

        for i, doc in enumerate(docs, 1):
            doc_str = f'\n<document index="{i}">'

            doc_str += "\n  <metadata>"
            for key, value in doc.metadata.items():
                doc_str += f"\n    <{key}>{value}</{key}>"
            doc_str += "\n  </metadata>"

            doc_str += f"\n  <content>\n{doc.page_content}\n  </content>"

            doc_str += "\n</document>"

            context_parts.append(doc_str)

        context_parts.append("\n</documents>")

        return "".join(context_parts)

    def generate_response(self, prompt: str, context: str, conversation_context: str = "", callbacks: Optional[list] = None) -> str:
        # Enhanced prompt template that includes conversation context
        if conversation_context:
            template = """Bạn là WDM-AI-TEMIS, trợ lý AI thông minh chuyên phân tích tài liệu và hỗ trợ người dùng.

LỊCH SỬ HỘI THOẠI:
{conversation_context}

NỘI DUNG TÀI LIỆU:
{context}

CÂU HỎI HIỆN TẠI: {question}

Hướng dẫn trả lời:
- Nếu câu hỏi về thông tin cá nhân hoặc cuộc hội thoại trước: sử dụng lịch sử hội thoại
- Nếu câu hỏi về tài liệu: sử dụng nội dung tài liệu  
- Trả lời tự nhiên, thân thiện bằng tiếng Việt
- Tham khảo cuộc hội thoại trước khi cần thiết
- Chỉ nói không biết khi cả lịch sử hội thoại và tài liệu đều không có thông tin

Trả lời:"""
        else:
            template = GENERATE_PROMPT
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"] + (["conversation_context"] if conversation_context else []),
        )

        llm = ChatVertexAI(
            model_name="gemini-2.0-flash",
            temperature=0.3,
        )

        chain = prompt_template | llm
        
        invoke_params = {"context": context, "question": prompt}
        if conversation_context:
            invoke_params["conversation_context"] = conversation_context
            
        response = chain.invoke(
            invoke_params,
            config={"callbacks": callbacks}
        )
        # Extract content from AIMessage if needed
        if hasattr(response, "content"):
            content = response.content
            # Handle case where content might be a list
            if isinstance(content, list):
                return str(content)
            return str(content) if content is not None else ""
        return str(response)

    def __call__(self, query: str, filter: bool = True, use_conversation: bool = True) -> dict:
        
        # Conversation Management
        conversation_context = ""
        conversation_id = None
        
        if use_conversation and self.conversation_manager:
            # Add user message to conversation
            try:
                if not self.conversation_manager.current_conversation:
                    conversation_id = self.conversation_manager.start_conversation()
                else:
                    conversation_id = self.conversation_manager.current_conversation
                
                self.conversation_manager.add_user_message(query)
                conversation_context = self.conversation_manager.get_current_context()
            except Exception as e:
                logger.warning(f"Failed to manage conversation: {e}")
                conversation_context = ""
        
        if not self.langfuse:
            # Fallback to original behavior if Langfuse is not configured
            analysis = self.query_analysis(query, available_sources=self.get_unique_sources())
            filter_sources = analysis.sources if filter and analysis.sources else None
            filter_types = analysis.types if filter and analysis.types else None
            docs = self.retrieve_documents(query, filter_sources, filter_types)
            context = self.prepare_context(docs)
            response = self.generate_response(query, context, conversation_context)
            
            # Add assistant response to conversation
            if use_conversation and self.conversation_manager:
                try:
                    self.conversation_manager.add_assistant_message(
                        response, 
                        metadata={
                            "retrieved_docs_count": len(docs),
                            "filter_sources": filter_sources,
                            "filter_types": filter_types
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to add assistant message: {e}")
            
            return {
                "response": response, 
                "context": context, 
                "docs": docs, 
                "query": query, 
                "analysis": analysis,
                "conversation_id": conversation_id,
                "conversation_context": conversation_context
            }

        
        with self.langfuse.start_as_current_span(
            name="rag-query",
            input={"query": query},
            metadata={
                "embedding_type": self.embedding_type,
                "embedding_model": self.embedding_model,
                "hybrid_search": self.enable_hybrid_search,
                "chunk_type": self.chunk_type,
                "conversation_enabled": use_conversation and self.conversation_manager is not None,
                "conversation_id": conversation_id,
            }
        ) as trace:
            # 1. Query Analysis Step
            with trace.start_as_current_span(name="query-analysis") as analysis_span:
                # Import CallbackHandler inside the method to get the handler
                from langfuse.langchain import CallbackHandler
                langfuse_handler = CallbackHandler()
                
                analysis = self.query_analysis(
                    query,
                    available_sources=self.get_unique_sources(),
                    available_types=["text", "table"],
                    callbacks=[langfuse_handler]
                )
                analysis_span.update(
                    output=analysis.dict()
                )
                
            filter_sources = analysis.sources if filter and analysis.sources else None
            filter_types = analysis.types if filter and analysis.types else None
            
            # 2. Retrieval Step
            with trace.start_as_current_span(name="retrieval") as retrieval_span:
                docs = self.retrieve_documents(
                    query=query, filter_sources=filter_sources, filter_types=filter_types
                )
                retrieval_span.update(
                    input={"query": query, "filter_sources": filter_sources, "filter_types": filter_types},
                    output={
                        "retrieved_documents_count": len(docs),
                        "retrieved_sources": list(set(d.metadata.get("source", "Unknown") for d in docs))
                    }
                )

            context = self.prepare_context(docs)
            
            # 3. Generation Step
            with trace.start_as_current_span(name="generation") as generation_span:
                # Import CallbackHandler again for this step
                from langfuse.langchain import CallbackHandler
                langfuse_handler = CallbackHandler()
                
                response = self.generate_response(
                    query,
                    context,
                    conversation_context,
                    callbacks=[langfuse_handler]
                )

                generation_span.update(
                    input={
                        "query": query, 
                        "context_length": len(context),
                        "conversation_context_length": len(conversation_context)
                    },
                    output={"response": response}
                )

            # Add assistant response to conversation
            if use_conversation and self.conversation_manager:
                try:
                    self.conversation_manager.add_assistant_message(
                        response, 
                        metadata={
                            "retrieved_docs_count": len(docs),
                            "filter_sources": filter_sources,
                            "filter_types": filter_types,
                            "langfuse_trace_id": trace.id if hasattr(trace, 'id') else None
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to add assistant message: {e}")

            # Finalize the main trace
            trace.update(output={"final_response": response})

            return {
                "response": response,
                "context": context,
                "docs": docs,
                "query": query,
                "analysis": analysis,
                "conversation_id": conversation_id,
                "conversation_context": conversation_context
            }