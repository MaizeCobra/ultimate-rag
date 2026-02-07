"""Main RAG AI Agent implementation."""

from dataclasses import dataclass, field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.config import get_settings
from src.agent.memory import PostgresChatMemory
from src.agent.prompts import RAG_SYSTEM_PROMPT
from src.agent.tools import (
    vector_search_tool,
    list_docs_tool,
    get_content_tool,
    sql_query_tool,
)


@dataclass
class AgentResponse:
    """Response from the agent."""

    content: str
    tool_calls: List[dict] = field(default_factory=list)


class RAGAgent:
    """RAG AI Agent with tools and memory.

    Matches the n8n "RAG AI Agent" node behavior.
    """

    def __init__(self, session_id: str):
        """Initialize the agent.

        Args:
            session_id: Unique session ID for memory
        """
        settings = get_settings()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.7,
        )

        # Initialize memory
        self.memory = PostgresChatMemory(session_id=session_id, max_messages=10)

        # Define tools
        self.tools = [
            vector_search_tool,
            list_docs_tool,
            get_content_tool,
            sql_query_tool,
        ]

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    async def _get_chat_history(self) -> list:
        """Convert memory to LangChain message format."""
        messages = await self.memory.get_messages()
        history = []
        for msg in messages:
            if msg.role == "human":
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))
        return history

    async def invoke(self, user_input: str) -> AgentResponse:
        """Process a user message and return a response.

        Args:
            user_input: The user's message

        Returns:
            AgentResponse with the AI response
        """
        # Get chat history
        chat_history = await self._get_chat_history()

        # Save user message
        await self.memory.add_message("human", user_input)

        # Use ainvoke for async tool support
        result = await self.executor.ainvoke(
            {
                "input": user_input,
                "chat_history": chat_history,
            }
        )

        response_content = result.get("output", "")

        # Save AI response
        await self.memory.add_message("ai", response_content)

        return AgentResponse(content=response_content)


async def create_agent(session_id: str) -> RAGAgent:
    """Factory function to create an agent.

    Args:
        session_id: Unique session ID

    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent(session_id=session_id)
