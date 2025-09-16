"""
LangGraph Anthropic Agent for White Album Project

This module provides a LangGraph-based agent using Anthropic's Claude model
for intelligent processing and decision-making within the White Album ecosystem.
"""

import os
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    """State structure for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any]
    current_task: str
    task_history: List[str]


class WhiteAlbumLangGraphAgent:
    """
    LangGraph-based agent using Anthropic Claude for White Album project tasks.

    This agent can handle various music-related tasks, biographical research,
    and gaming interactions within the White Album ecosystem.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7):
        """
        Initialize the LangGraph agent with Anthropic Claude.

        Args:
            model_name: The Claude model to use
            temperature: Sampling temperature for the model
        """
        # Initialize the Anthropic model
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Create tools
        self.tools = [
            self._create_music_analysis_tool(),
            self._create_biographical_research_tool(),
            self._create_gaming_context_tool(),
            self._create_project_status_tool()
        ]

        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the graph
        self.graph = self._create_graph()

        # Memory for conversation persistence
        self.memory = MemorySaver()

        # Compile the graph with memory
        self.app = self.graph.compile(checkpointer=self.memory)

    @staticmethod
    def _create_music_analysis_tool():
        """Create a tool for music-related analysis and processing."""
        @tool
        def analyze_music_content(query: str) -> str:
            """
            Analyze music content, lyrics, or audio files in the White Album project.

            Args:
                query: The music analysis query or task

            Returns:
                Analysis results and recommendations
            """
            # This would integrate with your existing music tools and structures
            return f"Music analysis for: {query}. Integration with staged_raw_material and music structures pending."

        return analyze_music_content

    @staticmethod
    def _create_biographical_research_tool():
        """Create a tool for biographical research and context."""
        @tool
        def research_biographical_context(subject: str) -> str:
            """
            Research biographical information relevant to the White Album project.

            Args:
                subject: The person, band, or biographical subject to research

            Returns:
                Biographical research results and context
            """
            # This would integrate with your biographical_tools.py
            return f"Biographical research for: {subject}. Integration with biographical tools and reference data pending."

        return research_biographical_context

    @staticmethod
    def _create_gaming_context_tool():
        """Create a tool for gaming-related context and interactions."""
        @tool
        def process_gaming_context(context: str) -> str:
            """
            Process gaming context and interactions for the White Album RPG elements.

            Args:
                context: The gaming context or scenario to process

            Returns:
                Gaming context analysis and recommendations
            """
            # This would integrate with your gaming_tools.py
            return f"Gaming context processing for: {context}. Integration with gaming tools and palace game pending."

        return process_gaming_context

    @staticmethod
    def _create_project_status_tool():
        """Create a tool for project status and workflow management."""
        @tool
        def check_project_status(component: str = "general") -> str:
            """
            Check the status of White Album project components.

            Args:
                component: The specific component to check (general, music, biographical, gaming)

            Returns:
                Current project status and next steps
            """
            return f"Project status check for: {component}. Monitoring staged materials, training progress, and component integration."

        return check_project_status

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph for the agent."""

        # Define the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with the current state."""
        messages = state["messages"]

        # Create a system prompt that incorporates White Album context
        system_prompt = """You are an intelligent agent for the White Album project, a comprehensive music creation and storytelling system. 

Your role is to assist with:
- Music analysis and processing of staged raw materials
- Biographical research and context development
- Gaming elements and RPG workflow integration
- Project coordination and status tracking

You have access to tools for music analysis, biographical research, gaming context, and project status. Use these tools when appropriate to provide comprehensive assistance.

Current task: {current_task}
Context: {context}
"""

        # Add system context to messages
        full_messages = [
            HumanMessage(content=system_prompt.format(
                current_task=state.get("current_task", "general assistance"),
                context=str(state.get("context", {}))
            ))
        ] + messages

        response = self.llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue to tools
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def process_message(self, message: str, context: Dict[str, Any] = None, task: str = "general") -> str:
        """
        Process a message through the agent.

        Args:
            message: The input message
            context: Additional context for the agent
            task: The current task type

        Returns:
            The agent's response
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": context or {},
            "current_task": task,
            "task_history": []
        }

        # Create a unique thread ID for this conversation
        thread_id = {"configurable": {"thread_id": "white_album_session"}}

        # Run the agent
        result = self.app.invoke(initial_state, config=thread_id)

        # Extract the final response
        return result["messages"][-1].content

    def stream_response(self, message: str, context: Dict[str, Any] = None, task: str = "general"):
        """
        Stream the agent's response for real-time interaction.

        Args:
            message: The input message
            context: Additional context for the agent
            task: The current task type

        Yields:
            Streaming response chunks
        """
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": context or {},
            "current_task": task,
            "task_history": []
        }

        thread_id = {"configurable": {"thread_id": "white_album_session"}}

        for chunk in self.app.stream(initial_state, config=thread_id):
            yield chunk


# Factory function for easy agent creation
def create_white_album_agent(model_name: str = "claude-3-5-sonnet-20241022", **kwargs) -> WhiteAlbumLangGraphAgent:
    """
    Factory function to create a White Album LangGraph agent.

    Args:
        model_name: The Claude model to use
        **kwargs: Additional configuration options

    Returns:
        Configured WhiteAlbumLangGraphAgent instance
    """
    return WhiteAlbumLangGraphAgent(model_name=model_name, **kwargs)
