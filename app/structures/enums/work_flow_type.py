from enum import Enum


class WorkflowType(str, Enum):
    """
    Types of workflows for Rainbow agents.
    1. SINGLE_AGENT: A single color agent handles the entire task.
    2. CHAIN: A sequence of color agents, each building on the previous one's output.
    3. PARALLEL: Multiple color agents work simultaneously on the same input.
    4. FULL_SPECTRUM: All color agents collaborate in a complex workflow.
    """

    SINGLE_AGENT = "single_agent"
    CHAIN = "chain"
    PARALLEL = "parallel"
    FULL_SPECTRUM = "full_spectrum"
