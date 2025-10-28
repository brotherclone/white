from enum import Enum


class WorkflowType(Enum):
    SINGLE_AGENT = "single_agent"  # Just one color agent
    CHAIN = "chain"  # Sequential: Black → Red → Orange → etc.
    PARALLEL = "parallel"  # Multiple agents on same input
    FULL_SPECTRUM = "full_spectrum"  # All agents in complex workflow
