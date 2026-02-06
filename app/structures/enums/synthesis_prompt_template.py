from enum import Enum


class SynthesisPromptTemplate(str, Enum):
    """
    Different synthesis prompt templates that emphasize different aspects
    of the chromatic transformation, producing varied final outputs.
    """

    TEMPORAL = "temporal"  # Past/present/future emphasis
    ONTOLOGICAL = "ontological"  # Real/imagined/forgotten emphasis
    EMOTIONAL = "emotional"  # Tone/mood integration focus
    STRUCTURAL = "structural"  # Musical architecture focus
