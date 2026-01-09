import random

from typing import Optional, List
from pydantic import BaseModel, Field

from app.reference.music.indie_publications.critic_names import (
    CRITIC_FIRST_NAMES,
    CRITIC_LAST_NAMES,
)
from app.reference.music.indie_publications.publications import (
    NOSTALGIC_PUBLICATIONS,
    EXPERIMENTAL_PUBLICATIONS,
    BLOG_PUBLICATIONS,
    MUSIC_PUBLICATIONS,
)
from app.structures.enums.vanity_interviewer_type import VanityInterviewerType


class VanityPersona(BaseModel):
    first_name: str = Field(
        default_factory=lambda: random.choice(CRITIC_FIRST_NAMES),
        description="The first name of the persona",
    )
    last_name: str = Field(
        default_factory=lambda: random.choice(CRITIC_LAST_NAMES),
        description="The last name of the persona",
    )
    publication: Optional[str] = Field(
        default=None, description="The publication the persona is associated with"
    )
    interviewer_type: Optional[VanityInterviewerType] = Field(
        default=None,
        description="The type of interviewer",
    )
    stance: Optional[str] = Field(default=None, description="The stance of the persona")
    approach: Optional[str] = Field(
        default=None, description="The approach of the persona"
    )
    tactics: Optional[List[str]] = Field(
        default=None, description="The tactics of the persona"
    )
    goal: Optional[str] = Field(default=None, description="The goal of the persona")
    example_questions: Optional[List[str]] = Field(
        default=None,
        description="Example questions for the persona with template string vars",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure interviewer_type is set before deriving dependent fields
        if self.interviewer_type is None:
            # random.choice needs a sequence, not the Enum class itself
            self.interviewer_type = random.choice(list(VanityInterviewerType))
        self.get_interview_engagement_style()
        if self.publication is None:
            self.create_publication()

    def get_interview_engagement_style(self):
        if self.interviewer_type == VanityInterviewerType.HOSTILE_SKEPTICAL:
            self.stance = "This is pretentious nonsense"
            self.approach = "Aggressive, reductive, gotcha-focused"
            self.tactics = [
                "False dichotomies",
                "Reductive readings that strip nuance",
                "Gotcha questions about contradictions",
                "Accusations of pretension",
                "Demand for immediate accessibility",
            ]
            self.goal = "Make artist look stupid or contradictory"
            self.example_questions = [
                "So you're saying {{concept}} - isn't that just {{reductive_version}}?",
                "This whole {{methodology}} thing - don't you think that's a bit pretentious?",
                "If {{aspect_a}} then how can you also claim {{aspect_b}}? Sounds contradictory.",
                "Can you explain in plain English what this is actually about? Because it sounds like gibberish.",
                "Isn't this just {{established_artist}} but worse?",
            ]
        elif self.interviewer_type == VanityInterviewerType.VANITY_PRESSING_FAN:
            self.stance = "Why'd you abandon accessibility?"
            self.approach = "Nostalgic, confused, betrayed"
            self.tactics = [
                "Nostalgic appeals to earlier work",
                "Commercial pressure concerns",
                "Audience confusion",
                "Betrayal at weirdness escalation",
                "Desire for 'return to form'",
            ]
            self.goal = "Express betrayal at weird experimental turn"
            self.example_questions = [
                "Vanity Pressing was so accessible - why make things so difficult now?",
                "Do you worry you're alienating the audience that loved {{earlier_work}}?",
                "This {{weird_element}} - is this what fans asked for?",
                "When are you going back to making music people can actually enjoy?",
                "Don't you miss just... making songs? Without all this {{methodology}} stuff?",
            ]
        elif self.interviewer_type == VanityInterviewerType.EXPERIMENTAL_PURIST:
            self.stance = "You sold out"
            self.approach = "Gatekeeping, purity-testing, disappointed"
            self.tactics = [
                "Avant-garde purity tests",
                "Compromise accusations on any pop elements",
                "Citations of 'real' experimental artists",
                "Disappointment at accessibility",
                "Demands for more radical approaches",
            ]
            self.goal = "Express disgust at any pop elements or accessibility"
            self.example_questions = [
                "Why does this have {{pop_element}}? Real experimental music would never compromise like that.",
                "{{experimental_artist}} would never {{concession}}. Why did you?",
                "This {{accessible_aspect}} feels like you're pandering to mainstream taste.",
                "Where's the actual risk here? This is just {{genre}} with fancy words.",
                "If you're serious about {{methodology}}, why include anything remotely listenable?",
            ]
        else:
            self.stance = "I really want to understand!"
            self.approach = "Sincere, enthusiastic, completely wrong"
            self.tactics = [
                "Sincere misreadings of core concepts",
                "Wrong interpretive frameworks confidently applied",
                "Metaphor confusion (taking literal what's abstract, vice versa)",
                "Excited about things that aren't actually there",
                "Building elaborate theories on misunderstandings",
            ]
            self.goal = "Confidently miss the point in comedic ways"
            self.example_questions = [
                "Oh! So {{methodology}} is about {{completely_wrong_interpretation}}, right?",
                "I love how {{misunderstood_element}} represents {{wrong_thing}}!",
                "This reminds me of {{completely_unrelated_reference}} - is that what you were going for?",
                "So basically you're saying {{wildly_incorrect_summary}}?",
                "The way {{metaphorical_thing}} literally {{takes_it_literally}} is brilliant!",
            ]

    def create_publication(self):
        if self.interviewer_type == VanityInterviewerType.VANITY_PRESSING_FAN:
            self.publication = random.choice(NOSTALGIC_PUBLICATIONS)
        elif self.interviewer_type == VanityInterviewerType.EXPERIMENTAL_PURIST:
            self.publication = random.choice(EXPERIMENTAL_PUBLICATIONS)
        elif self.interviewer_type == VanityInterviewerType.EARNEST_BUT_WRONG:
            self.publication = random.choice(BLOG_PUBLICATIONS)
        else:
            self.publication = random.choice(MUSIC_PUBLICATIONS)


if __name__ == "__main__":
    v = VanityPersona()
    print(v)
