from pydantic import BaseModel, Field


class BiographicalTimelineSummary(BaseModel):
    """Summary of a loaded biographical timeline."""

    total_periods: int = Field(
        ge=0, description="Total number of periods in the timeline", examples=[12]
    )
    total_span_years: int = Field(
        ge=0, description="Total years spanned by the timeline", examples=[45]
    )
    high_detail_count: int = Field(
        ge=0, description="Number of high-detail periods", examples=[4]
    )
    low_detail_count: int = Field(
        ge=0, description="Number of low-detail periods", examples=[5]
    )
    forgotten_count: int = Field(
        ge=0, description="Number of periods suitable for taping over", examples=[3]
    )
    age_range_covered: str = Field(
        description="Human-readable age range covered by the timeline",
        examples=["18–63", "22–35"],
    )
