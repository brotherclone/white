from pydantic import BaseModel


class BiographicalTimelineSummary(BaseModel):
    """Summary of loaded biographical timeline."""

    total_periods: int
    total_span_years: int
    high_detail_count: int
    low_detail_count: int
    forgotten_count: int
    age_range_covered: str
