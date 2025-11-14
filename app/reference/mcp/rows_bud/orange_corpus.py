"""
Orange Mythos Corpus Manager
Polars-based storage for Sussex County mythology
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl


class OrangeMythosCorpus:
    """Polars-based corpus for mythologizable stories"""

    # Schema definition
    SCHEMA = {
        "story_id": pl.Utf8,
        "headline": pl.Utf8,
        "date": pl.Date,
        "source": pl.Utf8,
        "location": pl.Utf8,
        "text": pl.Utf8,
        "tags": pl.List(pl.Utf8),
        "mythologization_score": pl.Float64,
        "symbolic_object_category": pl.Utf8,
        "symbolic_object_desc": pl.Utf8,
        "symbolic_object_inserted_at": pl.Datetime,
        "gonzo_text": pl.Utf8,
        "gonzo_perspective": pl.Utf8,
        "gonzo_intensity": pl.Int8,
        "gonzo_rewrite_at": pl.Datetime,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
    }

    def __init__(self, corpus_dir: Path):
        """
        Initialize corpus manager

        Args:
            corpus_dir: Directory for corpus storage
        """
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(exist_ok=True, parents=True)

        self.parquet_file = self.corpus_dir / "corpus.parquet"
        self.exports_dir = self.corpus_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)

        self.df = self._load_or_create()

    def _load_or_create(self) -> pl.DataFrame:
        """Load existing corpus or create new with schema"""
        if self.parquet_file.exists():
            df = pl.read_parquet(self.parquet_file)
            print(f"📚 Loaded corpus: {len(df)} stories")
            return df

        print("📚 Creating new corpus")
        return pl.DataFrame(schema=self.SCHEMA)

    def _save(self):
        """Persist corpus to parquet"""
        self.df.write_parquet(self.parquet_file)

    def add_story(
        self,
        headline: str,
        date: str,  # "YYYY-MM-DD"
        source: str,
        text: str,
        location: str,
        tags: List[str],
    ) -> tuple[str, float]:
        """
        Add a new story to corpus

        Returns:
            (story_id, mythologization_score)
        """
        story_id = str(uuid.uuid4())
        now = datetime.now()

        # Calculate score
        score = self._calculate_mythologization_score(tags, text, location)

        # Parse date
        try:
            date_parsed = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            date_parsed = datetime(1987, 1, 1).date()  # Fallback to mid-period

        new_row = pl.DataFrame(
            {
                "story_id": [story_id],
                "headline": [headline],
                "date": [date_parsed],
                "source": [source],
                "location": [location],
                "text": [text],
                "tags": [tags],
                "mythologization_score": [score],
                "symbolic_object_category": [None],
                "symbolic_object_desc": [None],
                "symbolic_object_inserted_at": [None],
                "gonzo_text": [None],
                "gonzo_perspective": [None],
                "gonzo_intensity": [None],
                "gonzo_rewrite_at": [None],
                "created_at": [now],
                "updated_at": [now],
            }
        )

        # Concat to main dataframe
        self.df = pl.concat([self.df, new_row], how="vertical")
        self._save()

        print(f"   Added story: {story_id} (score: {score:.2f})")
        return story_id, score

    def get_story(self, story_id: str) -> dict | None:
        """Get story by ID as dict"""
        result = self.df.filter(pl.col("story_id") == story_id)

        if len(result) == 0:
            return None

        # Convert to dict, handling None values properly
        story_dict = result.to_dicts()[0]

        # Convert date to string for JSON compatibility
        if story_dict.get("date"):
            story_dict["date"] = story_dict["date"].isoformat()

        return story_dict

    def insert_symbolic_object(
        self, story_id: str, category: str, description: str, updated_text: str
    ):
        """Update story with symbolic object insertion"""
        now = datetime.now()

        # Update the matching row
        self.df = self.df.with_columns(
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(updated_text))
            .otherwise(pl.col("text"))
            .alias("text"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(category))
            .otherwise(pl.col("symbolic_object_category"))
            .alias("symbolic_object_category"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(description))
            .otherwise(pl.col("symbolic_object_desc"))
            .alias("symbolic_object_desc"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(now))
            .otherwise(pl.col("symbolic_object_inserted_at"))
            .alias("symbolic_object_inserted_at"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(now))
            .otherwise(pl.col("updated_at"))
            .alias("updated_at"),
        )

        self._save()
        print(f"   Object inserted into story: {story_id}")

    def add_gonzo_rewrite(
        self, story_id: str, gonzo_text: str, perspective: str, intensity: int
    ):
        """Add gonzo rewritten version"""
        now = datetime.now()

        self.df = self.df.with_columns(
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(gonzo_text))
            .otherwise(pl.col("gonzo_text"))
            .alias("gonzo_text"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(perspective))
            .otherwise(pl.col("gonzo_perspective"))
            .alias("gonzo_perspective"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(intensity))
            .otherwise(pl.col("gonzo_intensity"))
            .alias("gonzo_intensity"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(now))
            .otherwise(pl.col("gonzo_rewrite_at"))
            .alias("gonzo_rewrite_at"),
            pl.when(pl.col("story_id") == story_id)
            .then(pl.lit(now))
            .otherwise(pl.col("updated_at"))
            .alias("updated_at"),
        )

        self._save()
        print(f"   Gonzo rewrite added to story: {story_id}")

    def search(
        self,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0,
        location: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        needs_mythologizing: bool = False,
    ) -> List[dict]:
        """
        Search corpus with filters

        Returns:
            List of story dicts sorted by score descending
        """
        result = self.df

        # Filter by tags (must have ALL specified tags)
        if tags:
            for tag in tags:
                result = result.filter(pl.col("tags").list.contains(tag))

        # Filter by score
        if min_score > 0:
            result = result.filter(pl.col("mythologization_score") >= min_score)

        # Filter by location
        if location:
            result = result.filter(
                pl.col("location").str.to_lowercase().str.contains(location.lower())
            )

        # Filter by date range
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            result = result.filter(pl.col("date") >= start)

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            result = result.filter(pl.col("date") <= end)

        # Filter for stories needing mythologization
        if needs_mythologizing:
            result = result.filter(pl.col("gonzo_text").is_null())

        # Sort by score descending
        result = result.sort("mythologization_score", descending=True)

        # Convert to dicts
        stories = result.to_dicts()

        # Convert dates to strings
        for story in stories:
            if story.get("date"):
                story["date"] = story["date"].isoformat()

        return stories

    def get_stats(self) -> dict:
        """Get corpus statistics"""
        if len(self.df) == 0:
            return {"total_stories": 0, "status": "empty_corpus"}

        # Date range
        date_min = self.df["date"].min()
        date_max = self.df["date"].max()

        # Tag distribution
        tags_exploded = self.df.select(pl.col("tags").explode()).to_series()
        tag_counts = tags_exploded.value_counts().sort("count", descending=True)
        tag_distribution = {row["tags"]: row["count"] for row in tag_counts.to_dicts()}

        # Mythologization status
        with_objects = self.df.filter(
            pl.col("symbolic_object_desc").is_not_null()
        ).height

        with_gonzo = self.df.filter(pl.col("gonzo_text").is_not_null()).height

        # Score stats
        scores = self.df["mythologization_score"]

        return {
            "total_stories": len(self.df),
            "date_range": {
                "start": date_min.isoformat() if date_min else None,
                "end": date_max.isoformat() if date_max else None,
            },
            "tag_distribution": tag_distribution,
            "mythologization_status": {
                "with_symbolic_objects": with_objects,
                "with_gonzo_rewrites": with_gonzo,
                "ready_to_mythologize": len(self.df) - with_gonzo,
            },
            "scores": {
                "average": float(scores.mean()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "median": float(scores.median()),
            },
            "status": "success",
        }

    def export_json(self, filename: str = "corpus_export.json"):
        """Export the entire corpus as JSON for human inspection"""
        output_path = self.exports_dir / filename

        # Convert to JSON-friendly format
        export_df = self.df.with_columns(
            [
                pl.col("date").cast(pl.Utf8),
                pl.col("created_at").cast(pl.Utf8),
                pl.col("updated_at").cast(pl.Utf8),
                pl.col("symbolic_object_inserted_at").cast(pl.Utf8),
                pl.col("gonzo_rewrite_at").cast(pl.Utf8),
            ]
        )

        export_df.write_json(output_path)
        print(f"📄 Exported JSON: {output_path}")
        return str(output_path)

    def export_for_training(self, filename: str = "orange_corpus_training.parquet"):
        """
        Export mythologized stories for the training pipeline

        Only includes stories with both original text and gonzo version
        """
        output_path = self.exports_dir / filename

        # Filter for complete mythologization
        training_df = self.df.filter(
            pl.col("gonzo_text").is_not_null()
            & pl.col("symbolic_object_desc").is_not_null()
        )

        # Select relevant columns for training
        training_df = training_df.select(
            [
                "story_id",
                "headline",
                "date",
                "location",
                "tags",
                "text",
                "gonzo_text",
                "symbolic_object_category",
                "symbolic_object_desc",
                "gonzo_intensity",
                "mythologization_score",
            ]
        )

        training_df.write_parquet(output_path)
        print(f"📊 Exported training data: {output_path} ({len(training_df)} stories)")
        return str(output_path)

    @staticmethod
    def _calculate_mythologization_score(
        tags: List[str], text: str, location: str
    ) -> float:
        """
        Calculate mythologization score (0.0-1.0)

        Scoring factors:
        - Has 2+ required tags: +0.3
        - Has 3+ tags: +0.2
        - Unexplained element: +0.2
        - Youth/teenage angle: +0.15
        - Location specificity: +0.15
        """
        score = 0.0

        tag_set = set(tags)
        required_tags = {
            "rock_bands",
            "youth_crime",
            "unexplained",
            "mental_health",
            "psychedelics",
        }

        # Tag coverage
        tag_count = len(tag_set & required_tags)
        if tag_count >= 2:
            score += 0.3
        if tag_count >= 3:
            score += 0.2

        # Unexplained phenomena bonus
        if "unexplained" in tag_set:
            score += 0.2

        # Youth angle bonus
        text_lower = text.lower()
        if "youth_crime" in tag_set or any(
            word in text_lower for word in ["teenage", "teen", "youth", "young"]
        ):
            score += 0.15

        # Location specificity bonus
        location_lower = location.lower()
        if "township" in location_lower or "borough" in location_lower:
            score += 0.15

        return min(score, 1.0)

    def get_high_value_stories(self, limit: int = 10) -> List[dict]:
        """Get top mythologizable stories that haven't been gonzo'd yet"""
        result = (
            self.df.filter(pl.col("gonzo_text").is_null())
            .sort("mythologization_score", descending=True)
            .limit(limit)
        )

        stories = result.to_dicts()
        for story in stories:
            if story.get("date"):
                story["date"] = story["date"].isoformat()

        return stories

    def analyze_temporal_patterns(self) -> dict:
        """Analyze when mysterious events cluster (for mythology mining)"""
        # Group by year
        yearly = (
            self.df.with_columns(pl.col("date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                [
                    pl.count().alias("story_count"),
                    pl.mean("mythologization_score").alias("avg_score"),
                    pl.col("tags").flatten().unique().alias("all_tags"),
                ]
            )
            .sort("year")
        )

        # Find peak mythology years
        patterns = yearly.to_dicts()

        return {
            "yearly_distribution": patterns,
            "peak_years": yearly.sort("story_count", descending=True)
            .limit(5)
            .to_dicts(),
            "highest_scoring_years": yearly.sort("avg_score", descending=True)
            .limit(5)
            .to_dicts(),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_corpus(corpus_dir: str = "./mythology_corpus") -> OrangeMythosCorpus:
    """Get or create the corpus instance (singleton pattern)"""
    return OrangeMythosCorpus(Path(corpus_dir))
