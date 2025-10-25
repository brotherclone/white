import random
from typing import Optional


from pydantic import BaseModel

from app.agents.enums.book_condition import BookCondition
from app.agents.enums.publisher_type import PublisherType
from app.agents.enums.book_genre import BookGenre
from app.agents.models.book_data import BookData
from app.util.string_utils import truncate_word_safe
from app.reference.books.book_topics import SCIFI_TOPICS, SEXPLOITATION_TOPICS, OCCULT_TOPICS, CULT_TOPICS, \
    BILDUNGSROMAN_TOPICS, NOIR_TOPICS, PSYCHEDELIC_TOPICS
from app.reference.books.book_authors import OCCULT_FIRST_NAMES, OCCULT_LAST_NAMES, SCIFI_FIRST_NAMES, SCIFI_LAST_NAMES, \
    SEXPLOITATION_FIRST_NAMES, SEXPLOITATION_LAST_NAMES, CULT_FIRST_NAMES, CULT_LAST_NAMES, \
    BILDUNGSROMAN_FIRST_NAMES, BILDUNGSROMAN_LAST_NAMES, NOIR_FIRST_NAMES, NOIR_LAST_NAMES, \
    PSYCHEDELIC_FIRST_NAMES, PSYCHEDELIC_LAST_NAMES, CREDENTIALS
from app.reference.books.book_publishers import OCCULT_PUBLISHERS, SCIFI_PUBLISHERS, SEXPLOITATION_PUBLISHERS, \
    CULT_PUBLISHERS, BILDUNGSROMAN_PUBLISHERS, NOIR_PUBLISHERS, PSYCHEDELIC_PUBLISHERS
from app.reference.books.book_misc import ACQUISITION_NOTES, SUPPRESSION_STORIES

GENRE_WEIGHTS = {
    BookGenre.OCCULT: 10,
    BookGenre.SCIFI: 25,
    BookGenre.SEXPLOITATION: 15,
    BookGenre.CULT: 20,
    BookGenre.BILDUNGSROMAN: 15,
    BookGenre.NOIR: 10,
    BookGenre.PSYCHEDELIC: 5,
}

class BookMaker(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def select_genre() -> BookGenre:
        """Weighted random genre selection"""
        genres = list(GENRE_WEIGHTS.keys())
        weights = list(GENRE_WEIGHTS.values())
        return random.choices(genres, weights=weights)[0]

    @staticmethod
    def get_topics_for_genre(g: BookGenre) -> list:
        """Return topic list for a given genre"""
        mapping = {
            BookGenre.OCCULT: OCCULT_TOPICS,
            BookGenre.SCIFI: SCIFI_TOPICS,
            BookGenre.SEXPLOITATION: SEXPLOITATION_TOPICS,
            BookGenre.CULT: CULT_TOPICS,
            BookGenre.BILDUNGSROMAN: BILDUNGSROMAN_TOPICS,
            BookGenre.NOIR: NOIR_TOPICS,
            BookGenre.PSYCHEDELIC: PSYCHEDELIC_TOPICS,
        }
        return mapping[g]

    @staticmethod
    def get_authors_for_genre(book_genre: BookGenre) -> tuple[list, list]:
        """Return first/last name lists for genre"""
        mapping = {
            BookGenre.OCCULT: (OCCULT_FIRST_NAMES, OCCULT_LAST_NAMES),
            BookGenre.SCIFI: (SCIFI_FIRST_NAMES, SCIFI_LAST_NAMES),
            BookGenre.SEXPLOITATION: (SEXPLOITATION_FIRST_NAMES, SEXPLOITATION_LAST_NAMES),
            BookGenre.CULT: (CULT_FIRST_NAMES, CULT_LAST_NAMES),
            BookGenre.BILDUNGSROMAN: (BILDUNGSROMAN_FIRST_NAMES, BILDUNGSROMAN_LAST_NAMES),
            BookGenre.NOIR: (NOIR_FIRST_NAMES, NOIR_LAST_NAMES),
            BookGenre.PSYCHEDELIC: (PSYCHEDELIC_FIRST_NAMES, PSYCHEDELIC_LAST_NAMES),
        }
        return mapping[book_genre]

    @staticmethod
    def get_publishers_for_genre(book_genre: BookGenre) -> list:
        """Return publisher list for genre"""
        mapping = {
            BookGenre.OCCULT: OCCULT_PUBLISHERS,
            BookGenre.SCIFI: SCIFI_PUBLISHERS,
            BookGenre.SEXPLOITATION: SEXPLOITATION_PUBLISHERS,
            BookGenre.CULT: CULT_PUBLISHERS,
            BookGenre.BILDUNGSROMAN: BILDUNGSROMAN_PUBLISHERS,
            BookGenre.NOIR: NOIR_PUBLISHERS,
            BookGenre.PSYCHEDELIC: PSYCHEDELIC_PUBLISHERS,
        }
        return mapping[book_genre]

    @staticmethod
    def generate_title(topic: str, book_genre: BookGenre) -> tuple[str, Optional[str]]:
        """Convert topic into book title with optional subtitle"""
        if book_genre in [BookGenre.SCIFI, BookGenre.SEXPLOITATION, BookGenre.CULT,
                          BookGenre.BILDUNGSROMAN, BookGenre.NOIR, BookGenre.PSYCHEDELIC]:
            if ':' in topic:
                parts = topic.split(':', 1)
                return parts[0].strip(), parts[1].strip()
            return topic, None
        title_templates = [
            lambda t: (t.split(':')[0] if ':' in t else t,
                       t.split(':')[1].strip() if ':' in t else None),
            lambda t: (f"The {t}", "A Practical Guide"),
            lambda t: (f"{t}", "Theory and Application"),
            lambda t: (f"Toward a New Understanding of {t}", None),
        ]
        template = random.choice(title_templates)
        return template(topic)

    @staticmethod
    def generate_author(book_genre: BookGenre) -> tuple[str, Optional[str]]:
        """Generate author name and credentials"""
        first_names, last_names = BookMaker.get_authors_for_genre(book_genre)
        first = random.choice(first_names)
        last = random.choice(last_names)

        # Sometimes use initials
        if random.random() < 0.3 and '.' not in first:
            first = f"{first[0]}."

        name = f"{first} {last}"

        # Credentials less common for fiction
        creds = None
        if book_genre == BookGenre.OCCULT:
            if random.random() < 0.7:
                creds = random.choice(CREDENTIALS)
        elif book_genre in [BookGenre.BILDUNGSROMAN, BookGenre.PSYCHEDELIC]:
            if random.random() < 0.3:
                creds = random.choice(CREDENTIALS)

        return name, creds

    @staticmethod
    def generate_catalog_number(year: int, index: int, g: BookGenre) -> str:
        """Generate Red Agent catalog number with genre code"""
        # Format: RA-[YEAR]-[GENRE_CODE]-[INDEX]
        genre_codes = {
            BookGenre.OCCULT: "OCC",
            BookGenre.SCIFI: "SCI",
            BookGenre.SEXPLOITATION: "SEX",
            BookGenre.CULT: "CLT",
            BookGenre.BILDUNGSROMAN: "BLD",
            BookGenre.NOIR: "NOR",
            BookGenre.PSYCHEDELIC: "PSY",
        }
        code = genre_codes[g]
        return f"RA-{year}-{code}-{index:04d}"

    @staticmethod
    def generate_abstract(topic: str, book_genre: BookGenre) -> str:
        """Generate genre-appropriate description"""

        if book_genre == BookGenre.SCIFI:
            templates = [
                f"A mind-bending exploration of {topic.lower()} that questions the nature "
                f"of reality itself. Classic hard SF with philosophical depth.",
                f"In a future where {topic.lower()} has transformed society, one individual "
                f"must confront what it means to be human. Nominated for the Nebula Award.",
                f"A haunting meditation on {topic.lower()} that bridges the gap between "
                f"golden age optimism and new wave experimentation.",
            ]
        elif book_genre == BookGenre.SEXPLOITATION:
            templates = [
                f"A steamy tale of desire in the digital age. {topic} pulses with "
                f"electric tension and forbidden pleasures. Adults only.",
                f"Where flesh meets technology, boundaries dissolve. {topic} explores "
                f"the intimate intersections of body and machine.",
                f"Lurid, provocative, unforgettable. {topic} pushed the boundaries "
                f"of what pulp fiction could be.",
            ]
        elif book_genre == BookGenre.CULT:
            templates = [
                f"An experimental narrative that deconstructs {topic.lower()} through "
                f"fractured prose and nonlinear time. Not for the faint of heart.",
                f"Banned in three countries upon release. {topic} remains one of the most "
                f"controversial works of the underground literary movement.",
                f"A fever dream of language and consciousness. {topic} defies categorization.",
            ]
        elif book_genre == BookGenre.BILDUNGSROMAN:
            templates = [
                f"A coming-of-age story for the digital generation. {topic} captures "
                f"the strange beauty of growing up online.",
                f"Tender, brutal, and deeply human. {topic} traces one person's journey "
                f"toward understanding in an increasingly mediated world.",
                f"A modern classic of identity formation. {topic} explores what it means "
                f"to become yourself when the self is distributed across networks.",
            ]
        elif book_genre == BookGenre.NOIR:
            templates = [
                f"In the shadowy corridors of Silicon Valley, a hardboiled detective "
                f"investigates {topic.lower()}. Femme fatales and server rooms collide.",
                f"Murder, mystery, and malware. {topic} is classic noir for the information age.",
                f"A twisting tale of corruption and code. {topic} updates the detective "
                f"story for the digital era.",
            ]
        elif book_genre == BookGenre.PSYCHEDELIC:
            templates = [
                f"A consciousness-expanding journey through {topic.lower()}. "
                f"Reality tunnels, reality TV, reality itself—all up for grabs.",
                f"Turn on, boot up, drop out. {topic} chronicles the intersection "
                f"of psychedelic culture and emerging technology.",
                f"From Leary to cyberspace: {topic} maps the evolution of consciousness "
                f"in the age of information.",
            ]
        else:
            templates = [
                f"This groundbreaking work explores the intersection of {topic.lower()}, "
                f"challenging conventional assumptions and revealing hidden patterns.",
                f"Drawing on suppressed research, the author presents a framework "
                f"for understanding {topic.lower()}. Includes detailed protocols.",
            ]

        return random.choice(templates)

    @staticmethod
    def generate_quote(topic: str, author: str, book_genre: BookGenre) -> str:
        """Generate a notable quote from the work"""

        if book_genre == BookGenre.SCIFI:
            quotes = [
                f'"The future is already here—it\'s just not evenly distributed." - {author}, on {topic.lower()}',
                f'"We were so busy asking if we could, we never stopped to ask if we should."',
                f'"Reality is that which, when you stop believing in it, doesn\'t go away."',
            ]
        elif book_genre == BookGenre.SEXPLOITATION:
            quotes = [
                f'"Where does the body end and the machine begin? In the heat of passion, who cares?"',
                f'"Touch me through the interface. Show me what bandwidth can do."',
                f'"Every connection leaves its mark. Some connections leave scars."',
            ]
        elif book_genre == BookGenre.CULT:
            quotes = [
                f'"Language is a virus from outer space." - {author}',
                f'"The word is now a virus. The flu virus may once have been a healthy lung cell."',
                f'"Nothing is true, everything is permitted."',
            ]
        elif book_genre == BookGenre.NOIR:
            quotes = [
                f'"It was a dark and stormy network."',
                f'"She walked in like bad code—beautiful, dangerous, and bound to crash the system."',
                f'"In this city, everyone\'s guilty of something. I just had to find out what."',
            ]
        else:
            quotes = [
                f'"Once you see it, you cannot unsee the patterns everywhere."',
                f'"What we call reality is merely a subset of what is possible."',
            ]

        return random.choice(quotes)

    @classmethod
    def generate_random_book(cls, index: Optional[int] = None,
                             force_genre: Optional[BookGenre] = None) -> BookData:
        """Generate a complete random book from the collection"""

        # Select genre
        the_genre = force_genre if force_genre else cls.select_genre()

        # Get genre-specific content
        topics = cls.get_topics_for_genre(the_genre)
        topic = random.choice(topics)

        title, subtitle = cls.generate_title(topic, the_genre)
        author, credentials = cls.generate_author(the_genre)

        year = random.randint(1947, 2023)
        publishers = cls.get_publishers_for_genre(the_genre)
        publisher = random.choice(publishers)

        pages = random.randint(127, 847)
        condition = random.choice(list(BookCondition))
        idx = index if index is not None else random.randint(1, 9999)
        catalog_num = cls.generate_catalog_number(year, idx, the_genre)

        has_isbn = random.random() < 0.4
        isbn = f"978-{random.randint(0, 9)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(0, 9)}" if has_isbn else None

        translated = random.random() < 0.2
        original_lang = random.choice(["German", "French", "Russian", "Japanese", "Italian"]) if translated else None
        translator = f"{random.choice(SCIFI_FIRST_NAMES)} {random.choice(SCIFI_LAST_NAMES)}" if translated else None

        danger_weights = {
            BookGenre.OCCULT: [1, 2, 3, 4, 5],
            BookGenre.SCIFI: [1, 1, 2, 2, 3],
            BookGenre.SEXPLOITATION: [1, 2, 2, 3, 3],
            BookGenre.CULT: [2, 3, 3, 4, 4],
            BookGenre.BILDUNGSROMAN: [1, 1, 1, 2, 2],
            BookGenre.NOIR: [1, 2, 2, 2, 3],
            BookGenre.PSYCHEDELIC: [2, 3, 3, 4, 4],
        }
        danger = random.choice(danger_weights[the_genre])

        tags = [word.lower() for word in title.split()[:3] if len(word) > 3]
        tags.append(the_genre.value)

        has_suppression = random.random() < 0.3
        suppression = random.choice(SUPPRESSION_STORIES) if has_suppression else None

        acquisition_year = random.randint(year, 2024)
        acquisition_date = f"{random.choice(['January', 'March', 'June', 'October'])} {acquisition_year}"
        acquisition_note = random.choice(ACQUISITION_NOTES)

        return BookData(
            title=title,
            subtitle=subtitle,
            author=author,
            author_credentials=credentials,
            year=year,
            publisher=publisher,
            publisher_type=PublisherType.OCCULT,  # Using existing enum
            edition=random.choice(["1st", "2nd", "3rd", "Revised", "Expanded", "Mass Market"]),
            pages=pages,
            isbn=isbn,
            catalog_number=catalog_num,
            condition=condition,
            acquisition_date=acquisition_date,
            acquisition_notes=acquisition_note,
            language="English",
            translated_from=original_lang,
            translator=translator,
            tags=tags,
            danger_level=danger,
            abstract=cls.generate_abstract(topic, the_genre),
            notable_quote=cls.generate_quote(topic, author, the_genre),
            suppression_history=suppression,
            related_works=[]
        )

    @classmethod
    def format_bibliography_entry(cls, the_book: BookData) -> str:
        """Format book as bibliography entry"""
        entry = f"{the_book.author}"
        if the_book.author_credentials:
            entry += f", {the_book.author_credentials}"
        entry += f" ({the_book.year}). *{the_book.title}"
        if the_book.subtitle:
            entry += f": {the_book.subtitle}"
        entry += f"*"
        if the_book.translator:
            entry += f" (Trans. {the_book.translator})"
        entry += f". {the_book.edition} ed. {the_book.publisher}"
        if the_book.isbn:
            entry += f". ISBN {the_book.isbn}"
        entry += f". [{the_book.catalog_number}]"
        return entry

    @classmethod
    def format_card_catalog(cls, book_to_catalog: BookData) -> str:
        """Format as old-school library card catalog entry with perfect alignment (70 chars wide)"""
        width = 70

        def pad(content):
            return f"║ {content:<{width - 3}}║\n"

        card = "\n"
        card += "╔" + "═" * (width - 2) + "╗\n"
        card += pad("LIGHT READER COLLECTION - RESTRICTED ACCESS")
        card += "╠" + "═" * (width - 2) + "╣\n"
        card += pad("")
        card += pad(f"CATALOG #: {book_to_catalog.catalog_number}")
        card += pad(f"DANGER LEVEL: {'⚠ ' * book_to_catalog.danger_level}")
        card += pad("")

        title_line = f"TITLE: {book_to_catalog.title}"
        if len(title_line) > width - 3:
            title_line = truncate_word_safe(title_line, width - 2)
        card += pad(title_line)

        if book_to_catalog.subtitle:
            subtitle_line = f"       {book_to_catalog.subtitle}"
            if len(subtitle_line) > width - 3:
                subtitle_line = truncate_word_safe(subtitle_line, width - 2)
            card += pad(subtitle_line)

        card += pad("")
        card += pad(f"AUTHOR: {book_to_catalog.author}")
        if book_to_catalog.author_credentials:
            card += pad(f"        {book_to_catalog.author_credentials}")

        card += pad("")
        card += pad(f"PUBLISHED: {book_to_catalog.year} - {book_to_catalog.publisher}")
        card += pad(f"EDITION: {book_to_catalog.edition}")
        card += pad(f"PAGES: {book_to_catalog.pages}")

        card += pad("")
        card += pad(f"CONDITION: {book_to_catalog.condition.value}")
        card += pad(f"ACQUIRED: {book_to_catalog.acquisition_date}")

        if book_to_catalog.suppression_history:
            card += pad("")
            card += pad("⚠ SUPPRESSION NOTICE:")
            supp_lines = book_to_catalog.suppression_history
            if len(supp_lines) > width - 3:
                # Word wrap suppression history
                words = supp_lines.split()
                line = ""
                for word in words:
                    if len(line + word) < width - 5:
                        line += word + " "
                    else:
                        card += pad(f"  {line.strip()}")
                        line = word + " "
                if line:
                    card += pad(f"  {line.strip()}")
            else:
                card += pad(f"  {supp_lines}")

        card += pad("")
        card += "╚" + "═" * (width - 2) + "╝\n"
        return card


if __name__ == "__main__":
    # Example usage - generate books from each genre
    print("=== LIGHT READING COLLECTION SAMPLES ===\n")

    for genre in BookGenre:
        print(f"\n{'=' * 70}")
        print(f"GENRE: {genre.value.upper()}")
        print('=' * 70)
        book = BookMaker.generate_random_book(force_genre=genre)
        print(BookMaker.format_card_catalog(book))

    # Generate some random books
    print(f"\n{'=' * 70}")
    print("RANDOM SELECTIONS")
    print('=' * 70)
    for i in range(3):
        book = BookMaker.generate_random_book(index=i + 1)
        print(BookMaker.format_bibliography_entry(book))
        print()