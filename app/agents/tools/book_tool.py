import random
from typing import Optional

from pydantic import BaseModel

from app.agents.enums.book_condition import BookCondition
from app.agents.enums.publisher_type import PublisherType
from app.agents.models.book_data import BookData

TOPICS = [
    "Typhonian Workings in post-industrial soundscapes",
    "Sigil dynamics in electromagnetic interference patterns",
    "Cyber-goetia: Summoning through network protocols",
    "Egregoric entities in distributed computing",
    "The Enochian grammar of programming languages",
    "Chaos magic optimization algorithms",
    "Tulpamancy and artificial consciousness bootstrapping",
    "Servitor construction through procedural generation",
    "Hypersigils in version control systems",
    "The Voudon information theory of Claude Shannon",
    "Orgone accumulators and data center cooling",
    "Scalar wave modulation in fiber optics",
    "Reich's orgone and carrier wave coherence",
    "Torsion field topology in quantum computing",
    "Morphogenetic field programming",
    "Bioelectric pattern languages",
    "The electromagnetic spectrum of thoughtforms",
    "Kirlian photography of electrical circuits",
    "Radionics and software-defined radio",
    "Psychotronic weapons and user interface design",
    "Pre-diluvian computing substrates",
    "Atlantean information architecture",
    "Vedic descriptions of neural networks",
    "Sumerian clay tablets as read-only memory",
    "Egyptian hieroglyphs as compression algorithms",
    "Antikythera mechanism source code",
    "Library of Alexandria's backup protocols",
    "Gobekli Tepe as astronomical computer",
    "Mayan calendar systems and Unix time",
    "Dogon astronomical databases",
    "Glossolalia compilation techniques",
    "Xenolinguistic parsing strategies",
    "Angelic languages and error correction codes",
    "The grammar of light language transmissions",
    "Dolphin communication protocols",
    "Plant signaling syntax",
    "Mycelial network packet structure",
    "Bee waggle dance as routing algorithm",
    "Bird migration geodatabases",
    "Whale song compression formats",
    "Remote viewing session protocols",
    "Telepathic bandwidth measurements",
    "Precognitive data mining",
    "Retrocausality in version control",
    "Astral projection coordinate systems",
    "Out-of-body debugging techniques",
    "Thoughtography and screen capture",
    "Psychometry of hard drives",
    "Apportation and quantum teleportation",
    "Bilocation in distributed systems",
    "Biophoton emission spectra",
    "Cellular automata in embryogenesis",
    "DNA as quaternary storage medium",
    "Biocrystals and memory formation",
    "Fungal intelligence architectures",
    "Slime mold problem-solving algorithms",
    "Bacterial quorum sensing protocols",
    "Plant neurobiology signal processing",
    "Tardigrade data preservation strategies",
    "Extremophile adaptation algorithms",
    "Necromantic data recovery",
    "Vampiric energy extraction from power grids",
    "Lycanthropic transformation protocols",
    "Demonic possession and process injection",
    "Curse propagation through social networks",
    "Hex encoding in the original sense",
    "Evil eye and surveillance countermeasures",
    "Black mirror scrying and screen technology",
    "Familiar spirits and AI assistants",
    "Witch marks as cryptographic signatures",
    "Chronesthesia and timestamp manipulation",
    "Déjà vu recursion patterns",
    "Retrocausality debugging",
    "Prophecy and predictive analytics",
    "Time loop detection algorithms",
    "Temporal lensing in databases",
    "Grandfather paradox resolution protocols",
    "Butterfly effect mitigation strategies",
    "Akashic records as blockchain",
    "Time crystals and clock synchronization",
    "Bigfoot territory mapping algorithms",
    "Mothman prediction models",
    "Chupacabra behavioral analysis",
    "Black-eyed children facial recognition evasion",
    "Shadow people and render distance",
    "Tulpa construction through collective attention",
    "Egregores in social media platforms",
    "Phantom hitchhiker route optimization",
    "Men in Black counterintelligence protocols",
    "Alien abduction memory recovery tools",
    "Alchemy as materials science preprocessing",
    "Astrology as personality hashing functions",
    "Numerology and prime number distributions",
    "Sacred geometry in circuit design",
    "Mandala generation algorithms",
    "Yantra patterns in chip architecture",
    "Geomancy and network topology",
    "Dowsing and signal detection",
    "Crystallomancy and semiconductor physics",
    "Haruspicy applied to cable management",
]

FIRST_NAMES = [
    "Aleister", "Dion", "Israel", "Éliphas", "Austin", "Kenneth",
    "Manly", "Helena", "Rudolf", "Wilhelm", "Carl", "Sigmund",
    "Nikola", "Wilhelm", "Viktor", "Rupert", "Timothy", "Robert Anton",
    "Philip K.", "Terence", "John", "Fritjof", "Stanislav", "David",
    "Graham", "Jacques", "Buckminster", "Marshall", "Douglas", "Norbert"
]

LAST_NAMES = [
    "Crowley", "Fortune", "Regardie", "Lévi", "Spare", "Grant",
    "Hall", "Blavatsky", "Steiner", "Reich", "Jung", "Freud",
    "Tesla", "Burroughs", "Schauberger", "Sheldrake", "Leary", "Wilson",
    "Dick", "McKenna", "Dee", "Capra", "Grof", "Bohm",
    "Hancock", "Vallée", "Fuller", "McLuhan", "Hofstadter", "Wiener"
]

CREDENTIALS = [
    "Ph.D.", "M.Sc.", "D.Sc.", "M.D.", "Sc.D.",
    "F.R.S.", "Member of the Golden Dawn", "33° Mason",
    "Former NSA Researcher", "Declassified Analyst",
    "Independent Scholar", "Autodidact", "Visiting Fellow",
    "Professor Emeritus", "Former MIT Researcher",
    "Rogue Anthropologist", "Disgraced Physicist",
    "Expelled from CERN", "Banned from Publication"
]

UNIVERSITY_PRESSES = [
    "Miskatonic University Press",
    "Invisible College Press",
    "Arkham Academic Publishers",
    "Unseen University Press",
    "Institute for Fortean Studies",
    "Academy of Suppressed Sciences",
]

OCCULT_PUBLISHERS = [
    "Starfire Publishing",
    "Fulgur Limited",
    "Teitan Press",
    "Scarlet Imprint",
    "Three Hands Press",
    "Xoanon Publishing",
    "Revelore Press",
    "Anathema Publishing",
]

UNDERGROUND_PRESSES = [
    "Black Sun Press",
    "Samizdat Editions",
    "Mimeograph Underground",
    "Xerox Liberation Front",
    "Censored Publications Collective",
    "The Invisible Press",
]

VANITY_PUBLISHERS = [
    "Self-Published",
    "Author's Own Imprint",
    "Privately Printed",
    "Limited Private Edition",
    "Proof Copy Only",
]

SUPPRESSED_PUBLISHERS = [
    "Recovered from Library Fire",
    "Reconstructed Edition",
    "Suppressed by Order",
    "Banned Edition Society",
    "Underground Reprint",
]

GOVERNMENT_SOURCES = [
    "Declassified: Project Stargate",
    "Released Under FOIA",
    "CIA Document Archive",
    "Leaked NSA Research",
    "Redacted Military Report",
]

ACQUISITION_NOTES = [
    "Found in estate sale, previous owner unknown",
    "Traded for at underground book fair",
    "Recovered from abandoned university library",
    "Anonymous package, no return address",
    "Purchased from seller who seemed nervous",
    "Found in condemned building",
    "Inherited from disappeared colleague",
    "Confiscated copy, released to public domain",
    "Photocopied from library before fire",
    "Acquired at auction, provenance unclear",
    "Gift from anonymous benefactor",
    "Smuggled out of restricted archive",
]

SUPPRESSION_STORIES = [
    "Banned by university ethics board in 1973",
    "Publisher ceased operations under mysterious circumstances",
    "Author disappeared before publication",
    "Recalled shortly after release, reason undisclosed",
    "Subject of government investigation",
    "Denounced by multiple professional societies",
    "Library copies systematically destroyed",
    "Publisher served with cease and desist",
    "Author recanted findings under pressure",
    "Classified upon discovery of implications",
]


class BookMaker(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def generate_title(topic: str) -> tuple[str, Optional[str]]:
        """Convert topic into book title with optional subtitle"""

        title_templates = [
            lambda t: (t.split(':')[0] if ':' in t else t,
                       t.split(':')[1].strip() if ':' in t else None),
            lambda t: (f"The {t}", "A Practical Guide"),
            lambda t: (f"{t}", "Theory and Application"),
            lambda t: (f"Toward a New Understanding of {t}", None),
            lambda t: (f"{t}", "Forbidden Knowledge Revealed"),
            lambda t: (f"The Secret History of {t}", None),
            lambda t: (f"{t}", "An Underground Manual"),
            lambda t: (f"Collected Papers on {t}", None),
            lambda t: (f"{t}", "Suppressed Research 1947-1991"),
            lambda t: (f"Beyond {t}", "Implications for Modern Practice"),
        ]

        template = random.choice(title_templates)
        return template(topic)

    @staticmethod
    def generate_author() -> tuple[str, Optional[str]]:

        """Generate author name and credentials"""

        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        if random.random() < 0.3:
            first = f"{first[0]}."
        name = f"{first} {last}"
        creds = None
        if random.random() < 0.7:
            creds = random.choice(CREDENTIALS)

        return name, creds

    @staticmethod
    def generate_publisher(pub_type: PublisherType) -> str:

        """Get publisher name by type"""

        mapping = {
            PublisherType.UNIVERSITY: UNIVERSITY_PRESSES,
            PublisherType.OCCULT: OCCULT_PUBLISHERS,
            PublisherType.SAMIZDAT: UNDERGROUND_PRESSES,
            PublisherType.VANITY: VANITY_PUBLISHERS,
            PublisherType.LOST: SUPPRESSED_PUBLISHERS,
            PublisherType.GOVERNMENT: GOVERNMENT_SOURCES,
        }
        return random.choice(mapping[pub_type])

    @staticmethod
    def generate_catalog_number(year: int, index: int) -> str:
        """Generate Red Agent catalog number"""
        # Format: RA-[YEAR]-[DANGER]-[INDEX]
        danger = random.randint(1, 5)
        return f"RA-{year}-D{danger}-{index:04d}"

    @staticmethod
    def generate_abstract(topic: str) -> str:
        # ToDO: More abstracts
        """Generate academic-style abstract"""

        templates = [
            f"This groundbreaking work explores the intersection of {topic.lower()}, "
            f"challenging conventional assumptions and revealing hidden patterns that "
            f"have eluded mainstream researchers for decades.",

            f"Drawing on suppressed research and unconventional methodologies, the author "
            f"presents a comprehensive framework for understanding {topic.lower()}. "
            f"Includes detailed protocols and safety warnings.",

            f"A systematic investigation into {topic.lower()}, combining theoretical "
            f"foundations with practical applications. Notable for its controversial "
            f"conclusions regarding the nature of information and reality.",

            f"This rare volume documents extensive field research into {topic.lower()}, "
            f"presenting findings that were deemed too dangerous for mainstream publication. "
            f"Includes appendices with raw data and experimental protocols.",
        ]
        return random.choice(templates)

    @staticmethod
    def generate_quote(topic: str, author: str) -> str:
        # ToDO: More quotes

        """Generate a notable quote from the work"""

        quotes = [
            f'"The implications of {topic.lower()} extend far beyond what academic gatekeepers are willing to acknowledge."',
            f'"Once you understand {topic.lower()}, you cannot unsee the patterns everywhere."',
            f'"They tried to suppress this research, but the truth has a way of emerging."',
            f'"What we call reality is merely a subset of what is possible through {topic.lower()}."',
            f'"The resistance I encountered proved I was on the right track."',
        ]
        return f"{random.choice(quotes)} - {author}"

    @classmethod
    def generate_random_book(cls, index: Optional[int] = None) -> BookData:

        """Generate a complete random book from the collection"""

        topic = random.choice(TOPICS)
        title, subtitle = cls.generate_title(topic)
        author, credentials = cls.generate_author()
        year = random.randint(1947, 2023)
        pub_type = random.choice(list(PublisherType))
        publisher = cls.generate_publisher(pub_type)
        pages = random.randint(127, 847)
        condition = random.choice(list(BookCondition))
        idx = index if index is not None else random.randint(1, 9999)
        catalog_num = cls.generate_catalog_number(year, idx)
        has_isbn = random.random() < 0.4  # Only 40% have ISBNs
        isbn = f"978-{random.randint(0, 9)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(0, 9)}" if has_isbn else None
        translated = random.random() < 0.2  # 20% are translations
        original_lang = random.choice(["German", "French", "Russian", "Latin", "Sanskrit"]) if translated else None
        translator = f"{random.choice(cls.FIRST_NAMES)} {random.choice(cls.LAST_NAMES)}" if translated else None
        danger = random.randint(1, 5)
        tags = [word.lower() for word in topic.split()[:3]]
        has_suppression = random.random() < 0.4
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
            publisher_type=pub_type,
            edition=random.choice(["1st", "2nd", "3rd", "Revised", "Expanded", "Suppressed 1st"]),
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
            abstract=cls.generate_abstract(topic),
            notable_quote=cls.generate_quote(topic, author),
            suppression_history=suppression,
            related_works=[]
        )

    @classmethod
    def format_bibliography_entry(cls, book: BookData) -> str:
        """Format book as bibliography entry"""
        entry = f"{book.author}"
        if book.author_credentials:
            entry += f", {book.author_credentials}"
        entry += f" ({book.year}). *{book.title}"
        if book.subtitle:
            entry += f": {book.subtitle}"
        entry += f"*"
        if book.translator:
            entry += f" (Trans. {book.translator})"
        entry += f". {book.edition} ed. {book.publisher}"
        if book.isbn:
            entry += f". ISBN {book.isbn}"
        entry += f". [{book.catalog_number}]"
        return entry

    @classmethod
    def format_card_catalog(cls, book_to_catalog: BookData) -> str:
        """Format as old-school library card catalog entry with perfect alignment (70 chars wide)"""
        width = 70
        def pad(content):
            return f"║ {content:<{width-3}}║\n"
        card = "\n"
        card += "╔" + "═" * (width-2) + "╗\n"
        card += pad("RED AGENT COLLECTION - RESTRICTED ACCESS")
        card += "╠" + "═" * (width-2) + "╣\n"
        card += pad("")
        card += pad(f"CATALOG #: {book_to_catalog.catalog_number}")
        card += pad(f"DANGER LEVEL: {'⚠' * book_to_catalog.danger_level}")
        card += pad("")
        card += pad(f"TITLE: {book_to_catalog.title}")
        if book_to_catalog.subtitle:
            card += pad(f"{book_to_catalog.subtitle}")
        card += pad("")
        card += pad(f"AUTHOR: {book_to_catalog.author}")
        if book_to_catalog.author_credentials:
            card += pad(f"{book_to_catalog.author_credentials}")
        card += pad("")
        card += pad(f"PUBLISHED: {book_to_catalog.year} - {book_to_catalog.publisher}")
        card += pad(f"EDITION: {book_to_catalog.edition}")
        card += pad(f"PAGES: {book_to_catalog.pages}")
        card += pad("")
        card += pad(f"CONDITION: {book_to_catalog.condition.value}")
        card += pad(f"ACQUIRED: {book_to_catalog.acquisition_date}")
        card += pad("")
        if book_to_catalog.suppression_history:
            card += pad("⚠ SUPPRESSION NOTICE:")
            card += pad(f"{book_to_catalog.suppression_history}")
            card += pad("")
        card += "╚" + "═" * (width-2) + "╝\n"
        return card

if __name__ == "__main__":
    # Example usage
    book = BookMaker.generate_random_book(index=1)
    print(BookMaker.format_bibliography_entry(book))
    print(BookMaker.format_card_catalog(book))