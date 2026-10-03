import os
import re
import ollama

from embedding import embed_query
from vectorstore import search


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")

client = ollama.Client(
    host=f"http://{OLLAMA_HOST}:11434"
)

MODEL_NAME = "llama3.2:3b"

NOT_COVERED = "IT policy does not cover this."


SYSTEM_PROMPT = """
You are an IT Policy Assistant.

The application has already retrieved policy passages
that are relevant to the user's question.

Answer using ONLY the supplied IT policy context.

Rules:

1. Use only information explicitly stated in the context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. Answer the user's question directly.
5. If a policy statement applies to a group of items,
   it also applies to each item named in that group.
6. Do not incorrectly say something is not covered when
   the requested item is explicitly included in a broader policy statement.
7. Do not change who performs an action.
8. Do not add responsibilities that are not explicitly stated.
9. If the question asks who is eligible or entitled to receive a device,
   do not treat the scope of the policy as proof of eligibility.
10. Keep the answer clear and concise.
11. Do not mention page numbers or section names.
    The application adds the source separately.
"""


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "who",
    "which",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "should",
    "would",
    "does",
    "do",
    "did",
    "of",
    "to",
    "for",
    "from",
    "in",
    "on",
    "at",
    "by",
    "with",
    "and",
    "or",
    "first",
    "this",
    "that",
    "it",
    "they",
    "their",
    "them",
    "employee",
    "employees",
    "user",
    "users",
    "company",
    "policy",
    "requirements",
    "requirement",
    "rules",
    "rule",
}


# Common spelling mistakes / wording users may type
TYPO_MAP = {
    "approvle": "approval",
    "aproval": "approval",
    "aproove": "approve",
    "poilcy": "policy",
    "polciy": "policy",
    "mounter": "monitor",
    "moniter": "monitor",
    "montior": "monitor",
    "tack": "take",
    "recieve": "receive",
    "reseave": "receive",
    "recive": "receive",
    "lapotp": "laptop",
    "latpop": "laptop",
}


PROCUREMENT_WORDS = {
    "buy",
    "buying",
    "purchase",
    "purchasing",
    "procure",
    "procurement",
    "approval",
    "approve",
    "approved",
    "authorization",
    "authorisation",
}


DEVICE_WORDS = {
    "laptop",
    "laptops",
    "computer",
    "computers",
    "device",
    "devices",
    "monitor",
    "monitors",
    "screen",
    "screens",
    "display",
    "displays",
    "accessory",
    "accessories",
}


DEVICE_ASSIGNMENT_WORDS = {
    "take",
    "takes",
    "get",
    "gets",
    "receive",
    "receives",
    "eligible",
    "eligibility",
    "entitled",
    "entitlement",
    "assigned",
    "assignment",
    "allowed",
}


PASSWORD_WORDS = {
    "password",
    "passwords",
    "passphrase",
    "passphrases",
    "credential",
    "credentials",
}


REMOTE_WORDS = {
    "remote",
    "remotely",
    "vpn",
    "offsite",
}


TOPIC_ANCHORS = {
    "procurement": [
        "it procurement",
        "procurement authorization",
        "hardware",
        "software",
        "purchase",
        "purchased",
        "equipment",
        "it operations manager",
        "line manager",
    ],

    "end_user_devices": [
        "end user it devices",
        "devices",
        "laptop",
        "laptops",
        "monitor",
        "screen",
        "accessories",
        "shared devices",
        "type of devices",
        "device",
    ],

    "password": [
        "password",
        "passwords",
        "90 days",
        "special character",
        "failed password",
        "account locks",
        "24 passwords",
    ],

    "remote_access": [
        "remote access",
        "vpn",
        "virtual private network",
        "remotely",
        "remote connection",
        "remote access privileges",
    ],
}


def normalize_user_text(text):
    """
    Fix common spelling mistakes before retrieval.
    """

    normalized = text.lower().strip()

    for wrong, correct in TYPO_MAP.items():
        normalized = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct,
            normalized
        )

    return normalized


def split_questions(text):
    """
    Split multiple questions when question marks are used.
    """

    text = text.strip()

    parts = re.findall(
        r"[^?]+(?:\?|$)",
        text
    )

    questions = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return questions or [text]


def tokenize(text):
    """
    Convert text into useful searchable words.
    """

    text = normalize_user_text(text)

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }


def detect_topic(question):
    """
    Identify the likely policy area.
    """

    words = tokenize(question)

    # Password
    if words.intersection(PASSWORD_WORDS):
        return "password"

    # Remote access
    if words.intersection(REMOTE_WORDS):
        return "remote_access"

    # Explicit procurement wording wins
    if words.intersection(PROCUREMENT_WORDS):
        return "procurement"

    # Device assignment / entitlement
    if (
        words.intersection(DEVICE_WORDS)
        and words.intersection(DEVICE_ASSIGNMENT_WORDS)
    ):
        return "end_user_devices"

    # Screen / monitor questions
    if words.intersection({
        "monitor",
        "monitors",
        "screen",
        "screens",
        "display",
        "displays",
    }):
        return "end_user_devices"

    # Generic laptop/device policy question
    if (
        words.intersection(DEVICE_WORDS)
        and "policy" in normalize_user_text(question)
    ):
        return "end_user_devices"

    return None


def expand_query(question):
    """
    Create a stronger semantic-search query.
    """

    normalized = normalize_user_text(question)

    topic = detect_topic(
        normalized
    )

    if topic == "procurement":
        return (
            f"{normalized}. "
            "IT procurement requirements for purchasing hardware, "
            "software, laptops, computers, devices or equipment. "
            "IT Procurement Authorization, approval, line manager "
            "and IT Operations Manager."
        )

    if topic == "end_user_devices":
        return (
            f"{normalized}. "
            "End user IT devices and accessories policy. "
            "Laptop, monitor, screen, computer and device eligibility, "
            "assignment, allocation, entitlement, categories and approval."
        )

    if topic == "password":
        return (
            f"{normalized}. "
            "Company password requirements, password rules, "
            "password length, complexity, expiration, reuse, "
            "lockout and security requirements."
        )

    if topic == "remote_access":
        return (
            f"{normalized}. "
            "Company remote access requirements, VPN rules, "
            "remote connection, approval and security requirements."
        )

    return normalized


def normalize_question_for_generation(question):
    """
    Rewrite short vague phrases into clearer questions
    before sending them to Ollama.
    """

    normalized = normalize_user_text(
        question
    )

    words = tokenize(
        normalized
    )

    topic = detect_topic(
        normalized
    )

    if len(words) > 5:
        return normalized

    if topic == "procurement":

        if "software" in words:
            return (
                "What are the IT policy requirements "
                "for purchasing software?"
            )

        if words.intersection({
            "laptop",
            "laptops",
            "computer",
            "computers",
        }):
            return (
                "What are the IT policy requirements "
                "for purchasing a laptop or computer?"
            )

        return (
            "What are the IT procurement requirements "
            "and required approvals?"
        )

    if topic == "end_user_devices":

        if words.intersection({
            "monitor",
            "monitors",
            "screen",
            "screens",
            "display",
            "displays",
        }):
            return (
                "What does the IT policy say about monitors, "
                "screens or displays provided to users?"
            )

        if words.intersection({
            "take",
            "takes",
            "get",
            "gets",
            "receive",
            "receives",
            "eligible",
            "eligibility",
            "entitled",
        }):
            return (
                "Who is eligible to receive or be assigned a laptop "
                "according to the end-user IT devices policy?"
            )

        return (
            "What does the IT policy say about laptops "
            "and end-user IT devices?"
        )

    if topic == "password":
        return (
            "What are the password requirements "
            "in the IT policy?"
        )

    if topic == "remote_access":
        return (
            "What are the IT policy requirements "
            "for remote access?"
        )

    return normalized


def is_identity_question(question):
    """
    Detect questions asking what the assistant is.
    """

    text = normalize_user_text(
        question
    )

    identity_phrases = [
        "who are you",
        "what are you",
        "what is this",
        "what do you do",
        "what can you do",
        "tell me about yourself",
    ]

    return any(
        phrase in text
        for phrase in identity_phrases
    )


def retrieve_documents(
    question,
    top_k=8
):
    """
    Retrieve relevant policy chunks.
    """

    normalized_question = normalize_user_text(
        question
    )

    original_words = tokenize(
        normalized_question
    )

    expanded_question = expand_query(
        normalized_question
    )

    # Short questions use the stronger expanded query only
    if (
        len(original_words) <= 4
        and expanded_question != normalized_question
    ):
        search_queries = [
            expanded_question
        ]

    else:
        search_queries = [
            normalized_question
        ]

        if expanded_question != normalized_question:
            search_queries.append(
                expanded_question
            )

    combined_results = []
    seen = set()

    for search_query in search_queries:

        query_vector = embed_query(
            search_query
        )

        results = search(
            query_vector,
            top_k=top_k
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            page = metadata.get(
                "page",
                "Unknown"
            )

            section = metadata.get(
                "section",
                "Unknown"
            )

            key = (
                document,
                page,
                section
            )

            if key in seen:
                continue

            seen.add(key)

            combined_results.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

    combined_results.sort(
        key=lambda item: (
            item["distance"]
            if item["distance"] is not None
            else 999
        )
    )

    return combined_results


def is_noisy_candidate(candidate):
    """
    Detect short / TOC / OCR-noise chunks.
    """

    text = candidate[
        "document"
    ].strip()

    if len(text) < 120:
        return True

    if re.search(
        r"\.{4,}",
        text
    ):
        return True

    if re.search(
        r"(.)\1{5,}",
        text
    ):
        return True

    return False


def topic_score(
    topic,
    candidate
):
    """
    Score how strongly a chunk matches the detected topic.
    """

    if topic is None:
        return 0

    metadata = candidate[
        "metadata"
    ]

    text = (
        candidate["document"]
        + " "
        + str(
            metadata.get(
                "section",
                ""
            )
        )
    ).lower()

    return sum(
        1
        for anchor in TOPIC_ANCHORS[
            topic
        ]
        if anchor in text
    )


def is_relevant(
    question,
    candidates
):
    """
    Reject clearly unrelated questions before Ollama.
    """

    if not candidates:
        return False

    topic = detect_topic(
        question
    )

    if topic is not None:

        for candidate in candidates[:8]:

            if is_noisy_candidate(
                candidate
            ):
                continue

            if topic_score(
                topic,
                candidate
            ) >= 2:
                return True

    expanded_question = expand_query(
        question
    )

    question_words = tokenize(
        expanded_question
    )

    if not question_words:
        return False

    for candidate in candidates[:6]:

        if is_noisy_candidate(
            candidate
        ):
            continue

        metadata = candidate[
            "metadata"
        ]

        searchable_text = (
            candidate["document"]
            + " "
            + str(
                metadata.get(
                    "section",
                    ""
                )
            )
        )

        document_words = tokenize(
            searchable_text
        )

        overlap = question_words.intersection(
            document_words
        )

        distance = candidate.get(
            "distance"
        )

        if len(overlap) >= 2:
            return True

        if (
            len(overlap) >= 1
            and distance is not None
            and distance <= 1.35
        ):
            return True

    best_distance = candidates[0].get(
        "distance"
    )

    if (
        best_distance is not None
        and best_distance <= 0.90
    ):
        return True

    return False


def select_context_candidates(
    question,
    candidates,
    max_chunks=2
):
    """
    Select the strongest clean chunks for Ollama.
    """

    if not candidates:
        return []

    topic = detect_topic(
        question
    )

    question_words = tokenize(
        expand_query(
            question
        )
    )

    scored = []

    for candidate in candidates:

        if is_noisy_candidate(
            candidate
        ):
            continue

        metadata = candidate[
            "metadata"
        ]

        searchable_text = (
            candidate["document"]
            + " "
            + str(
                metadata.get(
                    "section",
                    ""
                )
            )
        )

        document_words = tokenize(
            searchable_text
        )

        overlap = len(
            question_words.intersection(
                document_words
            )
        )

        candidate_topic_score = topic_score(
            topic,
            candidate
        )

        distance = candidate.get(
            "distance"
        )

        if distance is None:
            distance = 999

        scored.append(
            (
                -candidate_topic_score,
                -overlap,
                distance,
                candidate,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2]
        )
    )

    if not scored:
        return candidates[:max_chunks]

    best = scored[0][3]

    selected = [
        best
    ]

    best_section = str(
        best["metadata"].get(
            "section",
            ""
        )
    )

    # Prefer chunks from the same policy section
    for _, _, _, candidate in scored[1:]:

        if len(selected) >= max_chunks:
            break

        candidate_section = str(
            candidate["metadata"].get(
                "section",
                ""
            )
        )

        if candidate_section == best_section:
            selected.append(
                candidate
            )

    # Fill remaining slots if necessary
    if len(selected) < max_chunks:

        for _, _, _, candidate in scored[1:]:

            if len(selected) >= max_chunks:
                break

            if candidate not in selected:
                selected.append(
                    candidate
                )

    return selected


def build_context(
    candidates
):
    """
    Build the policy context sent to Ollama.
    """

    context_parts = []

    for candidate in candidates:

        document = candidate[
            "document"
        ]

        metadata = candidate[
            "metadata"
        ]

        page = metadata.get(
            "page",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE:
Page: {page}
Section: {section}

CONTENT:
{document}
"""
        )

    return "\n\n---\n\n".join(
        context_parts
    )


def get_sources(
    candidates
):
    """
    Create a unique source list.
    """

    sources = []

    for candidate in candidates:

        metadata = candidate[
            "metadata"
        ]

        page = metadata.get(
            "page",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "Unknown"
        )

        source = (
            f"Page {page}, "
            f"Section {section}"
        )

        if source not in sources:
            sources.append(
                source
            )

    return sources


def answer_means_not_covered(answer):
    """
    Detect when Ollama is effectively saying
    that the policy does not answer the question.
    """

    text = answer.lower().strip()

    not_covered_phrases = [
        "there is no information",
        "there is no information in the policy context",
        "there are no it policy requirements",
        "no information in the policy",
        "no information in the policy context",
        "no it policy requirements",
        "the policy does not specify",
        "the provided context does not specify",
        "the context does not specify",
        "not specified in the policy",
        "does not provide information",
        "does not contain information",
    ]

    return any(
        phrase in text
        for phrase in not_covered_phrases
    )


def asks_device_eligibility(question):
    """
    Detect questions asking who can receive/get/take a device.
    """

    words = tokenize(
        question
    )

    return (
        detect_topic(question) == "end_user_devices"
        and bool(
            words.intersection(
                DEVICE_ASSIGNMENT_WORDS
            )
        )
    )


def get_direct_policy_answer(
    question,
    candidates
):
    """
    Handle explicit policy rules that the small LLM
    occasionally misinterprets.

    This only activates when the retrieved policy text
    explicitly contains the rule.
    """

    words = tokenize(
        question
    )

    if detect_topic(
        question
    ) != "procurement":
        return None

    context_text = " ".join(
        candidate["document"].lower()
        for candidate in candidates
    )

    # Software purchase rule
    if (
        "software" in words
        and "hardware and software cannot be purchased" in context_text
        and "procurement authorization" in context_text
    ):
        return (
            "Software cannot be purchased without a completed "
            "IT Procurement Authorization. The authorization must be "
            "approved by the user's line manager, and the IT Operations "
            "Manager decides whether to approve, decline, or amend the "
            "purchase requirements."
        )

    return None


def answer_single_question(
    question,
    top_k=8
):
    """
    Answer one policy question.
    """

    candidates = retrieve_documents(
        question,
        top_k=top_k
    )

    if not is_relevant(
        question,
        candidates
    ):
        return NOT_COVERED

    context_candidates = (
        select_context_candidates(
            question,
            candidates,
            max_chunks=2
        )
    )

    # Do not confuse policy scope with device eligibility
    if asks_device_eligibility(
        question
    ):

        eligibility_text = " ".join(
            candidate["document"].lower()
            for candidate in context_candidates
        )

        explicit_eligibility_phrases = [
            "eligible to receive",
            "entitled to receive",
            "entitled to a laptop",
            "assigned a laptop",
            "shall receive a laptop",
            "will receive a laptop",
            "provided with a laptop",
        ]

        if not any(
            phrase in eligibility_text
            for phrase in explicit_eligibility_phrases
        ):
            return NOT_COVERED

    # Grounded direct answer for explicit rules
    # that the 3B model has previously misinterpreted
    direct_answer = get_direct_policy_answer(
        question,
        context_candidates
    )

    if direct_answer:

        sources = get_sources(
            context_candidates
        )

        source_text = "\n".join(
            f"- {source}"
            for source in sources
        )

        return (
            f"{direct_answer}\n\n"
            f"**Source:**\n"
            f"{source_text}"
        )

    context = build_context(
        context_candidates
    )

    generation_question = (
        normalize_question_for_generation(
            question
        )
    )

    user_prompt = f"""
IT POLICY CONTEXT:

{context}

QUESTION:

{generation_question}

Using only the policy context above, answer the question directly.
"""

    response = client.chat(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        keep_alive="10m",

        options={
            "num_ctx": 2048,
            "temperature": 0.0,
        },
    )

    answer = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    if answer_means_not_covered(
        answer
    ):
        return NOT_COVERED

    sources = get_sources(
        context_candidates
    )

    if sources:

        source_text = "\n".join(
            f"- {source}"
            for source in sources
        )

        return (
            f"{answer}\n\n"
            f"**Source:**\n"
            f"{source_text}"
        )

    return answer


def answer_one(
    question
):
    """
    Handle one question.
    """

    if is_identity_question(
        question
    ):
        return (
            "I am the IT Policy Assistant. "
            "I help users find and understand information "
            "from the company's IT policy."
        )

    return answer_single_question(
        question
    )


def answer_question(
    question
):
    """
    Handle one or multiple questions.
    """

    questions = split_questions(
        question
    )

    if len(questions) == 1:

        return answer_one(
            questions[0]
        )

    answers = []

    for index, individual_question in enumerate(
        questions,
        start=1
    ):

        answer = answer_one(
            individual_question
        )

        answers.append(
            f"### Question {index}\n"
            f"**{individual_question}**\n\n"
            f"{answer}"
        )

    return "\n\n".join(
        answers
    )


if __name__ == "__main__":

    print(
        "Welcome to the IT Policy Assistant."
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        question = input(
            "Ask your question: "
        ).strip()

        # Ignore empty Enter presses
        if not question:
            continue

        if question.lower() == "exit":
            break

        answer = answer_question(
            question
        )

        print("\n")
        print(answer)

        print(
            "\n"
            + "=" * 60
            + "\n"
        )