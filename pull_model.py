import sys
import time

from rag import client, MODEL_NAME


# How long to wait for the Ollama server to start
WAIT_TIMEOUT_SECONDS = 120


def wait_for_ollama(timeout=WAIT_TIMEOUT_SECONDS):
    """
    Wait until the Ollama server accepts requests.

    docker-compose depends_on only waits for the container
    to start, not for the server inside it to be ready.
    """

    deadline = time.time() + timeout

    while True:

        try:
            client.list()
            return True

        except Exception:

            if time.time() >= deadline:
                return False

            print("Waiting for Ollama...")
            time.sleep(2)


def model_exists(model):

    # Ollama stores untagged models as "<name>:latest"
    names = {model}

    if ":" not in model:
        names.add(f"{model}:latest")

    installed = {
        item.model
        for item in client.list().models
    }

    return bool(names & installed)


def pull_model_if_missing(model=MODEL_NAME):
    """
    Download the Ollama model only when it is not already present.
    """

    if not wait_for_ollama():

        print(
            "WARNING: Ollama is not reachable. "
            f"Could not check model: {model}"
        )

        return False

    if model_exists(model):

        print(f"Ollama model already available: {model}")

        return True

    print(f"Pulling Ollama model: {model}")

    last_status = None

    for progress in client.pull(model, stream=True):

        status = progress.get("status")

        # Only print when the status changes to keep logs short
        if status != last_status:
            print(status)
            last_status = status

    print(f"Ollama model ready: {model}")

    return True


if __name__ == "__main__":

    if not pull_model_if_missing():
        sys.exit(1)
