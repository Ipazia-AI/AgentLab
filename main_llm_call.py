from agentlab.llm import tracking
from agentlab.llm.response_api import APIPayload, OpenAIResponseModelArgs


def main():
    args = OpenAIResponseModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    model = args.make_model()
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What the capital of France?")]

    with tracking.set_tracker() as tracker:
        payload = APIPayload(messages=messages)
        parsed_output = model(payload)

        print(tracker.stats)
        print(parsed_output)


if __name__ == "__main__":  # necessary for dask backend
    main()
