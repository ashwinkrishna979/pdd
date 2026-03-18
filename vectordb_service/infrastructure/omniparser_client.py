import replicate


def parse_frame(
    image_path: str, model_version: str, client: replicate.Client
) -> dict:
    """Run OmniParser on a frame image via Replicate."""
    with open(image_path, "rb") as f:
        output = client.run(model_version, input={"image": f})
    return output
