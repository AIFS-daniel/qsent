def humanize_name(test_name: str) -> str:
    without_prefix = test_name.removeprefix("test_")
    sentence = without_prefix.replace("_", " ")
    return sentence.capitalize()
