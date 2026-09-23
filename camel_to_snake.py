import re


def camel_to_snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def convert_keys(data):
    if isinstance(data, dict):
        return {
            camel_to_snake(key): convert_keys(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [convert_keys(item) for item in data]

    return data