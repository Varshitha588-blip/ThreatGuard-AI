
from datasets import load_dataset

DATASET_NAME = "ansulev/phishing-email-dataset"


def main():
    print("Downloading/loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nAvailable splits:")
    print(dataset)

    data = dataset["train"]

    print("\nColumn names:")
    print(data.column_names)

    print("\nFirst two examples:")
    for example in data.select(range(min(2, len(data)))):
        for key, value in example.items():
            if isinstance(value, str):
                print(f"{key}: {value[:300]!r}")
            else:
                print(f"{key}: {value}")

    print("\nLabel counts:")
    label_column = "Email Type"

    if label_column in data.column_names:
        counts = {}
        for label in data[label_column]:
            counts[str(label)] = counts.get(str(label), 0) + 1
        print(counts)
    else:
        print("Inspect the printed columns to identify the label field.")


if __name__ == "__main__":
    main()
