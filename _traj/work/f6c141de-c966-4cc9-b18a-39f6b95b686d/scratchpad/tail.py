if __name__ == "__main__":
    modes = [("default-1", OLD), ("max-ilp-1", NEW), ("default-2", OLD), ("max-ilp-2", NEW)]
    only = os.environ.get("AB_ONLY")
    texts = {label: run(label, exe) for label, exe in modes if not only or label in only.split(",")}
    labels = list(texts)
    for a in labels[1:]:
        print(f"texts: {labels[0]} == {a}:", texts[labels[0]] == texts[a], flush=True)
