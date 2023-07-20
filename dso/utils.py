import json


def LoadJSON(fn):
    with open(fn, "r") as f:
        return json.load(f)


def WriteJSON(fn, d):
    with open(fn, "w") as f:
        json.dump(d, f)