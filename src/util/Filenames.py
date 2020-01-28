import hashlib


def md5hash(v: str) -> str:
    return hashlib.md5(remove_path(v).strip().encode()).hexdigest()


def remove_path(name: str) -> str:
    return name.split('/')[-1]


def extract_model_name(name: str) -> str:
    s = name.split('.')[0].split('_')
    return '_'.join(s[1:])


def extract_score(pred: str) -> float:
    split = pred.split('_')
    n = split[-1].split('.')
    return float('.'.join(n[:2]))


def extract_hash(orig_name: str) -> str:
    return orig_name.split('_')[-2]
