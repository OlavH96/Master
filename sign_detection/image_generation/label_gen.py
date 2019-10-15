import os

def get_labels():
    SIGN_PATH = 'images/signs'
    names = os.listdir(SIGN_PATH)
    return sorted([path for path in names if os.path.isdir(os.path.join(SIGN_PATH, path))])


def gen_labelmap():
    template = \
"""item {
    id: %r
    name: '%s'
}"""
    content = '\n\n'.join([template % (i+1, name) for i, name in enumerate(get_labels())])
    with open('labelmap.pbtxt', 'w') as f:
        f.write(content + '\n')


if __name__ == '__main__':
    gen_labelmap()
