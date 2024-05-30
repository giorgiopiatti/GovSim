import uuid

import randomname


def random_name():
    name = randomname.get_name()
    name += "-" + str(uuid.uuid4())[:4]
    return name


if __name__ == "__main__":
    print(random_name())
