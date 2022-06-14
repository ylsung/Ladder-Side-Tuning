import json
import sys

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}

def str2bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    return None

json_file = sys.argv[1]

attribute = sys.argv[2]
attribute_type = sys.argv[3]
attribute_value = sys.argv[4]

attribute_value = eval(attribute_type)(attribute_value)

with open(json_file, "r") as f:
    d = json.load(f)


d[attribute] = attribute_value

with open(json_file, "w") as f:
    json.dump(d, f, indent=0)