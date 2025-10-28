import json
from pathlib import Path

def strip_images_text(raw: str) -> str:
    s = raw
    i = 0
    out = []
    in_string = False
    escape = False
    n = len(s)

    while i < n:
        ch = s[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            # possible start of a key
            # check if this starts with "images"
            if s.startswith('"images"', i):
                # Start position of the property in the output buffer
                prop_start_in_out = len(out)
                # Skip over "images" key and colon and whitespace
                k = i + len('"images"')
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == ':':
                    k += 1
                while k < n and s[k].isspace():
                    k += 1
                # Expect array
                if k < n and s[k] == '[':
                    # scan to matching ']'
                    depth = 0
                    in_str2 = False
                    esc2 = False
                    while k < n:
                        c2 = s[k]
                        if in_str2:
                            if esc2:
                                esc2 = False
                            elif c2 == '\\':
                                esc2 = True
                            elif c2 == '"':
                                in_str2 = False
                        else:
                            if c2 == '"':
                                in_str2 = True
                            elif c2 == '[':
                                depth += 1
                            elif c2 == ']':
                                depth -= 1
                                if depth == 0:
                                    k += 1
                                    break
                        k += 1
                    # Skip whitespace and a trailing comma if present (removes separator after images)
                    while k < n and s[k].isspace():
                        k += 1
                    if k < n and s[k] == ',':
                        k += 1
                        while k < n and s[k].isspace():
                            k += 1
                    # Drop the property from output buffer
                    out = out[:prop_start_in_out]
                    i = k
                    continue
            # normal string start
            in_string = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return ''.join(out)


def extract_chunk_objects(raw: str) -> list:
    key = '"chunks"'
    i = raw.find(key)
    if i == -1:
        raise ValueError('"chunks" key not found')
    j = raw.find('[', i)
    if j == -1:
        raise ValueError('Opening [ for chunks not found')
    depth = 0
    in_str = False
    esc = False
    k = j
    while k < len(raw):
        ch = raw[k]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    break
        k += 1
    if depth != 0:
        # Fallback: take until end of file
        array_text = raw[j+1:]
    else:
        array_text = raw[j+1:k]

    objs = []
    in_str = False
    esc = False
    brace = 0
    start = None
    idx = 0
    while idx < len(array_text):
        ch = array_text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if brace == 0:
                    start = idx
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0 and start is not None:
                    objs.append(array_text[start:idx+1])
                    start = None
        idx += 1
    return objs


def main():
    target = Path(__file__).with_name("out.json")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")

    text = target.read_text(encoding="utf-8")
    obj_texts = extract_chunk_objects(text)

    cleaned = []
    for otext in obj_texts:
        oclean = strip_images_text(otext)
        try:
            obj = json.loads(oclean)
        except Exception:
            try:
                obj = json.loads(otext)
                if isinstance(obj, dict) and 'images' in obj:
                    obj.pop('images', None)
            except Exception:
                continue
        cleaned.append(obj)

    target.write_text(json.dumps({"chunks": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


